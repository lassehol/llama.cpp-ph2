// ggml-cuda8-cont-smoke.cu
//
// G55 regression guard: the permuted-src0 CONT smoke that would have caught
// the flat-copy bug.
//
// CONT exists ONLY to make a NON-contiguous tensor contiguous (post-permute,
// in attention). The original kernel flat-copied src0->data as one contiguous
// byte run, ignoring src0->nb[], and silently scrambled every permuted input.
// The existing CONT smoke feeds contiguous data, where a flat copy happens to
// be correct, so it never caught it. This smoke feeds DELIBERATELY
// non-contiguous (permuted) layouts and checks the strided-gather kernel reads
// src0 through its real byte strides.
//
// Standalone .cu (compiled with ggml-cuda8-cont.cu, no ggml headers): the
// launcher is declared extern "C" and all layouts are built by hand as plain
// float arrays, so nvcc parses it cleanly and no ggml_tensor struct is needed.
//
// Model of CONT: dst is packed/contiguous with dst shape == src0's (permuted)
// ne[]. For each logical index (i0,i1,i2,i3), the element lives in physical
// memory at byte offset i0*nb0 + i1*nb1 + i2*nb2 + i3*nb3; the kernel gathers
// it into the packed dst position. A flat copy (the bug) would instead read
// physical memory in order, producing the wrong element at every position
// where nb[] != the packed strides of ne[].
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

// Kernel launcher under test (extern "C" in the kernels archive / cont.cu).
extern "C" int ggml_cuda8_cont_f32_launch(
        const void * src0, float * dst,
        int ne0, int ne1, int ne2, int ne3,
        size_t nb0, size_t nb1, size_t nb2, size_t nb3);

// CPU reference: strided gather into a packed dst, in the exact ggml element
// order (dim 0 fastest). This is deliberately written independently of the
// kernel - agreement between the two is the check.
static void cpu_cont_reference(
        const std::vector<float> & phys,   // physical backing store, floats
        std::vector<float> & dst,          // packed output, ne0*ne1*ne2*ne3
        int ne0, int ne1, int ne2, int ne3,
        size_t nb0, size_t nb1, size_t nb2, size_t nb3) {  // strides in BYTES
    dst.assign((size_t) ne0 * ne1 * ne2 * ne3, 0.0f);
    size_t out = 0;
    for (int i3 = 0; i3 < ne3; ++i3) {
        for (int i2 = 0; i2 < ne2; ++i2) {
            for (int i1 = 0; i1 < ne1; ++i1) {
                for (int i0 = 0; i0 < ne0; ++i0) {
                    const size_t byte_off =
                        (size_t) i0 * nb0 + (size_t) i1 * nb1 +
                        (size_t) i2 * nb2 + (size_t) i3 * nb3;
                    dst[out++] = phys[byte_off / sizeof(float)];
                }
            }
        }
    }
}

struct Case {
    const char * label;
    int ne0, ne1, ne2, ne3;                 // logical (post-permute) shape
    size_t nb0, nb1, nb2, nb3;              // strides into phys, in BYTES
    size_t phys_floats;                     // size of the physical backing store
};

static bool run_case(const Case & c) {
    const size_t n_out = (size_t) c.ne0 * c.ne1 * c.ne2 * c.ne3;

    // Physical backing store, deterministic distinct values so any wrong-order
    // read shows up as a mismatch, not a coincidental equal value.
    std::vector<float> phys(c.phys_floats);
    for (size_t i = 0; i < phys.size(); ++i) {
        phys[i] = (float) i * 0.5f - 3.0f;
    }

    std::vector<float> ref;
    cpu_cont_reference(phys, ref, c.ne0, c.ne1, c.ne2, c.ne3,
                       c.nb0, c.nb1, c.nb2, c.nb3);

    float * d_src = NULL;
    float * d_dst = NULL;
    cudaMalloc(&d_src, phys.size() * sizeof(float));
    cudaMalloc(&d_dst, n_out       * sizeof(float));
    cudaMemcpy(d_src, &phys[0], phys.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_dst, 0xFF, n_out * sizeof(float));   // poison: unwritten slots show up

    const int rc = ggml_cuda8_cont_f32_launch(
        d_src, d_dst,
        c.ne0, c.ne1, c.ne2, c.ne3,
        c.nb0, c.nb1, c.nb2, c.nb3);

    bool ok = true;
    if (rc != 0) {
        std::printf("  %-40s FAIL (launcher returned %d)\n", c.label, rc);
        ok = false;
    } else {
        std::vector<float> got(n_out);
        cudaMemcpy(&got[0], d_dst, n_out * sizeof(float), cudaMemcpyDeviceToHost);
        long bad = -1;
        for (size_t i = 0; i < n_out; ++i) {
            // CONT is a pure copy - values must match exactly, no tolerance.
            if (got[i] != ref[i]) { bad = (long) i; break; }
        }
        if (bad >= 0) {
            std::printf("  %-40s FAIL at %ld: got=%.6g want=%.6g\n",
                        c.label, bad, (double) got[bad], (double) ref[bad]);
            ok = false;
        } else {
            std::printf("  %-40s PASS (%zu elems, exact)\n", c.label, n_out);
        }
    }

    cudaFree(d_src);
    cudaFree(d_dst);
    return ok;
}

// Helper: strides (bytes) for a CONTIGUOUS tensor of the given shape.
static void contig_strides(int ne0, int ne1, int ne2, int /*ne3*/,
                           size_t & nb0, size_t & nb1, size_t & nb2, size_t & nb3) {
    nb0 = sizeof(float);
    nb1 = nb0 * ne0;
    nb2 = nb1 * ne1;
    nb3 = nb2 * ne2;
}

int main() {
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) {
        std::fprintf(stderr, "no CUDA device 0\n");
        return 1;
    }
    std::printf("ggml-cuda8-cont-smoke: %s (cc %d.%d)\n\n", prop.name, prop.major, prop.minor);

    bool ok = true;
    std::vector<Case> cases;

    // ---- Case 1: contiguous (regression check the fix didn't break it) ----
    // nb[] equals the packed strides of ne[], so gather == flat copy here.
    {
        Case c; c.label = "contiguous [8,4,2,1] (control)";
        c.ne0 = 8; c.ne1 = 4; c.ne2 = 2; c.ne3 = 1;
        contig_strides(c.ne0, c.ne1, c.ne2, c.ne3, c.nb0, c.nb1, c.nb2, c.nb3);
        c.phys_floats = (size_t) c.ne0 * c.ne1 * c.ne2 * c.ne3;
        cases.push_back(c);
    }

    // ---- Case 2: permute dims 1<->2 of a contiguous [d0,d1,d2,d3] ----
    // Original contiguous shape [d0=8, d1=4, d2=3, d3=1], physical strides:
    //   nb_orig = [4, 8*4, 8*4*3? ...] -> compute from d0,d1,d2.
    // Permuted logical shape = [d0, d2, d1, d3]; nb reordered to match.
    // This is the canonical attention permute (reshape+permute of Q/K/V):
    // nb[1] now jumps by the ORIGINAL dim-2 stride, so it is non-contiguous.
    {
        const int d0 = 8, d1 = 4, d2 = 3, d3 = 1;
        const size_t o0 = sizeof(float);
        const size_t o1 = o0 * d0;
        const size_t o2 = o1 * d1;
        const size_t o3 = o2 * d2;
        Case c; c.label = "permute(1,2) [8,4,3,1]->[8,3,4,1]";
        c.ne0 = d0; c.ne1 = d2; c.ne2 = d1; c.ne3 = d3;   // logical permuted shape
        c.nb0 = o0; c.nb1 = o2; c.nb2 = o1; c.nb3 = o3;     // strides follow the swap
        c.phys_floats = (size_t) d0 * d1 * d2 * d3;
        cases.push_back(c);
    }

    // ---- Case 3: permute dims 0<->1 (dim-0 itself becomes strided) ----
    // Exercises a non-unit nb0 - the case a flat copy gets most wrong, since
    // even the fastest-varying dimension no longer steps by one float.
    {
        const int d0 = 6, d1 = 5, d2 = 2, d3 = 1;
        const size_t o0 = sizeof(float);
        const size_t o1 = o0 * d0;
        const size_t o2 = o1 * d1;
        const size_t o3 = o2 * d2;
        Case c; c.label = "permute(0,1) [6,5,2,1]->[5,6,2,1]";
        c.ne0 = d1; c.ne1 = d0; c.ne2 = d2; c.ne3 = d3;
        c.nb0 = o1; c.nb1 = o0; c.nb2 = o2; c.nb3 = o3;     // nb0 != sizeof(float)
        c.phys_floats = (size_t) d0 * d1 * d2 * d3;
        cases.push_back(c);
    }

    // ---- Case 4: larger, attention-shaped permute with batch dim ----
    // [head_dim=16, n_head=8, n_tok=4] contiguous, permuted to
    // [head_dim, n_tok, n_head] (the shape CONT actually sees before probs.V).
    {
        const int hd = 16, nh = 8, nt = 4, nb_ = 1;
        const size_t o0 = sizeof(float);
        const size_t o1 = o0 * hd;   // n_head stride
        const size_t o2 = o1 * nh;   // n_tok stride
        const size_t o3 = o2 * nt;   // batch stride
        Case c; c.label = "attn permute [16,8,4,1]->[16,4,8,1]";
        c.ne0 = hd; c.ne1 = nt; c.ne2 = nh; c.ne3 = nb_;
        c.nb0 = o0; c.nb1 = o2; c.nb2 = o1; c.nb3 = o3;
        c.phys_floats = (size_t) hd * nh * nt * nb_;
        cases.push_back(c);
    }

    std::printf("== CONT strided-gather (permuted src0) ==\n");
    for (size_t i = 0; i < cases.size(); ++i) {
        ok &= run_case(cases[i]);
    }

    std::printf("\nggml-cuda8-cont-smoke: %s\n",
                ok ? "SUCCESS (all cases exact)" : "FAIL (a case mismatched)");
    return ok ? 0 : 1;
}
