// ggml-cuda8-set-rows-smoke.cu  -  G43: SET_ROWS standalone smoke test
//
// Cases:
//   1. KV-cache shape        - 2D scatter, the form llama-kv-cache.cpp emits
//   2. out-of-order indices  - scatter, not a contiguous block copy
//   3. padded dst rows       - dst row stride > nc
//   4. 3D with index broadcast - ne02 > 1, ne11 == 1
//   5. out-of-range index    - must be skipped, not written out of bounds
//
// Case 5 matters because the CPU path asserts on a bad index and a kernel
// cannot; the kernel skips instead, so this checks it does not corrupt
// neighbouring memory.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <stdint.h>

extern "C" int ggml_cuda8_op_set_rows_f32(
        const void * src0, const void * idx, void * dst,
        int nc, int nr, int ne02, int ne03,
        int ne11, int ne12, int ne1,
        size_t nb01, size_t nb02, size_t nb03,
        size_t nb10, size_t nb11, size_t nb12,
        size_t nb1,  size_t nb2,  size_t nb3);

#define CHECK_CUDA(x) do { \
    cudaError_t err_ = (x); \
    if (err_ != cudaSuccess) { \
        fprintf(stderr, "CUDA %s:%d  %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err_)); exit(1); } \
} while (0)

// Reference: ggml_compute_forward_set_rows_impl, F32 -> F32.
static void set_rows_ref(const float * src0, const int64_t * idx, float * dst,
                         int nc, int nr, int ne02, int ne03,
                         int ne11, int ne12, int ne1,
                         int s01, int s02, int s03,
                         int d1,  int d2,  int d3) {
    for (int i03 = 0; i03 < ne03; i03++) {
        for (int i02 = 0; i02 < ne02; i02++) {
            for (int i = 0; i < nr; i++) {
                const int i12 = i03 % ne12;
                const int i11 = i02 % ne11;
                const int64_t i1 = idx[(size_t) i + (size_t) i11 * nr + (size_t) i12 * nr * ne11];
                if (i1 < 0 || i1 >= ne1) continue;
                const float * s = src0 + (size_t) i  * s01 + (size_t) i02 * s02 + (size_t) i03 * s03;
                float * d       = dst  + (size_t) i1 * d1  + (size_t) i02 * d2  + (size_t) i03 * d3;
                for (int c = 0; c < nc; c++) d[c] = s[c];
            }
        }
    }
}

static int run_case(const char * label,
                    int nc, int nr, int ne02, int ne03, int ne11, int ne12,
                    int ne1, int d1_pad,
                    const int64_t * h_idx_in) {
    const int s01 = nc, s02 = nc * nr, s03 = nc * nr * ne02;
    const int d1  = nc + d1_pad;
    const int d2  = d1 * ne1, d3 = d2 * ne02;

    const size_t src_n = (size_t) nr * ne02 * ne03 * nc;
    const size_t dst_n = (size_t) d3 * ne03;
    const size_t idx_n = (size_t) nr * ne11 * ne12;

    float   * h_src = (float *)   malloc(src_n * sizeof(float));
    int64_t * h_idx = (int64_t *) malloc(idx_n * sizeof(int64_t));
    float   * h_out = (float *)   malloc(dst_n * sizeof(float));
    float   * h_ref = (float *)   malloc(dst_n * sizeof(float));

    for (size_t i = 0; i < src_n; i++) h_src[i] = (float) (i % 997) * 0.125f + 1.0f;
    memcpy(h_idx, h_idx_in, idx_n * sizeof(int64_t));

    // Poison both, so untouched destination rows must stay poisoned.
    for (size_t i = 0; i < dst_n; i++) { h_ref[i] = -12345.0f; h_out[i] = -12345.0f; }
    set_rows_ref(h_src, h_idx, h_ref, nc, nr, ne02, ne03, ne11, ne12, ne1,
                 s01, s02, s03, d1, d2, d3);

    float   * d_src = NULL; int64_t * d_idx = NULL; float * d_dst = NULL;
    CHECK_CUDA(cudaMalloc(&d_src, src_n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_idx, idx_n * sizeof(int64_t)));
    CHECK_CUDA(cudaMalloc(&d_dst, dst_n * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_src, h_src, src_n * sizeof(float),   cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_idx, h_idx, idx_n * sizeof(int64_t), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_dst, h_out, dst_n * sizeof(float),   cudaMemcpyHostToDevice));

    const size_t f = sizeof(float);
    int rc = ggml_cuda8_op_set_rows_f32(
        d_src, d_idx, d_dst,
        nc, nr, ne02, ne03, ne11, ne12, ne1,
        s01 * f, s02 * f, s03 * f,
        sizeof(int64_t), (size_t) nr * sizeof(int64_t), (size_t) nr * ne11 * sizeof(int64_t),
        d1 * f, d2 * f, d3 * f);

    int pass = 0;
    if (rc != 0) {
        printf("  %-30s launch FAILED rc=%d\n", label, rc);
    } else {
        CHECK_CUDA(cudaMemcpy(h_out, d_dst, dst_n * sizeof(float), cudaMemcpyDeviceToHost));
        size_t bad = (size_t) -1;
        for (size_t i = 0; i < dst_n; i++) {
            if (h_out[i] != h_ref[i]) { bad = i; break; }
        }
        pass = (bad == (size_t) -1);
        if (pass) printf("  %-30s PASS\n", label);
        else      printf("  %-30s FAIL at %zu: got %g want %g\n",
                         label, bad, (double) h_out[bad], (double) h_ref[bad]);
    }

    cudaFree(d_src); cudaFree(d_idx); cudaFree(d_dst);
    free(h_src); free(h_idx); free(h_out); free(h_ref);
    return pass;
}

int main() {
    printf("ggml-cuda8-set-rows-smoke: starting\n");
    int ok = 1;

    // 1. KV-cache shape: 8 tokens scattered into a 64-row cache.
    {
        int64_t idx[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };
        ok &= run_case("kv-cache 2D, sequential", 128, 8, 1, 1, 1, 1, 64, 0, idx);
    }

    // 2. Out-of-order: a real cache does not write contiguous rows.
    {
        int64_t idx[8] = { 40, 3, 17, 63, 0, 9, 31, 22 };
        ok &= run_case("scattered indices", 128, 8, 1, 1, 1, 1, 64, 0, idx);
    }

    // 3. Padded destination rows.
    {
        int64_t idx[8] = { 12, 5, 44, 1, 60, 33, 7, 19 };
        ok &= run_case("padded dst rows", 128, 8, 1, 1, 1, 1, 64, 5, idx);
    }

    // 4. 3D with index broadcast over dim 2 (ne11 == 1).
    {
        int64_t idx[4] = { 6, 2, 9, 0 };
        ok &= run_case("3D, idx broadcast ne02=3", 32, 4, 3, 1, 1, 1, 16, 0, idx);
    }

    // 5. Out-of-range index must be skipped, not written.
    {
        int64_t idx[8] = { 0, 1, 999, 3, -4, 5, 6, 7 };
        ok &= run_case("out-of-range idx skipped", 128, 8, 1, 1, 1, 1, 64, 0, idx);
    }

    printf("ggml-cuda8-set-rows-smoke: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
