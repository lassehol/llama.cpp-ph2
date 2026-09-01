// ggml-cuda8-mulmat-f32-smoke.cpp
//
// G42: standalone smoke test for the batched F32xF32 MUL_MAT kernel
// (ggml_cuda8_mul_mat_f32_f32_launch) - the attention matmuls (K.Q,
// probs.V). Verified the same way G40/G45/G41 were: an independently
// re-derived CPU reference, checked against the kernel, rather than the
// kernel checked against itself.
//
// Compiled as plain C++ (.cpp), NOT .cu, for the same reason
// ggml-cuda8-softmax-ext-smoke.cpp is .cpp: it needs the full
// ggml_tensor/ggml_cuda8_context struct definitions (ggml.h /
// ggml-cuda8-context.h) to build the supported()-rejection test cases,
// and those headers do not parse cleanly under nvcc's front end. This
// file contains no device code - only host-side CUDA runtime calls and a
// call into the extern "C" launcher already compiled into
// ggml-cuda8-kernels by nvcc.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include "ggml.h"
#include "ggml-cuda8-context.h"
#include "ggml-cuda8-mulmat-f32.h"

extern "C" int ggml_cuda8_mul_mat_f32_f32_launch(
    const float * src0,
    const float * src1,
    float * dst,
    int ne00,
    int ne01, int ne02, int ne03,
    int ne11, int ne12, int ne13,
    size_t nb01, size_t nb02, size_t nb03,
    size_t nb11, size_t nb12, size_t nb13,
    size_t nb1,  size_t nb2,  size_t nb3);

static bool approx(float a, float b, float tol) {
    return std::fabs(a - b) <= tol * (1.0f + std::fabs(b));
}

// Independent CPU reference. Takes strides in element units (not bytes) for
// host-side simplicity; the launcher itself takes byte strides, converted
// at the call site. Deliberately re-derives the batched/broadcast indexing
// from scratch rather than sharing any code with the kernel.
static void cpu_mulmat_f32_reference(
        const std::vector<float> & src0, size_t nb01e, size_t nb02e, size_t nb03e,
        const std::vector<float> & src1, size_t nb11e, size_t nb12e, size_t nb13e,
        std::vector<float> & dst,        size_t nb1e,  size_t nb2e,  size_t nb3e,
        int ne00, int ne01, int ne02, int ne03,
        int ne11, int ne12, int ne13) {
    const int r2 = ne12 / ne02;
    const int r3 = ne13 / ne03;
    for (int i13 = 0; i13 < ne13; ++i13) {
        const int i03 = i13 / r3;
        for (int i12 = 0; i12 < ne12; ++i12) {
            const int i02 = i12 / r2;
            for (int i11 = 0; i11 < ne11; ++i11) {
                for (int i01 = 0; i01 < ne01; ++i01) {
                    const size_t off0 = (size_t) i01 * nb01e + (size_t) i02 * nb02e + (size_t) i03 * nb03e;
                    const size_t off1 = (size_t) i11 * nb11e + (size_t) i12 * nb12e + (size_t) i13 * nb13e;
                    double sum = 0.0;
                    for (int c = 0; c < ne00; ++c) {
                        sum += (double) src0[off0 + c] * (double) src1[off1 + c];
                    }
                    const size_t off_dst = (size_t) i11 * nb1e + (size_t) i12 * nb2e + (size_t) i13 * nb3e;
                    dst[off_dst + i01] = (float) sum;
                }
            }
        }
    }
}

struct TestConfig {
    const char * label;
    int ne00, ne01, ne02, ne03;
    int ne11, ne12, ne13;
    // If true, deliberately lay out src1 (and dst) with a non-"packed"
    // stride for dim 2 - inserting a gap between i12 slices - to confirm
    // the kernel genuinely respects the passed-in strides rather than
    // assuming a packed layout. This stands in for a permuted view: real
    // attention permutes reorder which dimension is fastest-varying in
    // memory, which manifests here as "the natural in-order stride is not
    // what's actually used."
    bool permuted_src1;
};

static bool run_case(const TestConfig & cfg) {
    const int ne00 = cfg.ne00, ne01 = cfg.ne01, ne02 = cfg.ne02, ne03 = cfg.ne03;
    const int ne11 = cfg.ne11, ne12 = cfg.ne12, ne13 = cfg.ne13;

    // src0: packed layout always (only src1/dst get the permutation stress
    // case, to keep the test's own bookkeeping simple - the kernel treats
    // src0 and src1 identically with respect to striding, so exercising it
    // on one operand is sufficient to validate the stride-handling code
    // path itself).
    const size_t nb01e = (size_t) ne00;
    const size_t nb02e = nb01e * ne01;
    const size_t nb03e = nb02e * ne02;
    const size_t n_src0 = nb03e * ne03;

    // src1 / dst: packed by default, or padded per-i12-slice if
    // permuted_src1 is set (extra 16 floats of slack after each i12 slice,
    // never read/written by a correct implementation - only a kernel that
    // ignored nb12/nb2 and assumed packing would read/write the wrong
    // offsets here).
    const size_t pad = cfg.permuted_src1 ? 16 : 0;
    const size_t nb11e = (size_t) ne00;
    const size_t nb12e = nb11e * ne11 + pad;
    const size_t nb13e = nb12e * ne12;
    const size_t n_src1 = nb13e * ne13;

    const size_t nb1e = (size_t) ne01;
    const size_t nb2e = nb1e * ne11 + pad;
    const size_t nb3e = nb2e * ne12;
    const size_t n_dst = nb3e * ne13;

    std::vector<float> h_src0(n_src0), h_src1(n_src1), h_dst_ref(n_dst, -999.0f), h_dst(n_dst);
    for (size_t i = 0; i < n_src0; ++i) {
        h_src0[i] = (float) ((int) (i * 2654435761u) % 2001) / 500.0f - 2.0f;
    }
    for (size_t i = 0; i < n_src1; ++i) {
        h_src1[i] = (float) ((int) (i * 2971215073u) % 2001) / 500.0f - 2.0f;
    }

    cpu_mulmat_f32_reference(
        h_src0, nb01e, nb02e, nb03e,
        h_src1, nb11e, nb12e, nb13e,
        h_dst_ref, nb1e, nb2e, nb3e,
        ne00, ne01, ne02, ne03, ne11, ne12, ne13);

    float * d_src0 = NULL, * d_src1 = NULL, * d_dst = NULL;
    cudaMalloc(&d_src0, n_src0 * sizeof(float));
    cudaMalloc(&d_src1, n_src1 * sizeof(float));
    cudaMalloc(&d_dst,  n_dst  * sizeof(float));
    cudaMemcpy(d_src0, &h_src0[0], n_src0 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_src1, &h_src1[0], n_src1 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_dst, 0xFF, n_dst * sizeof(float));

    int rc = ggml_cuda8_mul_mat_f32_f32_launch(
        d_src0, d_src1, d_dst,
        ne00, ne01, ne02, ne03, ne11, ne12, ne13,
        nb01e * sizeof(float), nb02e * sizeof(float), nb03e * sizeof(float),
        nb11e * sizeof(float), nb12e * sizeof(float), nb13e * sizeof(float),
        nb1e  * sizeof(float), nb2e  * sizeof(float), nb3e  * sizeof(float));

    bool ok = true;
    if (rc != 0) {
        std::printf("  %-46s FAIL (launcher returned %d)\n", cfg.label, rc);
        ok = false;
    } else {
        cudaMemcpy(&h_dst[0], d_dst, n_dst * sizeof(float), cudaMemcpyDeviceToHost);
        double max_err = 0.0;
        long bad = -1;
        // Only compare the "live" (i11,i12,i13) slices at their real
        // offsets - padding slots are deliberately never written by the
        // kernel and hold stale 0xFF poison, which is expected and correct,
        // not a bug, so they must be excluded from comparison.
        for (int i13 = 0; i13 < ne13 && bad < 0; ++i13) {
            for (int i12 = 0; i12 < ne12 && bad < 0; ++i12) {
                for (int i11 = 0; i11 < ne11 && bad < 0; ++i11) {
                    const size_t off = (size_t) i11 * nb1e + (size_t) i12 * nb2e + (size_t) i13 * nb3e;
                    for (int i01 = 0; i01 < ne01; ++i01) {
                        const double err = std::fabs((double) h_dst[off + i01] - (double) h_dst_ref[off + i01]);
                        if (err > max_err) max_err = err;
                        if (!approx(h_dst[off + i01], h_dst_ref[off + i01], 3e-3f)) {
                            bad = (long) (off + i01);
                            break;
                        }
                    }
                }
            }
        }
        if (bad >= 0) {
            std::printf("  %-46s FAIL at flat_off=%ld got=%.6g want=%.6g (max_err=%.3e)\n",
                        cfg.label, bad, (double) h_dst[bad], (double) h_dst_ref[bad], max_err);
            ok = false;
        } else {
            std::printf("  %-46s PASS (max_err=%.3e)\n", cfg.label, max_err);
        }
    }

    cudaFree(d_src0);
    cudaFree(d_src1);
    cudaFree(d_dst);
    return ok;
}

// supported()-gate rejection cases, exercised directly (no GPU touch).
static ggml_tensor make_fake_4d(ggml_type type, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3) {
    ggml_tensor t;
    std::memset(&t, 0, sizeof(t));
    t.type = type;
    t.ne[0] = ne0; t.ne[1] = ne1; t.ne[2] = ne2; t.ne[3] = ne3;
    t.nb[0] = sizeof(float);
    t.nb[1] = t.nb[0] * ne0;
    t.nb[2] = t.nb[1] * ne1;
    t.nb[3] = t.nb[2] * ne2;
    t.data = (void *) 0x1000;  // non-NULL sentinel
    return t;
}

static bool test_rejection(const char * label, bool expect_supported,
                            ggml_tensor * src0, ggml_tensor * src1, ggml_tensor * dst) {
    const int got = ggml_cuda8_supported_mul_mat_f32_f32(NULL, src0, src1, dst);
    const bool pass = (got != 0) == expect_supported;
    std::printf("  %-46s expected=%-5s got=%-5s %s\n",
                label, expect_supported ? "true" : "false",
                got ? "true" : "false", pass ? "PASS" : "FAIL");
    return pass;
}

int main() {
    std::printf("ggml-cuda8-mulmat-f32-smoke: starting\n\n");
    bool ok = true;

    std::printf("== Numerical correctness (GPU vs. independent CPU reference) ==\n");
    TestConfig configs[] = {
        { "non-batched (ne02=ne12=1)",              64, 8,  1, 1,  4,  1, 1, false },
        { "batched, no broadcast (r2=1, 4 heads)",   64, 8,  4, 1,  4,  4, 1, false },
        { "GQA broadcast (ne02=2, ne12=8, r2=4)",    64, 16, 2, 1,  4,  8, 1, false },
        { "batch dim (ne03=2, ne13=2)",              64, 8,  2, 2,  4,  2, 2, false },
        { "large ne00=1024 (multi-iter reduction)", 1024, 8, 2, 1,  4,  2, 1, false },
        { "permuted src1/dst (non-packed nb2)",      64, 8,  2, 1,  4,  2, 1, true  },
        { "oversized total_rows (>65535, grid.y)",  32, 300, 1, 1,  1, 220, 1, false },
    };
    for (size_t i = 0; i < sizeof(configs) / sizeof(configs[0]); ++i) {
        ok &= run_case(configs[i]);
    }

    std::printf("\n== supported() rejection cases ==\n");
    ggml_tensor src0_ok   = make_fake_4d(GGML_TYPE_F32, 64, 8, 2, 1);
    ggml_tensor src1_ok   = make_fake_4d(GGML_TYPE_F32, 64, 4, 8, 1);   // r2=4, valid GQA broadcast
    ggml_tensor dst_ok    = make_fake_4d(GGML_TYPE_F32, 8,  4, 8, 1);
    ggml_tensor src1_bad_reduce = make_fake_4d(GGML_TYPE_F32, 63, 4, 8, 1); // ne00 mismatch
    ggml_tensor src1_bad_broadcast = make_fake_4d(GGML_TYPE_F32, 64, 4, 5, 1); // ne12 not multiple of ne02
    ggml_tensor dst_bad_shape = make_fake_4d(GGML_TYPE_F32, 8, 4, 5, 1); // wrong ne2

    ggml_tensor src0_noncontig = make_fake_4d(GGML_TYPE_F32, 64, 8, 2, 1);
    src0_noncontig.nb[0] = sizeof(float) * 2;  // dim-0 not contiguous - out of scope, must be refused

    ok &= test_rejection("valid batched + broadcast", true, &src0_ok, &src1_ok, &dst_ok);
    ok &= test_rejection("reduction dim (ne00) mismatch", false, &src0_ok, &src1_bad_reduce, &dst_ok);
    ok &= test_rejection("broadcast ratio not integer", false, &src0_ok, &src1_bad_broadcast, &dst_ok);
    ok &= test_rejection("dst shape mismatch", false, &src0_ok, &src1_ok, &dst_bad_shape);
    ok &= test_rejection("src0 dim-0 non-contiguous (refused)", false, &src0_noncontig, &src1_ok, &dst_ok);

    std::printf("\nggml-cuda8-mulmat-f32-smoke: %s\n", ok ? "SUCCESS" : "FAIL");
    return ok ? 0 : 1;
}
