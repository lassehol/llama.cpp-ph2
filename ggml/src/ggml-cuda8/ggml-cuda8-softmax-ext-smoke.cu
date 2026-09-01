// ggml-cuda8-softmax-ext-smoke.cu
//
// G41: standalone smoke test for the masked/scaled/ALiBi-biased softmax
// kernel (ggml_cuda8_softmax_ext_f32_launch), verified the same way G45
// (ROPE NeoX) and G40 (SwiGLU) were: an independent CPU-side transcription
// of ggml's own algorithm, checked against the kernel across several
// configurations, rather than the kernel checked against itself.
//
// CPU reference mirrors ggml_compute_forward_soft_max_f32 (ggml-cpu/ops.cpp):
//   v[c]   = src[c]*scale + (mask ? slope(head)*mask[c] : 0)
//   dst[c] = softmax(v)[c]
// with the standard ALiBi slope construction (see the .cu file header for
// the exact formula).
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <float.h>
#include "ggml.h"
#include "ggml-cuda8-context.h"
#include "ggml-cuda8-softmax-ext.h"

extern "C" int ggml_cuda8_softmax_ext_f32_launch(
        const float * src,
        const float * mask,
        float * dst,
        int ne00, int ne01, int ne02, int ne03,
        size_t nb01, size_t nb02, size_t nb03,
        size_t dst_nb1, size_t dst_nb2, size_t dst_nb3,
        int mask_ne1, int mask_ne2, int mask_ne3,
        size_t mask_nb1, size_t mask_nb2, size_t mask_nb3,
        float scale, float max_bias);

static bool approx(float a, float b, float tol) {
    return std::fabs(a - b) <= tol * (1.0f + std::fabs(b));
}

// Independent CPU reference. Deliberately re-derives slope/softmax from
// scratch rather than sharing any code path with the kernel - agreement
// between two independently-written implementations is the actual check.
static void cpu_softmax_ext_reference(
        const std::vector<float> & src,          // ne00*ne01*ne02*ne03, packed
        const std::vector<float> * mask,         // ne00*mask_ne1*mask_ne2*mask_ne3, packed, or NULL
        std::vector<float> & dst_ref,
        int ne00, int ne01, int ne02, int ne03,
        int mask_ne1, int mask_ne2, int mask_ne3,
        float scale, float max_bias) {
    dst_ref.assign((size_t) ne00 * ne01 * ne02 * ne03, 0.0f);
    int n_head_log2 = 1;
    while (n_head_log2 * 2 <= ne02) n_head_log2 *= 2;
    const float m0 = std::pow(2.0f, -(max_bias) / (float) n_head_log2);
    const float m1 = std::pow(2.0f, -(max_bias * 0.5f) / (float) n_head_log2);
    for (int i03 = 0; i03 < ne03; ++i03) {
        for (int i02 = 0; i02 < ne02; ++i02) {
            float slope = 1.0f;
            if (mask != NULL && max_bias > 0.0f) {
                const int h = i02;
                slope = (h < n_head_log2)
                    ? std::pow(m0, (float) (h + 1))
                    : std::pow(m1, (float) (2 * (h - n_head_log2) + 1));
            }
            for (int i01 = 0; i01 < ne01; ++i01) {
                const size_t row_off = ((size_t) i03 * ne02 + i02) * ne01 * ne00 + (size_t) i01 * ne00;
                const float * row_mask = NULL;
                if (mask != NULL) {
                    const int m1i = i01 % mask_ne1;
                    const int m2i = i02 % mask_ne2;
                    const int m3i = i03 % mask_ne3;
                    const size_t mask_off = ((size_t) m3i * mask_ne2 + m2i) * mask_ne1 * ne00 + (size_t) m1i * ne00;
                    row_mask = &(*mask)[mask_off];
                }
                std::vector<float> v(ne00);
                float vmax = -FLT_MAX;
                for (int c = 0; c < ne00; ++c) {
                    v[c] = src[row_off + c] * scale + (row_mask ? slope * row_mask[c] : 0.0f);
                    vmax = std::max(vmax, v[c]);
                }
                float sum = 0.0f;
                for (int c = 0; c < ne00; ++c) {
                    v[c] = std::exp(v[c] - vmax);
                    sum += v[c];
                }
                const float inv_sum = sum > 0.0f ? 1.0f / sum : 0.0f;
                for (int c = 0; c < ne00; ++c) {
                    dst_ref[row_off + c] = v[c] * inv_sum;
                }
            }
        }
    }
}

struct TestConfig {
    const char * label;
    int ne00, ne01, ne02, ne03;
    bool use_mask;
    int mask_ne2, mask_ne3;  // mask_ne1 always == ne01 (no broadcast in that dim)
    float scale;
    float max_bias;
};

static bool run_case(const TestConfig & cfg) {
    const int ne00 = cfg.ne00, ne01 = cfg.ne01, ne02 = cfg.ne02, ne03 = cfg.ne03;
    const size_t n_src = (size_t) ne00 * ne01 * ne02 * ne03;

    std::vector<float> h_src(n_src);
    for (size_t i = 0; i < n_src; ++i) {
        // Deterministic pseudo-random spread across [-6, 6] so max/overflow
        // handling is actually exercised, not just near-zero inputs.
        h_src[i] = (float) ((int) (i * 2654435761u) % 12001) / 1000.0f - 6.0f;
    }

    std::vector<float> h_mask;
    int mask_ne1 = 0, mask_ne2 = 0, mask_ne3 = 0;
    if (cfg.use_mask) {
        mask_ne1 = ne01;
        mask_ne2 = cfg.mask_ne2;
        mask_ne3 = cfg.mask_ne3;
        const size_t n_mask = (size_t) ne00 * mask_ne1 * mask_ne2 * mask_ne3;
        h_mask.resize(n_mask);
        for (size_t i = 0; i < n_mask; ++i) {
            // Half the columns "masked" (large negative), half open (0) -
            // exercises both branches of the max/softmax reduction.
            h_mask[i] = ((i * 2971215073u) % 2 == 0) ? -1.0e4f : 0.0f;
        }
    }

    std::vector<float> ref;
    cpu_softmax_ext_reference(h_src, cfg.use_mask ? &h_mask : NULL, ref,
                               ne00, ne01, ne02, ne03,
                               mask_ne1, mask_ne2, mask_ne3,
                               cfg.scale, cfg.max_bias);

    float * d_src = NULL, * d_dst = NULL, * d_mask = NULL;
    const size_t bytes_src = n_src * sizeof(float);
    cudaMalloc(&d_src, bytes_src);
    cudaMalloc(&d_dst, bytes_src);
    cudaMemcpy(d_src, &h_src[0], bytes_src, cudaMemcpyHostToDevice);
    cudaMemset(d_dst, 0xFF, bytes_src);
    if (cfg.use_mask) {
        const size_t bytes_mask = h_mask.size() * sizeof(float);
        cudaMalloc(&d_mask, bytes_mask);
        cudaMemcpy(d_mask, &h_mask[0], bytes_mask, cudaMemcpyHostToDevice);
    }

    const size_t nb01 = (size_t) ne00 * sizeof(float);
    const size_t nb02 = nb01 * ne01;
    const size_t nb03 = nb02 * ne02;
    const size_t mask_nb1 = (size_t) ne00 * sizeof(float);
    const size_t mask_nb2 = cfg.use_mask ? mask_nb1 * mask_ne1 : 0;
    const size_t mask_nb3 = cfg.use_mask ? mask_nb2 * mask_ne2 : 0;

    int rc = ggml_cuda8_softmax_ext_f32_launch(
        d_src, d_mask, d_dst,
        ne00, ne01, ne02, ne03,
        nb01, nb02, nb03,
        nb01, nb02, nb03,
        mask_ne1, mask_ne2, mask_ne3,
        mask_nb1, mask_nb2, mask_nb3,
        cfg.scale, cfg.max_bias);

    bool ok = true;
    if (rc != 0) {
        std::printf("  %-40s FAIL (launcher returned %d)\n", cfg.label, rc);
        ok = false;
    } else {
        std::vector<float> h_dst(n_src);
        cudaMemcpy(&h_dst[0], d_dst, bytes_src, cudaMemcpyDeviceToHost);
        double max_err = 0.0;
        size_t bad = (size_t) -1;
        for (size_t i = 0; i < n_src; ++i) {
            const double err = std::fabs((double) h_dst[i] - (double) ref[i]);
            if (err > max_err) { max_err = err; }
            if (!approx(h_dst[i], ref[i], 1e-4f) && bad == (size_t) -1) {
                bad = i;
            }
        }
        if (bad != (size_t) -1) {
            std::printf("  %-40s FAIL at i=%zu got=%.6g want=%.6g (max_err=%.3e)\n",
                        cfg.label, bad, (double) h_dst[bad], (double) ref[bad], max_err);
            ok = false;
        } else {
            std::printf("  %-40s PASS (max_err=%.3e)\n", cfg.label, max_err);
        }
    }

    cudaFree(d_src);
    cudaFree(d_dst);
    if (d_mask) cudaFree(d_mask);
    return ok;
}

// Rejection cases exercise ggml_cuda8_supported_softmax_ext_f32() directly
// (declared in ggml-cuda8-softmax-ext.h, included above) - these do not
// touch the GPU, they check the dispatcher-level gate refuses what it
// should refuse (F16 mask, mismatched mask shapes) before a kernel is ever
// launched against them.
static ggml_tensor make_fake_2d(ggml_type type, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3) {
    ggml_tensor t;
    std::memset(&t, 0, sizeof(t));
    t.type = type;
    t.ne[0] = ne0; t.ne[1] = ne1; t.ne[2] = ne2; t.ne[3] = ne3;
    const size_t elsize = (type == GGML_TYPE_F16) ? 2 : sizeof(float);
    t.nb[0] = elsize;
    t.nb[1] = t.nb[0] * ne0;
    t.nb[2] = t.nb[1] * ne1;
    t.nb[3] = t.nb[2] * ne2;
    t.data = (void *) 0x1000;  // non-NULL sentinel; supported() only checks != NULL here
    return t;
}

static bool test_rejection(const char * label, bool expect_supported,
                            ggml_tensor * src0, ggml_tensor * src1, ggml_tensor * dst) {
    const int got = ggml_cuda8_supported_softmax_ext_f32(NULL, src0, src1, dst);
    const bool pass = (got != 0) == expect_supported;
    std::printf("  %-40s expected=%-5s got=%-5s %s\n",
                label, expect_supported ? "true" : "false",
                got ? "true" : "false", pass ? "PASS" : "FAIL");
    return pass;
}

int main() {
    std::printf("ggml-cuda8-softmax-ext-smoke: starting\n\n");
    bool ok = true;

    std::printf("== Numerical correctness (GPU vs. independent CPU reference) ==\n");
    TestConfig configs[] = {
        { "no mask, scale=1, max_bias=0 (degenerate)", 64, 8, 2, 1, false, 0, 0, 1.0f, 0.0f },
        { "no mask, scale=0.125",                       64, 8, 2, 1, false, 0, 0, 0.125f, 0.0f },
        { "mask (broadcast over heads), scale=1",       64, 8, 4, 1, true,  1, 1, 1.0f, 0.0f },
        { "mask (broadcast over heads), scale=0.125",   64, 8, 4, 1, true,  1, 1, 0.125f, 0.0f },
        { "mask + ALiBi, 4 heads (pow2)",               64, 8, 4, 1, true,  1, 1, 0.125f, 8.0f },
        { "mask + ALiBi, 6 heads (non-pow2)",           64, 8, 6, 1, true,  1, 1, 0.125f, 8.0f },
        { "mask, per-head mask (ne2==ne02)",             64, 8, 4, 1, true,  4, 1, 0.125f, 0.0f },
        { "mask, batch>1 (ne03=2)",                      64, 8, 4, 2, true,  1, 1, 0.125f, 4.0f },
        { "large cols (n_kv=1024), mask + ALiBi",       1024, 4, 8, 1, true, 1, 1, 0.125f, 8.0f },
    };
    for (size_t i = 0; i < sizeof(configs) / sizeof(configs[0]); ++i) {
        ok &= run_case(configs[i]);
    }

    std::printf("\n== supported() rejection cases ==\n");
    ggml_tensor f32_64x8x4x1 = make_fake_2d(GGML_TYPE_F32, 64, 8, 4, 1);
    ggml_tensor f32_dst      = make_fake_2d(GGML_TYPE_F32, 64, 8, 4, 1);
    ggml_tensor mask_f32_ok  = make_fake_2d(GGML_TYPE_F32, 64, 8, 1, 1);
    ggml_tensor mask_f16_bad = make_fake_2d(GGML_TYPE_F16, 64, 8, 1, 1);
    ggml_tensor mask_bad_ne1 = make_fake_2d(GGML_TYPE_F32, 64, 7, 1, 1);   // ne1 != src0 ne1
    ggml_tensor mask_bad_ne2 = make_fake_2d(GGML_TYPE_F32, 64, 8, 3, 1);   // ne2 neither 1 nor ne02

    ok &= test_rejection("no mask (valid)", true, &f32_64x8x4x1, NULL, &f32_dst);
    ok &= test_rejection("F32 mask, broadcast ne2=1 (valid)", true, &f32_64x8x4x1, &mask_f32_ok, &f32_dst);
    ok &= test_rejection("F16 mask (refused, not F16-ready)", false, &f32_64x8x4x1, &mask_f16_bad, &f32_dst);
    ok &= test_rejection("mask ne1 mismatch (refused)", false, &f32_64x8x4x1, &mask_bad_ne1, &f32_dst);
    ok &= test_rejection("mask ne2 neither 1 nor ne02 (refused)", false, &f32_64x8x4x1, &mask_bad_ne2, &f32_dst);

    std::printf("\nggml-cuda8-softmax-ext-smoke: %s\n", ok ? "SUCCESS" : "FAIL");
    return ok ? 0 : 1;
}
