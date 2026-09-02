// ggml-cuda8-q6k-bench.cpp
//
// G61: correctness-checked microbenchmark for the Q6_K MUL_MAT kernel.
//
// Companion to ggml-cuda8-q4k-bench.cpp. Q6_K got the same 256-thread
// utilization rewrite Q4_K did (~7.5x in the model), but via a trickier
// per-value mapping (the switch(k) quadrant decode), which is exactly the
// kind of place a subtle indexing bug hides. This gives Q6_K the same
// standalone guard: verify against an independent CPU reference AND report
// effective GB/s, in one fast standalone run - no model rebuild needed.
//
// Compiled as plain C++ (.cpp): calls the extern "C" launcher already in
// libggml-cuda8-kernels.a; does not touch ggml.h / nvcc-hostile headers.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <vector>
#include <time.h>

// ---- Q6_K block layout: MUST match ggml-cuda8-q6k.cu exactly ----------------
#define QK_K 256
struct block_q6_K {
    uint8_t  ql[QK_K / 2];      // 128 bytes: lower 4 bits
    uint8_t  qh[QK_K / 4];      //  64 bytes: upper 2 bits
    int8_t   scales[QK_K / 16]; //  16 bytes: 8-bit scales
    uint16_t d;                 //   2 bytes: fp16 super-block scale
};

// Kernel launcher under test (extern "C" in the kernels archive).
extern "C" int ggml_cuda8_op_mul_mat_q6k_f32(
    const void * src0, const float * src1, float * dst,
    int ne00, int ne01, int ne11,
    int nb01, int nb11);

// ---- host fp16 -> f32 -------------------------------------------------------
static float fp16_bits_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    const uint32_t exp  = (h >> 10) & 0x1F;
    const uint32_t mant = h & 0x3FF;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) { bits = sign; }
        else {
            int e = -1; uint32_t m = mant;
            do { m <<= 1; ++e; } while ((m & 0x400) == 0);
            m &= 0x3FF;
            bits = sign | ((uint32_t)(127 - 15 - e) << 23) | (m << 13);
        }
    } else if (exp == 0x1F) {
        bits = sign | 0x7F800000 | (mant << 13);
    } else {
        bits = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    float f; std::memcpy(&f, &bits, sizeof(f)); return f;
}
static uint16_t f32_to_fp16_bits(float f) {
    uint32_t x; std::memcpy(&x, &f, sizeof(x));
    const uint32_t sign = (x >> 16) & 0x8000;
    int32_t exp = (int32_t)((x >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = x & 0x7FFFFF;
    if (exp <= 0) return (uint16_t) sign;
    if (exp >= 0x1F) return (uint16_t)(sign | 0x7C00);
    return (uint16_t)(sign | (exp << 10) | (mant >> 13));
}

// Independent CPU reference for one output row. Re-derives the Q6_K dequant
// from scratch using the SAME interleaved layout the serial kernel documented
// (q1..q4 quadrants per l, across two halves of 128), so agreement between
// this and the GPU kernel is a genuine two-implementation check.
static float cpu_dot_q6k_row(const block_q6_K * row, const float * s1, int nb) {
    double sum = 0.0;
    for (int b = 0; b < nb; ++b) {
        const block_q6_K * blk = &row[b];
        const float d = fp16_bits_to_f32(blk->d);
        const float * s1b = s1 + b * QK_K;
        for (int half = 0; half < 2; ++half) {
            const uint8_t * ql = blk->ql + half * 64;
            const uint8_t * qh = blk->qh + half * 32;
            const int8_t  * sc = blk->scales + half * 8;
            const float   * s1h = s1b + half * 128;
            for (int l = 0; l < 32; ++l) {
                const int is = l / 16;
                const int8_t q1 = (int8_t)((ql[l]      & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int8_t q3 = (int8_t)((ql[l]      >>  4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int8_t q4 = (int8_t)((ql[l + 32] >>  4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                sum += (double)(d * (float) sc[is + 0] * (float) q1) * (double) s1h[l +  0];
                sum += (double)(d * (float) sc[is + 2] * (float) q2) * (double) s1h[l + 32];
                sum += (double)(d * (float) sc[is + 4] * (float) q3) * (double) s1h[l + 64];
                sum += (double)(d * (float) sc[is + 6] * (float) q4) * (double) s1h[l + 96];
            }
        }
    }
    return (float) sum;
}

static void fill_deterministic(uint8_t * p, size_t n, uint32_t seed) {
    uint32_t s = seed ? seed : 1u;
    for (size_t i = 0; i < n; ++i) {
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
        p[i] = (uint8_t)(s & 0xFF);
    }
}

static bool bench_shape(int ne00, int ne01, int ne11, int iters) {
    if (ne00 % QK_K != 0) {
        std::fprintf(stderr, "ne00=%d not a multiple of %d\n", ne00, QK_K);
        return false;
    }
    const int nb = ne00 / QK_K;
    const size_t weight_blocks = (size_t) ne01 * nb;
    const size_t weight_bytes  = weight_blocks * sizeof(block_q6_K);

    std::vector<block_q6_K> h_w(weight_blocks);
    fill_deterministic((uint8_t *) h_w.data(), weight_bytes, 0x5678u);
    // Sane fp16 d, and int8 scales in a modest range so the reference is
    // well-conditioned (raw xorshift scales could be large negatives).
    for (size_t i = 0; i < weight_blocks; ++i) {
        h_w[i].d = f32_to_fp16_bits(0.03f + 0.0005f * (float)(i % 16));
        for (int k = 0; k < QK_K / 16; ++k) {
            h_w[i].scales[k] = (int8_t)(1 + (int)((i + k) % 8));   // 1..8
        }
    }

    std::vector<float> h_src1((size_t) ne00 * ne11);
    for (size_t i = 0; i < h_src1.size(); ++i) {
        h_src1[i] = ((float)((i * 2654435761u) % 2001) / 1000.0f) - 1.0f;
    }

    std::vector<float> ref((size_t) ne01 * ne11);
    for (int col = 0; col < ne11; ++col) {
        const float * s1 = &h_src1[(size_t) col * ne00];
        for (int row = 0; row < ne01; ++row) {
            ref[(size_t) col * ne01 + row] = cpu_dot_q6k_row(&h_w[(size_t) row * nb], s1, nb);
        }
    }

    void * d_w = NULL; float * d_s1 = NULL; float * d_dst = NULL;
    cudaMalloc(&d_w,   weight_bytes);
    cudaMalloc(&d_s1,  h_src1.size() * sizeof(float));
    cudaMalloc(&d_dst, ref.size()    * sizeof(float));
    cudaMemcpy(d_w,  h_w.data(),    weight_bytes,                cudaMemcpyHostToDevice);
    cudaMemcpy(d_s1, h_src1.data(), h_src1.size()*sizeof(float), cudaMemcpyHostToDevice);

    const int nb01 = nb * (int) sizeof(block_q6_K);
    const int nb11 = ne00 * (int) sizeof(float);

    cudaMemset(d_dst, 0, ref.size() * sizeof(float));
    if (ggml_cuda8_op_mul_mat_q6k_f32(d_w, d_s1, d_dst, ne00, ne01, ne11, nb01, nb11) != 0) {
        std::fprintf(stderr, "  launcher returned error\n");
        cudaFree(d_w); cudaFree(d_s1); cudaFree(d_dst);
        return false;
    }
    std::vector<float> got(ref.size());
    cudaMemcpy(got.data(), d_dst, ref.size()*sizeof(float), cudaMemcpyDeviceToHost);
    double max_err = 0.0, max_rel = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double e = std::fabs((double) got[i] - (double) ref[i]);
        if (e > max_err) max_err = e;
        const double denom = std::fabs((double) ref[i]) + 1e-6;
        if (e / denom > max_rel) max_rel = e / denom;
    }

    cudaDeviceSynchronize();
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int it = 0; it < iters; ++it) {
        ggml_cuda8_op_mul_mat_q6k_f32(d_w, d_s1, d_dst, ne00, ne01, ne11, nb01, nb11);
    }
    cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC, &t1);
    const double ms_total = (t1.tv_sec - t0.tv_sec)*1e3 + (t1.tv_nsec - t0.tv_nsec)/1e6;
    const double ms_call  = ms_total / (double) iters;
    const double gbps     = ((double) weight_bytes / (ms_call / 1e3)) / 1e9;

    // Absolute tolerance scaled by accumulation length (same rationale as the
    // q4k bench: fp16 scales give ~1e-4 rounding; a real layout bug overshoots
    // by orders of magnitude).
    const double abs_tol = 1e-3 * (double) ne00;
    const bool ok = (max_err < abs_tol);
    std::printf("  ne00=%-5d ne01=%-5d ne11=%-3d nb=%-3d | "
                "max_err=%.3e rel=%.3e %s | %.3f ms/call | %.2f GB/s (weights=%.2f MiB)\n",
                ne00, ne01, ne11, nb, max_err, max_rel, ok ? "OK " : "BAD",
                ms_call, gbps, (double) weight_bytes / (1024.0*1024.0));

    cudaFree(d_w); cudaFree(d_s1); cudaFree(d_dst);
    return ok;
}

int main() {
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) {
        std::fprintf(stderr, "no CUDA device 0\n");
        return 1;
    }
    std::printf("ggml-cuda8-q6k-bench: %s (cc %d.%d, %.0f GB/s peak, %d SMs)\n\n",
                prop.name, prop.major, prop.minor,
                2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1e6,
                prop.multiProcessorCount);

    bool ok = true;
    const int iters = 50;
    std::printf("== matvec (ne11=1, token generation) ==\n");
    ok &= bench_shape(1024, 1024, 1, iters);
    ok &= bench_shape(1024, 3072, 1, iters);
    ok &= bench_shape(3072, 1024, 1, iters);
    ok &= bench_shape(2048, 2048, 1, iters);

    std::printf("\n== batched (ne11>1, prompt prefill) ==\n");
    ok &= bench_shape(1024, 1024, 8,  iters);
    ok &= bench_shape(1024, 1024, 32, iters);

    std::printf("\nggml-cuda8-q6k-bench: %s\n",
                ok ? "SUCCESS (all shapes correct)" : "FAIL (a shape produced wrong output)");
    return ok ? 0 : 1;
}
