// ggml-cuda8-q4k-bench.cpp
//
// G58: correctness-checked microbenchmark for the Q4_K MUL_MAT kernel.
//
// Purpose: make K-quant matmul optimization safe and fast to iterate.
// Instead of a full two-stage model rebuild + eyeballing token output
// (which cannot catch subtle numerical corruption - see the transposed-dst
// and flat-CONT silent bugs), this times the kernel AND verifies it against
// an independent CPU reference in one standalone run.
//
// Compiled as plain C++ (.cpp): it calls the extern "C" launcher already in
// libggml-cuda8-kernels.a and does not touch ggml.h / nvcc-hostile headers.
//
// Reports, for each shape:
//   - max_err vs CPU reference   (MUST stay ~0; a regression here = wrong kernel)
//   - ms/call                    (wall time per kernel invocation, averaged)
//   - effective GB/s             (weight bytes read / time; the number to raise)
//
// The GB/s figure is what the whole optimization targets: current kernel is
// expected to show a tiny fraction of the GTX 560's 128 GB/s because only
// nb = ne00/256 of its 256 threads/block do any work (thread underutilization).
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <vector>

// ---- Q4_K block layout: MUST match ggml-cuda8-q4k.cu exactly ----------------
#define QK_K 256
#define K_SCALE_SIZE 12
struct block_q4_K {
    uint16_t d;                    // fp16 bits: super-block scale
    uint16_t dmin;                 // fp16 bits: super-block min
    uint8_t  scales[K_SCALE_SIZE]; // 12 bytes: packed 6-bit scales+mins
    uint8_t  qs[QK_K / 2];         // 128 bytes: 4-bit quants
};

// Kernel launcher under test (extern "C" in the kernels archive).
extern "C" int ggml_cuda8_op_mul_mat_q4k_f32(
    const void * src0, const float * src1, float * dst,
    int ne00, int ne01, int ne11,
    int nb01, int nb11);

// ---- host fp16 -> f32 (matches __half2float for normal/zero values) ---------
static float fp16_bits_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    const uint32_t exp  = (h >> 10) & 0x1F;
    const uint32_t mant = h & 0x3FF;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) { bits = sign; }                 // +/- 0
        else {                                          // subnormal
            int e = -1;
            uint32_t m = mant;
            do { m <<= 1; ++e; } while ((m & 0x400) == 0);
            m &= 0x3FF;
            bits = sign | ((uint32_t)(127 - 15 - e) << 23) | (m << 13);
        }
    } else if (exp == 0x1F) {                           // inf/nan
        bits = sign | 0x7F800000 | (mant << 13);
    } else {
        bits = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

// f32 -> fp16 bits (round-to-nearest-even, enough for test data).
static uint16_t f32_to_fp16_bits(float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    const uint32_t sign = (x >> 16) & 0x8000;
    int32_t exp = (int32_t)((x >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = x & 0x7FFFFF;
    if (exp <= 0) return (uint16_t) sign;                // flush tiny to signed zero
    if (exp >= 0x1F) return (uint16_t)(sign | 0x7C00);   // overflow to inf
    return (uint16_t)(sign | (exp << 10) | (mant >> 13));
}

// Same 6-bit scale/min unpack as the kernel's get_scale_min_k4.
static void get_scale_min_k4(int j, const uint8_t * scales, uint8_t & sc, uint8_t & mn) {
    if (j < 4) {
        sc = scales[j]     & 63;
        mn = scales[j + 4] & 63;
    } else {
        sc = (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4);
        mn = (scales[j + 4] >> 4)  | ((scales[j]     >> 6) << 4);
    }
}

// Independent CPU reference for one output row: dequantize each Q4_K block and
// dot with the activation vector, using the CORRECT ggml element order.
static float cpu_dot_q4k_row(const block_q4_K * row, const float * s1, int nb) {
    double sum = 0.0;
    for (int b = 0; b < nb; ++b) {
        const block_q4_K * blk = &row[b];
        const float d    = fp16_bits_to_f32(blk->d);
        const float dmin = fp16_bits_to_f32(blk->dmin);
        const float * s1b = s1 + b * QK_K;
        for (int j = 0; j < 8; ++j) {
            uint8_t sc, mn;
            get_scale_min_k4(j, blk->scales, sc, mn);
            const float scale = d * (float) sc;
            const float mmin  = dmin * (float) mn;
            const int g = j / 2;
            const int is_high = j & 1;
            for (int l = 0; l < 32; ++l) {
                const uint8_t qbyte = blk->qs[g * 32 + l];
                const uint8_t q = is_high ? (qbyte >> 4) : (qbyte & 0xF);
                const float w = scale * (float) q - mmin;
                sum += (double) w * (double) s1b[j * 32 + l];
            }
        }
    }
    return (float) sum;
}

static void fill_deterministic(uint8_t * p, size_t n, uint32_t seed) {
    uint32_t s = seed ? seed : 1u;
    for (size_t i = 0; i < n; ++i) {
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;   // xorshift32
        p[i] = (uint8_t)(s & 0xFF);
    }
}

static bool bench_shape(int ne00, int ne01, int ne11, int iters) {
    if (ne00 % QK_K != 0) {
        std::fprintf(stderr, "ne00=%d not a multiple of %d\n", ne00, QK_K);
        return false;
    }
    const int nb = ne00 / QK_K;                       // blocks per row
    const size_t weight_blocks = (size_t) ne01 * nb;
    const size_t weight_bytes  = weight_blocks * sizeof(block_q4_K);

    std::vector<block_q4_K> h_w(weight_blocks);
    fill_deterministic((uint8_t *) h_w.data(), weight_bytes, 0x1234u);
    // Give d/dmin sane fp16 magnitudes so the reference is well-scaled
    // (raw xorshift bytes could land on inf/nan fp16 patterns).
    for (size_t i = 0; i < weight_blocks; ++i) {
        h_w[i].d    = f32_to_fp16_bits(0.05f + 0.001f * (float)(i % 16));
        h_w[i].dmin = f32_to_fp16_bits(0.02f);
    }

    std::vector<float> h_src1((size_t) ne00 * ne11);
    for (size_t i = 0; i < h_src1.size(); ++i) {
        h_src1[i] = ((float)((i * 2654435761u) % 2001) / 1000.0f) - 1.0f;
    }

    // CPU reference (ggml dst layout: dst[col*ne01 + row]).
    std::vector<float> ref((size_t) ne01 * ne11);
    for (int col = 0; col < ne11; ++col) {
        const float * s1 = &h_src1[(size_t) col * ne00];
        for (int row = 0; row < ne01; ++row) {
            ref[(size_t) col * ne01 + row] =
                cpu_dot_q4k_row(&h_w[(size_t) row * nb], s1, nb);
        }
    }

    // Device buffers.
    void * d_w = NULL; float * d_s1 = NULL; float * d_dst = NULL;
    cudaMalloc(&d_w,   weight_bytes);
    cudaMalloc(&d_s1,  h_src1.size() * sizeof(float));
    cudaMalloc(&d_dst, ref.size()    * sizeof(float));
    cudaMemcpy(d_w,  h_w.data(),    weight_bytes,                  cudaMemcpyHostToDevice);
    cudaMemcpy(d_s1, h_src1.data(), h_src1.size()*sizeof(float),   cudaMemcpyHostToDevice);

    const int nb01 = nb * (int) sizeof(block_q4_K);   // row stride, bytes
    const int nb11 = ne00 * (int) sizeof(float);      // src1 col stride, bytes

    // Correctness: one call, compare.
    cudaMemset(d_dst, 0, ref.size() * sizeof(float));
    if (ggml_cuda8_op_mul_mat_q4k_f32(d_w, d_s1, d_dst, ne00, ne01, ne11, nb01, nb11) != 0) {
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

    // Timing: iters calls, wall time (launcher syncs internally).
    cudaDeviceSynchronize();
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int it = 0; it < iters; ++it) {
        ggml_cuda8_op_mul_mat_q4k_f32(d_w, d_s1, d_dst, ne00, ne01, ne11, nb01, nb11);
    }
    cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC, &t1);
    const double ms_total = (t1.tv_sec - t0.tv_sec)*1e3 + (t1.tv_nsec - t0.tv_nsec)/1e6;
    const double ms_call  = ms_total / (double) iters;
    const double gbps     = ((double) weight_bytes / (ms_call / 1e3)) / 1e9;

    const bool ok = (max_rel < 1e-3);
    std::printf("  ne00=%-5d ne01=%-5d ne11=%-3d nb=%-3d | "
                "max_err=%.3e rel=%.3e %s | %.3f ms/call | %.2f GB/s (weights=%.2f MiB)\n",
                ne00, ne01, ne11, nb,
                max_err, max_rel, ok ? "OK " : "BAD",
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
    std::printf("ggml-cuda8-q4k-bench: %s (cc %d.%d, %.0f GB/s peak, %d SMs)\n\n",
                prop.name, prop.major, prop.minor,
                2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1e6,
                prop.multiProcessorCount);

    bool ok = true;
    const int iters = 50;
    std::printf("== matvec (ne11=1, token generation) ==\n");
    ok &= bench_shape(1024, 1024, 1, iters);   // Qwen3-0.6B attn projection shape
    ok &= bench_shape(1024, 3072, 1, iters);   // up/gate proj
    ok &= bench_shape(3072, 1024, 1, iters);   // down proj
    ok &= bench_shape(2048, 2048, 1, iters);

    std::printf("\n== batched (ne11>1, prompt prefill) ==\n");
    ok &= bench_shape(1024, 1024, 8,  iters);
    ok &= bench_shape(1024, 1024, 32, iters);

    std::printf("\nggml-cuda8-q4k-bench: %s\n", ok ? "SUCCESS (all shapes correct)" : "FAIL (a shape produced wrong output)");
    return ok ? 0 : 1;
}
