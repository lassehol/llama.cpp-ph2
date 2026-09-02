// ggml-cuda8-set-rows-f16-smoke.cu
//
// G49: correctness gate for the F32-src -> F16-dst SET_ROWS store
// (ggml_cuda8_op_set_rows_f16). Proves the F16 KV-cache WRITE path before any
// of the read-side G49 work. Standalone .cu: declares the launcher extern "C",
// builds all tensors by hand as plain arrays, no ggml headers.
//
// What it checks:
//   1. Each scattered value lands at dst[idx[i]] as the correct F16 encoding
//      of the F32 source (compared against an independent host F32->F16->F32
//      round-trip, so both sides quantize identically).
//   2. The int64 index scatter maps rows correctly (non-identity permutation).
//   3. Out-of-range indices are skipped, not written (poison stays intact).
//   4. Multi-channel (ne02>1) addressing works.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <vector>

// Launcher under test (extern "C" in the kernels archive).
extern "C" int ggml_cuda8_op_set_rows_f16(
        const void * src0, const void * idx, void * dst,
        int nc, int nr, int ne02, int ne03,
        int ne11, int ne12, int ne1,
        size_t nb01, size_t nb02, size_t nb03,
        size_t nb10, size_t nb11, size_t nb12,
        size_t nb1,  size_t nb2,  size_t nb3);

// Host F32 -> F16 bits (round-to-nearest-even), matching __float2half well
// enough for exact-encoding comparison of well-scaled test values.
static uint16_t f32_to_fp16_bits(float f) {
    uint32_t x; std::memcpy(&x, &f, sizeof(x));
    const uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = (int32_t)((x >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = x & 0x7FFFFFu;
    if (exp <= 0) {
        // subnormal/zero flush: fine for the modest test magnitudes used here
        return (uint16_t) sign;
    }
    if (exp >= 0x1F) return (uint16_t)(sign | 0x7C00u);
    // round-to-nearest-even on the 13 dropped mantissa bits
    const uint32_t round_bit = (mant >> 12) & 1u;
    const uint32_t sticky    = (mant & 0xFFFu) != 0u;
    uint16_t h = (uint16_t)(sign | ((uint32_t)exp << 10) | (mant >> 13));
    if (round_bit && (sticky || (h & 1u))) h += 1;   // ties-to-even
    return h;
}
static float fp16_bits_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    const uint32_t exp  = (h >> 10) & 0x1F;
    const uint32_t mant = h & 0x3FF;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) bits = sign;
        else {
            int e = -1; uint32_t m = mant;
            do { m <<= 1; ++e; } while ((m & 0x400) == 0);
            m &= 0x3FF;
            bits = sign | ((uint32_t)(127 - 15 - e) << 23) | (m << 13);
        }
    } else if (exp == 0x1F) {
        bits = sign | 0x7F800000u | (mant << 13);
    } else {
        bits = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    float f; std::memcpy(&f, &bits, sizeof(f)); return f;
}

static bool run_case(const char * label, int nc, int nr, int ne02, int ne1,
                     const std::vector<int64_t> & idx_map /* size nr */) {
    // Single batch dim, no index broadcast complexity: ne03=1, ne11=ne12=1.
    const int ne03 = 1;
    const int ne11 = 1, ne12 = 1;

    // src0: F32 [nc, nr, ne02] packed
    const size_t n_src = (size_t) nc * nr * ne02;
    std::vector<float> h_src(n_src);
    for (size_t i = 0; i < n_src; ++i) {
        // spread across a well-conditioned F16 range
        h_src[i] = ((float)((int)(i * 2654435761u % 4001)) / 1000.0f) - 2.0f;
    }

    // idx: I64 [nr]  (row i of src -> row idx_map[i] of dst)
    std::vector<int64_t> h_idx = idx_map;

    // dst: F16 [nc, ne1, ne02] packed, poison-filled (0xBEEF pattern)
    const size_t n_dst = (size_t) nc * ne1 * ne02;
    std::vector<uint16_t> h_dst(n_dst, 0xBEEF);

    // F16 byte strides for dst
    const size_t nb1_dst = (size_t) nc * sizeof(uint16_t);          // per dst row
    const size_t nb2_dst = nb1_dst * ne1;                          // per channel
    const size_t nb3_dst = nb2_dst * 1;
    // F32 byte strides for src0
    const size_t nb01 = (size_t) nc * sizeof(float);
    const size_t nb02 = nb01 * nr;
    const size_t nb03 = nb02 * ne02;
    // idx byte strides (I64, [nr])
    const size_t nb10 = sizeof(int64_t);
    const size_t nb11 = nb10 * nr;   // unused (ne11=1) but must be valid
    const size_t nb12 = nb11;

    float   * d_src = NULL; int64_t * d_idx = NULL; uint16_t * d_dst = NULL;
    cudaMalloc(&d_src, n_src * sizeof(float));
    cudaMalloc(&d_idx, h_idx.size() * sizeof(int64_t));
    cudaMalloc(&d_dst, n_dst * sizeof(uint16_t));
    cudaMemcpy(d_src, &h_src[0], n_src * sizeof(float),          cudaMemcpyHostToDevice);
    cudaMemcpy(d_idx, &h_idx[0], h_idx.size()*sizeof(int64_t),   cudaMemcpyHostToDevice);
    cudaMemcpy(d_dst, &h_dst[0], n_dst * sizeof(uint16_t),       cudaMemcpyHostToDevice);

    const int rc = ggml_cuda8_op_set_rows_f16(
        d_src, d_idx, d_dst,
        nc, nr, ne02, ne03, ne11, ne12, ne1,
        nb01, nb02, nb03,
        nb10, nb11, nb12,
        nb1_dst, nb2_dst, nb3_dst);

    bool ok = true;
    if (rc != 0) {
        std::printf("  %-42s FAIL (launcher returned %d)\n", label, rc);
        ok = false;
    } else {
        std::vector<uint16_t> got(n_dst);
        cudaMemcpy(&got[0], d_dst, n_dst * sizeof(uint16_t), cudaMemcpyDeviceToHost);

        // Build the expected dst: start from poison, then scatter F16(src).
        std::vector<uint16_t> exp(n_dst, 0xBEEF);
        for (int i2 = 0; i2 < ne02; ++i2) {
            for (int i = 0; i < nr; ++i) {
                const int64_t d = h_idx[i];
                if (d < 0 || d >= ne1) continue;   // skipped, stays poison
                for (int c = 0; c < nc; ++c) {
                    const float sv = h_src[((size_t) i2 * nr + i) * nc + c];
                    exp[((size_t) i2 * ne1 + d) * nc + c] = f32_to_fp16_bits(sv);
                }
            }
        }

        long bad = -1;
        for (size_t k = 0; k < n_dst; ++k) {
            if (got[k] == exp[k]) continue;
            // allow +/-1 ULP in the F16 encoding (host vs GPU rounding corner)
            const int diff = (int) got[k] - (int) exp[k];
            if (diff >= -1 && diff <= 1 && exp[k] != 0xBEEF) continue;
            bad = (long) k; break;
        }
        if (bad >= 0) {
            std::printf("  %-42s FAIL at %ld: got=0x%04X (%.4f) want=0x%04X (%.4f)\n",
                        label, bad, got[bad], fp16_bits_to_f32(got[bad]),
                        exp[bad], fp16_bits_to_f32(exp[bad]));
            ok = false;
        } else {
            std::printf("  %-42s PASS (%zu F16 elems)\n", label, n_dst);
        }
    }

    cudaFree(d_src); cudaFree(d_idx); cudaFree(d_dst);
    return ok;
}

int main() {
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) {
        std::fprintf(stderr, "no CUDA device 0\n");
        return 1;
    }
    std::printf("ggml-cuda8-set-rows-f16-smoke: %s (cc %d.%d)\n\n",
                prop.name, prop.major, prop.minor);

    bool ok = true;

    std::printf("== F32 src -> F16 dst store ==\n");

    // 1. identity map, single channel
    {
        std::vector<int64_t> m(4); for (int i = 0; i < 4; ++i) m[i] = i;
        ok &= run_case("identity [nc=8, nr=4, ne02=1, ne1=4]", 8, 4, 1, 4, m);
    }
    // 2. permuted scatter (non-identity), dst larger than src rows
    {
        std::vector<int64_t> m(4); m[0]=3; m[1]=0; m[2]=5; m[3]=1;
        ok &= run_case("permute [nc=8, nr=4, ne02=1, ne1=8]", 8, 4, 1, 8, m);
    }
    // 3. out-of-range indices skipped (one negative, one >= ne1)
    {
        std::vector<int64_t> m(4); m[0]=1; m[1]=-1; m[2]=2; m[3]=99;
        ok &= run_case("out-of-range skip [nc=8, nr=4, ne1=4]", 8, 4, 1, 4, m);
    }
    // 4. multi-channel (ne02>1)
    {
        std::vector<int64_t> m(3); m[0]=2; m[1]=0; m[2]=1;
        ok &= run_case("multi-channel [nc=16, nr=3, ne02=4, ne1=3]", 16, 3, 4, 3, m);
    }
    // 5. larger row width
    {
        std::vector<int64_t> m(8); for (int i = 0; i < 8; ++i) m[i] = (i * 3) % 8;
        ok &= run_case("wide [nc=128, nr=8, ne02=2, ne1=8]", 128, 8, 2, 8, m);
    }

    std::printf("\nggml-cuda8-set-rows-f16-smoke: %s\n",
                ok ? "SUCCESS (all cases match F16 reference)" : "FAIL");
    return ok ? 0 : 1;
}
