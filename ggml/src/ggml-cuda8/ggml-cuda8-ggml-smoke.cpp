// ggml/src/ggml-cuda8/ggml-cuda8-ggml-smoke.cpp
//
// G0 GGML-shaped smoke test for CUDA8/Fermi.
//
// This does not use real ggml_tensor objects yet.
// Instead, it mimics the first target GGML MUL_MAT case:
//
//   src0: Q8_0 matrix [rows, cols]
//   src1: F32 vector [cols]
//   dst:  F32 vector [rows]
//
// The test:
//   1. creates deterministic F32 matrix A and vector x
//   2. packs A into Q8_0 blocks using GGML-like Q8_0 layout
//   3. computes CPU Q8_0 reference
//   4. uploads packed A and x to GPU
//   5. calls ggml_cuda8_mul_mat_q8_0_f32()
//   6. compares GPU output against CPU Q8_0 reference

#include "ggml-cuda8-mulmat.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cuda_runtime.h>

static bool cuda_check_bool(cudaError_t err, const char * expr, const char * file, int line) {
    if (err != cudaSuccess) {
        std::fprintf(stderr,
            "ggml-cuda8-ggml-smoke: CUDA error %s (%d) for %s at %s:%d\n",
            cudaGetErrorString(err), (int) err, expr, file, line);
        return false;
    }

    return true;
}

#define CUDA_CHECK_BOOL(call) \
    do { if (!cuda_check_bool((call), #call, __FILE__, __LINE__)) return false; } while (0)

static uint16_t fp32_to_fp16_bits(float value) {
    union { float f; uint32_t u; } in;
    in.f = value;

    const uint32_t f = in.u;
    const uint32_t sign = (f >> 16) & 0x8000u;
    int32_t exp = (int32_t) ((f >> 23) & 0xFFu) - 127 + 15;
    uint32_t mant = f & 0x007FFFFFu;

    if (exp <= 0) {
        if (exp < -10) {
            return (uint16_t) sign;
        }

        mant |= 0x00800000u;

        const int shift = 14 - exp;
        uint32_t half_mant = mant >> shift;

        if ((mant >> (shift - 1)) & 1u) {
            half_mant += 1u;
        }

        return (uint16_t) (sign | half_mant);
    }

    if (exp >= 31) {
        return (uint16_t) (sign | 0x7C00u);
    }

    uint32_t half = sign | ((uint32_t) exp << 10) | (mant >> 13);

    if (mant & 0x00001000u) {
        half += 1u;
    }

    return (uint16_t) half;
}

static float fp16_bits_to_fp32(uint16_t h) {
    const uint32_t h_exp  = h & 0x7C00u;
    const uint32_t h_sig  = h & 0x03FFu;
    const uint32_t h_sign = h & 0x8000u;

    const uint32_t f_sign = h_sign << 16;
    uint32_t f_exp;
    uint32_t f_sig;

    union { uint32_t u; float f; } out;

    if (h_exp == 0) {
        if (h_sig == 0) {
            out.u = f_sign;
            return out.f;
        }

        uint32_t sig = h_sig;
        int exp = -14;

        while ((sig & 0x0400u) == 0) {
            sig <<= 1;
            --exp;
        }

        sig &= 0x03FFu;

        f_exp = (uint32_t) (exp + 127) << 23;
        f_sig = sig << 13;

        out.u = f_sign | f_exp | f_sig;
        return out.f;
    }

    if (h_exp == 0x7C00u) {
        f_exp = 0xFFu << 23;
        f_sig = h_sig << 13;

        out.u = f_sign | f_exp | f_sig;
        return out.f;
    }

    const int exp = (int) (h_exp >> 10) - 15;
    f_exp = (uint32_t) (exp + 127) << 23;
    f_sig = h_sig << 13;

    out.u = f_sign | f_exp | f_sig;
    return out.f;
}

static void fill_inputs(std::vector<float> & A, std::vector<float> & x, int rows, int cols) {
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            const int v = (r * 17 + c * 31 + 7) % 251;
            A[(size_t) r * cols + c] = ((float) v - 125.0f) * 0.003f;
        }
    }

    for (int c = 0; c < cols; ++c) {
        const int v = (c * 13 + 5) % 197;
        x[c] = ((float) v - 98.0f) * 0.004f;
    }
}

static void pack_q8_0(
    const std::vector<float> & A,
    std::vector<ggml_cuda8_q8_0_block> & Aq,
    int rows,
    int cols
) {
    const int blocks_per_row =
        (cols + GGML_CUDA8_QK8_0 - 1) / GGML_CUDA8_QK8_0;

    Aq.resize((size_t) rows * blocks_per_row);

    for (int r = 0; r < rows; ++r) {
        for (int ib = 0; ib < blocks_per_row; ++ib) {
            ggml_cuda8_q8_0_block & block = Aq[(size_t) r * blocks_per_row + ib];

            float max_abs = 0.0f;

            for (int k = 0; k < GGML_CUDA8_QK8_0; ++k) {
                const int col = ib * GGML_CUDA8_QK8_0 + k;

                float v = 0.0f;
                if (col < cols) {
                    v = A[(size_t) r * cols + col];
                }

                max_abs = std::max(max_abs, std::fabs(v));
            }

            float id = 0.0f;
            float d  = 0.0f;

            if (max_abs > 0.0f) {
                d  = max_abs / 127.0f;
                id = 127.0f / max_abs;
            }

            block.d = fp32_to_fp16_bits(d);

            for (int k = 0; k < GGML_CUDA8_QK8_0; ++k) {
                const int col = ib * GGML_CUDA8_QK8_0 + k;

                float v = 0.0f;
                if (col < cols) {
                    v = A[(size_t) r * cols + col];
                }

                int q = 0;

                if (max_abs > 0.0f) {
                    q = (int) std::floor(v * id + (v >= 0.0f ? 0.5f : -0.5f));
                }

                if (q > 127) {
                    q = 127;
                }

                if (q < -127) {
                    q = -127;
                }

                block.qs[k] = (int8_t) q;
            }
        }
    }
}

static void cpu_q8_0_mul_mat_ref(
    const std::vector<ggml_cuda8_q8_0_block> & Aq,
    const std::vector<float> & x,
    std::vector<float> & y,
    int rows,
    int cols
) {
    const int blocks_per_row =
        (cols + GGML_CUDA8_QK8_0 - 1) / GGML_CUDA8_QK8_0;

    for (int r = 0; r < rows; ++r) {
        const ggml_cuda8_q8_0_block * row_blocks =
            &Aq[(size_t) r * blocks_per_row];

        float sum = 0.0f;

        for (int ib = 0; ib < blocks_per_row; ++ib) {
            const ggml_cuda8_q8_0_block & block = row_blocks[ib];
            const float d = fp16_bits_to_fp32(block.d);

            const int base_col = ib * GGML_CUDA8_QK8_0;
            float block_sum = 0.0f;

            for (int k = 0; k < GGML_CUDA8_QK8_0; ++k) {
                const int col = base_col + k;

                if (col < cols) {
                    block_sum += ((float) block.qs[k]) * x[col];
                }
            }

            sum += d * block_sum;
        }

        y[r] = sum;
    }
}

static void compute_error(
    const std::vector<float> & ref,
    const std::vector<float> & got,
    double & max_abs_err,
    double & max_rel_err
) {
    max_abs_err = 0.0;
    max_rel_err = 0.0;

    for (size_t i = 0; i < ref.size(); ++i) {
        const double r = (double) ref[i];
        const double g = (double) got[i];

        const double abs_err = std::fabs(g - r);
        const double denom = std::max(1e-9, std::fabs(r));
        const double rel_err = abs_err / denom;

        max_abs_err = std::max(max_abs_err, abs_err);
        max_rel_err = std::max(max_rel_err, rel_err);
    }
}

static bool run_one_shape(int rows, int cols) {
    const int blocks_per_row =
        (cols + GGML_CUDA8_QK8_0 - 1) / GGML_CUDA8_QK8_0;

    const size_t n_A  = (size_t) rows * cols;
    const size_t n_x  = (size_t) cols;
    const size_t n_y  = (size_t) rows;
    const size_t n_Aq = (size_t) rows * blocks_per_row;

    const size_t bytes_Aq = n_Aq * sizeof(ggml_cuda8_q8_0_block);
    const size_t bytes_x  = n_x  * sizeof(float);
    const size_t bytes_y  = n_y  * sizeof(float);

    std::printf("\n");
    std::printf("G0 MUL_MAT-style test: rows=%d cols=%d blocks_per_row=%d\n",
        rows, cols, blocks_per_row);

    std::vector<float> A(n_A);
    std::vector<float> x(n_x);
    std::vector<float> y_cpu(n_y, 0.0f);
    std::vector<float> y_gpu(n_y, 0.0f);
    std::vector<ggml_cuda8_q8_0_block> Aq;

    fill_inputs(A, x, rows, cols);
    pack_q8_0(A, Aq, rows, cols);
    cpu_q8_0_mul_mat_ref(Aq, x, y_cpu, rows, cols);

    ggml_cuda8_q8_0_block * d_Aq = NULL;
    float * d_x = NULL;
    float * d_y = NULL;

    CUDA_CHECK_BOOL(cudaMalloc((void **) &d_Aq, bytes_Aq));
    CUDA_CHECK_BOOL(cudaMalloc((void **) &d_x,  bytes_x));
    CUDA_CHECK_BOOL(cudaMalloc((void **) &d_y,  bytes_y));

    CUDA_CHECK_BOOL(cudaMemcpy(d_Aq, &Aq[0], bytes_Aq, cudaMemcpyHostToDevice));
    CUDA_CHECK_BOOL(cudaMemcpy(d_x,  &x[0],  bytes_x,  cudaMemcpyHostToDevice));
    CUDA_CHECK_BOOL(cudaMemset(d_y, 0, bytes_y));

    const int rc = ggml_cuda8_mul_mat_q8_0_f32(d_Aq, d_x, d_y, rows, cols);

    if (rc != 0) {
        std::fprintf(stderr, "ggml-cuda8-ggml-smoke: mul_mat shim failed rc=%d\n", rc);
        cudaFree(d_Aq);
        cudaFree(d_x);
        cudaFree(d_y);
        return false;
    }

    CUDA_CHECK_BOOL(cudaMemcpy(&y_gpu[0], d_y, bytes_y, cudaMemcpyDeviceToHost));

    cudaFree(d_Aq);
    cudaFree(d_x);
    cudaFree(d_y);

    double max_abs_err = 0.0;
    double max_rel_err = 0.0;

    compute_error(y_cpu, y_gpu, max_abs_err, max_rel_err);

    std::printf("max_abs_err vs CPU Q8_0 ref: %.9g\n", max_abs_err);
    std::printf("max_rel_err vs CPU Q8_0 ref: %.9g\n", max_rel_err);

    const double abs_tol = 2e-4;
    const double rel_tol = 2e-4;

    if (max_abs_err > abs_tol && max_rel_err > rel_tol) {
        std::fprintf(stderr,
            "ggml-cuda8-ggml-smoke: FAIL rows=%d cols=%d abs=%g rel=%g\n",
            rows, cols, max_abs_err, max_rel_err);
        return false;
    }

    std::printf("PASS\n");
    return true;
}

int main(int argc, char ** argv) {
    (void) argc;
    (void) argv;

    std::printf("ggml-cuda8-ggml-smoke: starting\n");

    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);

    if (err != cudaSuccess) {
        std::fprintf(stderr, "cudaGetDeviceCount failed: %s (%d)\n",
            cudaGetErrorString(err), (int) err);
        return 1;
    }

    if (count <= 0) {
        std::fprintf(stderr, "ggml-cuda8-ggml-smoke: no CUDA devices found\n");
        return 1;
    }

    err = cudaSetDevice(0);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "cudaSetDevice failed: %s (%d)\n",
            cudaGetErrorString(err), (int) err);
        return 1;
    }

    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, 0);

    if (err != cudaSuccess) {
        std::fprintf(stderr, "cudaGetDeviceProperties failed: %s (%d)\n",
            cudaGetErrorString(err), (int) err);
        return 1;
    }

    std::printf("CUDA device 0: %s | cc %d.%d | global mem %.1f MiB\n",
        prop.name,
        prop.major,
        prop.minor,
        (double) prop.totalGlobalMem / (1024.0 * 1024.0)
    );

    bool ok = true;

    ok = ok && run_one_shape(4,    32);
    ok = ok && run_one_shape(128,  64);
    ok = ok && run_one_shape(256, 128);
    ok = ok && run_one_shape(512, 512);

    if (!ok) {
        std::fprintf(stderr, "ggml-cuda8-ggml-smoke: FAILED\n");
        return 1;
    }

    std::printf("\n");
    std::printf("ggml-cuda8-ggml-smoke: SUCCESS\n");
    return 0;
}
