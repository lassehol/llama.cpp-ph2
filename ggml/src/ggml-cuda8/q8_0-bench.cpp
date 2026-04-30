// ggml/src/ggml-cuda8/q8_0-bench.cpp
//
// Standalone CUDA8/Fermi Q8_0 MMV benchmark.
//
// Compares:
//   - CPU F32 reference
//   - CPU Q8_0 reference
//   - GPU Q8_0 qblock
//   - GPU Q8_0 col_parallel
//   - GPU Q8_0 selected wrapper

#include "q8_0-mmv.cuh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

typedef int (*q8_0_mmv_func_t)(
    const ggml_cuda8_q8_0_block * d_Aq,
    const float * d_x,
    float * d_y,
    int rows,
    int cols
);

struct bench_shape {
    int rows;
    int cols;
    int gpu_iters;
};

struct gpu_result {
    bool ok;
    double avg_sec;
    double gflops_equiv;
    double max_abs_err_vs_qref;
    double max_rel_err_vs_qref;
};

static bool cuda_check_bool(cudaError_t err, const char * expr, const char * file, int line) {
    if (err != cudaSuccess) {
        std::fprintf(stderr,
            "q8_0-bench: CUDA error %s (%d) for %s at %s:%d\n",
            cudaGetErrorString(err), (int) err, expr, file, line);
        return false;
    }
    return true;
}

#define CUDA_CHECK_BOOL(call) \
    do { if (!cuda_check_bool((call), #call, __FILE__, __LINE__)) return false; } while (0)

static double seconds_now() {
    typedef std::chrono::high_resolution_clock clock_type;
    return std::chrono::duration<double>(clock_type::now().time_since_epoch()).count();
}

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

static void cpu_mmv_f32(
    const std::vector<float> & A,
    const std::vector<float> & x,
    std::vector<float> & y,
    int rows,
    int cols
) {
    for (int r = 0; r < rows; ++r) {
        const float * Arow = &A[(size_t) r * cols];

        float sum = 0.0f;
        for (int c = 0; c < cols; ++c) {
            sum += Arow[c] * x[c];
        }

        y[r] = sum;
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

            float d = 0.0f;
            float id = 0.0f;

            if (max_abs > 0.0f) {
                d = max_abs / 127.0f;
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

static void cpu_mmv_q8_0(
    const std::vector<ggml_cuda8_q8_0_block> & Aq,
    const std::vector<float> & x,
    std::vector<float> & y,
    int rows,
    int cols
) {
    const int blocks_per_row =
        (cols + GGML_CUDA8_QK8_0 - 1) / GGML_CUDA8_QK8_0;

    for (int r = 0; r < rows; ++r) {
        float sum = 0.0f;

        const ggml_cuda8_q8_0_block * row_blocks =
            &Aq[(size_t) r * blocks_per_row];

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

static bool run_gpu_variant(
    const char * name,
    q8_0_mmv_func_t fn,
    const ggml_cuda8_q8_0_block * d_Aq,
    const float * d_x,
    float * d_y,
    const std::vector<float> & y_qref,
    std::vector<float> & y_gpu,
    int rows,
    int cols,
    int iters,
    gpu_result & result
) {
    result.ok = false;
    result.avg_sec = 0.0;
    result.gflops_equiv = 0.0;
    result.max_abs_err_vs_qref = 0.0;
    result.max_rel_err_vs_qref = 0.0;

    const size_t bytes_y = (size_t) rows * sizeof(float);

    CUDA_CHECK_BOOL(cudaMemset(d_y, 0, bytes_y));

    for (int i = 0; i < 2; ++i) {
        if (fn(d_Aq, d_x, d_y, rows, cols) != 0) {
            std::fprintf(stderr, "q8_0-bench: %s warmup failed\n", name);
            return false;
        }
    }

    CUDA_CHECK_BOOL(cudaDeviceSynchronize());

    const double t0 = seconds_now();

    for (int i = 0; i < iters; ++i) {
        if (fn(d_Aq, d_x, d_y, rows, cols) != 0) {
            std::fprintf(stderr, "q8_0-bench: %s failed at iter %d\n", name, i);
            return false;
        }
    }

    CUDA_CHECK_BOOL(cudaDeviceSynchronize());

    const double t1 = seconds_now();

    CUDA_CHECK_BOOL(cudaMemcpy(&y_gpu[0], d_y, bytes_y, cudaMemcpyDeviceToHost));

    double max_abs_err = 0.0;
    double max_rel_err = 0.0;

    compute_error(y_qref, y_gpu, max_abs_err, max_rel_err);

    const double avg_sec = (t1 - t0) / (double) iters;
    const double ops_equiv = 2.0 * (double) rows * (double) cols;

    result.ok = true;
    result.avg_sec = avg_sec;
    result.gflops_equiv = ops_equiv / avg_sec / 1.0e9;
    result.max_abs_err_vs_qref = max_abs_err;
    result.max_rel_err_vs_qref = max_rel_err;

    return true;
}

static bool check_q8_result(
    const char * name,
    const gpu_result & r,
    int rows,
    int cols
) {
    const double abs_tol = 2e-4;
    const double rel_tol = 2e-4;

    if (r.max_abs_err_vs_qref > abs_tol && r.max_rel_err_vs_qref > rel_tol) {
        std::fprintf(stderr,
            "q8_0-bench: FAIL %s rows=%d cols=%d abs=%g rel=%g\n",
            name, rows, cols,
            r.max_abs_err_vs_qref,
            r.max_rel_err_vs_qref
        );
        return false;
    }

    return true;
}

static bool run_one_shape(const bench_shape & s) {
    const int rows = s.rows;
    const int cols = s.cols;

    const int blocks_per_row =
        (cols + GGML_CUDA8_QK8_0 - 1) / GGML_CUDA8_QK8_0;

    const size_t n_A = (size_t) rows * cols;
    const size_t n_x = (size_t) cols;
    const size_t n_y = (size_t) rows;
    const size_t n_Aq = (size_t) rows * blocks_per_row;

    const size_t bytes_A_f32 = n_A * sizeof(float);
    const size_t bytes_Aq    = n_Aq * sizeof(ggml_cuda8_q8_0_block);
    const size_t bytes_x     = n_x * sizeof(float);
    const size_t bytes_y     = n_y * sizeof(float);

    std::printf("\n");
    std::printf("============================================================\n");
    std::printf("ggml-cuda8/q8_0-bench: rows=%d cols=%d blocks_per_row=%d\n",
        rows, cols, blocks_per_row);
    std::printf("A F32 size: %.2f MiB | A Q8_0 size: %.2f MiB | x: %.2f KiB | y: %.2f KiB\n",
        (double) bytes_A_f32 / (1024.0 * 1024.0),
        (double) bytes_Aq    / (1024.0 * 1024.0),
        (double) bytes_x     / 1024.0,
        (double) bytes_y     / 1024.0
    );
    std::printf("sizeof(q8_0 block): %zu bytes\n", sizeof(ggml_cuda8_q8_0_block));

    std::vector<float> A(n_A);
    std::vector<float> x(n_x);
    std::vector<float> y_f32_ref(n_y, 0.0f);
    std::vector<float> y_q8_ref(n_y, 0.0f);
    std::vector<float> y_gpu(n_y, 0.0f);
    std::vector<ggml_cuda8_q8_0_block> Aq;

    fill_inputs(A, x, rows, cols);

    const double pack_t0 = seconds_now();
    pack_q8_0(A, Aq, rows, cols);
    const double pack_t1 = seconds_now();

    const double cpu_f32_t0 = seconds_now();
    cpu_mmv_f32(A, x, y_f32_ref, rows, cols);
    const double cpu_f32_t1 = seconds_now();

    const double cpu_q8_t0 = seconds_now();
    cpu_mmv_q8_0(Aq, x, y_q8_ref, rows, cols);
    const double cpu_q8_t1 = seconds_now();

    double quant_abs_err = 0.0;
    double quant_rel_err = 0.0;
    compute_error(y_f32_ref, y_q8_ref, quant_abs_err, quant_rel_err);

    ggml_cuda8_q8_0_block * d_Aq = NULL;
    float * d_x = NULL;
    float * d_y = NULL;

    CUDA_CHECK_BOOL(cudaMalloc((void **) &d_Aq, bytes_Aq));
    CUDA_CHECK_BOOL(cudaMalloc((void **) &d_x,  bytes_x));
    CUDA_CHECK_BOOL(cudaMalloc((void **) &d_y,  bytes_y));

    CUDA_CHECK_BOOL(cudaMemcpy(d_Aq, &Aq[0], bytes_Aq, cudaMemcpyHostToDevice));
    CUDA_CHECK_BOOL(cudaMemcpy(d_x,  &x[0],  bytes_x,  cudaMemcpyHostToDevice));
    CUDA_CHECK_BOOL(cudaMemset(d_y, 0, bytes_y));

    gpu_result qblock;
    gpu_result col_parallel;
    gpu_result selected;

    bool ok = true;

    if (!run_gpu_variant("GPU Q8_0 qblock", ggml_cuda8_q8_0_mmv_f32_qblock,
            d_Aq, d_x, d_y, y_q8_ref, y_gpu, rows, cols, s.gpu_iters, qblock)) {
        ok = false;
    }

    if (ok && !run_gpu_variant("GPU Q8_0 col_parallel", ggml_cuda8_q8_0_mmv_f32_col_parallel,
            d_Aq, d_x, d_y, y_q8_ref, y_gpu, rows, cols, s.gpu_iters, col_parallel)) {
        ok = false;
    }

    if (ok && !run_gpu_variant("GPU Q8_0 selected", ggml_cuda8_q8_0_mmv_f32,
            d_Aq, d_x, d_y, y_q8_ref, y_gpu, rows, cols, s.gpu_iters, selected)) {
        ok = false;
    }

    cudaFree(d_Aq);
    cudaFree(d_x);
    cudaFree(d_y);

    if (!ok) {
        return false;
    }

    const double ops_equiv = 2.0 * (double) rows * (double) cols;
    const double cpu_f32_sec = cpu_f32_t1 - cpu_f32_t0;
    const double cpu_q8_sec  = cpu_q8_t1  - cpu_q8_t0;
    const double pack_sec    = pack_t1    - pack_t0;

    const double cpu_f32_gflops = ops_equiv / cpu_f32_sec / 1.0e9;
    const double cpu_q8_gflops  = ops_equiv / cpu_q8_sec  / 1.0e9;

    std::printf("pack Q8_0:       %.6f s\n", pack_sec);
    std::printf("CPU F32 ref:     %.6f s | %.4f GFLOP/s equiv\n",
        cpu_f32_sec, cpu_f32_gflops);
    std::printf("CPU Q8_0 ref:    %.6f s | %.4f GFLOP/s equiv\n",
        cpu_q8_sec, cpu_q8_gflops);

    std::printf("GPU Q8_0 qblock:       %.6f s | %.4f GFLOP/s equiv | speedup vs CPU Q8 %.3fx | abs %.9g | rel %.9g\n",
        qblock.avg_sec,
        qblock.gflops_equiv,
        cpu_q8_sec / qblock.avg_sec,
        qblock.max_abs_err_vs_qref,
        qblock.max_rel_err_vs_qref);

    std::printf("GPU Q8_0 col_parallel: %.6f s | %.4f GFLOP/s equiv | speedup vs CPU Q8 %.3fx | abs %.9g | rel %.9g\n",
        col_parallel.avg_sec,
        col_parallel.gflops_equiv,
        cpu_q8_sec / col_parallel.avg_sec,
        col_parallel.max_abs_err_vs_qref,
        col_parallel.max_rel_err_vs_qref);

    std::printf("GPU Q8_0 selected:     %.6f s | %.4f GFLOP/s equiv | speedup vs CPU Q8 %.3fx | abs %.9g | rel %.9g\n",
        selected.avg_sec,
        selected.gflops_equiv,
        cpu_q8_sec / selected.avg_sec,
        selected.max_abs_err_vs_qref,
        selected.max_rel_err_vs_qref);

    std::printf("quant err vs F32 ref: abs %.9g | rel %.9g\n",
        quant_abs_err, quant_rel_err);

    if (!check_q8_result("qblock", qblock, rows, cols)) {
        return false;
    }

    if (!check_q8_result("col_parallel", col_parallel, rows, cols)) {
        return false;
    }

    if (!check_q8_result("selected", selected, rows, cols)) {
        return false;
    }

    if (cols >= 32) {
        std::printf("selector expected: col_parallel\n");
    } else {
        std::printf("selector expected: qblock\n");
    }

    std::printf("PASS\n");
    return true;
}

int main(int argc, char ** argv) {
    (void) argc;
    (void) argv;

    std::printf("ggml-cuda8-q8_0-bench: starting\n");

    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "q8_0-bench: cudaGetDeviceCount failed: %s (%d)\n",
            cudaGetErrorString(err), (int) err);
        return 1;
    }

    if (count <= 0) {
        std::fprintf(stderr, "ggml-cuda8-q8_0-bench: no CUDA devices found\n");
        return 1;
    }

    err = cudaSetDevice(0);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "q8_0-bench: cudaSetDevice failed: %s (%d)\n",
            cudaGetErrorString(err), (int) err);
        return 1;
    }

    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, 0);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "q8_0-bench: cudaGetDeviceProperties failed: %s (%d)\n",
            cudaGetErrorString(err), (int) err);
        return 1;
    }

    std::printf("CUDA device 0: %s | cc %d.%d | global mem %.1f MiB\n",
        prop.name,
        prop.major,
        prop.minor,
        (double) prop.totalGlobalMem / (1024.0 * 1024.0)
    );

    const bench_shape shapes[] = {
        { 128,   64,  30 },
        { 128,  128,  30 },
        { 256,  256,  20 },
        { 512,  512,  10 },
        { 1024, 512,  10 },
        { 1024, 1024, 5  },
        { 2048, 1024, 3  },
        { 4096, 1024, 2  },
    };

    const int n_shapes = (int) (sizeof(shapes) / sizeof(shapes[0]));

    bool ok = true;

    for (int i = 0; i < n_shapes; ++i) {
        if (!run_one_shape(shapes[i])) {
            ok = false;
            break;
        }
    }

    if (!ok) {
        std::fprintf(stderr, "ggml-cuda8-q8_0-bench: FAILED\n");
        return 1;
    }

    std::printf("\n");
    std::printf("ggml-cuda8-q8_0-bench: SUCCESS\n");
    return 0;
}
