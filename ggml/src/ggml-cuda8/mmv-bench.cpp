// ggml/src/ggml-cuda8/mmv-bench.cpp
//
// Standalone CUDA8/Fermi MMV benchmark for ggml-cuda8.
//
// Compares:
//   - CPU reference
//   - GPU naive kernel
//   - GPU block-per-row reduction kernel
//   - GPU selected wrapper
//
// Build target:
//   ggml-cuda8-mmv-bench

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

extern "C" int ggml_cuda8_mmv_f32(
    const float * d_A,
    const float * d_x,
    float * d_y,
    int rows,
    int cols
);

extern "C" int ggml_cuda8_mmv_f32_naive(
    const float * d_A,
    const float * d_x,
    float * d_y,
    int rows,
    int cols
);

extern "C" int ggml_cuda8_mmv_f32_block(
    const float * d_A,
    const float * d_x,
    float * d_y,
    int rows,
    int cols
);

typedef int (*mmv_func_t)(
    const float * d_A,
    const float * d_x,
    float * d_y,
    int rows,
    int cols
);

#define CUDA_CHECK(call)                                                            \
    do {                                                                            \
        cudaError_t err__ = (call);                                                 \
        if (err__ != cudaSuccess) {                                                 \
            std::fprintf(stderr,                                                    \
                "mmv-bench: CUDA error %s (%d) at %s:%d\n",                         \
                cudaGetErrorString(err__), (int) err__, __FILE__, __LINE__);        \
            return false;                                                           \
        }                                                                           \
    } while (0)

struct bench_shape {
    int rows;
    int cols;
    int gpu_iters;
};

struct gpu_result {
    bool ok;
    double avg_sec;
    double gflops;
    double max_abs_err;
    double max_rel_err;
};

static void fill_inputs(std::vector<float> & A, std::vector<float> & x, int rows, int cols) {
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            const int v = (r * 17 + c * 31 + 7) % 23;
            A[(size_t) r * cols + c] = ((float) v - 11.0f) * 0.01f;
        }
    }

    for (int c = 0; c < cols; ++c) {
        const int v = (c * 13 + 5) % 19;
        x[c] = ((float) v - 9.0f) * 0.02f;
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

static double seconds_now() {
    typedef std::chrono::high_resolution_clock clock_type;
    return std::chrono::duration<double>(clock_type::now().time_since_epoch()).count();
}

static void compute_error(
    const std::vector<float> & ref,
    const std::vector<float> & got,
    double & max_abs_err,
    double & max_rel_err
) {
    max_abs_err = 0.0;
    max_rel_err = 0.0;

    const size_t n = ref.size();

    for (size_t i = 0; i < n; ++i) {
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
    mmv_func_t fn,
    const float * d_A,
    const float * d_x,
    float * d_y,
    const std::vector<float> & y_cpu,
    std::vector<float> & y_gpu,
    int rows,
    int cols,
    int iters,
    gpu_result & result
) {
    result.ok = false;
    result.avg_sec = 0.0;
    result.gflops = 0.0;
    result.max_abs_err = 0.0;
    result.max_rel_err = 0.0;

    const size_t bytes_y = (size_t) rows * sizeof(float);

    CUDA_CHECK(cudaMemset(d_y, 0, bytes_y));

    // Warmup.
    for (int i = 0; i < 2; ++i) {
        if (fn(d_A, d_x, d_y, rows, cols) != 0) {
            std::fprintf(stderr, "mmv-bench: %s warmup failed\n", name);
            return false;
        }
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    const double t0 = seconds_now();

    for (int i = 0; i < iters; ++i) {
        if (fn(d_A, d_x, d_y, rows, cols) != 0) {
            std::fprintf(stderr, "mmv-bench: %s failed at iter %d\n", name, i);
            return false;
        }
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    const double t1 = seconds_now();

    CUDA_CHECK(cudaMemcpy(&y_gpu[0], d_y, bytes_y, cudaMemcpyDeviceToHost));

    const double total_sec = t1 - t0;
    const double avg_sec = total_sec / (double) iters;
    const double ops = 2.0 * (double) rows * (double) cols;
    const double gflops = ops / avg_sec / 1.0e9;

    double max_abs_err = 0.0;
    double max_rel_err = 0.0;

    compute_error(y_cpu, y_gpu, max_abs_err, max_rel_err);

    result.ok = true;
    result.avg_sec = avg_sec;
    result.gflops = gflops;
    result.max_abs_err = max_abs_err;
    result.max_rel_err = max_rel_err;

    return true;
}

static bool check_result_tolerance(
    const char * name,
    const gpu_result & r,
    int rows,
    int cols
) {
    // The block reduction changes accumulation order, so error can be slightly
    // larger than the naive kernel. Keep tolerance realistic for F32 reduction.
    const double abs_tol = 1e-4;
    const double rel_tol = 1e-4;

    if (r.max_abs_err > abs_tol && r.max_rel_err > rel_tol) {
        std::fprintf(stderr,
            "mmv-bench: FAIL %s rows=%d cols=%d abs_err=%g rel_err=%g\n",
            name, rows, cols, r.max_abs_err, r.max_rel_err
        );
        return false;
    }

    return true;
}

static bool run_one_shape(const bench_shape & s) {
    const int rows = s.rows;
    const int cols = s.cols;

    const size_t n_A = (size_t) rows * cols;
    const size_t n_x = (size_t) cols;
    const size_t n_y = (size_t) rows;

    const size_t bytes_A = n_A * sizeof(float);
    const size_t bytes_x = n_x * sizeof(float);
    const size_t bytes_y = n_y * sizeof(float);

    std::printf("\n");
    std::printf("============================================================\n");
    std::printf("ggml-cuda8/mmv-bench: rows=%d cols=%d\n", rows, cols);
    std::printf("A size: %.2f MiB | x size: %.2f KiB | y size: %.2f KiB\n",
        (double) bytes_A / (1024.0 * 1024.0),
        (double) bytes_x / 1024.0,
        (double) bytes_y / 1024.0
    );

    std::vector<float> h_A(n_A);
    std::vector<float> h_x(n_x);
    std::vector<float> h_y_cpu(n_y, 0.0f);
    std::vector<float> h_y_gpu(n_y, 0.0f);

    fill_inputs(h_A, h_x, rows, cols);

    const double cpu_t0 = seconds_now();
    cpu_mmv_f32(h_A, h_x, h_y_cpu, rows, cols);
    const double cpu_t1 = seconds_now();

    const double cpu_sec = cpu_t1 - cpu_t0;
    const double ops = 2.0 * (double) rows * (double) cols;
    const double cpu_gflops = ops / cpu_sec / 1.0e9;

    float * d_A = NULL;
    float * d_x = NULL;
    float * d_y = NULL;

    CUDA_CHECK(cudaMalloc((void **) &d_A, bytes_A));
    CUDA_CHECK(cudaMalloc((void **) &d_x, bytes_x));
    CUDA_CHECK(cudaMalloc((void **) &d_y, bytes_y));

    CUDA_CHECK(cudaMemcpy(d_A, &h_A[0], bytes_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, &h_x[0], bytes_x, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_y, 0, bytes_y));

    gpu_result naive;
    gpu_result block;
    gpu_result selected;

    bool ok = true;

    if (!run_gpu_variant("GPU naive", ggml_cuda8_mmv_f32_naive,
            d_A, d_x, d_y, h_y_cpu, h_y_gpu, rows, cols, s.gpu_iters, naive)) {
        ok = false;
    }

    if (ok && !run_gpu_variant("GPU block", ggml_cuda8_mmv_f32_block,
            d_A, d_x, d_y, h_y_cpu, h_y_gpu, rows, cols, s.gpu_iters, block)) {
        ok = false;
    }

    if (ok && !run_gpu_variant("GPU selected", ggml_cuda8_mmv_f32,
            d_A, d_x, d_y, h_y_cpu, h_y_gpu, rows, cols, s.gpu_iters, selected)) {
        ok = false;
    }

    cudaFree(d_A);
    cudaFree(d_x);
    cudaFree(d_y);

    if (!ok) {
        return false;
    }

    std::printf("CPU reference: %.6f s | %.4f GFLOP/s\n", cpu_sec, cpu_gflops);

    std::printf("GPU naive:     %.6f s | %.4f GFLOP/s | speedup %.3fx | abs %.9g | rel %.9g\n",
        naive.avg_sec,
        naive.gflops,
        cpu_sec / naive.avg_sec,
        naive.max_abs_err,
        naive.max_rel_err
    );

    std::printf("GPU block:     %.6f s | %.4f GFLOP/s | speedup %.3fx | abs %.9g | rel %.9g\n",
        block.avg_sec,
        block.gflops,
        cpu_sec / block.avg_sec,
        block.max_abs_err,
        block.max_rel_err
    );

    std::printf("GPU selected:  %.6f s | %.4f GFLOP/s | speedup %.3fx | abs %.9g | rel %.9g\n",
        selected.avg_sec,
        selected.gflops,
        cpu_sec / selected.avg_sec,
        selected.max_abs_err,
        selected.max_rel_err
    );

    if (!check_result_tolerance("GPU naive", naive, rows, cols)) {
        return false;
    }

    if (!check_result_tolerance("GPU block", block, rows, cols)) {
        return false;
    }

    if (!check_result_tolerance("GPU selected", selected, rows, cols)) {
        return false;
    }

    if (cols >= 32) {
        std::printf("selector expected: block\n");
    } else {
        std::printf("selector expected: naive\n");
    }

    std::printf("PASS\n");
    return true;
}

int main(int argc, char ** argv) {
    (void) argc;
    (void) argv;

    std::printf("ggml-cuda8-mmv-bench: starting\n");

    int count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&count));

    if (count <= 0) {
        std::fprintf(stderr, "ggml-cuda8-mmv-bench: no CUDA devices found\n");
        return 1;
    }

    CUDA_CHECK(cudaSetDevice(0));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    std::printf("CUDA device 0: %s | cc %d.%d | global mem %.1f MiB\n",
        prop.name,
        prop.major,
        prop.minor,
        (double) prop.totalGlobalMem / (1024.0 * 1024.0)
    );

    const bench_shape shapes[] = {
        // Include shapes below and above the selector threshold.
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
        std::fprintf(stderr, "ggml-cuda8-mmv-bench: FAILED\n");
        return 1;
    }

    std::printf("\n");
    std::printf("ggml-cuda8-mmv-bench: SUCCESS\n");
    return 0;
}

