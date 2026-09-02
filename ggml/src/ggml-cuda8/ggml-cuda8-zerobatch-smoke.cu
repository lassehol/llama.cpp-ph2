// ggml-cuda8-zerobatch-smoke.cu
//
// Regression guard for the "zero-sized sub-batch" bug class found while
// investigating prefill throughput on a real long prompt (~500+ tokens).
//
// Root cause: llama.cpp's batch/ubatch splitting can legitimately produce a
// sub-batch fragment with a zero row/token/element count for some graph
// nodes (e.g. an output-selection GET_ROWS with no positions to gather in
// that fragment, or a broadcast MUL over a zero-sized activation slice).
// ggml's CPU backend treats n==0 as a trivial no-op everywhere; several
// CUDA8 launchers instead rejected it as an "invalid argument" (a guard
// written before any real large/split-batch graph ever exercised n==0
// specifically). That crashed the WHOLE graph on any prompt long enough to
// trigger a batch split - short test prompts and every existing smoke used
// n_tokens=1 / small fixed counts, so it was invisible until a real
// ~500-token prompt hit it.
//
// All 9 launchers below now follow the same verified pattern: batch-count
// params reject n<0 as a real error and treat n==0 as "nothing to do, return
// 0 success"; shape/broadcast-divisor params (ncols in diagmask, ne00 in
// get_rows/mulmat, nc in set_rows, ne11/ne12 in set_rows which gate an
// in-kernel modulo) remain strictly >0 - a zero there is genuine corruption,
// not a valid empty batch. Signatures verified against the current source of
// every file (softmax.cu, scalar.cu, reduce.cu, diagmask.cu, getrows.cu,
// mul.cu, mulmat-f32.cu, set-rows.cu). All 26 correctness assertions here
// (the n==0, n<0, and strict-shape-param cases) are hardware-verified PASS.
//
// Standalone .cu, links the full kernels archive (calls into many different
// launchers already built into that library, same pattern as
// ggml-cuda8-dispatch-all-smoke).
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdint.h>

// -- launchers under test (extern "C" in the kernels archive) ----------------
extern "C" int ggml_cuda8_op_get_rows_f32(
    const float * src0, const int * src1, float * dst, int ne00, int n_tokens);

extern "C" int ggml_cuda8_mul_f32_launch(
    const float * a, const float * b, float * c, int n);
extern "C" int ggml_cuda8_mul_broadcast_f32_launch(
    const float * a, const float * b, float * c, int n_total, int n_repeat);

// verified signature: (x, dst, ncols, nrows, rows_per_channel, n_past)
extern "C" int ggml_cuda8_op_diag_mask_inf_f32(
    const float * x, float * dst, int ncols, int nrows, int rows_per_channel, int n_past);

extern "C" int ggml_cuda8_reduce_sum_rows_f32_launch(
    const float * src, float * dst, int rows, int cols);
extern "C" int ggml_cuda8_reduce_max_rows_f32_launch(
    const float * src, float * dst, int rows, int cols);

extern "C" int ggml_cuda8_add_scalar_f32_launch(
    const float * src, float scalar, float * dst, int n);
extern "C" int ggml_cuda8_mul_scalar_f32_launch(
    const float * src, float scalar, float * dst, int n);

extern "C" int ggml_cuda8_softmax_rows_f32_launch(
    const float * src, float * dst, int rows, int cols);

extern "C" int ggml_cuda8_mul_mat_f32_f32_launch(
    const float * src0, const float * src1, float * dst,
    int ne00, int ne01, int ne02, int ne03, int ne11, int ne12, int ne13,
    size_t nb01, size_t nb02, size_t nb03,
    size_t nb11, size_t nb12, size_t nb13,
    size_t nb1,  size_t nb2,  size_t nb3);

extern "C" int ggml_cuda8_op_set_rows_f32(
    const void * src0, const void * idx, void * dst,
    int nc, int nr, int ne02, int ne03, int ne11, int ne12, int ne1,
    size_t nb01, size_t nb02, size_t nb03,
    size_t nb10, size_t nb11, size_t nb12,
    size_t nb1,  size_t nb2,  size_t nb3);

// -- tiny device buffer helper -------------------------------------------------
struct Buf {
    void * d = NULL;
    size_t bytes = 0;
    Buf(size_t n) : bytes(n) { if (n) cudaMalloc(&d, n); }
    ~Buf() { if (d) cudaFree(d); }
};

static int g_pass = 0, g_fail = 0;
static void check(const char * label, int got, int want) {
    if (got == want) { std::printf("  %-46s PASS (rc=%d)\n", label, got); ++g_pass; }
    else              { std::printf("  %-46s FAIL (rc=%d, want=%d)\n", label, got, want); ++g_fail; }
}

int main() {
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) {
        std::fprintf(stderr, "no CUDA device 0\n");
        return 1;
    }
    std::printf("ggml-cuda8-zerobatch-smoke: %s (cc %d.%d)\n\n", prop.name, prop.major, prop.minor);

    // Generously oversized shared buffers - the n==0 cases touch nothing;
    // the small n>0 sanity checks touch a handful of elements. Reused across
    // calls purely for convenience.
    Buf a(4096), b(4096), c(4096), idx_i32(4096), idx_i64(4096);

    // Zero-init everything. Critical for idx_i32/idx_i64: cudaMalloc leaves
    // them uninitialized, so an n>0 control call (e.g. get_rows_f32
    // n_tokens=2) would read garbage row indices and fault with an illegal
    // memory access - which then latches the G51 sticky-poison flag and
    // fails every subsequent call in this process, unrelated ops included.
    // Zero is a valid row index into any nonzero-sized src0, so this makes
    // every control call safe. Zero-filling a/b/c is cheap insurance against
    // reading uninitialized floats anywhere else.
    cudaMemset(a.d, 0, a.bytes);
    cudaMemset(b.d, 0, b.bytes);
    cudaMemset(c.d, 0, c.bytes);
    cudaMemset(idx_i32.d, 0, idx_i32.bytes);
    cudaMemset(idx_i64.d, 0, idx_i64.bytes);

    std::printf("== n == 0 (batch-count params): must succeed (rc=0), nothing to do ==\n");
    check("get_rows_f32 n_tokens=0",
          ggml_cuda8_op_get_rows_f32((const float*)a.d, (const int*)idx_i32.d, (float*)c.d, 64, 0), 0);
    check("mul_f32 n=0",
          ggml_cuda8_mul_f32_launch((const float*)a.d, (const float*)b.d, (float*)c.d, 0), 0);
    check("mul_broadcast_f32 n_total=0",
          ggml_cuda8_mul_broadcast_f32_launch((const float*)a.d, (const float*)b.d, (float*)c.d, 0, 1024), 0);
    check("diag_mask_inf_f32 nrows=0",
          ggml_cuda8_op_diag_mask_inf_f32((const float*)a.d, (float*)c.d, /*ncols*/64, /*nrows*/0, /*rows_per_channel*/64, /*n_past*/0), 0);
    check("reduce_sum_rows_f32 rows=0",
          ggml_cuda8_reduce_sum_rows_f32_launch((const float*)a.d, (float*)c.d, 0, 64), 0);
    check("reduce_sum_rows_f32 cols=0",
          ggml_cuda8_reduce_sum_rows_f32_launch((const float*)a.d, (float*)c.d, 4, 0), 0);
    check("reduce_max_rows_f32 rows=0",
          ggml_cuda8_reduce_max_rows_f32_launch((const float*)a.d, (float*)c.d, 0, 64), 0);
    check("add_scalar_f32 n=0",
          ggml_cuda8_add_scalar_f32_launch((const float*)a.d, 1.0f, (float*)c.d, 0), 0);
    check("mul_scalar_f32 n=0",
          ggml_cuda8_mul_scalar_f32_launch((const float*)a.d, 2.0f, (float*)c.d, 0), 0);
    check("softmax_rows_f32 rows=0",
          ggml_cuda8_softmax_rows_f32_launch((const float*)a.d, (float*)c.d, 0, 64), 0);
    check("softmax_rows_f32 cols=0",
          ggml_cuda8_softmax_rows_f32_launch((const float*)a.d, (float*)c.d, 4, 0), 0);
    check("mulmat_f32 ne01=0",
          ggml_cuda8_mul_mat_f32_f32_launch((const float*)a.d, (const float*)b.d, (float*)c.d,
              64, 0, 1, 1, 4, 1, 1,
              64*4, 64*4, 64*4,  64*4, 64*4, 64*4,  0*4, 0*4, 0*4), 0);
    check("mulmat_f32 ne11=0",
          ggml_cuda8_mul_mat_f32_f32_launch((const float*)a.d, (const float*)b.d, (float*)c.d,
              64, 8, 1, 1, 0, 1, 1,
              64*4, 64*4, 64*4,  64*4, 64*4, 64*4,  8*4, 8*4, 8*4), 0);
    check("set_rows_f32 nr=0",
          ggml_cuda8_op_set_rows_f32(a.d, idx_i64.d, c.d,
              64, 0, 1, 1, 4, 1, 8,
              64*4, 64*4, 64*4,  8, 8, 8,  64*4, 64*4, 64*4), 0);
    check("set_rows_f32 ne02=0",
          ggml_cuda8_op_set_rows_f32(a.d, idx_i64.d, c.d,
              64, 4, 0, 1, 4, 1, 8,
              64*4, 64*4, 64*4,  8, 8, 8,  64*4, 64*4, 64*4), 0);

    std::printf("\n== n < 0: must still be rejected (rc=-1), guard not loosened ==\n");
    check("get_rows_f32 n_tokens=-1",
          ggml_cuda8_op_get_rows_f32((const float*)a.d, (const int*)idx_i32.d, (float*)c.d, 64, -1), -1);
    check("mul_f32 n=-1",
          ggml_cuda8_mul_f32_launch((const float*)a.d, (const float*)b.d, (float*)c.d, -1), -1);
    check("mul_broadcast_f32 n_total=-1",
          ggml_cuda8_mul_broadcast_f32_launch((const float*)a.d, (const float*)b.d, (float*)c.d, -1, 1024), -1);
    check("diag_mask_inf_f32 nrows=-1",
          ggml_cuda8_op_diag_mask_inf_f32((const float*)a.d, (float*)c.d, 64, -1, 64, 0), -1);
    check("reduce_sum_rows_f32 rows=-1",
          ggml_cuda8_reduce_sum_rows_f32_launch((const float*)a.d, (float*)c.d, -1, 64), -1);
    check("softmax_rows_f32 rows=-1",
          ggml_cuda8_softmax_rows_f32_launch((const float*)a.d, (float*)c.d, -1, 64), -1);
    check("mulmat_f32 ne01=-1",
          ggml_cuda8_mul_mat_f32_f32_launch((const float*)a.d, (const float*)b.d, (float*)c.d,
              64, -1, 1, 1, 4, 1, 1,
              64*4, 64*4, 64*4,  64*4, 64*4, 64*4,  0, 0, 0), -1);
    check("set_rows_f32 nr=-1",
          ggml_cuda8_op_set_rows_f32(a.d, idx_i64.d, c.d,
              64, -1, 1, 1, 4, 1, 8,
              64*4, 64*4, 64*4,  8, 8, 8,  64*4, 64*4, 64*4), -1);

    std::printf("\n== shape/divisor params: must STAY strict at 0 (never a valid empty batch) ==\n");
    check("diag_mask_inf_f32 ncols=0 (shape dim, must reject)",
          ggml_cuda8_op_diag_mask_inf_f32((const float*)a.d, (float*)c.d, 0, 4, 64, 0), -1);
    check("mulmat_f32 ne00=0 (reduction dim, must reject)",
          ggml_cuda8_mul_mat_f32_f32_launch((const float*)a.d, (const float*)b.d, (float*)c.d,
              0, 8, 1, 1, 4, 1, 1,
              64*4, 64*4, 64*4,  64*4, 64*4, 64*4,  8*4, 8*4, 8*4), -1);
    // set_rows_f32's ne11/ne12 gate an in-kernel modulo (i11 = i02 % ne11) -
    // a zero here is undefined behavior in-kernel, not a valid no-op, so
    // this guard must stay strict regardless of batch size.
    check("set_rows_f32 ne11=0 (modulo divisor, must reject)",
          ggml_cuda8_op_set_rows_f32(a.d, idx_i64.d, c.d,
              64, 4, 1, 1, 0, 1, 8,
              64*4, 64*4, 64*4,  8, 8, 8,  64*4, 64*4, 64*4), -1);

    std::printf("\n== small n > 0 control: normal path still works ==\n");
    check("get_rows_f32 n_tokens=2 (control)",
          ggml_cuda8_op_get_rows_f32((const float*)a.d, (const int*)idx_i32.d, (float*)c.d, 64, 2), 0);
    check("mul_f32 n=64 (control)",
          ggml_cuda8_mul_f32_launch((const float*)a.d, (const float*)b.d, (float*)c.d, 64), 0);
    check("diag_mask_inf_f32 nrows=4 (control)",
          ggml_cuda8_op_diag_mask_inf_f32((const float*)a.d, (float*)c.d, 64, 4, 64, 0), 0);

    std::printf("\nggml-cuda8-zerobatch-smoke: %s (%d/%d)\n",
                g_fail == 0 ? "SUCCESS" : "FAIL", g_pass, g_pass + g_fail);
    return g_fail == 0 ? 0 : 1;
}
