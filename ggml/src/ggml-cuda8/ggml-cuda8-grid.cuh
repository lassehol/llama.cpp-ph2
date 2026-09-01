// ggml/src/ggml-cuda8/ggml-cuda8-grid.cuh
//
// G38: Fermi grid-dimension limits.
//
// On compute capability 2.x the maximum grid.x is 65535. It only becomes
// 2^31-1 from sm_30. At 256 threads/block that ceiling is reached at
// ~16.7M work items - exactly a 4096x4096 tensor, i.e. an ordinary weight
// matrix in any real model.
//
// Failure mode: the launch is rejected with cudaErrorInvalidConfiguration.
// Every launcher in this backend checks cudaGetLastError() straight after the
// launch, so this surfaces as a dispatch error rather than as wrong numbers.
// Loud, but it means any model with a large enough tensor simply cannot run.
//
// Two patterns keep us under the limit. Both need the kernel-side loop AND the
// host-side clamp - clamping alone would silently drop the work that does not
// fit in the first 65535 blocks.
//
//   element-wise, one thread per item:
//       const int stride = blockDim.x * gridDim.x;
//       for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) { ... }
//       launch with ggml_cuda8_grid_1d(n, block_size)
//
//   one block per row:
//       for (int row = blockIdx.x; row < nrows; row += gridDim.x) { ... }
//       launch with ggml_cuda8_grid_rows(nrows)
//
// Kernels holding shared state across the row loop (reduction scratch) must
// __syncthreads() at the end of each iteration, before the next iteration
// overwrites it. Note also that a row loop makes early `return` on an
// out-of-range row wrong - a returning thread would skip the __syncthreads()
// its block-mates are waiting on. Bound the loop instead of returning.
//
// ggml-cuda8-q4k.cu, -q6k.cu and q8_0-mmv.cu instead use a 2D grid
// (blockIdx.x + blockIdx.y * gridDim.x). That is equally correct and is left
// alone. New code should prefer the stride loops: they also bound the number
// of resident blocks, which suits a 7-SM Fermi part better than spawning one
// block per row of a large tensor.

#ifndef GGML_CUDA8_GRID_CUH
#define GGML_CUDA8_GRID_CUH

#define GGML_CUDA8_MAX_GRID_X 65535

// Blocks needed for n work items, clamped to the Fermi limit.
// Pair with a grid-stride loop in the kernel.
static inline int ggml_cuda8_grid_1d(long long n_items, int block_size) {
    if (n_items < 1 || block_size < 1) {
        return 1;
    }
    const long long blocks = (n_items + block_size - 1) / block_size;
    if (blocks > GGML_CUDA8_MAX_GRID_X) {
        return GGML_CUDA8_MAX_GRID_X;
    }
    return (int) blocks;
}

// Blocks for one-block-per-row kernels, clamped to the Fermi limit.
// Pair with a row-stride loop in the kernel.
static inline int ggml_cuda8_grid_rows(long long n_rows) {
    if (n_rows < 1) {
        return 1;
    }
    if (n_rows > GGML_CUDA8_MAX_GRID_X) {
        return GGML_CUDA8_MAX_GRID_X;
    }
    return (int) n_rows;
}

#endif // GGML_CUDA8_GRID_CUH
