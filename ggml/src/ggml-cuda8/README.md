# ggml-cuda8


<!-- G16_STATUS_START -->
## G16 status: real GGML graph-builder graph_compute coverage

Status: **PASS on GTX 560 / CUDA 8 / Fermi**.

G16 extends the CUDA8 backend validation from synthetic `ggml_cgraph` dispatch tests to **real GGML graph-builder-created graphs**. The validated path is:

```text
ggml_init / ggml_new_tensor_* / ggml_add / ggml_mul / ggml_soft_max / ggml_sum_rows
  -> ggml_new_graph / ggml_build_forward_expand
  -> backend-owned CUDA8 buffer residency rebinding
  -> ggml_backend_i.graph_compute
  -> CUDA8 residency-aware dispatcher
```

Validated G16 checkpoints:

- **G16A**: real graph-builder `ADD_F32` graph.
- **G16B**: real graph-builder `ADD_F32 -> MUL_SCALAR_F32` graph.
- **G16C**: real graph-builder `ADD_F32 -> MUL_SCALAR_F32 -> SOFTMAX_ROWS_F32` graph.
- **G16D**: real graph-builder attention-like graph: `ADD_F32 -> MUL_SCALAR_F32 -> SOFTMAX_ROWS_F32 -> REDUCE_SUM_ROWS_F32`.

Implementation notes:

- G16 graph-builder smoke targets use the standalone-GC source subset:

```text
../ggml.c
../ggml-quants.c
../ggml-threading.cpp
-ffunction-sections
-fdata-sections
-Wl,--gc-sections
```

- The standalone-GC pattern intentionally avoids linking the full upstream `ggml` target, because the full target pulls newer C++ sources such as `gguf.cpp` that are not compatible with the current CUDA8/Fermi legacy build mode.
- G16D uses a real `ggml_sum_rows()` graph-builder node. For CUDA8 dispatcher compatibility, the real `sum_rows` output is rebound as a compact **1D F32 vector of length `rows`**.
- The G16D focused regression also verifies G16A/B/C, the existing synthetic attention-like graph_compute smoke, and the dispatch-all kernel smoke.
<!-- G16_STATUS_END -->

<!-- G17_STATUS_START -->
## G17 status: real GGML graph-builder Q8_0 MUL_MAT coverage

Status: **PASS on GTX 560 / CUDA 8 / Fermi**.

G17 extends the real GGML graph-builder `graph_compute` path from F32 elementwise and row-wise graphs into the quantized matvec path.

Validated G17 checkpoints:

- **G17A**: real graph-builder `ggml_mul_mat(Q8_0, F32)` smoke introduced.
- **G17A2**: `ggml_backend_i.graph_compute` routes real `GGML_OP_MUL_MAT` nodes to the existing CUDA8 dispatcher operation `GGML_CUDA8_OP_MUL_MAT_Q8_0_F32_VEC`.
- **G17C**: real graph-builder Q8_0 MMV smoke now validates against a host-packed Q8_0 quantization reference instead of the earlier simplified `d = 1.0` block case.
- **G17D**: main regression and README status refreshed for the packed Q8_0 graph-builder path.

Validated path:

    A_f32
      -> host pack_q8_0(A_f32)
      -> Aq Q8_0 blocks

    ggml_new_tensor_2d(..., GGML_TYPE_Q8_0, cols, rows)
    ggml_new_tensor_1d(..., GGML_TYPE_F32, cols)
    ggml_mul_mat(Q8_0_matrix, F32_vector)
      -> ggml_new_graph / ggml_build_forward_expand
      -> backend-owned CUDA8 buffer residency rebinding
      -> ggml_backend_i.graph_compute
      -> GGML_OP_MUL_MAT router
      -> GGML_CUDA8_OP_MUL_MAT_Q8_0_F32_VEC
      -> CUDA8/Fermi Q8_0 MMV kernel

Supported real graph-builder MUL_MAT layout at this checkpoint:

    src0: GGML_TYPE_Q8_0 matrix, shape [cols, rows]
    src1: GGML_TYPE_F32 vector, shape [cols]
    dst:  GGML_TYPE_F32 vector, shape [rows]

Implementation notes:

- The G17 Q8_0 graph-builder smoke uses the same standalone-GC graph-builder target pattern as G16:

    ../ggml.c
    ../ggml-quants.c
    ../ggml-threading.cpp
    -ffunction-sections
    -fdata-sections
    -Wl,--gc-sections

- G17A2 adds a minimal `GGML_OP_MUL_MAT` case in `ggml-cuda8-ggml-backend.cpp`.
- The dispatcher still performs the final layout/type support check via `ggml_cuda8_dispatch_supported(...)`.
- G17C validates the graph-builder Q8_0 path using host-packed Q8_0 blocks and a CPU Q8_0 dequantized reference.
- G17C focused regression passes:
  - packed Q8_0 graph-builder MMV smoke,
  - real graph-builder attention-like G16D smoke,
  - dispatch-all CUDA8 kernel smoke.
<!-- G17_STATUS_END -->

<!-- G18_STATUS_START -->
## G18 status: real GGML graph-builder quantized pipeline coverage

Status: **PASS on GTX 560 / CUDA 8 / Fermi**.

G18 extends the real GGML graph-builder quantized path from standalone Q8_0 matvec into mixed quantized/F32 multi-node pipelines.

Validated G18 checkpoints:

- **G18A**: real graph-builder two-op quantized pipeline passes:
  - `h = ggml_mul_mat(Q8_0, x)`
  - `y = ggml_add(h, bias)`
- **G18B**: main regression and README status refreshed for the G18A pipeline.
- **G18C**: real graph-builder three-op quantized pipeline passes:
  - `h = ggml_mul_mat(Q8_0, x)`
  - `scaled = ggml_mul(h, scale)`
  - `y = ggml_add(scaled, bias)`
- **G18D**: main regression and README status refreshed for the G18C pipeline.

Validated G18C graph:

    Aq     : GGML_TYPE_Q8_0 matrix, shape [cols, rows]
    x      : GGML_TYPE_F32 vector, shape [cols]
    h      : GGML_TYPE_F32 vector, shape [rows]
    scale  : GGML_TYPE_F32 scalar, shape [1]
    scaled : GGML_TYPE_F32 vector, shape [rows]
    bias   : GGML_TYPE_F32 vector, shape [rows]
    y      : GGML_TYPE_F32 vector, shape [rows]

Validated G18C path:

    A_f32
      -> host pack_q8_0(A_f32)
      -> Aq Q8_0 blocks

    h      = ggml_mul_mat(Aq, x)
    scaled = ggml_mul(h, scale)
    y      = ggml_add(scaled, bias)

    ggml_build_forward_expand(y)
      -> real GGML graph with three nodes
      -> node 0: GGML_OP_MUL_MAT
      -> node 1: GGML_OP_MUL
      -> node 2: GGML_OP_ADD
      -> ggml_backend_i.graph_compute
      -> GGML_CUDA8_OP_MUL_MAT_Q8_0_F32_VEC
      -> GGML_CUDA8_OP_MUL_SCALAR_F32
      -> GGML_CUDA8_OP_ADD_F32
      -> CUDA8/Fermi kernels

Implementation notes:

- G18C reuses the packed Q8_0 reference strategy validated in G17C.
- G18C validates that the F32 output of a real graph-builder Q8_0 `MUL_MAT` node can feed directly into a scalar F32 `MUL` node and then into an F32 `ADD` node.
- The G18C smoke uses the same standalone-GC graph-builder target pattern as G16/G17/G18A:

    ../ggml.c
    ../ggml-quants.c
    ../ggml-threading.cpp
    -ffunction-sections
    -fdata-sections
    -Wl,--gc-sections

- G18C focused regression passes:
  - Q8_0 MUL_MAT -> MUL_SCALAR -> ADD graph-builder pipeline smoke,
  - Q8_0 MUL_MAT -> ADD graph-builder pipeline smoke,
  - packed Q8_0 graph-builder MMV smoke,
  - real graph-builder attention-like G16D smoke,
  - dispatch-all CUDA8 kernel smoke.
<!-- G18_STATUS_END -->

<!-- G19_STATUS_START -->
## G19 status: real GGML graph-builder quantized softmax pipeline coverage

Status: **PASS on GTX 560 / CUDA 8 / Fermi**.

G19 extends the real GGML graph-builder quantized pipeline from linear-style F32 post-ops into a row-wise softmax endpoint.

Validated G19 checkpoints:

- **G19A**: real graph-builder four-op quantized pipeline passes:
  - `h = ggml_mul_mat(Q8_0, x)`
  - `scaled = ggml_mul(h, scale)`
  - `biased = ggml_add(scaled, bias)`
  - `prob = ggml_soft_max(biased)`
- **G19B**: main regression and README status refreshed for the G19A pipeline.

Validated G19A graph:

    Aq     : GGML_TYPE_Q8_0 matrix, shape [cols, rows]
    x      : GGML_TYPE_F32 vector, shape [cols]
    h      : GGML_TYPE_F32 vector, shape [rows]
    scale  : GGML_TYPE_F32 scalar, shape [1]
    scaled : GGML_TYPE_F32 vector, shape [rows]
    bias   : GGML_TYPE_F32 vector, shape [rows]
    biased : GGML_TYPE_F32 vector, shape [rows]
    prob   : GGML_TYPE_F32 vector, shape [rows]

Validated G19A path:

    A_f32
      -> host pack_q8_0(A_f32)
      -> Aq Q8_0 blocks

    h      = ggml_mul_mat(Aq, x)
    scaled = ggml_mul(h, scale)
    biased = ggml_add(scaled, bias)
    prob   = ggml_soft_max(biased)

    ggml_build_forward_expand(prob)
      -> real GGML graph with four nodes
      -> node 0: GGML_OP_MUL_MAT
      -> node 1: GGML_OP_MUL
      -> node 2: GGML_OP_ADD
      -> node 3: GGML_OP_SOFT_MAX
      -> ggml_backend_i.graph_compute
      -> GGML_CUDA8_OP_MUL_MAT_Q8_0_F32_VEC
      -> GGML_CUDA8_OP_MUL_SCALAR_F32
      -> GGML_CUDA8_OP_ADD_F32
      -> GGML_CUDA8_OP_SOFTMAX_ROWS_F32
      -> CUDA8/Fermi kernels

Implementation notes:

- G19A reuses the packed Q8_0 reference strategy validated in G17C.
- G19A validates that the F32 output of a real graph-builder Q8_0 `MUL_MAT` node can feed through scalar F32 `MUL`, F32 `ADD`, and row-wise `SOFT_MAX`.
- The G19A smoke uses the same standalone-GC graph-builder target pattern as G16/G17/G18:

    ../ggml.c
    ../ggml-quants.c
    ../ggml-threading.cpp
    -ffunction-sections
    -fdata-sections
    -Wl,--gc-sections

- G19A focused regression passes:
  - Q8_0 MUL_MAT -> MUL_SCALAR -> ADD -> SOFTMAX graph-builder pipeline smoke,
  - Q8_0 MUL_MAT -> MUL_SCALAR -> ADD graph-builder pipeline smoke,
  - Q8_0 MUL_MAT -> ADD graph-builder pipeline smoke,
  - packed Q8_0 graph-builder MMV smoke,
  - real graph-builder attention-like G16D smoke,
  - dispatch-all CUDA8 kernel smoke.
<!-- G19_STATUS_END -->

<!-- G20_STATUS_START -->
## G20 status: real GGML graph-builder quantized softmax + sum_rows pipeline coverage

Status: **PASS on GTX 560 / CUDA 8 / Fermi**.

G20 extends the real GGML graph-builder quantized softmax pipeline by adding a post-softmax row-sum reduction endpoint.

Validated G20 checkpoints:

- **G20A**: real graph-builder five-op quantized pipeline passes:
  - `h = ggml_mul_mat(Q8_0, x)`
  - `scaled = ggml_mul(h, scale)`
  - `biased = ggml_add(scaled, bias)`
  - `prob = ggml_soft_max(biased)`
  - `row_sum = ggml_sum_rows(prob)`
- **G20B**: main regression and README status refreshed for the G20A pipeline.

Validated G20A graph:

    Aq      : GGML_TYPE_Q8_0 matrix, shape [cols, rows]
    x       : GGML_TYPE_F32 vector, shape [cols]
    h       : GGML_TYPE_F32 vector, shape [rows]
    scale   : GGML_TYPE_F32 scalar, shape [1]
    scaled  : GGML_TYPE_F32 vector, shape [rows]
    bias    : GGML_TYPE_F32 vector, shape [rows]
    biased  : GGML_TYPE_F32 vector, shape [rows]
    prob    : GGML_TYPE_F32 vector, shape [rows]
    row_sum : GGML_TYPE_F32 scalar/vector, shape [1]

Validated G20A path:

    A_f32
      -> host pack_q8_0(A_f32)
      -> Aq Q8_0 blocks

    h       = ggml_mul_mat(Aq, x)
    scaled  = ggml_mul(h, scale)
    biased  = ggml_add(scaled, bias)
    prob    = ggml_soft_max(biased)
    row_sum = ggml_sum_rows(prob)

    ggml_build_forward_expand(row_sum)
      -> real GGML graph with five nodes
      -> node 0: GGML_OP_MUL_MAT
      -> node 1: GGML_OP_MUL
      -> node 2: GGML_OP_ADD
      -> node 3: GGML_OP_SOFT_MAX
      -> node 4: GGML_OP_SUM_ROWS
      -> ggml_backend_i.graph_compute
      -> GGML_CUDA8_OP_MUL_MAT_Q8_0_F32_VEC
      -> GGML_CUDA8_OP_MUL_SCALAR_F32
      -> GGML_CUDA8_OP_ADD_F32
      -> GGML_CUDA8_OP_SOFTMAX_ROWS_F32
      -> GGML_CUDA8_OP_REDUCE_SUM_ROWS_F32
      -> CUDA8/Fermi kernels

Implementation notes:

- G20A reuses the packed Q8_0 reference strategy validated in G17C.
- G20A validates that the F32 output of a real graph-builder Q8_0 `MUL_MAT` node can feed through scalar F32 `MUL`, F32 `ADD`, row-wise `SOFT_MAX`, and row-sum reduction.
- The G20A smoke uses the same standalone-GC graph-builder target pattern as G16/G17/G18/G19:

    ../ggml.c
    ../ggml-quants.c
    ../ggml-threading.cpp
    -ffunction-sections
    -fdata-sections
    -Wl,--gc-sections

- G20A focused regression passes:
  - Q8_0 MUL_MAT -> MUL_SCALAR -> ADD -> SOFTMAX -> SUM_ROWS graph-builder pipeline smoke,
  - Q8_0 MUL_MAT -> MUL_SCALAR -> ADD -> SOFTMAX graph-builder pipeline smoke,
  - Q8_0 MUL_MAT -> MUL_SCALAR -> ADD graph-builder pipeline smoke,
  - Q8_0 MUL_MAT -> ADD graph-builder pipeline smoke,
  - packed Q8_0 graph-builder MMV smoke,
  - real graph-builder attention-like G16D smoke,
  - dispatch-all CUDA8 kernel smoke.
<!-- G20_STATUS_END -->

<!-- G21_STATUS_START -->
## G21 status: real GGML graph-builder quantized residual branch coverage

Status: **PASS on GTX 560 / CUDA 8 / Fermi**.

G21 validates a residual-style second F32 input branch after a real graph-builder Q8_0 projection.

Validated G21 checkpoints:

- **G21A**: real graph-builder residual-branch quantized pipeline passes:
  - `h = ggml_mul_mat(Q8_0, x)`
  - `y = ggml_add(h, residual)`
  - residual input branch isolation verified after `ggml_backend_i.graph_compute`
- **G21B**: main regression and README status refreshed for the G21A residual branch pipeline.

Validated G21A graph:

    Aq       : GGML_TYPE_Q8_0 matrix, shape [cols, rows]
    x        : GGML_TYPE_F32 vector, shape [cols]
    h        : GGML_TYPE_F32 vector, shape [rows]
    residual : GGML_TYPE_F32 vector, shape [rows]
    y        : GGML_TYPE_F32 vector, shape [rows]

Validated G21A path:

    A_f32
      -> host pack_q8_0(A_f32)
      -> Aq Q8_0 blocks

    h = ggml_mul_mat(Aq, x)
    y = ggml_add(h, residual)

    ggml_build_forward_expand(y)
      -> real GGML graph with two nodes
      -> node 0: GGML_OP_MUL_MAT
      -> node 1: GGML_OP_ADD
      -> ggml_backend_i.graph_compute
      -> GGML_CUDA8_OP_MUL_MAT_Q8_0_F32_VEC
      -> GGML_CUDA8_OP_ADD_F32
      -> CUDA8/Fermi kernels

Implementation notes:

- G21A reuses the packed Q8_0 reference strategy validated in G17C.
- G21A uses an explicit second F32 input branch named `residual`.
- G21A verifies that the residual input branch remains unchanged after graph dispatch.
- This is distinct from the earlier bias-add smoke because it explicitly validates input branch isolation for residual-style graph topology.
- The G21A smoke uses the same standalone-GC graph-builder target pattern as G16/G17/G18/G19/G20:

    ../ggml.c
    ../ggml-quants.c
    ../ggml-threading.cpp
    -ffunction-sections
    -fdata-sections
    -Wl,--gc-sections

- G21A focused regression passes:
  - Q8_0 MUL_MAT -> residual ADD graph-builder pipeline smoke,
  - Q8_0 MUL_MAT -> MUL_SCALAR -> ADD -> SOFTMAX -> SUM_ROWS graph-builder pipeline smoke,
  - Q8_0 MUL_MAT -> MUL_SCALAR -> ADD -> SOFTMAX graph-builder pipeline smoke,
  - Q8_0 MUL_MAT -> MUL_SCALAR -> ADD graph-builder pipeline smoke,
  - Q8_0 MUL_MAT -> ADD graph-builder pipeline smoke,
  - packed Q8_0 graph-builder MMV smoke,
  - real graph-builder attention-like G16D smoke,
  - dispatch-all CUDA8 kernel smoke.
<!-- G21_STATUS_END -->

<!-- G22_STATUS_START -->
## G22 status: real GGML graph-builder quantized residual softmax coverage

Status: **PASS on GTX 560 / CUDA 8 / Fermi**.

G22 extends the residual-branch quantized graph-builder path by feeding the residual-add output into row-wise softmax.

Validated G22 checkpoints:

- **G22A**: real graph-builder residual-softmax quantized pipeline passes:
  - `h = ggml_mul_mat(Q8_0, x)`
  - `biased = ggml_add(h, residual)`
  - `prob = ggml_soft_max(biased)`
  - residual input branch isolation verified after `ggml_backend_i.graph_compute`
- **G22B**: main regression and README status refreshed for the G22A residual-softmax pipeline.

Validated G22A graph:

    Aq       : GGML_TYPE_Q8_0 matrix, shape [cols, rows]
    x        : GGML_TYPE_F32 vector, shape [cols]
    h        : GGML_TYPE_F32 vector, shape [rows]
    residual : GGML_TYPE_F32 vector, shape [rows]
    biased   : GGML_TYPE_F32 vector, shape [rows]
    prob     : GGML_TYPE_F32 vector, shape [rows]

Validated G22A path:

    A_f32
      -> host pack_q8_0(A_f32)
      -> Aq Q8_0 blocks

    h      = ggml_mul_mat(Aq, x)
    biased = ggml_add(h, residual)
    prob   = ggml_soft_max(biased)

    ggml_build_forward_expand(prob)
      -> real GGML graph with three nodes
      -> node 0: GGML_OP_MUL_MAT
      -> node 1: GGML_OP_ADD
      -> node 2: GGML_OP_SOFT_MAX
      -> ggml_backend_i.graph_compute
      -> GGML_CUDA8_OP_MUL_MAT_Q8_0_F32_VEC
      -> GGML_CUDA8_OP_ADD_F32
      -> GGML_CUDA8_OP_SOFTMAX_ROWS_F32
      -> CUDA8/Fermi kernels

Implementation notes:

- G22A reuses the packed Q8_0 reference strategy validated in G17C.
- G22A combines the residual branch topology from G21A with the softmax endpoint from G19A.
- G22A verifies that the residual input branch remains unchanged after graph dispatch.
- The G22A smoke uses the same standalone-GC graph-builder target pattern as G16/G17/G18/G19/G20/G21:

    ../ggml.c
    ../ggml-quants.c
    ../ggml-threading.cpp
    -ffunction-sections
    -fdata-sections
    -Wl,--gc-sections

- G22A focused regression passes:
  - Q8_0 MUL_MAT -> residual ADD -> SOFTMAX graph-builder pipeline smoke,
  - Q8_0 MUL_MAT -> residual ADD graph-builder pipeline smoke,
  - Q8_0 MUL_MAT -> MUL_SCALAR -> ADD -> SOFTMAX -> SUM_ROWS graph-builder pipeline smoke,
  - Q8_0 MUL_MAT -> MUL_SCALAR -> ADD -> SOFTMAX graph-builder pipeline smoke,
  - Q8_0 MUL_MAT -> MUL_SCALAR -> ADD graph-builder pipeline smoke,
  - Q8_0 MUL_MAT -> ADD graph-builder pipeline smoke,
  - packed Q8_0 graph-builder MMV smoke,
  - real graph-builder attention-like G16D smoke,
  - dispatch-all CUDA8 kernel smoke.
<!-- G22_STATUS_END -->
\n\n<!-- G23_STATUS_START -->
## G23 status: real GGML graph-builder quantized residual softmax + sum_rows coverage

Status: **PASS on GTX 560 / CUDA 8 / Fermi**.

G23 extends the residual-softmax quantized graph-builder path by adding a post-softmax row-sum reduction endpoint.

Validated G23 checkpoints:

- **G23A**: real graph-builder residual-softmax-sumrows quantized pipeline passes:
  - `h = ggml_mul_mat(Q8_0, x)`
  - `biased = ggml_add(h, residual)`
  - `prob = ggml_soft_max(biased)`
  - `row_sum = ggml_sum_rows(prob)`
  - residual input branch isolation verified after `ggml_backend_i.graph_compute`
- **G23B**: main regression and README status refreshed for the G23A residual-softmax-sumrows pipeline.

Validated G23A graph:

    Aq       : GGML_TYPE_Q8_0 matrix, shape [cols, rows]
    x        : GGML_TYPE_F32 vector, shape [cols]
    h        : GGML_TYPE_F32 vector, shape [rows]
    residual : GGML_TYPE_F32 vector, shape [rows]
    biased   : GGML_TYPE_F32 vector, shape [rows]
    prob     : GGML_TYPE_F32 vector, shape [rows]
    row_sum  : GGML_TYPE_F32 scalar/vector, shape [1]

Validated G23A path:

    A_f32
      -> host pack_q8_0(A_f32)
      -> Aq Q8_0 blocks

    h       = ggml_mul_mat(Aq, x)
    biased  = ggml_add(h, residual)
    prob    = ggml_soft_max(biased)
    row_sum = ggml_sum_rows(prob)

    ggml_build_forward_expand(row_sum)
      -> real GGML graph with four nodes
      -> node 0: GGML_OP_MUL_MAT
      -> node 1: GGML_OP_ADD
      -> node 2: GGML_OP_SOFT_MAX
      -> node 3: GGML_OP_SUM_ROWS
      -> ggml_backend_i.graph_compute
      -> GGML_CUDA8_OP_MUL_MAT_Q8_0_F32_VEC
      -> GGML_CUDA8_OP_ADD_F32
      -> GGML_CUDA8_OP_SOFTMAX_ROWS_F32
      -> GGML_CUDA8_OP_REDUCE_SUM_ROWS_F32
      -> CUDA8/Fermi kernels

Implementation notes:

- G23A reuses the packed Q8_0 reference strategy validated in G17C.
- G23A combines the residual branch topology from G21A, the residual-softmax endpoint from G22A, and the post-softmax reduction endpoint from G20A.
- G23A verifies that the residual input branch remains unchanged after graph dispatch.
- The G23A smoke uses the same standalone-GC graph-builder target pattern as G16/G17/G18/G19/G20/G21/G22:

    ../ggml.c
    ../ggml-quants.c
    ../ggml-threading.cpp
    -ffunction-sections
    -fdata-sections
    -Wl,--gc-sections

- G23A focused regression passes:
  - Q8_0 MUL_MAT -> residual ADD -> SOFTMAX -> SUM_ROWS graph-builder pipeline smoke,
  - Q8_0 MUL_MAT -> residual ADD -> SOFTMAX graph-builder pipeline smoke,
  - Q8_0 MUL_MAT -> residual ADD graph-builder pipeline smoke,
  - Q8_0 MUL_MAT -> MUL_SCALAR -> ADD -> SOFTMAX -> SUM_ROWS graph-builder pipeline smoke,
  - Q8_0 MUL_MAT -> MUL_SCALAR -> ADD -> SOFTMAX graph-builder pipeline smoke,
  - Q8_0 MUL_MAT -> MUL_SCALAR -> ADD graph-builder pipeline smoke,
  - Q8_0 MUL_MAT -> ADD graph-builder pipeline smoke,
  - packed Q8_0 graph-builder MMV smoke,
  - real graph-builder attention-like G16D smoke,
  - dispatch-all CUDA8 kernel smoke.
<!-- G23_STATUS_END -->\n

<!-- G24_STATUS_START -->
## G24 status: real GGML graph-builder quantized scaled residual softmax + sum_rows coverage

Status: **PASS on GTX 560 / CUDA 8 / Fermi**.

G24 extends the residual-softmax-sumrows graph-builder path by inserting scalar scaling between the Q8_0 projection and the residual add.

Validated G24 checkpoints:

- **G24A**: real graph-builder scaled residual-softmax-sumrows quantized pipeline passes:
  - `h = ggml_mul_mat(Q8_0, x)`
  - `scaled = ggml_mul(h, scale)`
  - `biased = ggml_add(scaled, residual)`
  - `prob = ggml_soft_max(biased)`
  - `row_sum = ggml_sum_rows(prob)`
  - residual input branch isolation verified after `ggml_backend_i.graph_compute`
- **G24B**: main regression and README status refreshed for the G24A scaled residual-softmax-sumrows pipeline.

Validated G24A graph:

    Aq       : GGML_TYPE_Q8_0 matrix, shape [cols, rows]
    x        : GGML_TYPE_F32 vector, shape [cols]
    h        : GGML_TYPE_F32 vector, shape [rows]
    scale    : GGML_TYPE_F32 scalar, shape [1]
    scaled   : GGML_TYPE_F32 vector, shape [rows]
    residual : GGML_TYPE_F32 vector, shape [rows]
    biased   : GGML_TYPE_F32 vector, shape [rows]
    prob     : GGML_TYPE_F32 vector, shape [rows]
    row_sum  : GGML_TYPE_F32 scalar/vector, shape [1]

Validated G24A path:

    A_f32
      -> host pack_q8_0(A_f32)
      -> Aq Q8_0 blocks

    h       = ggml_mul_mat(Aq, x)
    scaled  = ggml_mul(h, scale)
    biased  = ggml_add(scaled, residual)
    prob    = ggml_soft_max(biased)
    row_sum = ggml_sum_rows(prob)

    ggml_build_forward_expand(row_sum)
      -> real GGML graph with five nodes
      -> node 0: GGML_OP_MUL_MAT
      -> node 1: GGML_OP_MUL
      -> node 2: GGML_OP_ADD
      -> node 3: GGML_OP_SOFT_MAX
      -> node 4: GGML_OP_SUM_ROWS
      -> ggml_backend_i.graph_compute
      -> GGML_CUDA8_OP_MUL_MAT_Q8_0_F32_VEC
      -> GGML_CUDA8_OP_MUL_SCALAR_F32
      -> GGML_CUDA8_OP_ADD_F32
      -> GGML_CUDA8_OP_SOFTMAX_ROWS_F32
      -> GGML_CUDA8_OP_REDUCE_SUM_ROWS_F32
      -> CUDA8/Fermi kernels

Implementation notes:

- G24A reuses the packed Q8_0 reference strategy validated in G17C.
- G24A combines the scalar post-projection scaling from G20A with the residual branch topology from G21A/G22A/G23A.
- G24A verifies that the residual input branch remains unchanged after graph dispatch.
- The G24A smoke uses the same standalone-GC graph-builder target pattern as G16/G17/G18/G19/G20/G21/G22/G23:

    ../ggml.c
    ../ggml-quants.c
    ../ggml-threading.cpp
    -ffunction-sections
    -fdata-sections
    -Wl,--gc-sections

- G24A focused regression passes:
  - Q8_0 MUL_MAT -> MUL_SCALAR -> residual ADD -> SOFTMAX -> SUM_ROWS graph-builder pipeline smoke,
  - Q8_0 MUL_MAT -> residual ADD -> SOFTMAX -> SUM_ROWS graph-builder pipeline smoke,
  - Q8_0 MUL_MAT -> residual ADD -> SOFTMAX graph-builder pipeline smoke,
  - Q8_0 MUL_MAT -> residual ADD graph-builder pipeline smoke,
  - Q8_0 MUL_MAT -> MUL_SCALAR -> ADD -> SOFTMAX -> SUM_ROWS graph-builder pipeline smoke,
  - packed Q8_0 graph-builder MMV smoke,
  - real graph-builder attention-like G16D smoke,
  - dispatch-all CUDA8 kernel smoke.
<!-- G24_STATUS_END -->

<!-- G25_STATUS_START -->
## G25 status: RMS_NORM kernel + dispatch wiring

Status: **PASS on GTX 560 / CUDA 8 / Fermi**.

G25 adds the RMS_NORM (Root Mean Square Normalization) kernel, the core normalization
operation used in LLaMA and other modern transformer architectures. The kernel uses
Fermi-safe shared-memory tree reduction (no warp shuffle).

Validated G25 checkpoints:

- **G25A**: standalone RMS_NORM F32 kernel smoke test passes:
  - 4 rows x 128 cols, eps=1e-5
  - max_err = 2.38e-07 (tolerance 1e-4)
  - Kernel: shared-mem reduction, 256 threads/block, one block per row
  - Dispatch wiring: GGML_OP_RMS_NORM -> GGML_CUDA8_OP_RMS_NORM_F32

- **G25B**: main regression and README status refreshed for G25A RMS_NORM kernel.

Validated G25A kernel:

    y_i = x_i * rsqrt( mean(x^2) + eps )

    Fermi-safe implementation:
    - extern __shared__ float sdata[] for partial sums
    - Tree reduction (no __shfl_down)
    - rsqrtf for normalization
    - 256 threads per block, 1 block per row

New files:
- ggml-cuda8-rms-norm.cu - kernel + extern "C" dispatch wrapper
- ggml-cuda8-rms-norm-smoke.cu - standalone smoke test

Dispatch pipeline:
- ggml-cuda8-dispatch.h - GGML_CUDA8_OP_RMS_NORM_F32 enum
- ggml-cuda8-dispatch.cpp - supported/execute routing
- ggml-cuda8-ggml-backend.cpp - GGML_OP_RMS_NORM -> GGML_CUDA8_OP_RMS_NORM_F32

Notes:
- RMS_NORM is the normalization used in LLaMA (replaces LayerNorm).
- eps is extracted from node->op_params (same as upstream ggml).
- The kernel is row-wise: each CUDA block processes one row.
- G25B focused regression passes:
  - standalone RMS_NORM kernel smoke,
  - Q8_0 MUL_MAT -> MUL_SCALAR -> residual ADD -> SOFTMAX -> SUM_ROWS graph-builder pipeline smoke,
  - packed Q8_0 graph-builder MMV smoke,
  - real graph-builder attention-like G16D smoke,
  - dispatch-all CUDA8 kernel smoke.
<!-- G25_STATUS_END -->

<!-- G26_STATUS_START -->
## G26 status: element-wise MUL kernel + dispatch wiring

Status: **PASS on GTX 560 / CUDA 8 / Fermi**.

G26 adds element-wise MUL (F32), used for post-RMSNorm weight scaling in LLaMA
and other transformer architectures. The existing GGML_OP_MUL case is extended
to support both scalar MUL (1-element src1) and element-wise MUL (same-shape src1).

Validated G26 checkpoints:

- **G26A**: standalone element-wise MUL F32 kernel smoke test passes:
  - n=512, max_err = 0.000000e+00 (tolerance 1e-6)
  - Kernel: 256 threads/block, c[i] = a[i] * b[i]
  - Dispatch wiring: GGML_OP_MUL (same-shape) -> GGML_CUDA8_OP_MUL_F32

- **G26B**: main regression and README status refreshed for G26A element-wise MUL.

Validated G26A kernel:

    c[i] = a[i] * b[i]    (element-wise, F32)

    Fermi-safe implementation:
    - Simple grid-stride loop, 256 threads per block
    - No shared memory, no reduction
    - cudaDeviceSynchronize after launch

New files:
- ggml-cuda8-mul.cu - element-wise MUL kernel + extern "C" launch wrapper
- ggml-cuda8-mul-smoke.cu - standalone smoke test

Dispatch pipeline:
- ggml-cuda8-dispatch.h - GGML_CUDA8_OP_MUL_F32 enum
- ggml-cuda8-dispatch.cpp - supported/execute routing
- ggml-cuda8-ggml-backend.cpp - GGML_OP_MUL extended:
    - src1 has 1 element -> GGML_CUDA8_OP_MUL_SCALAR_F32 (existing)
    - src1 same shape    -> GGML_CUDA8_OP_MUL_F32 (new, G26A)

Notes:
- Element-wise MUL is the weight-scaling step after RMSNorm in LLaMA.
- The combined RMS_NORM -> MUL pipeline (G27) will validate the full normalization path.
- G26B focused regression passes:
  - standalone element-wise MUL kernel smoke,
  - standalone RMS_NORM kernel smoke,
  - Q8_0 MUL_MAT -> MUL_SCALAR -> residual ADD -> SOFTMAX -> SUM_ROWS pipeline smoke,
  - packed Q8_0 graph-builder MMV smoke,
  - real graph-builder attention-like G16D smoke,
  - dispatch-all CUDA8 kernel smoke.
<!-- G26_STATUS_END -->

<!-- G27_STATUS_START -->
## G27 status: real GGML graph-builder RMS_NORM -> MUL pipeline

Status: **PASS on GTX 560 / CUDA 8 / Fermi**.

G27 validates the LLaMA normalization pattern as a real GGML graph dispatched
through the CUDA8 backend graph_compute path:

    y = rms_norm(x, eps) * w

This is the exact pre-attention / pre-FFN normalization step in LLaMA.

Validated G27 checkpoints:

- **G27A**: real GGML graph-builder RMS_NORM -> MUL (element-wise) pipeline passes:
  - n=512, eps=1e-5
  - max_err = 2.384186e-07 (tolerance 1e-4)
  - Graph: 2 nodes (op=25 RMS_NORM, op=7 MUL)
  - Both ops dispatched through ggml_backend_i.graph_compute
  - Device-resident buffers with residency tracking

- **G27B**: main regression and README status refreshed for G27A.

Validated G27A graph:

    node 0: GGML_OP_RMS_NORM  -> GGML_CUDA8_OP_RMS_NORM_F32
    node 1: GGML_OP_MUL       -> GGML_CUDA8_OP_MUL_F32

    x (F32, n=512)
     |
     ggml_rms_norm(x, eps=1e-5)
     |
     ggml_mul(normed, w)
     |
     y (F32, n=512)

Notes:
- This is the first multi-op pipeline combining normalization + element-wise ops.
- RMS_NORM uses Fermi-safe shared-memory tree reduction (no warp shuffle).
- MUL uses simple grid-stride element-wise kernel.
- op_params (eps) propagation validated through graph_compute dispatch.
- G27B focused regression passes:
  - RMS_NORM -> MUL graph-builder pipeline smoke,
  - standalone RMS_NORM kernel smoke,
  - standalone element-wise MUL kernel smoke,
  - Q8_0 MUL_MAT -> MUL_SCALAR -> residual ADD -> SOFTMAX -> SUM_ROWS pipeline smoke,
  - packed Q8_0 graph-builder MMV smoke,
  - real graph-builder attention-like G16D smoke,
  - dispatch-all CUDA8 kernel smoke.
<!-- G27_STATUS_END -->

<!-- G28_STATUS_START -->
## G28 status: ROPE (Rotary Positional Embedding) kernel + dispatch wiring

Status: **PASS on GTX 560 / CUDA 8 / Fermi**.

G28 adds the ROPE kernel, which applies rotary positional embeddings to query/key
tensors. This is the position-encoding mechanism used in LLaMA and most modern
transformer architectures, replacing absolute or learned positional embeddings.

Validated G28 checkpoints:

- **G28A**: standalone ROPE F32 kernel smoke test passes:
  - head_dim=64, n_heads=4, seq_len=8, n_dims=64
  - freq_base=10000.0, freq_scale=1.0
  - max_err = 5.960464e-07 (tolerance 1e-4)
  - Kernel: one thread per pair, cosf/sinf rotation, Fermi-safe
  - Dispatch wiring: GGML_OP_ROPE -> GGML_CUDA8_OP_ROPE_F32

- **G28B**: main regression and README status refreshed for G28A ROPE kernel.

Validated G28A kernel:

    For each pair (x0, x1) at position p:
      theta = p * freq_base^(-2*pair/n_dims) * freq_scale
      dst[2k]   = x0 * cos(theta) - x1 * sin(theta)
      dst[2k+1] = x0 * sin(theta) + x1 * cos(theta)

    Fermi-safe implementation:
    - One thread per pair of elements
    - cosf/sinf/powf (no warp shuffle, no shared memory needed)
    - 256 threads per block, grid covers all pairs across heads/positions
    - Pairs beyond n_dims pass through unchanged

    op_params layout (int32_t[15]):
      [0]=n_past  [1]=n_dims  [2]=mode
      [5]=freq_base  [6]=freq_scale  [7]=ext_factor

    Supported: mode=0 (basic ROPE), ext_factor=0 (no YaRN)

New files:
- ggml-cuda8-rope.cu - ROPE kernel + extern "C" dispatch wrapper
- ggml-cuda8-rope-smoke.cu - standalone smoke test

Dispatch pipeline:
- ggml-cuda8-dispatch.h - GGML_CUDA8_OP_ROPE_F32 enum
- ggml-cuda8-dispatch.cpp - supported/execute routing (extracts op_params)
- ggml-cuda8-ggml-backend.cpp - GGML_OP_ROPE -> GGML_CUDA8_OP_ROPE_F32
  (tensors passed without flattening; ROPE needs multi-dim ne[] + op_params)

Notes:
- ROPE is the first op requiring both F32 and I32 input tensors (src1 = positions).
- ROPE is the first op where tensors are NOT flattened in the backend mapping,
  because the kernel needs the full multi-dimensional shape.
- Basic mode only (mode=0, no YaRN, no mrope) -- sufficient for LLaMA inference.
- G28B focused regression passes:
  - standalone ROPE kernel smoke,
  - RMS_NORM -> MUL graph-builder pipeline smoke,
  - standalone RMS_NORM kernel smoke,
  - standalone element-wise MUL kernel smoke,
  - Q8_0 MUL_MAT -> MUL_SCALAR -> residual ADD -> SOFTMAX -> SUM_ROWS pipeline smoke,
  - packed Q8_0 graph-builder MMV smoke,
  - real graph-builder attention-like G16D smoke,
  - dispatch-all CUDA8 kernel smoke.
<!-- G28_STATUS_END -->




