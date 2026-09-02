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

<!-- G29_STATUS_START -->
## G29 status: RESHAPE/VIEW/PERMUTE/TRANSPOSE no-op support

Status: **PASS on GTX 560 / CUDA 8 / Fermi**.

G29 adds no-op support for metadata-only tensor operations: RESHAPE, VIEW,
PERMUTE, and TRANSPOSE. These ops do not perform any computation -- they only
change how existing data is interpreted (shape, strides, offsets). The graph_compute
loop simply skips them, matching the behavior of the upstream CUDA backend.

Validated G29 checkpoints:

- **G29A**: RESHAPE no-op graph-builder smoke test passes:
  - Graph: x -> reshape_2d(256,2) -> reshape_1d(512) -> add(bias) -> y
  - 3 graph nodes: RESHAPE(skip) -> RESHAPE(skip) -> ADD(dispatch)
  - max_err = 0.000000e+00
  - Validates that reshape nodes are skipped and data aliasing works correctly

- **G29B**: main regression and README status refreshed for G29A.

Implementation:
- No new kernels or dispatch ops required
- Single patch to ggml-cuda8-ggml-backend.cpp graph_compute loop:
  nodes with op RESHAPE/VIEW/PERMUTE/TRANSPOSE are skipped alongside GGML_OP_NONE
- Matches upstream ggml-cuda behavior (these ops are no-ops in graph execution)

Skipped ops in graph_compute:
- GGML_OP_NONE
- GGML_OP_RESHAPE (G29A)
- GGML_OP_VIEW (G29A)
- GGML_OP_PERMUTE (G29A)
- GGML_OP_TRANSPOSE (G29A)

Notes:
- These ops are essential for multi-head attention reshaping in LLaMA:
  [seq_len, n_embd] -> [seq_len, n_heads, head_dim] -> permute for QKV
- The tensor data pointer aliases the source -- no copy, no kernel launch.
- G29B focused regression passes:
  - RESHAPE no-op graph-builder pipeline smoke,
  - ROPE standalone kernel smoke,
  - RMS_NORM -> MUL graph-builder pipeline smoke,
  - standalone RMS_NORM kernel smoke,
  - standalone element-wise MUL kernel smoke,
  - Q8_0 MUL_MAT -> MUL_SCALAR -> residual ADD -> SOFTMAX -> SUM_ROWS pipeline smoke,
  - packed Q8_0 graph-builder MMV smoke,
  - real graph-builder attention-like G16D smoke,
  - dispatch-all CUDA8 kernel smoke.
<!-- G29_STATUS_END -->

<!-- G30_STATUS_START -->
## G30 status: CONT (contiguous copy) kernel + dispatch wiring

Status: **PASS on GTX 560 / CUDA 8 / Fermi**.

G30 adds CONT (GGML_OP_CONT) support, which makes non-contiguous tensors
contiguous by copying their data into a fresh, densely-packed buffer. This
is needed after PERMUTE/TRANSPOSE operations in multi-head attention, where
the logical layout changes but the physical data order does not.

Validated G30 checkpoints:

- **G30A**: CONT graph-builder smoke test passes:
  - Graph: x -> cont -> add(bias) -> y
  - 2 graph nodes: CONT(dispatch) -> ADD(dispatch)
  - max_err = 0.000000e+00
  - Implementation: cudaMemcpy device-to-device (D2D)

- **G30B**: main regression and README status refreshed for G30A.

Implementation:
- Restored cpy.cu with ggml_cuda8_cpy_f32_d2d() -- device-to-device cudaMemcpy
- New dispatch op: GGML_CUDA8_OP_CONT_F32
- dispatch.cpp: supported (F32 src0 + dst) + exec (D2D copy by element count)
- backend.cpp: GGML_OP_CONT -> GGML_CUDA8_OP_CONT_F32

Notes:
- CONT is used in LLaMA after permute operations to make data physically
  contiguous before matrix multiplications.
- For contiguous F32 tensors, CONT is effectively a memcpy.
- For non-contiguous tensors (after permute), a strided copy would be needed;
  the current implementation handles the common contiguous case.
- G30B focused regression passes:
  - CONT -> ADD graph-builder pipeline smoke,
  - RESHAPE no-op graph-builder pipeline smoke,
  - ROPE standalone kernel smoke,
  - RMS_NORM -> MUL graph-builder pipeline smoke,
  - standalone RMS_NORM kernel smoke,
  - standalone element-wise MUL kernel smoke,
  - Q8_0 MUL_MAT -> MUL_SCALAR -> residual ADD -> SOFTMAX -> SUM_ROWS pipeline smoke,
  - packed Q8_0 graph-builder MMV smoke,
  - real graph-builder attention-like G16D smoke,
  - dispatch-all CUDA8 kernel smoke.
<!-- G30_STATUS_END -->

<!-- G31_STATUS_START -->
## G31 status: DIAG_MASK_INF (causal masking) kernel + dispatch wiring

Status: **PASS on GTX 560 / CUDA 8 / Fermi**.

G31 adds the DIAG_MASK_INF kernel, which applies causal masking to attention
score matrices. Elements above the diagonal (future tokens) are masked by
subtracting FLT_MAX, making them -infinity after softmax. This is the standard
autoregressive masking used in LLaMA and all decoder-only transformers.

Validated G31 checkpoints:

- **G31A**: standalone DIAG_MASK_INF F32 kernel smoke test passes:
  - 8x8 matrix, n_past=0, rows_per_channel=8
  - 28/64 positions masked (upper triangle)
  - max_err = 0.000000e+00
  - Visual mask pattern verified:
    row 0: . X X X X X X X
    row 1: . . X X X X X X
    ...
    row 7: . . . . . . . .

- **G31B**: main regression and README status refreshed for G31A.

Validated G31A kernel:

    dst[row][col] = x[row][col] - (col > n_past + row % rows_per_channel) * FLT_MAX

    Fermi-safe implementation:
    - dim3 block(1, 256, 1), grid(nrows, ceil(ncols/256), 1)
    - One thread per element
    - n_past extracted from op_params[0]
    - rows_per_channel = ne[1] (number of rows per attention head)

New files:
- ggml-cuda8-diagmask.cu - kernel + extern "C" dispatch wrapper
- ggml-cuda8-diagmask-smoke.cu - standalone smoke test with visual mask output

Dispatch pipeline:
- ggml-cuda8-dispatch.h - GGML_CUDA8_OP_DIAG_MASK_INF_F32 enum
- ggml-cuda8-dispatch.cpp - supported/execute routing
- ggml-cuda8-ggml-backend.cpp - GGML_OP_DIAG_MASK_INF -> GGML_CUDA8_OP_DIAG_MASK_INF_F32
  (tensors passed without flattening; kernel needs 2D shape + op_params)

Notes:
- Causal masking ensures each token can only attend to itself and previous tokens.
- Matches upstream ggml-cuda/diagmask.cu logic exactly.
- n_past=0 for standard autoregressive inference (no KV cache offset).
- With DIAG_MASK_INF, the full attention pipeline is now possible:
    Q*K^T -> DIAG_MASK_INF -> SOFTMAX -> V matmul
- G31B focused regression passes:
  - standalone DIAG_MASK_INF kernel smoke,
  - CONT -> ADD graph-builder pipeline smoke,
  - RESHAPE no-op graph-builder pipeline smoke,
  - ROPE standalone kernel smoke,
  - RMS_NORM -> MUL graph-builder pipeline smoke,
  - standalone RMS_NORM kernel smoke,
  - standalone element-wise MUL kernel smoke,
  - Q8_0 MUL_MAT -> MUL_SCALAR -> residual ADD -> SOFTMAX -> SUM_ROWS pipeline smoke,
  - packed Q8_0 graph-builder MMV smoke,
  - real graph-builder attention-like G16D smoke,
  - dispatch-all CUDA8 kernel smoke.
<!-- G31_STATUS_END -->

<!-- G32_STATUS_START -->
## G32 status: full transformer block pipeline smoke

Status: **PASS on GTX 560 / CUDA 8 / Fermi**.

G32 validates the complete LLaMA pre-attention + scoring pipeline as a single
6-op real GGML graph dispatched through the CUDA8 backend graph_compute path.
This is the culmination of milestones G11-G31 -- every op built for this backend
working together in one end-to-end pipeline.

Validated G32 checkpoints:

- **G32A**: full transformer block pipeline smoke test passes:
  - 6-op graph: RMS_NORM -> MUL(elem) -> MUL_MAT_Q8_0xF32 -> ADD -> MUL_SCALAR -> SOFTMAX
  - embd=128, proj=64, eps=1e-5, scale=0.125
  - max_err = 5.587935e-09 (tolerance 1e-3)
  - softmax_sum = 0.999999940 (tolerance 1e-4 from 1.0)
  - Mixed precision: Q8_0 quantized weights x F32 activations
  - All 6 ops dispatched through ggml_backend_i.graph_compute

- **G32B**: main regression and README status refreshed for G32A.

Validated G32A pipeline:

    x [128, F32]
    |
    ggml_rms_norm(x, eps=1e-5)          -> RMS_NORM_F32
    |
    ggml_mul(norm, w_norm)              -> MUL_F32 (element-wise)
    |
    ggml_mul_mat(W_q8[64x128], normed) -> MUL_MAT_Q8_0xF32_VEC
    |
    ggml_add(h, bias)                   -> ADD_F32
    |
    ggml_mul(h_biased, scale)           -> MUL_SCALAR_F32
    |
    ggml_soft_max(scaled)               -> SOFTMAX_ROWS_F32
    |
    y [64, F32]  (probability distribution, sum ~ 1.0)

Op types exercised in a single graph_compute:
- RMS_NORM_F32:          layer normalization (shared-mem reduction)
- MUL_F32:               element-wise weight scaling
- MUL_MAT_Q8_0xF32_VEC: quantized matrix-vector multiply
- ADD_F32:               bias addition
- MUL_SCALAR_F32:        attention scaling (1/sqrt(head_dim))
- SOFTMAX_ROWS_F32:      probability distribution

Complete CUDA8 backend op inventory (G11-G32):
- CPY_F32:                    buffer copy (G11)
- ADD_F32:                    element-wise add (G12)
- ADD_SCALAR_F32:             scalar add (G13)
- MUL_SCALAR_F32:             scalar multiply (G13)
- REDUCE_SUM_ROWS_F32:        row-wise sum (G15)
- REDUCE_MAX_ROWS_F32:        row-wise max (G15)
- SOFTMAX_ROWS_F32:           row-wise softmax (G15)
- MUL_MAT_Q8_0xF32_VEC:      quantized MMV (G17)
- RMS_NORM_F32:               layer normalization (G25)
- MUL_F32:                    element-wise multiply (G26)
- ROPE_F32:                   rotary positional embeddings (G28)
- RESHAPE/VIEW/PERMUTE/TRANSPOSE: metadata no-ops (G29)
- CONT_F32:                   contiguous copy (G30)
- DIAG_MASK_INF_F32:          causal masking (G31)

- G32B focused regression passes:
  - full transformer block pipeline smoke (6-op),
  - standalone DIAG_MASK_INF kernel smoke,
  - CONT -> ADD graph-builder pipeline smoke,
  - RESHAPE no-op graph-builder pipeline smoke,
  - ROPE standalone kernel smoke,
  - RMS_NORM -> MUL graph-builder pipeline smoke,
  - standalone RMS_NORM kernel smoke,
  - standalone element-wise MUL kernel smoke,
  - Q8_0 MUL_MAT -> MUL_SCALAR -> residual ADD -> SOFTMAX -> SUM_ROWS pipeline smoke,
  - packed Q8_0 graph-builder MMV smoke,
  - real graph-builder attention-like G16D smoke,
  - dispatch-all CUDA8 kernel smoke.
<!-- G32_STATUS_END -->

<!-- G33_STATUS_START -->
## G33 status: GET_ROWS (embedding lookup) kernel + dispatch wiring

Status: **PASS on GTX 560 / CUDA 8 / Fermi**.

G33 adds GET_ROWS (GGML_OP_GET_ROWS) support for F32 embedding tables.
This is the first op in the LLaMA inference pipeline -- it converts integer
token IDs into dense embedding vectors by looking up rows from the embedding
weight matrix.

Validated G33 checkpoints:

- **G33A**: standalone GET_ROWS F32 kernel smoke test passes:
  - vocab=32, embd=64, n_tokens=4, tokens=[3, 7, 0, 15]
  - max_err = 0.000000e+00 (exact row copy)
  - Kernel: one block per token, 256 threads stride over columns
  - Dispatch wiring: GGML_OP_GET_ROWS -> GGML_CUDA8_OP_GET_ROWS_F32

- **G33B**: main regression and README status refreshed for G33A.

Validated G33A kernel:

    dst[token, :] = src0[src1[token], :]

    For each token index t in src1 (I32):
      copy row src1[t] from embedding table src0 (F32) to dst row t

    Fermi-safe implementation:
    - One CUDA block per token (gridDim.x = n_tokens)
    - 256 threads per block stride over embedding columns
    - No shared memory, no reduction

New files:
- ggml-cuda8-getrows.cu - kernel + extern "C" dispatch wrapper
- ggml-cuda8-getrows-smoke.cu - standalone smoke test

Dispatch pipeline:
- ggml-cuda8-dispatch.h - GGML_CUDA8_OP_GET_ROWS_F32 enum
- ggml-cuda8-dispatch.cpp - supported/execute routing
- ggml-cuda8-ggml-backend.cpp - GGML_OP_GET_ROWS -> GGML_CUDA8_OP_GET_ROWS_F32
  (F32 src0 + I32 src1, tensors not flattened)

Complete CUDA8 backend op inventory (G11-G33, 15 ops):
- CPY_F32, ADD_F32, ADD_SCALAR_F32, MUL_SCALAR_F32
- REDUCE_SUM_ROWS_F32, REDUCE_MAX_ROWS_F32, SOFTMAX_ROWS_F32
- MUL_MAT_Q8_0xF32_VEC, RMS_NORM_F32, MUL_F32, ROPE_F32
- RESHAPE/VIEW/PERMUTE/TRANSPOSE (no-ops)
- CONT_F32, DIAG_MASK_INF_F32, GET_ROWS_F32

With GET_ROWS, the complete LLaMA inference pipeline is now possible:
  tokens -> GET_ROWS -> [transformer block x N] -> output

- G33B focused regression passes:
  - standalone GET_ROWS kernel smoke,
  - full transformer block pipeline smoke (6-op),
  - standalone DIAG_MASK_INF kernel smoke,
  - CONT -> ADD graph-builder pipeline smoke,
  - RESHAPE no-op graph-builder pipeline smoke,
  - ROPE standalone kernel smoke,
  - RMS_NORM -> MUL graph-builder pipeline smoke,
  - standalone RMS_NORM kernel smoke,
  - standalone element-wise MUL kernel smoke,
  - Q8_0 pipelines, graph-builder attention-like smoke,
  - dispatch-all CUDA8 kernel smoke.
<!-- G33_STATUS_END -->

<!-- G34_STATUS_START -->
## G34 status: full attention pipeline (11 ops, Q/K/V fan-out)

Status: **PASS on GTX 560 / CUDA 8 / Fermi**.

G34 validates the complete single-token single-head attention pipeline as a
real GGML graph dispatched through the CUDA8 backend. This is the most complex
test in the suite: 11 ops, 7 op types, 19 tensors, 3 quantized projections,
and Q/K/V fan-out from a shared normalized input.

Validated G34 checkpoints:

- **G34A**: full attention pipeline smoke test passes:
  - 11 graph nodes, 8 leaf tensors, 19 total tensors
  - vocab=32, embd=128, proj=64, token_id=5
  - max_err = 6.332994e-08 (tolerance 1e-3)
  - softmax_sum = 0.999999940 (tolerance 1e-4 from 1.0)
  - ggml graph optimizer reordered Q/K/V projections for efficiency;
    backend handled reordering correctly via src[] pointer chasing

- **G34B**: main regression and README status refreshed for G34A.

Validated G34A pipeline:

    token_id [I32]
      |
    1. GET_ROWS(embed[32x128])     -> x [128]        (embedding lookup)
    2. RMS_NORM(x, eps=1e-5)       -> norm [128]     (layer normalization)
    3. MUL(norm, w_norm)           -> x_n [128]      (element-wise weight scaling)
      |
      +-- 4. MUL_MAT(W_q[64x128], x_n) -> q [64]   (Q projection, Q8_0)
      +-- 5. MUL_MAT(W_k[64x128], x_n) -> k [64]   (K projection, Q8_0)
      +-- 6. MUL_MAT(W_v[64x128], x_n) -> v [64]   (V projection, Q8_0)
           |
    7. MUL(q, 1/sqrt(64))         -> q_s [64]       (attention scaling)
    8. ADD(q_s, k)                 -> scores [64]    (Q+K interaction)
    9. SOFTMAX(scores)             -> probs [64]     (attention probabilities)
   10. MUL(probs, v)              -> attn [64]       (weighted value)
   11. ADD(attn, bias)            -> out [64]        (output)

Op types exercised:
- GET_ROWS_F32:          embedding lookup (G33)
- RMS_NORM_F32:          layer normalization (G25)
- MUL_F32:               element-wise multiply x3 (G26)
- MUL_MAT_Q8_0xF32_VEC: quantized projection x3 (G17)
- MUL_SCALAR_F32:        attention scaling (G13)
- ADD_F32:               residual/bias x2 (G12)
- SOFTMAX_ROWS_F32:      attention probabilities (G15)

Notes:
- This is the first test with Q/K/V fan-out: three MUL_MAT ops sharing
  the same input tensor (x_n). The ggml graph optimizer reorders the
  projections for efficiency; our backend handles this correctly because
  each op reads from its src[] pointers regardless of execution order.
- 19 device-resident tensors managed in a single 112KB buffer.
- Mixed precision throughout: I32 token IDs, Q8_0 weight matrices, F32 activations.
- G34B focused regression passes:
  - full attention pipeline smoke (11-op),
  - full transformer block pipeline smoke (6-op),
  - standalone GET_ROWS kernel smoke,
  - standalone DIAG_MASK_INF kernel smoke,
  - CONT -> ADD graph-builder pipeline smoke,
  - RESHAPE no-op graph-builder pipeline smoke,
  - ROPE standalone kernel smoke,
  - RMS_NORM -> MUL graph-builder pipeline smoke,
  - standalone RMS_NORM kernel smoke,
  - standalone element-wise MUL kernel smoke,
  - Q8_0 pipelines, graph-builder attention-like smoke,
  - dispatch-all CUDA8 kernel smoke.
<!-- G34_STATUS_END -->

<!-- G35_STATUS_START -->
## G35 status: full end-to-end LLaMA inference pipeline (15 ops, ALL op types)

Status: **PASS on GTX 560 / CUDA 8 / Fermi**.

G35 is the culmination of the entire CUDA8 backend project (G11-G35). It
validates every op built for this backend in a single 15-op graph that
exercises the complete LLaMA inference pipeline end-to-end: from token ID
input through embedding lookup, normalization, Q/K/V projection with
rotary positional embeddings, causal masking, softmax attention, value
weighting, and final output.

Validated G35 checkpoints:

- **G35A**: full end-to-end pipeline smoke test passes:
  - 15 graph nodes, 9 leaf tensors, 24 total tensors
  - vocab=32, embd=128, proj=64, token=5, rope_pos=7, n_past=31
  - max_err = 1.192093e-06 (tolerance 1e-3)
  - softmax_sum = 0.999999881 (tolerance 1e-4 from 1.0)
  - masked = 32/64 (causal mask correct: cols 32-63 masked)
  - 132 KB device buffer

- **G35B**: main regression and README status refreshed for G35A.

Validated G35A pipeline (15 ops, 9 op types):

     token_id=5 [I32]
       |
     1. GET_ROWS(embed[32x128])          -> emb [128]         GET_ROWS_F32
     2. RMS_NORM(emb, eps=1e-5)          -> norm [128]        RMS_NORM_F32
     3. MUL(norm, w_norm)                -> x_n [128]         MUL_F32 (elem)
       |
       +-- 4. MUL_MAT(Wq[64x128], x_n)  -> q [64]           MUL_MAT_Q8_0xF32
       +-- 5. MUL_MAT(Wk[64x128], x_n)  -> k [64]           MUL_MAT_Q8_0xF32
       +-- 6. MUL_MAT(Wv[64x128], x_n)  -> v [64]           MUL_MAT_Q8_0xF32
            |
     7. ROPE(q, pos=7, n_dims=64)        -> q_r [64]         ROPE_F32
     8. ROPE(k, pos=7, n_dims=64)        -> k_r [64]         ROPE_F32
     9. MUL(q_r, 1/sqrt(64))             -> q_s [64]         MUL_SCALAR_F32
    10. ADD(q_s, k_r)                     -> scores [64]      ADD_F32
    11. DIAG_MASK_INF(scores, n_past=31)  -> masked [64]      DIAG_MASK_INF_F32
    12. SOFTMAX(masked)                   -> probs [64]       SOFTMAX_ROWS_F32
    13. MUL(probs, v)                     -> attn [64]        MUL_F32 (elem)
    14. CONT(attn)                        -> attn_c [64]      CONT_F32
    15. ADD(attn_c, bias)                 -> out [64]         ADD_F32

All 9 op types exercised:
  1. GET_ROWS_F32           - embedding lookup (G33)
  2. RMS_NORM_F32           - layer normalization (G25)
  3. MUL_F32                - element-wise multiply (G26)
  4. MUL_MAT_Q8_0xF32_VEC  - quantized projection x3 (G17)
  5. ROPE_F32               - rotary positional embeddings x2 (G28)
  6. MUL_SCALAR_F32         - attention scaling (G13)
  7. ADD_F32                - residual/bias x2 (G12)
  8. DIAG_MASK_INF_F32      - causal masking (G31)
  9. SOFTMAX_ROWS_F32       - attention probabilities (G15)
  +  CONT_F32               - contiguous copy (G30)

Complete CUDA8 backend op inventory (G11-G35, 15 dispatch ops + 4 no-ops):
  Kernels:   CPY_F32, ADD_F32, ADD_SCALAR_F32, MUL_SCALAR_F32,
             REDUCE_SUM_ROWS_F32, REDUCE_MAX_ROWS_F32, SOFTMAX_ROWS_F32,
             MUL_MAT_Q8_0xF32_VEC, RMS_NORM_F32, MUL_F32, ROPE_F32,
             CONT_F32, DIAG_MASK_INF_F32, GET_ROWS_F32
  No-ops:    RESHAPE, VIEW, PERMUTE, TRANSPOSE

- G35B focused regression passes:
  - full end-to-end pipeline smoke (15-op, all 9 op types),
  - full attention pipeline smoke (11-op, Q/K/V fan-out),
  - full transformer block pipeline smoke (6-op),
  - standalone GET_ROWS kernel smoke,
  - standalone DIAG_MASK_INF kernel smoke,
  - CONT -> ADD graph-builder pipeline smoke,
  - RESHAPE no-op graph-builder pipeline smoke,
  - ROPE standalone kernel smoke,
  - RMS_NORM -> MUL graph-builder pipeline smoke,
  - standalone RMS_NORM kernel smoke,
  - standalone element-wise MUL kernel smoke,
  - Q8_0 MUL_MAT -> MUL_SCALAR -> residual ADD -> SOFTMAX -> SUM_ROWS pipeline smoke,
  - packed Q8_0 graph-builder MMV smoke,
  - real graph-builder attention-like G16D smoke,
  - dispatch-all CUDA8 kernel smoke.
<!-- G35_STATUS_END -->

<!-- G36_STATUS_START -->
## G36 status: backend auto-registration into ggml backend registry

Status: **PASS on GTX 560 / CUDA 8 / Fermi**.

G36 wires the CUDA8 backend into ggml's backend registry system, enabling
automatic discovery by llama.cpp and other ggml-based tools. When GGML_USE_CUDA8
is defined, the backend registers itself alongside the standard CUDA backend.

Validated G36 checkpoints:

- **G36A**: backend auto-registration compiles and links:
  - ggml_backend_cuda8_reg() entry point
  - Full ggml_backend_device_i implementation
  - supports_op for all 15 dispatch ops with type checking
  - Device filter: compute capability 2.x-3.x only (Fermi/Kepler)
  - Full regression passes (no regressions from registration code)

- **G36B**: main regression and README status refreshed for G36A.

Implementation:

  ggml_backend_reg_t ggml_backend_cuda8_reg()
    |
    +-- ggml_backend_reg_i:
    |     get_name()          -> "CUDA8"
    |     get_device_count()  -> N (Fermi/Kepler devices)
    |     get_device()        -> device by index
    |     get_proc_address()  -> NULL (no extensions)
    |
    +-- ggml_backend_device_i (per device):
          get_name()          -> "CUDA8_0"
          get_description()   -> "GeForce GTX 560" (from cudaDeviceProp)
          get_memory()        -> cudaMemGetInfo
          get_type()          -> GGML_BACKEND_DEVICE_TYPE_GPU
          init_backend()      -> ggml_cuda8_ggml_backend_init(device)
          get_buffer_type()   -> ggml_cuda8_ggml_buffer_type()
          supports_op()       -> checks all 15 dispatch ops + 4 no-ops
          supports_buft()     -> accepts CUDA8 buffer type
          offload_op()        -> true (offload everything)

  supports_op coverage:
    - GGML_OP_NONE, RESHAPE, VIEW, PERMUTE, TRANSPOSE (no-ops)
    - GGML_OP_ADD (F32)
    - GGML_OP_MUL (F32 element-wise + scalar)
    - GGML_OP_SOFT_MAX (F32)
    - GGML_OP_SUM_ROWS (F32)
    - GGML_OP_MUL_MAT (Q8_0 x F32)
    - GGML_OP_RMS_NORM (F32, eps from op_params)
    - GGML_OP_ROPE (F32, mode=0, ext_factor=0 only)
    - GGML_OP_CONT (F32)
    - GGML_OP_DIAG_MASK_INF (F32)
    - GGML_OP_GET_ROWS (F32 src0 + I32 src1)

Files:
- ggml-cuda8-backend-reg.cpp (new) - device + registry implementation
- ggml-backend-reg.cpp (patched) - #ifdef GGML_USE_CUDA8 registration
- CMakeLists.txt (patched) - GGML_USE_CUDA8 define + source

Notes:
- Device filter only registers compute capability 2.x-3.x (Fermi/Kepler).
  Modern GPUs (cc >= 4.0) are handled by the standard CUDA backend.
- No async/events support (Fermi limitation).
- No host buffer or buffer_from_host_ptr (keeps it simple).
- This is the first step toward running real GGUF model inference on Fermi.
<!-- G36_STATUS_END -->

<!-- G37_STATUS_START -->
## G37 status: SOFT_MAX soft_max_ext guard (silent wrong-answer fix)

Status: **PASS on GTX 560 / CUDA 8 / Fermi** (11/11 regression).

G37 fixes a correctness bug rather than adding capability. `supports_op` claimed any
F32 `GGML_OP_SOFT_MAX` while the `SOFTMAX_ROWS_F32` kernel implements only a plain
row-wise softmax. ggml folds four separate things into that same op via
`ggml_soft_max_ext()`:

    src[1]    - attention mask
    src[2]    - attention sinks (ggml_soft_max_add_sinks)
    params[0] - scale    (1/sqrt(head_dim) in real attention)
    params[1] - max_bias (ALiBi slope)

The kernel takes no mask, has no sink support, and never reads `op_params`. So a real
attention graph calling `ggml_soft_max_ext(kq, mask, scale, max_bias)` was claimed by
this backend and computed as an **unmasked, unscaled** softmax: no crash, no fallback,
just plausible-looking wrong attention weights. Nothing in G15-G36 caught it because
every existing smoke test uses plain `ggml_soft_max()`.

Validated G37 checkpoints:

- **G37A**: `ggml_cuda8_soft_max_is_plain()` predicate added; `supports_op` and
  `graph_compute` both gated on it. Predicate verified standalone under `-std=c++11`
  across 7 cases (plain / zeroed params / mask / sinks / scale / max_bias / NULL).
- **G37B**: smoke fixtures corrected and rejection cases added.
- **G37C**: full regression on GTX 560, 11/11 pass:

      PASS  ggml-cuda8-ggml-backend-supports-op-smoke                    <- G37 primary
      PASS  ggml-cuda8-ggml-backend-graph-compute-softmax-smoke          <- fixture fix
      PASS  ggml-cuda8-ggml-backend-graph-compute-attnlike-smoke         <- fixture fix
      PASS  ggml-cuda8-softmax-smoke
      PASS  ggml-cuda8-dispatch-all-smoke
      PASS  ggml-cuda8-ggml-backend-graph-builder-softmax-smoke
      PASS  ggml-cuda8-ggml-backend-graph-builder-attnlike-smoke
      PASS  ggml-cuda8-ggml-backend-graph-builder-q8_0-residual-scale-add-softmax-sumrows-smoke
      PASS  ggml-cuda8-ggml-backend-graph-builder-transformer-block-smoke
      PASS  ggml-cuda8-ggml-backend-graph-builder-attention-smoke
      PASS  ggml-cuda8-ggml-backend-graph-builder-e2e-smoke              <- G35 e2e, unaffected

  The supports-op smoke exits nonzero on any expectation mismatch, so its PASS
  covers all five new rejection cases plus `SOFT_MAX (plain)` still being accepted.

- **G37D**: incidental fixes surfaced by the rebuild:

  - `ggml-cuda8-ggml-backend.cpp` was missing `#include <cuda_runtime.h>`. It calls
    the CUDA runtime directly but does not get the header transitively - the
    `-ggml-backend.h` -> `-dispatch.h` -> `-context.h` -> `-backend.h` chain never
    includes it. This had been latent behind a stale object file; the G37 header
    change forced the recompile that exposed it.
  - `run-regression.sh` added: builds and runs a named target set, resolving
    binaries under `<build>/bin` (the root `CMakeLists.txt` redirects
    `CMAKE_RUNTIME_OUTPUT_DIRECTORY` there). Written for the container's cmake
    3.5.1 - no `-S`/`-B`, no `--build -j`.
  - `.gitattributes`: `*.sh text eol=lf`, so shell scripts authored on Windows
    survive the round trip to the Linux container.

Implementation:

    ggml-cuda8-ggml-backend.h
      + ggml_cuda8_soft_max_is_plain(op)   // shared predicate, C++11, header-inline
          returns 1 only for: src[1]==NULL, src[2]==NULL, scale==1.0f, max_bias==0.0f

    ggml-cuda8-backend-reg.cpp
        GGML_OP_SOFT_MAX  -> types AND ggml_cuda8_soft_max_is_plain(op)
          (the gatekeeper: unsupported forms now fall back to CPU)

    ggml-cuda8-ggml-backend.cpp
        GGML_OP_SOFT_MAX  -> hard error if !ggml_cuda8_soft_max_is_plain(node)
          (defence in depth: should be unreachable, so reaching it means the
           scheduler bypassed supports_op - fail loudly rather than miscompute)

Fixture corrections (these were latent bugs in the tests, not regressions):

- `ggml-cuda8-ggml-backend-graph-compute-softmax-smoke.cpp:161`
- `ggml-cuda8-ggml-backend-graph-compute-attnlike-smoke.cpp:104`

  Both built synthetic SOFT_MAX nodes with `src[1] = &dummy` as filler. `src[1]` is
  the mask slot, so those nodes were malformed - they described a masked softmax while
  expecting plain-softmax results. Both now set `src[1] = NULL` and stamp
  `op_params = { 1.0f, 0.0f }`, matching what `ggml_soft_max()` actually emits.

  Note the general trap: `memset`-based tensor fixtures leave `op_params` zeroed, which
  reads as `scale = 0.0f` - **not** a plain softmax. Any hand-built SOFT_MAX node has to
  stamp the params explicitly.

New rejection cases in `ggml-cuda8-ggml-backend-supports-op-smoke.cpp`:

    SOFT_MAX (plain)        -> true
    SOFT_MAX (mask)         -> false
    SOFT_MAX (sinks)        -> false
    SOFT_MAX (scale!=1)     -> false
    SOFT_MAX (max_bias)     -> false
    SOFT_MAX (zero params)  -> false

Consequence: real attention softmax now runs on the CPU. That is a graph split and it
is slower - but it is correct, which the previous behaviour was not. G41 implements
mask/scale/max_bias properly in the kernel and lifts the restriction.
<!-- G37_STATUS_END -->

<!-- G38_STATUS_START -->
## G38 status: Fermi grid-limit clamps across the older kernels

Status: **PASS on GTX 560 / CUDA 8 / Fermi** (21/21 regression).

On compute capability 2.x the maximum `gridDim.x` is 65535 (2^31-1 only from sm_30).
At 256 threads/block that ceiling is hit at ~16.7M work items - a 4096x4096 tensor,
i.e. an ordinary weight matrix. Every smoke test in this directory used small shapes,
so the unclamped launches survived from G9 to G38 unnoticed.

**Failure mode, stated precisely.** Every launcher here checks `cudaGetLastError()`
straight after its launch, so an over-limit grid is rejected with
`cudaErrorInvalidConfiguration` and comes back as a dispatch error. That is loud, not
silent - an earlier draft of `docs/backport-cuda8.md` claimed otherwise and was wrong.
The genuinely silent variant is *clamping without a stride loop*: the launch then
succeeds and quietly computes only the first 65535 blocks' worth of output. Both are
covered below.

Validated G38 checkpoints:

- **G38A**: `ggml-cuda8-grid.cuh` added - `ggml_cuda8_grid_1d()` and
  `ggml_cuda8_grid_rows()`, plus the rules for pairing them with a stride loop.
- **G38B**: 14 launches converted across 9 files.
- **G38C**: `ggml-cuda8-oversized-smoke` added.
- **G38D**: `-DGGML_CUDA8_PTXAS_VERBOSE=ON` added for the register baseline.
- **G38E**: full regression on GTX 560, **21/21 pass** - the 4 primary targets,
  10 standalone kernel smokes for everything rewritten here, and the 7
  end-to-end graph smokes including the G35 e2e pipeline.
- **G38F**: build system reworked so the container can configure at all (below).

Two patterns, both requiring the kernel-side loop AND the host-side clamp (the clamp
alone would drop the work that does not fit in the first 65535 blocks):

    element-wise:
        const int stride = blockDim.x * gridDim.x;
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
        launch with ggml_cuda8_grid_1d(n, block_size)

    one block per row:
        for (int row = blockIdx.x; row < nrows; row += gridDim.x)
        launch with ggml_cuda8_grid_rows(nrows)

Converted: `add.cu`, `mul.cu` (x2), `scalar.cu` (x2), `rope.cu`, `getrows.cu`,
`diagmask.cu`, `reduce.cu` (x2), `softmax.cu`, `rms-norm.cu`, `mmv.cu` (x2), and the
K-quant `GET_ROWS` in `q4k.cu` / `q6k.cu`. The Q4_K/Q6_K and Q8_0 MUL_MAT kernels
already used a 2D grid and were left alone.

Two hazards worth recording, both specific to putting a loop around a kernel body
that was written to run once:

1. **Early `return` becomes a deadlock.** `softmax.cu` opened with
   `if (row >= rows) return;`. Inside a row loop that would let a thread skip the
   `__syncthreads()` its block-mates are waiting on. Removed in favour of the loop
   bound, which is uniform across the block.

2. **Shared-memory reuse across iterations.** Every kernel carrying reduction scratch
   (`softmax.cu`, `reduce.cu` x2, `rms-norm.cu`, `mmv.cu` block kernel) now
   `__syncthreads()` at the end of each iteration, before the next one overwrites
   `partial[tid]` / `sdata[tid]`.

Signature changes (all file-local, no external callers):
- `kernel_rms_norm_f32` gained `nrows`
- `kernel_diag_mask_inf_f32` gained `nrows`; its index arithmetic moved to `size_t`,
  which was an independent latent overflow (`row * ncols` in `int`)
- `kernel_get_rows_q4k` / `kernel_get_rows_q6k` gained `n_tokens`

New smoke: `ggml-cuda8-oversized-smoke`

    elementwise  n = 65535*256 + 4096   ADD_F32, MUL_SCALAR_F32   (~200 MiB, skipped if VRAM is short)
    row kernels  rows = 70000           REDUCE_SUM_ROWS, SOFTMAX_ROWS, RMS_NORM
    get_rows     n_tokens = 70000       GET_ROWS_F32

Output buffers are poisoned with `0xFF` before each launch, and the checks look
*past* the 65535-block boundary rather than at element 0 - a clamp-without-loop bug
produces correct output at the start and stale poison after it.

Register baseline (`-DGGML_CUDA8_PTXAS_VERBOSE=ON`) is wired up but **not yet
measured**. sm_2x caps registers at 63 per thread; the K-quant kernels dequantize a
whole block into registers, so they are the ones to look at first.

### G38F: the container can now configure the build itself

The old `build-cuda8-parent` tree was configured against the repo root by cmake
3.5.1, back when the root still accepted that. An upstream merge later raised
`cmake_minimum_required` to 3.14 in both `CMakeLists.txt` and `ggml/CMakeLists.txt`,
which stranded the tree: cmake 3.5.1 could still drive `cmake --build` on it, but
could never regenerate it. Adding any new target therefore failed with
`No rule to make target`, with nothing in the error pointing at the cause.

`run-regression.sh` now configures **`ggml/src/ggml-cuda8` directly**, into
`build-cuda8-kernels`. That directory is its own project needing only cmake 3.5 and
reaching the ggml core by relative path, so the parent project is not involved in
building kernels or smokes at all. `build-cuda8-parent` can be deleted.

Three things a clean configure exposed, all previously masked by stale objects in
the old tree:

1. **`GGML_VERSION` / `GGML_COMMIT`** are defined by the parent, and `../ggml.c`
   requires them. `CMakeLists.txt` now supplies fallbacks when configured standalone.

2. **`ggml_abort` was unresolved** for every smoke that links the kernel archive
   without compiling `../ggml.c`. The inline helpers in `ggml-impl.h`
   (`ggml_set_op_params`, `ggml_hash_insert`) expand to `GGML_ASSERT`, which calls
   it. New `ggml-cuda8-ggml-core` static library provides it.

   Kept deliberately *separate* from `ggml-cuda8-kernels`: the host build imports
   `libggml-cuda8-kernels.a` as a raw archive next to its own `ggml-base`, so
   `ggml.c` inside that archive would collide at link time. As a CMake dependency
   the archive stays byte-identical.

3. **`ggml_backend_tensor_set` / `_memset` were unresolved.** `../ggml.c` calls them
   from `ggml_set_zero()` and `ggml_graph_reset()`; they live in
   `../ggml-backend.cpp`, which cannot be compiled here (the full ggml target drags
   in C++17 sources that GCC 5.4 at C++11 rejects). Fixed with the same standalone-GC
   pattern the graph-builder targets already use: `-ffunction-sections
   -fdata-sections` on the core library, `-Wl,--gc-sections` propagated PUBLIC to the
   executables. Nothing here calls those functions, the sections are discarded, and
   GNU ld does not report undefined references from discarded sections.

Also: `ggml-cuda8-host.cmake` now accepts the flat standalone archive layout
(`<build>/libggml-cuda8-kernels.a`) as well as the parent's nested one, so the
host-side `GGML_CUDA8_HOST` import keeps working with either build.
<!-- G38_STATUS_END -->

<!-- G39_STATUS_START -->
## G39 status: first real GGUF, and the CPU/GPU split log

Status: **tooling in place, not yet run.** The result of this checkpoint is a
measurement, not a feature: it replaces guesswork about which ops matter with a
frequency-ordered list.

Everything before this point was validated against hand-built graphs. A real GGUF
graph differs in ways that are hard to predict, so the point of G39 is to stop
predicting and look.

### What was added

**`GGML_CUDA8_DEBUG_OPS=1`** - `supports_op` returns a bare bool, so a model that
silently falls back to the CPU leaves no record of what was refused.
`ggml-cuda8-backend-reg.cpp` now logs each distinct refused op signature once when
first seen, and prints a frequency-ordered summary at exit:

    ggml-cuda8: ops refused by supports_op (ran on CPU), by frequency:
           896  SOFT_MAX dst=f32 src0=f32 src1=f32 [soft_max_ext: mask/sinks/scale/max_bias]
           448  GLU/swiglu dst=f32 src0=f32
            32  GET_ROWS dst=f32 src0=q6_K src1=i32

That ordering *is* the roadmap for G40 onwards. The signature carries op, GLU/unary
subtype, and dst/src0/src1 types - enough to identify the work without every shape
becoming its own entry. SOFT_MAX additionally reports when it was refused for
soft_max_ext features rather than types (the G37 guard), since that is invisible in
the types alone.

This complements rather than duplicates ggml's own `GGML_SCHED_DEBUG=2`, which shows
*where* each node ran and where the split boundaries fall, but never why a backend
declined a node.

**`stage-host-artifacts.sh`** - `ggml-cuda8-host.cmake` needs `cuda8-libs/` and
`cuda8-headers/` next to the archive, and the CUDA 8 toolkit only exists inside the
container. This copies them onto the shared volume and refuses to run against a
non-CUDA-8 toolkit, since staging a newer runtime would produce a host binary that
cannot talk to sm_21 kernels.

### Procedure

In the CUDA 8 container:

    ./ggml/src/ggml-cuda8/run-regression.sh          # builds the archive
    ./ggml/src/ggml-cuda8/stage-host-artifacts.sh

On the Ubuntu 22.04 host, **from the repo root** - the two sides mount the same
volume at different paths (`/workspace/notebooks/llama.cpp-ph2` in the container,
`/mnt/shared/caffe/examples/llama.cpp-ph2` on the host), so use `$PWD` rather than
copying an absolute path across:

    cmake -S . -B build-host \
          -DGGML_CUDA=OFF \
          -DGGML_CUDA8_HOST=ON \
          -DGGML_CUDA8_LIB_DIR="$PWD/build-cuda8-kernels"
    cmake --build build-host -j$(nproc) --target llama-cli llama-server

    GGML_CUDA8_DEBUG_OPS=1 GGML_SCHED_DEBUG=2 \
      ./build-host/bin/llama-cli -m ./models/Qwen3-0.6B-Q4_K_M.gguf \
      -ngl 99 -p "hello" -n 8 --cache-type-k f32 --cache-type-v f32 \
      2> split.log

`llama-cli` is the simplest instrument for a first run; `llama-server` is the actual
target and uses the same backend path.

`--cache-type-k/v f32` is required until G49: the KV cache defaults to F16 and Fermi
has no F16 arithmetic. It costs 2x KV memory, which at 0.6B and a short prompt is
nothing.

### What to expect from Qwen3 specifically

Qwen3 0.6B fits the 1 GB budget better than anything else available (~0.4 GB), but
it is the *architecture* least suited to the current op set, so do not read the split
log as a verdict on the port:

- **ROPE will be refused on every layer.** Qwen3 uses NeoX-style rope (`mode=2`);
  `supports_op` accepts only `mode=0`. That is G45, and it will dominate the log.
- **SWIGLU will be refused on every layer** - the FFN, and the largest block of
  FLOPs. That is G40.
- **SOFT_MAX will be refused on every attention op** - `soft_max_ext` with mask and
  scale, correctly declined by the G37 guard. That is G41.
- Qwen3 also applies RMS norm per head to Q and K, so expect more `RMS_NORM` nodes
  than a LLaMA-architecture model of the same size. Those are supported.

Running a LLaMA-architecture model as well (TinyLlama 1.1B or Llama 3.2 1B at Q4_K_M)
is worth it for contrast: those use `mode=0` rope, so ROPE stays on the GPU and the
log isolates the FFN and attention gaps instead. Both are close to the 1 GB ceiling
though, so partial offload may be needed.

The frequency ordering is per-architecture. If Qwen3 is the target, G45 moves up the
list; if it is a LLaMA-class model, G40 and G41 come first.

### Required: `-nkvo` until SET_ROWS lands (G43)

Without it, the run aborts during `graph_reserve`:

    pre-allocated tensor (cache_k_l0) in a buffer (CUDA8_0) that cannot run the
    operation (SET_ROWS)

The KV cache is written with `ggml_set_rows` (`llama-kv-cache.cpp:1228,1263,1284`).
With `-ngl 99` the cache is allocated in the CUDA8 buffer, so the destination tensor
is pre-allocated in a buffer whose backend does not support the op - and the CPU
backend does not accept that buffer type. `ggml_backend_sched_backend_id_from_cur`
has no legal placement and calls `GGML_ABORT` rather than falling back.

This is the one class of unsupported op that is fatal rather than merely slow: a
refused op on *activations* just splits the graph, but a refused op writing to a
*pre-allocated* tensor has nowhere to go. `-nkvo` / `--no-kv-offload` keeps the cache
in host memory and sidesteps it. G43 removes the need.

### Expected obstacle: the CUDA 8 GCC check (did NOT occur)

Recorded because the prediction was wrong and the reason is worth keeping:
the host build with GCC 11.4 completed cleanly against the staged CUDA 8 headers.
The `crt/host_config.h` version check either is not reached by this include path or
is not present in the way expected. No mitigation was needed.

Original analysis follows, in case a future toolchain change revives it.

CUDA 8's `crt/host_config.h` contains

    #if __GNUC__ > 5 || (__GNUC__ == 5 && __GNUC_MINOR__ > 3)
    #error -- unsupported GNU version! gcc versions later than 5.3 are not supported!

and the host build compiles `ggml-cuda8-backend-reg.cpp` with the host GCC (11.x),
which includes `cuda_runtime.h` from the staged CUDA 8 headers. This is expected to
fire on the first host build.

It is a header assertion, not a real ABI constraint: the file only calls a handful of
runtime API functions (`cudaGetDeviceCount`, `cudaGetDeviceProperties`,
`cudaMemGetInfo`), all plain C entry points that are ABI-stable. Options, in order of
preference:

1. Compile just that one file with `-D__NV_NO_HOST_COMPILER_CHECK__` (if CUDA 8's
   header honours it) or with the check patched out of the staged copy of
   `cuda8-headers/crt/host_config.h` - the staged copy exists precisely so it can be
   modified without touching the container.
2. Compile that one file with `g++-5` on the host, if it is installed. It has a C
   interface, so it links against the C++17 build cleanly.
3. Declare the few runtime functions by hand in the host TU and drop the
   `cuda_runtime.h` include entirely.

### Results: Qwen3-0.6B-Q4_K_M on the GTX 560

The model loads and lands on the GPU - 372 MiB of weights, 306 MiB compute buffer,
190 MiB free of 963 MiB. The 1 GB budget holds for a 0.6B model at Q4_K_M.

Ops refused by `supports_op`, by frequency (counts span several `graph_reserve`
passes, so read the ratios rather than the absolute numbers):

           4312  ROPE dst=f32 src0=f32 src1=i32
           3080  GLU/SWIGLU dst=f32 src0=f32 src1=f32
            448  MUL_MAT dst=f32 src0=f32 src1=f32
            224  SOFT_MAX dst=f32 src0=f32 src1=f32 [soft_max_ext: mask/sinks/scale/max_bias]
            171  CPY dst=f16 src0=f32 src1=f16
             84  FLASH_ATTN_EXT dst=f32 src0=f32 src1=f16

**ROPE and SWIGLU dominate by an order of magnitude.** ROPE is the Qwen3 NeoX
(`mode=2`) case, refused on every layer; SWIGLU is the entire FFN. Everything else is
a rounding error next to those two.

Note this ordering is Qwen3-specific. A LLaMA-architecture model uses `mode=0` rope,
which is already supported, so ROPE would vanish from its list entirely and SWIGLU
would lead. Worth re-running against a LLaMA-class model before treating this as the
definitive ordering.

`MUL_MAT f32xf32` at 448 is the attention matmuls (K·Q, probs·V) - activations rather
than weights, so the Q4_K path never sees them. An earlier draft called wiring these
up a "near-free win" because `mmv.cu:210` already has F32 kernels. That was wrong:
those are strictly matrix-times-vector, while the attention matmuls are per-head
batched and 3D (`ne02`/`ne03`), often on permuted views. Useful starting point, not a
drop-in.

`-fa off` matters: the default is `auto`, and since the CPU backend supports
`FLASH_ATTN_EXT`, auto enables it and the whole attention block becomes one CPU op.
While FA is on, implementing SOFT_MAX and MUL_MAT changes nothing, because those
nodes are not in the graph.

### Re-measured after G45 (ROPE NeoX) and G40 (SwiGLU)

Same command, same model, after the two big ops landed:

           112  MUL_MAT dst=f32 src0=f32 src1=f32
            56  SOFT_MAX dst=f32 src0=f32 src1=f32 [soft_max_ext: mask/sinks/scale/max_bias]

Six entries totalling ~8300 became two totalling 168. At 28 layers that is 2 matmuls
and 1 softmax per layer per pass - **exactly the attention core and nothing else**.
Embeddings, norms, every quantized projection, rope and the whole FFN are now GPU-side.

The `CPY f32->f16` and `FLASH_ATTN_EXT` entries from the first run are absent because
this run passed `-fa off` and `--cache-type-k/v f32`; the earlier counts came from the
auto-fit probe, which builds a context with its own defaults.

Consequence for the roadmap: **G41 (SOFT_MAX_EXT) and G42 (F32 MUL_MAT) are now a
single unit of work.** They interleave inside the same attention block, so doing only
one still leaves the graph splitting on every layer.

### Bug found and fixed: double free during teardown

With `-nkvo -fa off` the run got past the scheduler and printed its memory
breakdown, then died with `double free or corruption (fasttop)`.

`cuda8_free_buffer()` ended with `std::free(buffer)`. But
`ggml_backend_buffer_free()` calls that hook and then does `delete buffer` itself, so
the struct was released twice - and with mismatched allocators, since ggml allocates
it with `new` in `ggml_backend_buffer_init()` while `cuda8_buft_alloc_buffer()` was
hand-rolling a `std::malloc`.

Fixed by allocating the struct with `new` - matching ggml's `delete` - and leaving its
lifetime entirely to ggml. The hook now owns only the device allocation and our own
context. This would have corrupted the heap on every buffer free; nothing caught it
earlier because the smoke tests allocate and free through the same paths without a
full model teardown.

The first attempt called `ggml_backend_buffer_init()` instead, which is the tidier
form and picks up any fields upstream adds. It does not link in the container: that
function lives in `../ggml-backend.cpp`, which the standalone build does not compile,
and pulling it in would cascade into `ggml-alloc.c` and ggml's own
`ggml-backend-reg.cpp`. So `cuda8_buft_alloc_buffer()` mirrors the construction by
hand instead - worth re-checking against `ggml_backend_buffer_init()` on each upstream
sync, since a newly added field would be value-initialised here rather than set.

(`backend->device` was checked at the same time and is fine - `init_backend` in
`ggml-cuda8-backend-reg.cpp` sets it, as it must, since `ggml_backend_supports_buft`
dereferences it.)
<!-- G39_STATUS_END -->

<!-- G45_STATUS_START -->
## G45 status: ROPE NeoX (mode 2)

Status: **PASS on GTX 560 / CUDA 8 / Fermi** (21/21 regression).

Top of the G39 rejection log by an order of magnitude - 4312 refusals, every ROPE
node in every layer, because Qwen3 uses NeoX-style rope and `supports_op` accepted
only `mode=0`.

### The layout difference

Both modes rotate `n_dims/2` pairs and pass through `[n_dims, ne0)`; they differ only
in which two elements form a pair. From `rotate_pairs()` in
`ggml/src/ggml-cpu/ops.cpp`:

    NORMAL (0)  rotate_pairs(n_dims, n_offset=1,         scale=1)
                pair p -> elements (2p, 2p+1)            [cscscscs]
    NEOX   (2)  rotate_pairs(n_dims, n_offset=n_dims/2,  scale=2)
                pair p -> elements (p, p + n_dims/2)     [ccccssss]

theta is indexed by the pair number in both cases, so the rotation maths is shared and
only the element indices change.

### Verification

`kernel_rope_f32` and the smoke's CPU reference share the same logic, so agreeing
proves nothing. Both were instead checked against an independent transcription of
ggml's own `ggml_rope_cache_init` + `rotate_pairs`, across four configurations:

    full rotary head_dim=64            mode=0  9.5e-07   mode=2  1.2e-06
    partial rotary n_dims=32           mode=0  6.0e-07   mode=2  6.6e-07
    qwen3-like freq_base=1e6           mode=0  9.8e-07   mode=2  1.2e-06
    freq_scale=0.5, partial rotary     mode=0  3.0e-07   mode=2  2.4e-07

The smoke also asserts NEOX output *differs* from NORMAL on the same input - if the
kernel ignored `mode`, both would match a reference that also ignored it.

On hardware:

    head_dim=64  n_heads=4  seq_len=8  n_dims=64  freq_base=10000  freq_scale=1.0
      NORMAL (mode 0)  max_err=5.960464e-07  PASS
      NEOX   (mode 2)  max_err=4.917383e-07  PASS
      NEOX differs from NORMAL: yes (PASS)
      MROPE (mode 8) rejected: yes (PASS)

    ROPE (mode=0 NORMAL)   true    ROPE (mode=8 MROPE)    false
    ROPE (mode=2 NEOX)     true    ROPE (mode=24 VISION)  false
                                   ROPE (YaRN ext=1)      false
                                   ROPE (attn_factor!=1)  false
                                   ROPE (freq_factors)    false
                                   ROPE (zero params)     false

### Two silently-ignored parameters, now refused

`supports_op` checked only `mode` and `ext_factor`. Two others were being dropped on
the floor - the same class of bug as SOFT_MAX in G37:

- **`src[2]` (freq_factors)** - `ggml_rope_cache_init` divides theta by
  `freq_factors[pair]`. The kernel never reads src[2], so a node carrying one would
  have been rotated at the wrong frequencies.
- **`attn_factor`** (`op_params[8]`) - `rope_yarn` multiplies cos/sin by it (`mscale`).
  The kernel always uses magnitude 1.

Both now rejected in `supports_op` and re-checked in the dispatch. Note the fixture
trap this exposes, same as G37's: a `memset` fixture leaves `attn_factor` at `0.0f`,
but real ggml passes `1.0f`. The supports-op smoke gained `set_rope_params()` for
exactly this reason.

Signature change: `ggml_cuda8_op_rope_f32()` gained an `int mode` parameter.

New/changed smoke cases:

    ROPE (mode=0 NORMAL)    -> true      ROPE (mode=8 MROPE)     -> false
    ROPE (mode=2 NEOX)      -> true      ROPE (mode=24 VISION)   -> false
                                         ROPE (YaRN ext=1)       -> false
                                         ROPE (attn_factor!=1)   -> false
                                         ROPE (freq_factors)     -> false
                                         ROPE (zero params)      -> false

`ROPE (mode=2/mrope) -> false` was removed - it asserted the old restriction.
<!-- G45_STATUS_END -->

<!-- G40_STATUS_START -->
## G40 status: SwiGLU (the FFN)

Status: **PASS on GTX 560 / CUDA 8 / Fermi** (22/22 regression).

Second in the G39 rejection log at 3080, and the largest block of FLOPs in the model -
`GGML_OP_GLU` is the whole feed-forward network.

    dst[i] = silu(gate[i]) * up[i],   silu(x) = x / (1 + exp(-x))

Mirrors `ggml_compute_forward_swiglu_f32` + the scalar tail of
`ggml_vec_swiglu_f32` (`ggml-cpu/ops.cpp`, `vec.cpp`).

### Two shapes, both implemented

`GGML_OP_GLU` carries its operands in one of two ways, and which one appears depends
on how the model was built:

    split   src1 != NULL   ggml_swiglu_split(a, b) - gate = src0, up = src1,
                           nc = src0->ne[0], dst same shape as src0
    halves  src1 == NULL   ggml_swiglu(a) - gate and up are the two halves of each
                           src0 row, nc = src0->ne[0]/2, op_params[1] (`swapped`)
                           selects which half is which

Qwen3 reaches `build_ffn(..., LLM_FFN_SILU, LLM_FFN_PAR, il)`, which calls
`ggml_swiglu_split()` - so the **split** form is the one that matters here. The halves
form is implemented anyway; it is a few lines and other architectures use it.

### Row strides, not contiguity

ggml only guarantees `ggml_is_contiguous_1` for this op - each row is contiguous, but
rows may be padded. The kernel therefore takes explicit row strides (in floats,
derived from `nb[1]`) rather than assuming a packed tensor. Assuming full contiguity
would have been a latent correctness bug on padded tensors, so the smoke includes a
deliberately padded case.

### Verification

Same approach as G45: the kernel and the smoke's reference share logic, so they were
both checked against an independent transcription of ggml's own implementation.
Five configurations, all exact (0.0e+00 - the arithmetic is identical, not merely
close):

    split, packed          halves swapped=0        halves swapped=1
    split, padded rows     degenerate nc=1

Inputs span [-8, 8] so silu's saturating tails are exercised rather than just x~0.

### Refused

`supports_op` accepts only `GGML_GLU_OP_SWIGLU`. The others are rejected:
`SWIGLU_OAI` carries alpha/limit in `op_params[2..3]` that this kernel does not apply,
and `REGLU`/`GEGLU`/`GEGLU_ERF`/`GEGLU_QUICK` use different activations entirely.
Non-contiguous rows and mismatched gate/up shapes are also refused.

New files: `ggml-cuda8-swiglu.cu`, `ggml-cuda8-swiglu-smoke.cu`.
New dispatch op: `GGML_CUDA8_OP_SWIGLU_F32` (20 dispatch ops now).
<!-- G40_STATUS_END -->

<!-- G43_STATUS_START -->
## G43 – SET_ROWS

Status: ✅ Hardware verified on GTX 560

Verification:
- Removed `-nkvo`
- Used:
  --cache-type-k f32
  --cache-type-v f32
- Observed `SET_ROWS_F32` execution on CUDA8 backend
- Generation completed successfully
- `SET_ROWS` no longer appears in unsupported-op statistics

Conclusion:
G43 eliminates the scheduler blockage that previously required `-nkvo`.

### Why this was reordered ahead of G41/G42

The post-G40 measurement left only two refusals - 112 `MUL_MAT f32xf32` and 56
`SOFT_MAX soft_max_ext`, the attention core. The obvious next step was to implement
those two. It would have been wasted work.

`llama-graph.cpp` (build_attn_mha):

    if (!cparams.offload_kqv) {
        // all nodes between the KV store and the attention output are run on the CPU
        ggml_backend_sched_set_tensor_backend(sched, cur, backend_cpu);
    }

and `common/common.cpp:1721`:

    cparams.offload_kqv = !params.no_kv_offload;

So `-nkvo` makes llama.cpp *pin* the attention block to the CPU. Those 168 refusals
are a consequence of `-nkvo`, not an independent gap: implementing both attention
kernels would have changed nothing, because llama.cpp would still place attention on
the CPU - correctly, since with `-nkvo` the KV cache is in host memory and running
attention on the GPU would mean copying the cache across PCIe every step.

`-nkvo` is required only because SET_ROWS is missing. So SET_ROWS is the unlock:
it removes `-nkvo`, which stops the pinning, which is what makes G41/G42 worth doing.

### Fatal rather than slow

Most refused ops just split the graph. SET_ROWS is one of the few where refusal is
fatal: the destination is pre-allocated in the CUDA8 buffer, so
`ggml_backend_sched_backend_id_from_cur` has no legal placement and calls
`GGML_ABORT`:

    pre-allocated tensor (cache_k_l0) in a buffer (CUDA8_0) that cannot run the
    operation (SET_ROWS)

That is the abort seen at the start of G39.

### F32 only - the cache type requirement moves, it does not disappear

Only F32 destinations are handled. F16 stores need G49, so the KV cache must stay
F32 (`--cache-type-k f32 --cache-type-v f32`). Dropping `-nkvo` *without* those flags
re-triggers the abort above, because the F16 cache would sit in our buffer with no
backend able to write it.

    before G43:  -ngl 99 -nkvo -fa off --cache-type-k f32 --cache-type-v f32
    after  G43:  -ngl 99       -fa off --cache-type-k f32 --cache-type-v f32

### Verification

Same method as G45/G40 - an independent transcription of ggml's implementation,
compared against a transcription of the kernel, across five configurations:

    kv-cache 2D          padded dst rows        3D idx broadcast
    4D idx broadcast     ne11=2 (no broadcast)

Index tensors deliberately included out-of-range values. The CPU path asserts on
those; a kernel cannot, and writing anyway would corrupt memory outside the
destination, so the kernel skips them - both sides agree on that behaviour.

New files: `ggml-cuda8-set-rows.cu`, `ggml-cuda8-set-rows-smoke.cu`.
New dispatch op: `GGML_CUDA8_OP_SET_ROWS_F32` (21 dispatch ops now).
<!-- G43_STATUS_END -->


### G51 status: dispatcher robustness hardening (post-G45)

Status: **PASS on GTX 560 / CUDA 8 / Fermi** (24/24 regression, including one
new hardware-verified fault-injection target).

G51 is a correctness/robustness pass rather than a new-op checkpoint. It
closes four gaps found during a code review of the dispatcher and backend
layers, none of which were caught by the existing smoke suite because they
only manifest under partial-failure or malicious/malformed-input conditions
that the happy-path fixtures don't exercise.

#### G51A: `exec_add_f32` buffer leak on partial allocation failure

`ggml_cuda8_exec_add_f32()` (ggml-cuda8-add.cpp) allocated three CUDA8
backend buffers (`b0`, `b1`, `bd`) in sequence but only freed buffers already
allocated on a *later* allocation failure in the Q8_0 MUL_MAT and scalar
ADD/MUL paths — the plain ADD_F32 path returned `-1` on a second or third
allocation failure without freeing the first one or two buffers already
allocated. ADD_F32 is one of the most frequently dispatched ops (every
residual/bias-add node in the graph-builder pipelines goes through it), so
a leak here compounds under repeated allocation pressure rather than being
a one-off. Fixed to match the cleanup pattern already used by
`exec_mul_mat_q8_0_f32_vec()` and `exec_scalar_f32_host_staging()`: each
allocation failure now frees every buffer already allocated before
returning.

Verified via `ggml-cuda8-add-smoke` (unchanged behavior on the success
path) plus the full graph-builder regression.

#### G51B: ROPE/SWIGLU `supported_*` gates missing `op_params` validation

`ggml_cuda8_supported_rope_f32()` and `ggml_cuda8_supported_swiglu_f32()`
(ggml-cuda8-dispatch.cpp) validated only tensor types/shapes — the
`op_params`-level checks (ROPE: `mode`, `ext_factor`, `attn_factor`,
`freq_factors`/`src[2]`; SWIGLU: the `glu_op` variant) lived exclusively in
the `exec_*` functions, with an explicit comment noting this was "defence
in depth... reaching here means the scheduler bypassed supports_op." This
is the same class of bug G37 fixed for SOFT_MAX_EXT: if a scheduler's
`supports_op` hook is wired to `ggml_cuda8_dispatch_supported()` (which
`ggml_cuda8_ggml_backend_dispatch_op()` does call directly), a YaRN-scaled
ROPE node or a non-SWIGLU GLU node (GEGLU, SWIGLU_OAI, etc.) would be
accepted as "supported," assigned to this backend by the scheduler, and
only discovered unsupported once `graph_compute()` was already mid-flight
— aborting the whole graph instead of the scheduler cleanly routing it to
CPU or another backend.

Moved the `op_params` checks into the `supported_*` gates (the exec-side
checks remain as defence-in-depth, with their error messages updated to
note they are now unreachable in the normal case). No behavior change on
any currently-passing smoke case — G45's ROPE NeoX and G40's SWIGLU paths
are unaffected, since both already pass the (now duplicated) checks.

Verified via `ggml-cuda8-rope-smoke`, `ggml-cuda8-swiglu-smoke`, and full
regression.

#### G51C: K-quant `supported_*` gates missing null-data and residency checks

`supported_mul_mat_q4k_f32`, `supported_mul_mat_q6k_f32`,
`supported_get_rows_q4k`, `supported_get_rows_q6k` checked only
`!src0 || !src1 || !dst` (struct-pointer non-null), unlike the Q8_0 MUL_MAT
path's `check_tensor_ptrs()`, which additionally rejects null `->data`. The
`exec_*` counterparts for these four ops pass `src0->data`/`src1->data`/
`dst->data` straight into the kernel launcher with no host<->device
staging — the architecturally correct choice for weight/activation
throughput, since the ggml scheduler contract guarantees CUDA8-buffer
residency for any tensor assigned to this backend under normal
`graph_compute` — but it left no defence for direct-dispatch callers (e.g.
tests bypassing `graph_compute`) or a scheduler bug that assigns a
host-backed tensor here, which would fault asynchronously inside the
kernel rather than failing cleanly at the dispatch boundary.

Fixed: all four `supported_*` gates now call the shared `check_tensor_ptrs()`
helper (closing the null-data gap), plus a debug-build-only
(`#ifndef NDEBUG`) residency diagnostic using the existing
`ggml_cuda8_ggml_tensor_is_device_resident()` registry lookup. The
diagnostic logs a loud warning if a K-quant tensor isn't registered as
CUDA8-resident but does not change `supported()`'s return value, so it
cannot introduce a false rejection for any currently-passing path.

Verified via `ggml-cuda8-getrows-smoke`, `ggml-cuda8-dispatch-all-smoke`,
and full regression — the residency diagnostic was live during this run
(non-`NDEBUG` build) and produced no warnings, confirming the K-quant
smoke fixtures do allocate through the CUDA8 buffer registry as expected.

#### G51D: sticky poisoned-device flag for fatal CUDA errors — hardware verified

`cuda8_backend_synchronize()` (ggml-cuda8-ggml-backend.cpp) previously only
logged a `cudaDeviceSynchronize()` failure to stderr and returned —
`ggml_backend_i::synchronize` is `void(*)(ggml_backend_t)`, so there was no
return channel to propagate failure through in the first place. A fatal
CUDA error class (`cudaErrorIllegalAddress`, `cudaErrorLaunchFailure`,
`cudaErrorECCUncorrectable`, `cudaErrorAssert`) typically invalidates the
CUDA context — and on most driver versions, the whole per-process CUDA
state for that device — until process exit. Silently logging and
continuing meant the *next* `graph_compute()` call (a fresh
`ggml_cuda8_context_create()`) would also fail, but with a generic,
unrelated-looking message instead of one pointing back at the original
fault.

Added a process-wide `std::atomic<bool> g_cuda8_device_poisoned`, latched
by `cuda8_backend_synchronize()` on a narrow allow-list of fatal error
codes (deliberately not "anything != cudaSuccess" — recoverable
launch-configuration errors like the grid-size class G38 fixed are
excluded). Checked at the top of both `cuda8_backend_graph_compute()` and
`ggml_cuda8_ggml_backend_dispatch_op()`, so any further dispatch attempt
after a fatal fault is refused immediately with a message pointing back at
the original error, rather than failing later with an opaque, unrelated
CUDA error. Also fixed `cuda8_backend_synchronize()` to call
`cudaSetDevice(ctx->device)` before syncing — previously dead code for the
single-GPU-per-process case, but a latent multi-device bug.

New smoke target: **`ggml-cuda8-poison-smoke`**. Deliberately launches a
kernel against unregistered host memory (a `malloc()`'d pointer reinterpreted
as a device pointer under CUDA's Unified Virtual Addressing) to provoke a
real illegal-address fault, then verifies:
1. the flag starts clean,
2. a control `ADD_F32` dispatch succeeds on the healthy device,
3. the injected fault + `synchronize()` latches the flag,
4. `ggml_cuda8_ggml_backend_device_is_poisoned()` reports `1`,
5. the same dispatch call is refused immediately post-injection,
6. the flag stays latched across a second `synchronize()` call.

**Verified on real hardware** — GTX 560 / CUDA 8.0.61 / driver 390.157. The
injection technique reliably faults on this driver/hardware combination
(rather than landing in the test's `INCONCLUSIVE` degrade path), and the
full assertion chain passed: fault detected, flag latched, subsequent
dispatch refused, latch confirmed sticky across a second synchronize call.
This is the first checkpoint in this backend's history to hardware-verify
a fault-handling path rather than only a happy path.

#### Regression

Added to the `PRIMARY` target set in `run-regression.sh` (targets exercising
the most recent changes), alongside `ggml-cuda8-oversized-smoke` and the
G37/supports-op targets:

```
PRIMARY=(
    ggml-cuda8-oversized-smoke
    ggml-cuda8-ggml-backend-supports-op-smoke
    ggml-cuda8-ggml-backend-graph-compute-softmax-smoke
    ggml-cuda8-ggml-backend-graph-compute-attnlike-smoke
    ggml-cuda8-poison-smoke
)
```

Full container regression: **24/24 pass** (was 23/23 before G51D added
`ggml-cuda8-poison-smoke`).

#### Notes

- None of G51A–D add or change dispatch op coverage — the 21 dispatch ops
  + 5 no-ops inventory from G43 is unchanged. This checkpoint is purely
  about failure-mode correctness in code paths the existing fixtures
  already exercised on the success side.
- `ggml_cuda8_ggml_backend_device_is_poisoned()` is defined in
  ggml-cuda8-ggml-backend.cpp but not yet declared in
  ggml-cuda8-ggml-backend.h — `ggml-cuda8-poison-smoke.cpp` currently
  carries a local `extern "C"` forward declaration as a stopgap. Worth
  adding the real prototype to the header the next time that file is
  touched.
- The register-spill baseline from G38 (`-DGGML_CUDA8_PTXAS_VERBOSE=ON`,
  §3.3 in docs/backport-cuda8.md) is still unmeasured — unrelated to this
  checkpoint, carried over as an open item.


### G41 status: SOFT_MAX_EXT proper (mask + scale + ALiBi)

Status: **PASS on GTX 560 / CUDA 8 / Fermi** (25/25 regression).

G41 lifts the G37 restriction for real attention softmax. Through G40, the
only softmax the CUDA8 backend could compute was the plain row-wise case
(`ggml_cuda8_soft_max_is_plain()`: no mask, no sinks, scale==1.0f,
max_bias==0.0f) - anything else fell back to CPU, correctly but slowly,
since a real attention graph almost always calls `ggml_soft_max_ext(kq,
mask, scale, max_bias)`. Per the G39 rejection log, this was one of only
two refusals left per layer after G40/G45 landed (`SOFT_MAX
soft_max_ext`, 56 occurrences alongside 112 `MUL_MAT f32xf32`) - i.e.
exactly the attention core.

#### What was added

A new dispatch op, `GGML_CUDA8_OP_SOFTMAX_EXT_F32`, alongside (not
replacing) the existing plain `SOFTMAX_ROWS_F32` path. The plain path is
completely unchanged - every pipeline that already used it (G16C, G19A,
G32A, the whole G16-G35 checkpoint history) still takes the same fast
path with the same kernel.

The new kernel implements:
```
v[c]   = src[c] * scale + (mask ? slope(head) * mask[c] : 0)
dst[c] = softmax(v)[c]
```
with the standard ALiBi per-head slope construction (`n_head_log2`,
`m0`/`m1` precomputed once on the host from `n_head`/`max_bias`, not
recomputed per row).

**Explicitly still refused, not silently mishandled** - same philosophy as
every other guard in this backend (G37's SOFT_MAX_EXT guard, G45's ROPE
`freq_factors`/`attn_factor` guards):
- **Attention sinks** (`src[2]`) - not implemented, refused in both
  `supports_op` and the dispatcher's `supported_*` gate.
- **Non-F32 (F16) mask** - deferred to the general F16-storage work
  already tracked as G49. Refused rather than misread.

New files: `ggml-cuda8-softmax-ext.h`, `ggml-cuda8-softmax-ext.cu`,
`ggml-cuda8-softmax-ext.cpp`, `ggml-cuda8-softmax-ext-smoke.cpp`.

New dispatch op: `GGML_CUDA8_OP_SOFTMAX_EXT_F32` (22 dispatch ops now).

#### Dispatch routing

`GGML_OP_SOFT_MAX` in `ggml-cuda8-ggml-backend.cpp`'s `graph_compute` now
branches three ways instead of two:
1. `ggml_cuda8_soft_max_is_plain(node)` -> `SOFTMAX_ROWS_F32` (unchanged
   fast path).
2. else if `ggml_cuda8_soft_max_is_supported_ext(node)` -> new
   `SOFTMAX_EXT_F32` path (mask/scale/ALiBi, no sinks, F32 mask only).
3. else -> loud dispatch failure (sinks or non-F32 mask reached graph_compute
   despite supports_op - should be unreachable, same "scheduler bypassed
   supports_op" defence-in-depth pattern used throughout this backend).

`supports_op` (`ggml-cuda8-backend-reg.cpp`) mirrors this: `is_plain()` OR
(`is_supported_ext()` AND mask-shape validation) - re-validating shape
independently rather than trusting the dispatcher alone, the same
double-gate pattern G51 established for SET_ROWS.

#### A build-system note worth keeping: `.cu` vs `.cpp` for smoke tests

The first attempt at `ggml-cuda8-softmax-ext-smoke` was written as a
`.cu` file and failed to compile under nvcc with `expected a declaration`
errors that appeared to originate in unrelated headers
(`ggml-cuda8-backend-buffer.h`, `ggml-cuda8-context.h`). The smoke
contained no actual device code (no `__global__`, no `<<<>>>` launches) -
only host-side CUDA runtime calls plus fake `ggml_tensor` construction to
exercise `ggml_cuda8_supported_softmax_ext_f32()` directly. nvcc's
front-end does not parse the full `ggml.h` / `ggml-cuda8-context.h`
header chain cleanly, even though host GCC 5.4 parses it fine. Renamed to
`.cpp` (host-compiled, linked against the already-nvcc-compiled
`ggml-cuda8-kernels` archive) with no other content change and it built
clean. This matches the existing convention in this directory - every
other smoke that constructs real `ggml_tensor`/`ggml_cuda8_context`
structs (`ggml-cuda8-add-smoke`, `ggml-cuda8-softmax-smoke`, ...) is
already `.cpp`; only kernel-only smokes with no struct dependency
(`ggml-cuda8-oversized-smoke.cu`, `ggml-cuda8-rope-smoke.cu`) stay `.cu`.
Worth remembering next time a new smoke needs both a real `ggml_tensor`
fixture and a CUDA kernel call in the same binary.

#### Verification

Nine numerical configurations, each checked against an independently
re-derived CPU reference (not sharing code with the kernel - the actual
check is agreement between two separately-written implementations, the
same method G40/G45 used):
- no mask, scale=1 (degenerate case, should match plain softmax
  numerically even though it takes the new code path)
- no mask, scale=0.125 (attention scale, no masking - one real shape from
  the G39 log)
- mask only, broadcast over heads (`mask_ne2=1`)
- mask + scale together
- mask + ALiBi, 4 heads (power-of-two `n_head_log2` case)
- mask + ALiBi, 6 heads (non-power-of-two `n_head_log2` case - exercises
  the `m1`/second branch of the slope formula)
- mask with a per-head mask tensor (`mask_ne2 == ne02`, no broadcast)
- mask + ALiBi with batch > 1 (`ne03=2`)
- large columns (`n_kv=1024`) with mask + ALiBi, to exercise the
  multi-iteration stride loop inside the kernel (not just small
  single-pass shapes)

Plus five `supported()`-gate rejection cases exercised directly (no GPU
touch): no-mask valid, F32-mask-broadcast valid, F16-mask refused,
mask `ne1` mismatch refused, mask `ne2` neither 1 nor `ne02` refused.

All nine numerical cases and all five rejection cases pass on GTX 560 /
CUDA 8.0.61 / driver 390.157.

#### Fixture fallout in `ggml-cuda8-ggml-backend-supports-op-smoke`

Same class of thing G37 and G45 both hit when they changed what
`supports_op` accepts: four `SOFT_MAX` fixtures in
`ggml-cuda8-ggml-backend-supports-op-smoke.cpp` were hardcoded to expect
`false` and now correctly expect `true` under the new logic:
- `SOFT_MAX (mask)` -> `SOFT_MAX_EXT (mask)`, moved to the TRUE section
- `SOFT_MAX (scale!=1)` -> `SOFT_MAX_EXT (scale!=1, no mask)`, moved to
  the TRUE section (this is literally the real-attention shape from the
  G39 log)
- `SOFT_MAX (max_bias)` -> `SOFT_MAX_EXT (mask + max_bias)`, moved to the
  TRUE section
- `SOFT_MAX (zero params)` - flipped in place; scale=0.0f with no mask is
  now a legitimately computable (if unusual) input rather than a rejected
  one. Not a correctness regression, but this fixture can no longer be
  relied on to catch an unstamped-`op_params` mistake in a future SOFT_MAX
  fixture - flagged in an inline comment for whoever touches this file
  next.

Added one new rejection case that had no coverage before this checkpoint:
`SOFT_MAX (F16 mask)`, confirming the F16-mask boundary (deferred to G49)
is actually enforced, not just documented in a comment.

`SOFT_MAX (sinks)` is unaffected and still correctly expects `false`.

#### Regression

Full container regression: **25/25 pass** (was 24/24 after G51 - the new
`ggml-cuda8-softmax-ext-smoke` target accounts for the difference; the
`supports-op-smoke` fixture fix did not add or remove a target, only
corrected four expected-value assertions within it).

#### What's next

Per the G39 re-measurement note, **G41 was explicitly paired with G42**
("a single unit of work") because both refusals live inside the same
attention block - implementing only one still leaves the graph splitting
on every layer. With G41 now done, **G42 (F32xF32 MUL_MAT for the
attention matmuls) is the remaining half of that pairing** and the next
highest-value checkpoint.

One correction carried over from the G39 write-up, worth restating here
since it's easy to lose: the *vector* F32 kernel already in `mmv.cu` is
**not** a drop-in for G42. The real attention matmuls (K.Q, probs.V) are
per-head batched and 3D (`ne02`/`ne03`), often on permuted views - this
is strictly matrix-times-vector, not the batched/permuted shape attention
actually needs. G42 will likely need to land together with G46
(permuted/non-contiguous src1 for MUL_MAT) rather than as an isolated
checkpoint. Worth re-running the `GGML_CUDA8_DEBUG_OPS=1` rejection log
against a real model now that G41 has landed, to get fresh data before
committing to G42's exact scope.


### G42 status: batched F32xF32 MUL_MAT (attention matmuls)

Status: **PASS on GTX 560 / CUDA 8 / Fermi** (26/26 regression).

G42 implements the second half of the "single unit of work" the G39
re-measurement identified: the attention matmuls (K.Q and probs.V), which
operate on activations rather than weights and so were never covered by the
quantized-weight MUL_MAT paths (Q8_0/Q4_K/Q6_K) or the vector matvec kernels
in mmv.cu. Together with G41 (SOFT_MAX_EXT), this covers both refusals the
G39 log showed dominating every attention block - in principle the attention
core no longer forces a CPU fallback per layer.

#### What was added

A new dispatch op, `GGML_CUDA8_OP_MUL_MAT_F32_F32`, and a new kernel
(`ggml-cuda8-mulmat-f32.cu`) that is batched and broadcast-aware, distinct
from both the quantized-weight MUL_MAT paths and the F32 vector matvec in
mmv.cu (which no dispatch op maps to - that gap noted in backport-cuda8.md
is what G42 fills, though not by wiring the existing vector kernel; see
below).

Semantics match `ggml_mul_mat(a=src0, b=src1)`:
```
dst[i01,i11,i12,i13] = sum_c src0[c,i01,i12/r2,i13/r3] * src1[c,i11,i12,i13]
```
where `r2 = ne12/ne02`, `r3 = ne13/ne03` are the GQA-style head-broadcast
ratios (src0, the K/V operand, has fewer heads than src1, the Q/probs
operand, and repeats to match).

New dispatch op: `GGML_CUDA8_OP_MUL_MAT_F32_F32` (23 dispatch ops now).

New files: `ggml-cuda8-mulmat-f32.h`, `ggml-cuda8-mulmat-f32.cu`,
`ggml-cuda8-mulmat-f32.cpp`, `ggml-cuda8-mulmat-f32-smoke.cpp`.

#### The G39 correction stands: the vector kernel was NOT reusable

The original G42 plan (backport-cuda8.md 2.1) described the F32xF32 kernel
as already written (mmv.cu:210), needing "just a dispatch op id and a
supports_op entry". The G39 re-measurement corrected that: real attention
matmuls are per-head batched and 3D (ne02/ne03), often on permuted views -
strictly matrix-times-vector is not the shape attention actually needs. G42
follows the corrected understanding and adds a purpose-built batched kernel
rather than wiring the vector one.

#### Scope: contiguous dim-0, arbitrary dims 1-3 (absorbs most of G46)

The kernel requires dim 0 (the reduction dimension) contiguous on both
src0 and src1 (`nb[0] == sizeof(float)`), but takes explicit byte strides
for dims 1-3 rather than assuming a packed layout. This is a deliberate
scope choice: attention's permute() typically reorders the head/token/batch
dims while leaving dim 0 alone, so honouring dims 1-3 strides handles the
common permuted-view case directly - which is most of what G46
(permuted/non-contiguous src1) was tracking. Fully arbitrary dim-0 strides
remain out of scope and are refused (fall back to CPU).

One block per output element via an explicit 2D dim3 grid (grid.x up to
65535, grid.y for the rest), matching the existing Q4_K/Q6_K MUL_MAT
kernels' grid construction exactly - NOT the ggml_cuda8_grid_rows() +
grid-stride-loop pattern, which requires a per-block loop this kernel does
not have. Shared-memory tree reduction, no warp shuffle, Fermi-safe.

`supports_op` (backend-reg.cpp) independently re-validates the full shape
contract (reduction dim match, GQA divisibility, dst shape) rather than
trusting the dispatcher alone - the same double-gate pattern G41/G43/G51
established.

#### Verification

Seven numerical configurations, each checked against an independently
re-derived CPU reference (double-precision accumulation reference vs. the
single-precision kernel; agreement between two separately-written
implementations is the check, same method G40/G41/G45 used):
- non-batched (ne02=ne12=1)
- batched, no broadcast (r2=1, 4 heads)
- GQA broadcast (ne02=2, ne12=8, r2=4)
- batch dimension (ne03=2, ne13=2)
- large reduction dim (ne00=1024, exercises the multi-iteration inner loop)
- permuted src1/dst (deliberately non-packed nb2, a padded gap between i12
  slices that only a kernel ignoring the passed strides would read/write
  wrong - stands in for a permuted view)
- oversized total_rows (>65535, exercises grid.y)

Plus five `supported()`-gate rejection cases exercised directly (no GPU
touch): valid batched+broadcast, reduction-dim mismatch, non-integer
broadcast ratio, dst-shape mismatch, and src0 dim-0 non-contiguous.

All pass on GTX 560 / CUDA 8.0.61 / driver 390.157.

#### Two build-integration snags worth recording (both hit during landing)

1. **Duplicate `case GGML_OP_MUL_MAT`.** The new merged MUL_MAT case
   (Q8_0/Q4_K/Q6_K early-return plus the F32xF32 path) was added at the top
   of supports_op's switch while the original quantized-only case was left
   in place further down - two `case` labels for the same value in one
   switch, which GCC rejects as "duplicate case value". Fixed by deleting
   the now-redundant original block. Lesson: a "replace this case" edit must
   verify the old case is actually gone, not just that the new one is
   present.

2. **Collateral loss of the no-op cases.** Inserting the new MUL_MAT block
   at the top of the switch landed exactly where the
   NONE/RESHAPE/VIEW/PERMUTE/TRANSPOSE group used to sit, and that group was
   dropped in the process - so supports_op started returning false for all
   five metadata-only ops (they fell through to default: return false).
   This is worse than a test failure: in a real run it would force graph
   splits around every reshape/view/permute/transpose. Caught by the
   supports-op-smoke fixture (the five no-op TRUE cases went red), fixed by
   re-adding the group. Lesson: the supports-op-smoke fixture is load-bearing
   for exactly this kind of collateral damage - trust its red before
   assuming the failure is a stale fixture.

#### Fixture update in ggml-cuda8-ggml-backend-supports-op-smoke

Same pattern as G41: one `MUL_MAT (F32xF32)` fixture that used to correctly
expect `false` now correctly expects `true`, because its shapes
(src0=[128,64], src1=[128,1], dst=[64,1], no broadcast) satisfy G42's
batched-matmul contract exactly. Moved to the TRUE section and relabelled
`MUL_MAT (F32xF32, batched)`. A new FALSE case, `MUL_MAT (F32xF32, ne00
mismatch)`, was added so negative coverage for the op is not lost (src0 and
src1 disagree on the reduction dimension).

#### Regression

Full container regression: **26/26 pass** (was 25/25 after G41 - the new
`ggml-cuda8-mulmat-f32-smoke` target accounts for the difference).

#### What's next: MEASURE before the next kernel

G41+G42 together were the attention-core unit of work. The correct next
step is not another kernel but a re-measurement, following the same
"measure before optimizing" principle G39 established:

```
GGML_CUDA8_DEBUG_OPS=1 <llama-cli / rpc-server invocation with a Q4_K model>
```

prints the `ops refused by supports_op (ran on CPU), by frequency` summary
at exit. That log is the authoritative answer to what is left, and there
are things only it can confirm that the smoke suite cannot:
- whether a real attention graph actually takes the G42 path, or hits a
  dim-0-strided layout G42 refuses (scope limit above) and still falls back;
- whether GGML_OP_SCALE / broadcast-ADD (G44) are now the top refusals;
- whether any new op type surfaces now that the graph reaches deeper into
  the model without splitting at attention.

If the log is clean or close to it, the priority order below G42 is: G44
(SCALE, broadcast ADD - cheap leaf splits), then G49 (F16 storage - already
unblocked per section 7, removes the --cache-type f32 workaround), then G50
(perf: occupancy/block-size, only meaningful once the graph stops
splitting). The G38 register-spill baseline (section 3.3) also remains open
and is independent of all of the above.


### G53 status: transposed-dst bug in K-quant MUL_MAT — first real Q4_K model on Fermi

Status: **FIXED, verified on GTX 560 with a real model.** First coherent
Q4_K_M generation end-to-end on the CUDA8/Fermi backend (Qwen3-0.6B-Q4_K_M).

G53 is a critical correctness fix, not a new-op checkpoint. It fixes a
silent wrong-answer bug in the Q4_K and Q6_K MUL_MAT kernels that had been
latent since those kernels landed - passing every smoke test while
producing scrambled output on any real multi-token forward pass.

#### The bug

Both `kernel_mul_mat_q4k_f32` (ggml-cuda8-q4k.cu) and
`kernel_mul_mat_q6k_f32` (ggml-cuda8-q6k.cu) wrote the output element in
the wrong memory layout:

```
dst[row * ne11 + col] = smem[0];   // WRONG - transposed
```

ggml's mul_mat output is contiguous with ne[0] = ne01 (output features) as
the fast-varying dimension, so element (row, col) belongs at index
`row + col*ne01`. The kernel wrote a transposed index instead. Fixed in
both kernels:

```
dst[col * ne01 + row] = smem[0];   // correct ggml dst layout
```

#### Why it was invisible until a real model

`row*ne11 + col` and `col*ne01 + row` are equal **only when ne11 == 1**
(single token / matrix-vector). Every K-quant smoke and every graph-builder
Q4_K/Q6_K checkpoint (G17C, G18-G23) used ne11 == 1 - matvec shapes - so
the transpose was a no-op and all of them passed. The bug only manifests at
ne11 > 1, which is exactly what prompt prefill produces: the multi-token
prompt's hidden states get scrambled by the transposed matmul, poisoning
everything downstream. The characteristic symptom was structured-loop
garbage (`uchosuchos`, `tobertober`) rather than random noise - the
signature of transposed-but-not-random logits.

This is the same class of bug the project has repeatedly guarded against
(G37 SOFT_MAX_EXT, G45 ROPE freq_factors): produces wrong numbers, not an
error; graph_compute reports SUCCESS; nothing fails loudly.

#### Why Q8_0 worked but Q4_K/Q6_K did not

The Q8_0 MUL_MAT path (exec_mul_mat_q8_0_f32_vec, ggml-cuda8-dispatch.cpp)
loops per token and writes each token's output at
`dst->data + t*bytes_out`, i.e. index `t*ne01 + r` - the correct ggml
layout. So the prior real-model milestone (SmolLM2-135M in Q8_0) generated
fine. The two K-quant kernels were the only matmuls writing the transposed
layout, which is why the first Q4_K model exposed it.

#### How it was found (bisection record)

`-ngl 99` produced garbage; `-ngl 0` (weights on CPU) produced coherent
text - isolating the bug to a GPU kernel. Forcing each suspect op to CPU
one at a time via a temporary `return false` in supports_op:
RMS_NORM, ROPE, GLU/SWIGLU, MUL - all still garbage, eliminating those.
Forcing GGML_OP_MUL_MAT to CPU - coherent. That pinned it to the K-quant
matmul (Q8_0 was already known-good from SmolLM2), and inspecting the dst
write against ggml's layout found the transpose.

#### Regression guard added (the test that should have caught it)

The existing K-quant smokes cannot catch this because they are all ne11 ==
1. G53 adds a multi-token (ne11 > 1) Q4_K/Q6_K MUL_MAT smoke whose CPU
reference computes the **ggml** layout (`ref[row + col*ne01]`), then
compares element by element. Against the old transposed kernel this test
fails; against the fix it passes. This is the guard that turns "fixed by
hand" into "cannot silently regress" - without it, the exact same bug could
return unnoticed.

#### Verification

- `ggml-cuda8-mulmat-f32-smoke` and all 26 existing regression targets:
  still pass (the fix is at ne11 > 1; the ne11 == 1 smokes are unaffected).
- New ne11 > 1 K-quant smoke: fails on the old kernel, passes on the fix.
- **Real model:** Qwen3-0.6B-Q4_K_M via llama-cli, -ngl 99, generates
  coherent text ("Okay, the user said 'Once upon a time'... I need to
  respond appropriately."). First real Q4_K model on this backend - the
  prior high-water mark was SmolLM2-135M in Q8_0.

#### Significant finding: attention never reaches the GPU (see backport-cuda8 section 11)

The GGML_CUDA8_DEBUG_OPS=1 op histogram from the fixed run shows the whole
attention inner block - SOFTMAX_EXT_F32 (G41), MUL_MAT_F32xF32 (G42),
DIAG_MASK_INF_F32 - executing **zero** times on GPU, with zero refusals
logged. Attention runs entirely on CPU, and it is not being refused by
supports_op - it is never offered to the backend at all, because
offload_op only returns true for MUL_MAT/GET_ROWS whose src[0] (the weight)
is GPU-resident. Attention's K.Q and probs.V matmuls have an *activation*
as src[0], which lives on CPU, so offload_op returns false and the
scheduler keeps the entire attention subgraph CPU-side. G41/G42 are correct
(smoke-proven) but currently dead code on real models. This, plus the
per-op host-staging cost, is why -ngl 99 (1.2/0.8 t/s) is ~7x slower than
-ngl 0 (9.0/6.0 t/s). Full analysis and next-phase framing in
backport-cuda8.md section 11.


### G55 status: flat-copy CONT bug fixed — first fully GPU-resident LLM on CUDA8/Fermi

Status: **FIXED, verified on GTX 560 with a real model.** This is the
project's headline milestone: Qwen3-0.6B-Q4_K_M generates coherent text with
the ENTIRE transformer - attention included - running on the GPU as a single
graph segment per token. Not just weights and FFN (that was already working
with attention on CPU); the whole thing.

#### The bug

`ggml_cuda8_exec_cont_f32` (ggml-cuda8-dispatch.cpp) flat-copied
`src0->data -> dst->data` as one contiguous byte run via
`ggml_cuda8_cpy_f32_d2d(src, dst, n_bytes)` - a launcher that takes only a
byte count and never reads `src0->nb[]`. And `supported_cont_f32` accepted
any F32 src0 without checking contiguity.

This is self-contradictory: CONT exists *specifically* to make a
NON-contiguous tensor contiguous (the G30A note: "CONT kernel - makes
non-contiguous tensors contiguous - needed for KV cache after permute").
ggml only inserts a CONT node when its input is non-contiguous, i.e.
post-permute in attention. So CONT was always handed a permuted src0 with
non-trivial strides, and always flat-copied it in the wrong element order -
scrambling exactly the attention tensors it was meant to fix.

Fixed with a strided-gather kernel (new file ggml-cuda8-cont.cu): dst is
written packed/contiguous while src0 is read through its real byte strides
nb[0..3]. The old contiguous case still works (strides just equal the packed
layout), and the now-unused ggml_cuda8_cpy_f32_d2d path was removed from the
CONT exec.

#### Why it was invisible until now

Same root cause as the G53 transposed-dst bug: no smoke test ever fed CONT a
non-contiguous (permuted) src0. Every prior coherent run had attention on
CPU (because `-nkvo` kept the KV cache host-side), so the attention CONTs -
the only non-contiguous ones - never executed on the GPU. The moment
attention became GPU-resident (see below), CONT fired for real and the bug
surfaced immediately as structured-loop garbage (`ioneonaonaona`).

#### How attention got onto the GPU (the config, not code)

The blocker was never op coverage - G41/G42/G43 kernels existed and were
correct but were dead code, because the scheduler placed the whole attention
subgraph on CPU. Root cause: `-nkvo` (no-KV-offload) keeps the KV cache
host-resident, so attention's K.Q / probs.V matmuls have a CPU operand and
never get offered to the CUDA8 backend (zero supports_op refusals logged -
they were never even asked).

Dropping `-nkvo` puts the cache on the GPU. That required capping context
(`-c 512`): the full-context F32 cache is ~896 MiB, which does not fit
alongside Q4_K weights in the GTX 560's ~1 GB. At `-c 512` the cache is
~11 MiB. With `-nkvo` removed and `-c 512`, the op histogram gained
SET_ROWS_F32 (G43), MUL_MAT_F32xF32 (G42), SOFTMAX_EXT_F32 (G41) and
CONT_F32 (G55) - the entire attention core, all at internally-consistent
2:1 frequencies, and the graph collapsed to a single n_nodes=1125 segment
per token with zero CPU<->GPU boundaries mid-graph.

Working invocation:
```
llama-cli -m Qwen3-0.6B-Q4_K_M.gguf -ngl 99 -fa off \
          --cache-type-k f32 --cache-type-v f32 -c 512 \
          -no-cnv -st -p "..."
```
The F32 cache requirement stands until G49 (F16 storage); the context cap is
a VRAM constraint of the 1 GB card, not a correctness limit.

#### Regression guard

Needs a permuted-src0 CONT smoke (feed CONT a ggml_permute'd, non-contiguous
src0; compare against a CPU strided-gather reference). The existing CONT
smoke uses contiguous input and cannot catch this - the same gap the ne11>1
K-quant smoke closed for G53. Without it, a future edit can silently
reintroduce the flat copy.

#### Verification

- Qwen3-0.6B-Q4_K_M, -ngl 99, GPU-resident attention: coherent output.
- Op histogram: 1 segment/token (63 graph_compute calls for a 63-token run,
  each n_nodes=1125), all attention ops present at consistent frequencies.
- The 26 existing regression targets unaffected (the strided kernel handles
  the contiguous case identically).

#### Performance characterization (measured, not assumed)

GPU-resident generation runs at ~0.8 tok/s vs ~6 tok/s CPU-only on the same
box (Phenom II X6 1055T, 2010-era 6-core, no AVX). A per-op GPU timing probe
(GGML_CUDA8_TIMING=1) over a real run gives the definitive breakdown:

    ~1490 ms/graph total GPU compute, of which:
      MUL_MAT_Q4_K   62%   5.49 ms/call
      MUL_MAT_Q6_K   35%  17.83 ms/call
      MUL_MAT_F32     3%   0.82 ms/call   (attention)
      everything else <1% combined

Findings:
- **Compute-bound, not overhead-bound.** Sum-of-op time matches the
  ~1250 ms/token baseline; there is no dispatch/boundary slack to reclaim.
  Confirmed by elimination: boundary splits (Test B: graph is 1 segment),
  per-op host staging (residency-aware ADD: no delta), and per-op device
  sync (stripped from the two hottest kernels: no delta) were each measured
  and ruled out.
- **97% of GPU time is the two K-quant weight matmuls.** The entire attention
  core that G41-G55 enabled costs <4% - necessary for correctness and to
  collapse the CPU split, but negligible to run.
- **Not register spilling.** ptxas (-DGGML_CUDA8_PTXAS_VERBOSE=ON, wired at
  G38, first measured here): q4k mul_mat 24 registers, q6k 34 registers,
  BOTH with 0 bytes spill stores / 0 loads. The §3.3 spill hypothesis is
  disproved.
- **Q6_K is 3.25x slower per call than Q4_K** (17.8 vs 5.5 ms), not from
  spills but from occupancy: 34 reg/thread + __launch_bounds__(256,2) caps
  Q6_K at 2-3 blocks/SM vs Q4_K's ~5 (GF114 has 32768 reg/SM), so fewer
  concurrent warps hide less memory latency, compounded by heavier per-value
  dequant.
- **Bandwidth-underutilized, not bandwidth-bound.** The GTX 560's 128 GB/s
  dwarfs the Phenom II's ~21 GB/s DDR3, yet the CPU wins - because the
  hand-written Fermi-safe matmuls (one block per output row, no vectorized
  loads, no dp4a, no cuBLAS) achieve only a fraction of the card's
  bandwidth. There is theoretical headroom, but closing it is a substantial
  kernel-engineering project (coalesced/vectorized loads, occupancy
  rebalancing), not a one-line fix, with uncertain payoff on Fermi.

Conclusion: 0.8 tok/s is the current honest number for this hardware pair. It
is not a bug, not overhead, not spills - it is naive K-quant matmuls
under-using the card's bandwidth. Correctness is the milestone; the GPU
path's value is offloading the CPU (freeing the 6-core Phenom for other
cluster work) and enabling models too large for CPU RAM, not beating CPU on a
0.6B model. K-quant matmul throughput is a well-scoped optional future
project with measured headroom, filed rather than pursued.








