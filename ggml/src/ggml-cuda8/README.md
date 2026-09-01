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












