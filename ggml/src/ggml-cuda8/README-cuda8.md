# ggml-cuda8: Legacy CUDA 8 / Fermi Backend Experiment

This directory contains an experimental CUDA 8 backend path for legacy NVIDIA GPUs, primarily targeting Fermi-class hardware such as the GeForce GTX 560 / compute capability 2.1.

The current implementation is intentionally incremental and conservative. It is **not yet a full `ggml_backend_t` implementation**. Instead, it provides a validated CUDA8 execution layer with:

- CUDA device/backend identity probing
- device buffer abstraction
- backend context object
- manual `ggml_tensor` metadata based dispatcher
- selected CUDA8/Fermi-safe kernels
- standalone smoke and benchmark targets

The implementation is designed to avoid modern CUDA features that are unavailable on CUDA 8 / Fermi.

---

## Current Status

Validated on:

```text
GPU:     NVIDIA GeForce GTX 560
CC:      2.1
CUDA:    CUDA 8.0
CMake:   3.22.x for parent build

---

---

---

---

---

<!-- G15_STATUS_BEGIN -->
## CUDA8 G11/G12/G13/G14/G15 Backend Status

This CUDA8 backend experiment now has a validated GGML-shaped CUDA8 device-buffer bridge, CUDA8 pointer residency registry, residency-aware dispatcher, minimal `ggml_backend_t`-shaped CUDA8 backend object, backend-owned buffer graph smokes, compute-shaped backend callback probes, compile-checked real `ggml_backend_i` graph callback slots, and real `backend->iface.graph_compute(...)` execution for selected F32 graph nodes.

Validated GGML-facing buffer pieces:

```text
ggml_backend_buffer_type_t wrapper        PASS
ggml_backend_buffer_t wrapper             PASS
set_tensor/get_tensor                     PASS
memset_tensor/clear                       PASS
multiple tensor offsets in one buffer     PASS
pointer residency lookup                  PASS
unregister-on-free behavior               PASS
```

Validated CUDA8 dispatcher ops:

```text
CPY_F32                       PASS
ADD_F32                       PASS
ADD_SCALAR_F32                PASS
MUL_SCALAR_F32                PASS
REDUCE_SUM_ROWS_F32           PASS
REDUCE_MAX_ROWS_F32           PASS
SOFTMAX_ROWS_F32              PASS
MUL_MAT_Q8_0xF32_VEC          PASS
```

Validated direct CUDA8-buffer and backend-owned graph milestones:

```text
G11:
  direct device-resident CUDA8 buffer graphs:
    C = A + B
    D = C * scalar
    S = softmax_rows(D)
    attention-like buffer graph smoke

PASS:
  ggml-cuda8-ggml-buffer-device-graph-smoke
  ggml-cuda8-ggml-buffer-device-softmax-graph-smoke
  ggml-cuda8-ggml-buffer-device-attnlike-smoke
```

```text
G12:
  minimal ggml_backend_t-shaped CUDA8 object
  backend default CUDA8 buffer helper
  backend-owned CUDA8 buffer graphs

PASS:
  ggml-cuda8-ggml-backend-probe
  ggml-cuda8-ggml-backend-buffer-graph-smoke
  ggml-cuda8-ggml-backend-attnlike-smoke
```

```text
G13:
  compute-shaped backend callback helper
  callback forwards supported ops to the residency-aware dispatcher
  backend-owned attention-like graph through callback helper

PASS:
  ggml-cuda8-ggml-backend-compute-probe
  ggml-cuda8-ggml-backend-compute-attnlike-smoke
```

```text
G14:
  inspected this checkout's actual ggml_backend_i API shape
  populated real graph callback slots with safe stubs:
    graph_plan_create
    graph_plan_free
    graph_plan_update
    graph_plan_compute
    graph_compute
    graph_optimize
    synchronize

PASS:
  ggml-cuda8-ggml-backend-graph-api-probe
```

Validated real `ggml_backend_i.graph_compute` milestones:

```text
G15A:
  real graph_compute dispatch of one synthetic ADD_F32 graph node:
    C = A + B

PASS:
  ggml-cuda8-ggml-backend-graph-compute-add-smoke
```

```text
G15B:
  real graph_compute dispatch of a two-node graph:
    C = A + B
    D = C * scalar

  mapping:
    GGML_OP_ADD -> GGML_CUDA8_OP_ADD_F32
    GGML_OP_MUL with scalar src1 -> GGML_CUDA8_OP_MUL_SCALAR_F32

PASS:
  ggml-cuda8-ggml-backend-graph-compute-add-mul-smoke
```

```text
G15C:
  real graph_compute dispatch of a three-node softmax graph:
    C = A + B
    D = C * scalar
    S = softmax_rows(D)

  mapping:
    GGML_OP_ADD      -> GGML_CUDA8_OP_ADD_F32
    GGML_OP_MUL      -> GGML_CUDA8_OP_MUL_SCALAR_F32
    GGML_OP_SOFT_MAX -> GGML_CUDA8_OP_SOFTMAX_ROWS_F32

PASS:
  ggml-cuda8-ggml-backend-graph-compute-softmax-smoke
```

```text
G15D:
  real graph_compute dispatch of an attention-like graph using real GGML ops available in this checkout:
    scores  = A + B
    scaled  = scores * scale
    probs   = softmax_rows(scaled)
    row_sum = sum_rows(probs)

  mapping:
    GGML_OP_ADD      -> GGML_CUDA8_OP_ADD_F32
    GGML_OP_MUL      -> GGML_CUDA8_OP_MUL_SCALAR_F32
    GGML_OP_SOFT_MAX -> GGML_CUDA8_OP_SOFTMAX_ROWS_F32
    GGML_OP_SUM_ROWS -> GGML_CUDA8_OP_REDUCE_SUM_ROWS_F32

  note:
    row_max remains a host-side diagnostic only because this checkout does not expose a GGML max-row-values op.

PASS:
  ggml-cuda8-ggml-backend-graph-compute-attnlike-smoke
```

Preferred full G11/G12/G13/G14/G15 regression command:

```bash
cd /workspace/notebooks/llama.cpp-ph2
./run_g11_regression.sh
```

Expected final line:

```text
G15E regression SUCCESS
```

Current validated architecture:

```text
CUDA8 kernels
  -> ggml_cuda8_backend_buffer
  -> minimal ggml_backend_buffer_t wrapper
  -> CUDA8 pointer residency registry
  -> residency-aware dispatcher
  -> direct device-resident smoke graphs
  -> minimal ggml_backend_t-shaped CUDA8 object
  -> backend-owned CUDA8 buffer graphs
  -> compute-shaped backend callback graphs
  -> real ggml_backend_i graph callback stubs
  -> real graph_compute ADD_F32 graph
  -> real graph_compute ADD -> MUL_SCALAR graph
  -> real graph_compute ADD -> MUL_SCALAR -> SOFTMAX graph
  -> real graph_compute ADD -> MUL_SCALAR -> SOFTMAX -> SUM_ROWS attention-like graph
```

Suggested next steps:

```text
G15F:
  clean up patcher-generated backend graph_compute code into a stable hand-maintained implementation

G16A:
  move from manual synthetic cgraph construction toward graph_compute calls from real GGML graph builder / allocator flow

G15Q/G16Q:
  bring Q8_0 x F32 vector matmul into backend-owned graph_compute coverage
```
<!-- G15_STATUS_END -->
