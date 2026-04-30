#!/usr/bin/env python3
import os
import time

ROOT = "/workspace/notebooks/llama.cpp-ph2"
README = os.path.join(ROOT, "ggml/src/ggml-cuda8/README-cuda8.md")
START = "<!-- G15_STATUS_BEGIN -->"
END = "<!-- G15_STATUS_END -->"
SECTION = "<!-- G15_STATUS_BEGIN -->\n## CUDA8 G11/G12/G13/G14/G15 Backend Status\n\nThis CUDA8 backend experiment now has a validated GGML-shaped CUDA8 device-buffer bridge, CUDA8 pointer residency registry, residency-aware dispatcher, minimal `ggml_backend_t`-shaped CUDA8 backend object, backend-owned buffer graph smokes, compute-shaped backend callback probes, compile-checked real `ggml_backend_i` graph callback slots, and real `backend->iface.graph_compute(...)` execution for selected F32 graph nodes.\n\nValidated GGML-facing buffer pieces:\n\n```text\nggml_backend_buffer_type_t wrapper        PASS\nggml_backend_buffer_t wrapper             PASS\nset_tensor/get_tensor                     PASS\nmemset_tensor/clear                       PASS\nmultiple tensor offsets in one buffer     PASS\npointer residency lookup                  PASS\nunregister-on-free behavior               PASS\n```\n\nValidated CUDA8 dispatcher ops:\n\n```text\nCPY_F32                       PASS\nADD_F32                       PASS\nADD_SCALAR_F32                PASS\nMUL_SCALAR_F32                PASS\nREDUCE_SUM_ROWS_F32           PASS\nREDUCE_MAX_ROWS_F32           PASS\nSOFTMAX_ROWS_F32              PASS\nMUL_MAT_Q8_0xF32_VEC          PASS\n```\n\nValidated direct CUDA8-buffer and backend-owned graph milestones:\n\n```text\nG11:\n  direct device-resident CUDA8 buffer graphs:\n    C = A + B\n    D = C * scalar\n    S = softmax_rows(D)\n    attention-like buffer graph smoke\n\nPASS:\n  ggml-cuda8-ggml-buffer-device-graph-smoke\n  ggml-cuda8-ggml-buffer-device-softmax-graph-smoke\n  ggml-cuda8-ggml-buffer-device-attnlike-smoke\n```\n\n```text\nG12:\n  minimal ggml_backend_t-shaped CUDA8 object\n  backend default CUDA8 buffer helper\n  backend-owned CUDA8 buffer graphs\n\nPASS:\n  ggml-cuda8-ggml-backend-probe\n  ggml-cuda8-ggml-backend-buffer-graph-smoke\n  ggml-cuda8-ggml-backend-attnlike-smoke\n```\n\n```text\nG13:\n  compute-shaped backend callback helper\n  callback forwards supported ops to the residency-aware dispatcher\n  backend-owned attention-like graph through callback helper\n\nPASS:\n  ggml-cuda8-ggml-backend-compute-probe\n  ggml-cuda8-ggml-backend-compute-attnlike-smoke\n```\n\n```text\nG14:\n  inspected this checkout's actual ggml_backend_i API shape\n  populated real graph callback slots with safe stubs:\n    graph_plan_create\n    graph_plan_free\n    graph_plan_update\n    graph_plan_compute\n    graph_compute\n    graph_optimize\n    synchronize\n\nPASS:\n  ggml-cuda8-ggml-backend-graph-api-probe\n```\n\nValidated real `ggml_backend_i.graph_compute` milestones:\n\n```text\nG15A:\n  real graph_compute dispatch of one synthetic ADD_F32 graph node:\n    C = A + B\n\nPASS:\n  ggml-cuda8-ggml-backend-graph-compute-add-smoke\n```\n\n```text\nG15B:\n  real graph_compute dispatch of a two-node graph:\n    C = A + B\n    D = C * scalar\n\n  mapping:\n    GGML_OP_ADD -> GGML_CUDA8_OP_ADD_F32\n    GGML_OP_MUL with scalar src1 -> GGML_CUDA8_OP_MUL_SCALAR_F32\n\nPASS:\n  ggml-cuda8-ggml-backend-graph-compute-add-mul-smoke\n```\n\n```text\nG15C:\n  real graph_compute dispatch of a three-node softmax graph:\n    C = A + B\n    D = C * scalar\n    S = softmax_rows(D)\n\n  mapping:\n    GGML_OP_ADD      -> GGML_CUDA8_OP_ADD_F32\n    GGML_OP_MUL      -> GGML_CUDA8_OP_MUL_SCALAR_F32\n    GGML_OP_SOFT_MAX -> GGML_CUDA8_OP_SOFTMAX_ROWS_F32\n\nPASS:\n  ggml-cuda8-ggml-backend-graph-compute-softmax-smoke\n```\n\n```text\nG15D:\n  real graph_compute dispatch of an attention-like graph using real GGML ops available in this checkout:\n    scores  = A + B\n    scaled  = scores * scale\n    probs   = softmax_rows(scaled)\n    row_sum = sum_rows(probs)\n\n  mapping:\n    GGML_OP_ADD      -> GGML_CUDA8_OP_ADD_F32\n    GGML_OP_MUL      -> GGML_CUDA8_OP_MUL_SCALAR_F32\n    GGML_OP_SOFT_MAX -> GGML_CUDA8_OP_SOFTMAX_ROWS_F32\n    GGML_OP_SUM_ROWS -> GGML_CUDA8_OP_REDUCE_SUM_ROWS_F32\n\n  note:\n    row_max remains a host-side diagnostic only because this checkout does not expose a GGML max-row-values op.\n\nPASS:\n  ggml-cuda8-ggml-backend-graph-compute-attnlike-smoke\n```\n\nPreferred full G11/G12/G13/G14/G15 regression command:\n\n```bash\ncd /workspace/notebooks/llama.cpp-ph2\n./run_g11_regression.sh\n```\n\nExpected final line:\n\n```text\nG15E regression SUCCESS\n```\n\nCurrent validated architecture:\n\n```text\nCUDA8 kernels\n  -> ggml_cuda8_backend_buffer\n  -> minimal ggml_backend_buffer_t wrapper\n  -> CUDA8 pointer residency registry\n  -> residency-aware dispatcher\n  -> direct device-resident smoke graphs\n  -> minimal ggml_backend_t-shaped CUDA8 object\n  -> backend-owned CUDA8 buffer graphs\n  -> compute-shaped backend callback graphs\n  -> real ggml_backend_i graph callback stubs\n  -> real graph_compute ADD_F32 graph\n  -> real graph_compute ADD -> MUL_SCALAR graph\n  -> real graph_compute ADD -> MUL_SCALAR -> SOFTMAX graph\n  -> real graph_compute ADD -> MUL_SCALAR -> SOFTMAX -> SUM_ROWS attention-like graph\n```\n\nSuggested next steps:\n\n```text\nG15F:\n  clean up patcher-generated backend graph_compute code into a stable hand-maintained implementation\n\nG16A:\n  move from manual synthetic cgraph construction toward graph_compute calls from real GGML graph builder / allocator flow\n\nG15Q/G16Q:\n  bring Q8_0 x F32 vector matmul into backend-owned graph_compute coverage\n```\n<!-- G15_STATUS_END -->\n"

def read_file(path):
    with open(path, "r") as f:
        return f.read()

def write_file(path, data):
    with open(path, "w") as f:
        f.write(data)

if not os.path.exists(README):
    old = "# ggml-cuda8: Legacy CUDA 8 / Fermi Backend Experiment\n\n"
else:
    old = read_file(README)
    backup = README + ".g15e-backup-" + str(int(time.time()))
    write_file(backup, old)
    print("backup", backup)

# Remove older generated status blocks.
for old_start, old_end in [
    ("<!-- G14_STATUS_BEGIN -->", "<!-- G14_STATUS_END -->"),
    ("<!-- G13_STATUS_BEGIN -->", "<!-- G13_STATUS_END -->"),
    ("<!-- G12_STATUS_BEGIN -->", "<!-- G12_STATUS_END -->"),
    ("<!-- G11_STATUS_BEGIN -->", "<!-- G11_STATUS_END -->"),
    ("<!-- G11C_STATUS_BEGIN -->", "<!-- G11C_STATUS_END -->"),
]:
    if old_start in old and old_end in old:
        before = old.split(old_start, 1)[0]
        after = old.split(old_end, 1)[1]
        old = before.rstrip() + "\n\n" + after.lstrip()

if START in old and END in old:
    before = old.split(START, 1)[0]
    after = old.split(END, 1)[1]
    new = before.rstrip() + "\n\n" + SECTION.strip() + "\n" + after
else:
    if not old.endswith("\n"):
        old += "\n"
    new = old.rstrip() + "\n\n---\n\n" + SECTION.strip() + "\n"

write_file(README, new)
print("updated", README)
print("G15E README status update complete.")
