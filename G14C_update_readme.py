#!/usr/bin/env python3
import os
import time

ROOT = "/workspace/notebooks/llama.cpp-ph2"
README = os.path.join(ROOT, "ggml/src/ggml-cuda8/README-cuda8.md")
START = "<!-- G14_STATUS_BEGIN -->"
END = "<!-- G14_STATUS_END -->"
SECTION = "<!-- G14_STATUS_BEGIN -->\n## CUDA8 G11/G12/G13/G14 Device-Resident Backend Status\n\nThis CUDA8 backend experiment now has a validated GGML-shaped CUDA8 device-buffer bridge, CUDA8 pointer residency registry, residency-aware dispatcher, minimal `ggml_backend_t`-shaped CUDA8 backend object, backend-owned buffer graph smokes, compute-shaped backend callback probes, and compile-checked real `ggml_backend_i` graph callback stubs. The current implementation is still smoke-test oriented; real graph node execution through `backend->iface.graph_compute` is not enabled yet.\n\nValidated GGML-facing buffer pieces:\n\n```text\nggml_backend_buffer_type_t wrapper        PASS\nggml_backend_buffer_t wrapper             PASS\nset_tensor/get_tensor                     PASS\nmemset_tensor/clear                       PASS\nmultiple tensor offsets in one buffer     PASS\npointer residency lookup                  PASS\nunregister-on-free behavior               PASS\n```\n\nValidated device-resident dispatcher ops:\n\n```text\nADD_F32                  PASS\nADD_SCALAR_F32           PASS\nMUL_SCALAR_F32           PASS\nREDUCE_SUM_ROWS_F32      PASS\nREDUCE_MAX_ROWS_F32      PASS\nSOFTMAX_ROWS_F32         PASS\n```\n\nValidated direct CUDA8-buffer device-resident graphs:\n\n```text\nC = A + B\nD = C * scalar\n\nPASS:\n  ggml-cuda8-ggml-buffer-device-graph-smoke\n```\n\n```text\nC = A + B\nD = C * scalar\nS = softmax_rows(D)\n\nPASS:\n  ggml-cuda8-ggml-buffer-device-softmax-graph-smoke\n```\n\n```text\nscores  = A + B\nscaled  = scores * scale\nrow_max = reduce_max_rows(scaled)\nprobs   = softmax_rows(scaled)\nrow_sum = reduce_sum_rows(probs)\n\nPASS:\n  ggml-cuda8-ggml-buffer-device-attnlike-smoke\n```\n\nValidated minimal backend object and backend-owned buffer milestones:\n\n```text\nG12A:\n  ggml_backend_t-shaped CUDA8 object\n  backend name = CUDA8\n  local default-buffer-type helper returns CUDA8 buffer type\n  backend default buffer set/get roundtrip\n\nPASS:\n  ggml-cuda8-ggml-backend-probe\n```\n\n```text\nG12B:\n  backend default buffer type allocates CUDA8 buffer\n  backend-owned buffer feeds ADD -> MUL_SCALAR -> SOFTMAX graph\n\nPASS:\n  ggml-cuda8-ggml-backend-buffer-graph-smoke\n```\n\n```text\nG12C:\n  backend default buffer type allocates CUDA8 buffer\n  backend-owned buffer feeds attention-like graph:\n    scores  = A + B\n    scaled  = scores * scale\n    row_max = reduce_max_rows(scaled)\n    probs   = softmax_rows(scaled)\n    row_sum = reduce_sum_rows(probs)\n\nPASS:\n  ggml-cuda8-ggml-backend-attnlike-smoke\n```\n\nValidated compute-shaped backend callback milestones:\n\n```text\nG13A:\n  backend compute-shaped callback helper validates CUDA8 backend object\n  callback forwards supported ops to the residency-aware dispatcher\n  backend-owned buffer feeds ADD -> MUL_SCALAR -> SOFTMAX graph through callback helper\n\nPASS:\n  ggml-cuda8-ggml-backend-compute-probe\n```\n\n```text\nG13B:\n  backend compute-shaped callback helper forwards the full attention-like chain:\n    scores  = A + B\n    scaled  = scores * scale\n    row_max = reduce_max_rows(scaled)\n    probs   = softmax_rows(scaled)\n    row_sum = reduce_sum_rows(probs)\n\nPASS:\n  ggml-cuda8-ggml-backend-compute-attnlike-smoke\n```\n\nValidated real `ggml_backend_i` graph API milestones:\n\n```text\nG14A:\n  inspected this checkout's actual backend API shape\n  ggml_backend_i includes:\n    graph_plan_create\n    graph_plan_free\n    graph_plan_update\n    graph_plan_compute\n    graph_compute\n    graph_optimize\n  ggml_backend_i does not include get_default_buffer_type\n\nREPORT:\n  ggml/src/ggml-cuda8/G14A_backend_api_report.md\n  ggml/src/ggml-cuda8/G14A_backend_api_snapshot.txt\n```\n\n```text\nG14B:\n  populated real ggml_backend_i graph callback slots with safe stubs\n  compile-checked exact signatures from this checkout\n  directly called graph_plan_* / graph_compute / graph_optimize / synchronize stubs\n\nPASS:\n  ggml-cuda8-ggml-backend-graph-api-probe\n```\n\nPreferred full G11/G12/G13/G14 regression command:\n\n```bash\ncd /workspace/notebooks/llama.cpp-ph2\n./run_g11_regression.sh\n```\n\nExpected final line:\n\n```text\nG14C regression SUCCESS\n```\n\nImportant limitation:\n\n```text\nThis is still not real ggml_backend_t graph-compute execution.\nThe graph_compute and graph_plan_compute callbacks are currently safe no-op/success stubs.\nReal node dispatch still lives in the G13 helper path:\n  ggml_cuda8_ggml_backend_dispatch_op(...)\nThe next step should wire a tiny synthetic graph to graph_compute in a controlled G15A probe.\n```\n\nCurrent validated architecture:\n\n```text\nCUDA8 kernels\n  -> ggml_cuda8_backend_buffer\n  -> minimal ggml_backend_buffer_t wrapper\n  -> CUDA8 pointer residency registry\n  -> residency-aware dispatcher\n  -> direct device-resident smoke graphs\n  -> minimal ggml_backend_t-shaped CUDA8 object\n  -> backend-owned CUDA8 buffer graphs\n  -> compute-shaped backend callback graphs\n  -> real ggml_backend_i graph callback stubs\n```\n\nSuggested next steps:\n\n```text\nG15A:\n  real graph_compute dispatch of one tiny synthetic ADD_F32 graph\n  use existing residency-aware dispatcher under graph_compute\n  keep backend-owned CUDA8 buffer and explicit verification\n\nG14D:\n  update README/regression after graph API naming/API cleanup\n\nG11Q/G12Q/G13Q/G14Q:\n  bring Q8_0 x F32 vector matmul into the GGML buffer/residency/backend-owned/callback/graph API framework\n```\n<!-- G14_STATUS_END -->\n"


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
    backup = README + ".g14c-backup-" + str(int(time.time()))
    write_file(backup, old)
    print("backup", backup)

# Remove older canonical status blocks if present.
for old_start, old_end in [
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
print("G14C README status update complete.")
