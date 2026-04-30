#!/usr/bin/env python3
import os
import time

ROOT = "/workspace/notebooks/llama.cpp-ph2"
README = os.path.join(ROOT, "ggml/src/ggml-cuda8/README-cuda8.md")
START = "<!-- G13_STATUS_BEGIN -->"
END = "<!-- G13_STATUS_END -->"
SECTION = '<!-- G13_STATUS_BEGIN -->\n## CUDA8 G11/G12/G13 Device-Resident Backend Status\n\nThis CUDA8 backend experiment now has a validated GGML-shaped CUDA8 device-buffer bridge, CUDA8 pointer residency registry, residency-aware dispatcher, minimal `ggml_backend_t`-shaped CUDA8 backend object, backend-owned buffer graph smokes, and graph-compute-shaped backend callback probes. The current implementation is still smoke-test oriented and is not yet full `ggml_backend_i` graph-compute integration.\n\nValidated GGML-facing buffer pieces:\n\n```text\nggml_backend_buffer_type_t wrapper        PASS\nggml_backend_buffer_t wrapper             PASS\nset_tensor/get_tensor                     PASS\nmemset_tensor/clear                       PASS\nmultiple tensor offsets in one buffer     PASS\npointer residency lookup                  PASS\nunregister-on-free behavior               PASS\n```\n\nValidated device-resident dispatcher ops:\n\n```text\nADD_F32                  PASS\nADD_SCALAR_F32           PASS\nMUL_SCALAR_F32           PASS\nREDUCE_SUM_ROWS_F32      PASS\nREDUCE_MAX_ROWS_F32      PASS\nSOFTMAX_ROWS_F32         PASS\n```\n\nValidated direct CUDA8-buffer device-resident graphs:\n\n```text\nC = A + B\nD = C * scalar\n\nPASS:\n  ggml-cuda8-ggml-buffer-device-graph-smoke\n```\n\n```text\nC = A + B\nD = C * scalar\nS = softmax_rows(D)\n\nPASS:\n  ggml-cuda8-ggml-buffer-device-softmax-graph-smoke\n```\n\n```text\nscores  = A + B\nscaled  = scores * scale\nrow_max = reduce_max_rows(scaled)\nprobs   = softmax_rows(scaled)\nrow_sum = reduce_sum_rows(probs)\n\nPASS:\n  ggml-cuda8-ggml-buffer-device-attnlike-smoke\n```\n\nValidated minimal backend object and backend-owned buffer milestones:\n\n```text\nG12A:\n  ggml_backend_t-shaped CUDA8 object\n  backend name = CUDA8\n  local default-buffer-type helper returns CUDA8 buffer type\n  backend default buffer set/get roundtrip\n\nPASS:\n  ggml-cuda8-ggml-backend-probe\n```\n\n```text\nG12B:\n  backend default buffer type allocates CUDA8 buffer\n  backend-owned buffer feeds ADD -> MUL_SCALAR -> SOFTMAX graph\n\nPASS:\n  ggml-cuda8-ggml-backend-buffer-graph-smoke\n```\n\n```text\nG12C:\n  backend default buffer type allocates CUDA8 buffer\n  backend-owned buffer feeds attention-like graph:\n    scores  = A + B\n    scaled  = scores * scale\n    row_max = reduce_max_rows(scaled)\n    probs   = softmax_rows(scaled)\n    row_sum = reduce_sum_rows(probs)\n\nPASS:\n  ggml-cuda8-ggml-backend-attnlike-smoke\n```\n\nValidated compute-shaped backend callback milestones:\n\n```text\nG13A:\n  backend compute-shaped callback helper validates CUDA8 backend object\n  callback forwards supported ops to the residency-aware dispatcher\n  backend-owned buffer feeds ADD -> MUL_SCALAR -> SOFTMAX graph through callback helper\n\nPASS:\n  ggml-cuda8-ggml-backend-compute-probe\n```\n\n```text\nG13B:\n  backend compute-shaped callback helper forwards the full attention-like chain:\n    scores  = A + B\n    scaled  = scores * scale\n    row_max = reduce_max_rows(scaled)\n    probs   = softmax_rows(scaled)\n    row_sum = reduce_sum_rows(probs)\n\nPASS:\n  ggml-cuda8-ggml-backend-compute-attnlike-smoke\n```\n\nPreferred full G11/G12/G13 regression command:\n\n```bash\ncd /workspace/notebooks/llama.cpp-ph2\n./run_g11_regression.sh\n```\n\nExpected final line:\n\n```text\nG13C regression SUCCESS\n```\n\nImportant limitation:\n\n```text\nThis is still not full ggml_backend_t graph-compute integration.\nThe current bridge directly allocates the exposed ggml_backend / ggml_backend_buffer\nstructs from ggml-backend-impl.h and uses a local compute-shaped dispatch helper.\nThe next integration step should inspect the exact ggml_backend_i graph-compute API\nshape in this checkout before wiring into real backend graph execution.\n```\n\nCurrent validated architecture:\n\n```text\nCUDA8 kernels\n  -> ggml_cuda8_backend_buffer\n  -> minimal ggml_backend_buffer_t wrapper\n  -> CUDA8 pointer residency registry\n  -> residency-aware dispatcher\n  -> direct device-resident smoke graphs\n  -> minimal ggml_backend_t-shaped CUDA8 object\n  -> backend-owned CUDA8 buffer graphs\n  -> compute-shaped backend callback graphs\n```\n\nSuggested next steps:\n\n```text\nG14A:\n  inspect actual ggml_backend_i graph-compute API shape in this checkout and\n  create the smallest real backend graph-compute hook only if the interface supports it cleanly\n\nG13D:\n  update README/regression after any cleanup of helper naming/API surface\n\nG11Q/G12Q/G13Q:\n  bring Q8_0 x F32 vector matmul into the GGML buffer/residency/backend-owned/callback framework\n```\n<!-- G13_STATUS_END -->\n'


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
    backup = README + ".g13c-backup-" + str(int(time.time()))
    write_file(backup, old)
    print("backup", backup)

# Remove older canonical status blocks if present.
for old_start, old_end in [
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
print("G13C README status update complete.")
