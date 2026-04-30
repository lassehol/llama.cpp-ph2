#!/usr/bin/env python3
import os
import time

ROOT = "/workspace/notebooks/llama.cpp-ph2"
README = os.path.join(ROOT, "ggml/src/ggml-cuda8/README-cuda8.md")
START = "<!-- G11_STATUS_BEGIN -->"
END = "<!-- G11_STATUS_END -->"
SECTION = '<!-- G11_STATUS_BEGIN -->\n## G11 Device-Resident Execution Status\n\nThe CUDA8 backend experiment now has a validated GGML-shaped device-buffer bridge, a CUDA8 pointer residency registry, and a residency-aware dispatcher capable of running multi-op device-resident smoke graphs on CUDA 8 / Fermi-era hardware.\n\nValidated GGML-facing buffer pieces:\n\n```text\nggml_backend_buffer_type_t wrapper        PASS\nggml_backend_buffer_t wrapper             PASS\nset_tensor/get_tensor                     PASS\nmemset_tensor/clear                       PASS\nmultiple tensor offsets in one buffer     PASS\npointer residency lookup                  PASS\nunregister-on-free behavior               PASS\n```\n\nValidated device-resident dispatcher ops:\n\n```text\nADD_F32                  PASS\nADD_SCALAR_F32           PASS\nMUL_SCALAR_F32           PASS\nREDUCE_SUM_ROWS_F32      PASS\nREDUCE_MAX_ROWS_F32      PASS\nSOFTMAX_ROWS_F32         PASS\n```\n\nValidated device-resident smoke graphs:\n\n```text\nC = A + B\nD = C * scalar\n\nPASS:\n  ggml-cuda8-ggml-buffer-device-graph-smoke\n```\n\n```text\nC = A + B\nD = C * scalar\nS = softmax_rows(D)\n\nPASS:\n  ggml-cuda8-ggml-buffer-device-softmax-graph-smoke\n```\n\nValidated attention-like device-resident micrograph:\n\n```text\nscores  = A + B\nscaled  = scores * scale\nrow_max = reduce_max_rows(scaled)\nprobs   = softmax_rows(scaled)\nrow_sum = reduce_sum_rows(probs)\n\nPASS:\n  ggml-cuda8-ggml-buffer-device-attnlike-smoke\n```\n\nPreferred G11 regression command:\n\n```bash\ncd /workspace/notebooks/llama.cpp-ph2\n./run_g11_regression.sh\n```\n\nExpected final line:\n\n```text\nG11F regression SUCCESS\n```\n\nImportant limitation:\n\n```text\nThis is still not full ggml_backend_t registration.\nThe current bridge directly allocates the exposed ggml_backend_buffer struct\nfrom ggml-backend-impl.h to avoid linking full ggml-base in the CUDA8\ncontainer.\n```\n\nCurrent validated architecture:\n\n```text\nCUDA8 kernels\n  -> ggml_cuda8_backend_buffer\n  -> minimal ggml_backend_buffer_t wrapper\n  -> CUDA8 pointer residency registry\n  -> residency-aware dispatcher\n  -> device-resident multi-op smoke graphs\n```\n\nSuggested next steps:\n\n```text\nG12A:\n  minimal ggml_backend_t registration probe without pulling full ggml-base into\n  the CUDA8-only smoke build\n\nG11Q:\n  bring Q8_0 x F32 vector matmul into the GGML buffer/residency framework\n\nG11E-2:\n  add larger attention-like device-resident graph sizes and timing counters\n```\n<!-- G11_STATUS_END -->\n'


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
    backup = README + ".g11f-backup-" + str(int(time.time()))
    write_file(backup, old)
    print("backup", backup)

old_start = "<!-- G11C_STATUS_BEGIN -->"
old_end = "<!-- G11C_STATUS_END -->"
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
print("G11F README status update complete.")
