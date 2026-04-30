#!/usr/bin/env python3
# G16E_update_regression_readme.py
#
# G16E: refresh main CUDA8 regression script + README status for G16A/B/C/D.
#
# Updates:
#   - /workspace/notebooks/llama.cpp-ph2/run_g11_regression.sh
#   - /workspace/notebooks/llama.cpp-ph2/ggml/src/ggml-cuda8/README.md
#
# Idempotent; creates timestamped backups.

import re
import stat
import time
from pathlib import Path

ROOT = Path("/workspace/notebooks/llama.cpp-ph2")
REG = ROOT / "run_g11_regression.sh"
README = ROOT / "ggml/src/ggml-cuda8/README.md"

G16_TARGETS = [
    ("ggml-cuda8-ggml-backend-graph-builder-add-smoke",      "ggml-cuda8-ggml-backend-graph-builder-add-smoke"),
    ("ggml-cuda8-ggml-backend-graph-builder-add-mul-smoke",  "ggml-cuda8-ggml-backend-graph-builder-add-mul-smoke"),
    ("ggml-cuda8-ggml-backend-graph-builder-softmax-smoke",  "ggml-cuda8-ggml-backend-graph-builder-softmax-smoke"),
    ("ggml-cuda8-ggml-backend-graph-builder-attnlike-smoke", "ggml-cuda8-ggml-backend-graph-builder-attnlike-smoke"),
]

README_BLOCK = """<!-- G16_STATUS_START -->
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
"""

def read(path: Path) -> str:
    return path.read_text()

def write(path: Path, data: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(data)

def backup(path: Path, tag: str) -> None:
    if path.exists():
        b = path.with_name(path.name + ".{}-backup-{}".format(tag, int(time.time())))
        b.write_text(path.read_text())
        print("backup", b)

def refresh_regression() -> None:
    if not REG.exists():
        raise RuntimeError("main regression script not found: {}".format(REG))

    data = read(REG)
    orig = data
    backup(REG, "g16e")

    if "run_target" not in data:
        raise RuntimeError("run_g11_regression.sh does not contain run_target helper")

    missing_lines = []
    for target, exe in G16_TARGETS:
        if target not in data:
            missing_lines.append("run_target {} {}".format(target, exe))

    if missing_lines:
        insert_block = "\n# G16 real GGML graph-builder graph_compute smoke coverage\n" + "\n".join(missing_lines) + "\n"

        match = None
        for pat in [
            r"\necho\s+\".*SUCCESS.*\"\s*\n?$",
            r"\necho\s+'.*SUCCESS.*'\s*\n?$",
        ]:
            match = re.search(pat, data, flags=re.S)
            if match:
                break

        if match:
            data = data[:match.start()] + insert_block + data[match.start():]
        else:
            if not data.endswith("\n"):
                data += "\n"
            data += insert_block

    if data != orig:
        write(REG, data)
        REG.chmod(REG.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        print("patched", REG)
    else:
        print("main regression already contains G16A/B/C/D targets")

def refresh_readme() -> None:
    data = read(README) if README.exists() else "# ggml-cuda8\n\n"
    orig = data
    backup(README, "g16e")

    start = "<!-- G16_STATUS_START -->"
    end = "<!-- G16_STATUS_END -->"

    if start in data and end in data:
        data = re.sub(re.escape(start) + r".*?" + re.escape(end), README_BLOCK.strip(), data, flags=re.S)
    else:
        if not data.endswith("\n"):
            data += "\n"
        data += "\n" + README_BLOCK.strip() + "\n"

    if data != orig:
        write(README, data)
        print("patched", README)
    else:
        print("README G16 status already current")

def main() -> None:
    refresh_regression()
    refresh_readme()
    print("G16E regression + README refresh complete.")

if __name__ == "__main__":
    main()
