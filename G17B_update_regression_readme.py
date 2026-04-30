#!/usr/bin/env python3
# G17B_update_regression_readme.py
#
# G17B: refresh main CUDA8 regression script and README status for G17A/G17A2.
#
# Updates:
#   - /workspace/notebooks/llama.cpp-ph2/run_g11_regression.sh
#   - /workspace/notebooks/llama.cpp-ph2/ggml/src/ggml-cuda8/README.md
#
# Python 3.5-compatible: no f-strings.
# Markdown-safety note:
#   README_BLOCK deliberately avoids Markdown triple-backtick fences. It uses
#   indented code blocks instead, so this source can be embedded safely in
#   Markdown pages or chat without runaway code-block rendering.

import os
import re
import stat
import time

ROOT = "/workspace/notebooks/llama.cpp-ph2"
REG = os.path.join(ROOT, "run_g11_regression.sh")
README = os.path.join(ROOT, "ggml/src/ggml-cuda8/README.md")

G17_TARGET = "ggml-cuda8-ggml-backend-graph-builder-q8_0-mmv-smoke"
G17_EXE = "ggml-cuda8-ggml-backend-graph-builder-q8_0-mmv-smoke"

README_BLOCK = """<!-- G17_STATUS_START -->
## G17 status: real GGML graph-builder Q8_0 MUL_MAT coverage

Status: **PASS on GTX 560 / CUDA 8 / Fermi**.

G17 extends the real GGML graph-builder `graph_compute` path from F32 elementwise and row-wise graphs into the quantized matvec path.

Validated G17 checkpoints:

- **G17A**: real graph-builder `ggml_mul_mat(Q8_0, F32)` smoke.
- **G17A2**: `ggml_backend_i.graph_compute` now routes real `GGML_OP_MUL_MAT` nodes to the existing CUDA8 dispatcher operation `GGML_CUDA8_OP_MUL_MAT_Q8_0_F32_VEC`.

Validated path:

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

- The G17A smoke uses the same standalone-GC graph-builder target pattern as G16:

    ../ggml.c
    ../ggml-quants.c
    ../ggml-threading.cpp
    -ffunction-sections
    -fdata-sections
    -Wl,--gc-sections

- G17A2 adds a minimal `GGML_OP_MUL_MAT` case in `ggml-cuda8-ggml-backend.cpp`.
- The dispatcher still performs the final layout/type support check via `ggml_cuda8_dispatch_supported(...)`.
- G17A2 focused regression passes:
  - real graph-builder Q8_0 MMV smoke,
  - real graph-builder attention-like G16D smoke,
  - dispatch-all CUDA8 kernel smoke.
<!-- G17_STATUS_END -->
"""


def read_file(path):
    with open(path, "r") as f:
        return f.read()


def write_file(path, data):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d)
    with open(path, "w") as f:
        f.write(data)


def backup(path, tag):
    if os.path.exists(path):
        data = read_file(path)
        b = path + "." + tag + "-backup-" + str(int(time.time()))
        write_file(b, data)
        print("backup", b)
        return data
    return ""


def refresh_regression():
    if not os.path.exists(REG):
        raise RuntimeError("main regression script not found: {}".format(REG))

    data = read_file(REG)
    orig = data
    backup(REG, "g17b")

    if "run_target" not in data:
        raise RuntimeError("run_g11_regression.sh does not contain run_target helper")

    line = "run_target {} {}".format(G17_TARGET, G17_EXE)

    if G17_TARGET not in data:
        insert_block = "\n# G17 real GGML graph-builder Q8_0 MUL_MAT graph_compute smoke coverage\n" + line + "\n"

        g16_anchor = "run_target ggml-cuda8-ggml-backend-graph-builder-attnlike-smoke ggml-cuda8-ggml-backend-graph-builder-attnlike-smoke"
        pos = data.find(g16_anchor)
        if pos >= 0:
            endline = data.find("\n", pos)
            if endline >= 0:
                data = data[:endline + 1] + insert_block + data[endline + 1:]
            else:
                data = data + insert_block
        else:
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
        write_file(REG, data)
        os.chmod(REG, os.stat(REG).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        print("patched", REG)
    else:
        print("main regression already contains G17A target")


def refresh_readme():
    if os.path.exists(README):
        data = read_file(README)
    else:
        data = "# ggml-cuda8\n\n"

    orig = data
    backup(README, "g17b")

    start = "<!-- G17_STATUS_START -->"
    end = "<!-- G17_STATUS_END -->"

    if start in data and end in data:
        data = re.sub(re.escape(start) + r".*?" + re.escape(end), README_BLOCK.strip(), data, flags=re.S)
    else:
        if not data.endswith("\n"):
            data += "\n"
        data += "\n" + README_BLOCK.strip() + "\n"

    if data != orig:
        write_file(README, data)
        print("patched", README)
    else:
        print("README G17 status already current")


def main():
    refresh_regression()
    refresh_readme()
    print("G17B regression + README refresh complete.")


if __name__ == "__main__":
    main()
