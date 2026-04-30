#!/usr/bin/env python3
# G17D_update_regression_readme.py
#
# G17D: refresh main CUDA8 regression script + README status for G17C.
#
# Updates:
#   - /workspace/notebooks/llama.cpp-ph2/run_g11_regression.sh
#   - /workspace/notebooks/llama.cpp-ph2/ggml/src/ggml-cuda8/README.md
#
# Python 3.5-compatible: no f-strings.
# Markdown-safety note:
#   README_BLOCK avoids Markdown triple-backtick fences and uses indented code
#   blocks instead. This prevents runaway code-block rendering when copied into
#   pages, chat, or generated documentation.

import os
import re
import stat
import time

ROOT = "/workspace/notebooks/llama.cpp-ph2"
REG = os.path.join(ROOT, "run_g11_regression.sh")
README = os.path.join(ROOT, "ggml/src/ggml-cuda8/README.md")

G17_TARGET = "ggml-cuda8-ggml-backend-graph-builder-q8_0-mmv-smoke"
G17_EXE = "ggml-cuda8-ggml-backend-graph-builder-q8_0-mmv-smoke"

G17_RUN_LINE = "run_target {} {}".format(G17_TARGET, G17_EXE)
G17_COMMENT = "# G17 real GGML graph-builder Q8_0 MUL_MAT graph_compute smoke coverage"

README_BLOCK = """<!-- G17_STATUS_START -->
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


def remove_existing_g17_lines(data):
    out = []
    for line in data.splitlines():
        stripped = line.strip()
        if stripped == G17_COMMENT:
            continue
        if stripped == G17_RUN_LINE:
            continue
        out.append(line)
    return "\n".join(out) + "\n"


def insert_g17_target(data):
    insert_block = "\n" + G17_COMMENT + "\n" + G17_RUN_LINE + "\n"

    g16_anchor = "run_target ggml-cuda8-ggml-backend-graph-builder-attnlike-smoke ggml-cuda8-ggml-backend-graph-builder-attnlike-smoke"
    pos = data.find(g16_anchor)
    if pos >= 0:
        endline = data.find("\n", pos)
        if endline >= 0:
            return data[:endline + 1] + insert_block + data[endline + 1:]
        return data + insert_block

    g15_anchor = "run_target ggml-cuda8-ggml-backend-graph-compute-attnlike-smoke ggml-cuda8-ggml-backend-graph-compute-attnlike-smoke"
    pos = data.find(g15_anchor)
    if pos >= 0:
        endline = data.find("\n", pos)
        if endline >= 0:
            return data[:endline + 1] + insert_block + data[endline + 1:]
        return data + insert_block

    dispatch_anchor = "# Legacy/host-backed dispatcher coverage"
    pos = data.find(dispatch_anchor)
    if pos >= 0:
        return data[:pos] + insert_block + "\n" + data[pos:]

    raise RuntimeError("could not find a safe insertion anchor for G17 target")


def update_success_label(data):
    labels = [
        'echo "G15E regression SUCCESS"',
        'echo "CUDA8 G17B regression SUCCESS"',
        'echo "CUDA8 G17C regression SUCCESS"',
        'echo "CUDA8 G17D regression SUCCESS"',
    ]

    found = False
    for label in labels:
        if label in data:
            data = data.replace(label, 'echo "CUDA8 G17D regression SUCCESS"')
            found = True

    if found:
        return data

    if not data.endswith("\n"):
        data += "\n"
    data += '\necho "CUDA8 G17D regression SUCCESS"\n'
    return data


def refresh_regression():
    if not os.path.exists(REG):
        raise RuntimeError("main regression script not found: {}".format(REG))

    data = read_file(REG)
    orig = data
    backup(REG, "g17d")

    if "run_target()" not in data:
        raise RuntimeError("run_g11_regression.sh does not contain run_target function")

    data = remove_existing_g17_lines(data)
    data = insert_g17_target(data)
    data = update_success_label(data)

    if data != orig:
        write_file(REG, data)
        os.chmod(REG, os.stat(REG).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        print("patched", REG)
    else:
        print("main regression already current for G17D")


def refresh_readme():
    if os.path.exists(README):
        data = read_file(README)
    else:
        data = "# ggml-cuda8\n\n"

    orig = data
    backup(README, "g17d")

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
    print("G17D regression + README refresh complete.")


if __name__ == "__main__":
    main()
