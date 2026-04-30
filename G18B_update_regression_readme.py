#!/usr/bin/env python3
# G18B_update_regression_readme.py
#
# G18B: refresh main CUDA8 regression script + README status for G18A.
#
# Updates:
#   - /workspace/notebooks/llama.cpp-ph2/run_g11_regression.sh
#   - /workspace/notebooks/llama.cpp-ph2/ggml/src/ggml-cuda8/README.md
#
# Python 3.5-compatible: no f-strings.
# Markdown-safety note:
#   README_BLOCK avoids Markdown triple-backtick fences and uses indented code
#   blocks instead.

import os
import re
import stat
import time

ROOT = "/workspace/notebooks/llama.cpp-ph2"
REG = os.path.join(ROOT, "run_g11_regression.sh")
README = os.path.join(ROOT, "ggml/src/ggml-cuda8/README.md")

G17_Q8_TARGET = "ggml-cuda8-ggml-backend-graph-builder-q8_0-mmv-smoke"
G17_Q8_EXE = "ggml-cuda8-ggml-backend-graph-builder-q8_0-mmv-smoke"

G18_TARGET = "ggml-cuda8-ggml-backend-graph-builder-q8_0-add-smoke"
G18_EXE = "ggml-cuda8-ggml-backend-graph-builder-q8_0-add-smoke"

G17_COMMENT = "# G17 real GGML graph-builder Q8_0 MUL_MAT graph_compute smoke coverage"
G18_COMMENT = "# G18 real GGML graph-builder Q8_0 MUL_MAT -> ADD graph_compute smoke coverage"

G17_LINE = "run_target {} {}".format(G17_Q8_TARGET, G17_Q8_EXE)
G18_LINE = "run_target {} {}".format(G18_TARGET, G18_EXE)

README_BLOCK = """<!-- G18_STATUS_START -->
## G18 status: real GGML graph-builder quantized pipeline coverage

Status: **PASS on GTX 560 / CUDA 8 / Fermi**.

G18 extends the real GGML graph-builder quantized path from a standalone Q8_0 matvec into a two-op mixed quantized/F32 pipeline.

Validated G18 checkpoints:

- **G18A**: real graph-builder two-op quantized pipeline passes:
  - `h = ggml_mul_mat(Q8_0, x)`
  - `y = ggml_add(h, bias)`
- **G18B**: main regression and README status refreshed for the G18A pipeline.

Validated graph:

    Aq   : GGML_TYPE_Q8_0 matrix, shape [cols, rows]
    x    : GGML_TYPE_F32 vector, shape [cols]
    h    : GGML_TYPE_F32 vector, shape [rows]
    bias : GGML_TYPE_F32 vector, shape [rows]
    y    : GGML_TYPE_F32 vector, shape [rows]

Validated path:

    A_f32
      -> host pack_q8_0(A_f32)
      -> Aq Q8_0 blocks

    h = ggml_mul_mat(Aq, x)
    y = ggml_add(h, bias)

    ggml_build_forward_expand(y)
      -> real GGML graph with two nodes
      -> node 0: GGML_OP_MUL_MAT
      -> node 1: GGML_OP_ADD
      -> ggml_backend_i.graph_compute
      -> GGML_CUDA8_OP_MUL_MAT_Q8_0_F32_VEC
      -> GGML_CUDA8_OP_ADD_F32
      -> CUDA8/Fermi kernels

Implementation notes:

- G18A reuses the packed Q8_0 reference strategy validated in G17C.
- G18A validates that the F32 output of a real graph-builder Q8_0 `MUL_MAT` node can feed directly into a subsequent real graph-builder F32 `ADD` node.
- The G18A smoke uses the same standalone-GC graph-builder target pattern as G16/G17:

    ../ggml.c
    ../ggml-quants.c
    ../ggml-threading.cpp
    -ffunction-sections
    -fdata-sections
    -Wl,--gc-sections

- G18A focused regression passes:
  - Q8_0 MUL_MAT -> ADD graph-builder pipeline smoke,
  - packed Q8_0 graph-builder MMV smoke,
  - real graph-builder attention-like G16D smoke,
  - dispatch-all CUDA8 kernel smoke.
<!-- G18_STATUS_END -->
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


def remove_existing_q8_pipeline_lines(data):
    remove = set([
        G17_COMMENT,
        G18_COMMENT,
        G17_LINE,
        G18_LINE,
    ])

    out = []
    for line in data.splitlines():
        if line.strip() in remove:
            continue
        out.append(line)

    return "\n".join(out) + "\n"


def insert_q8_pipeline_targets(data):
    insert_block = (
        "\n"
        + G17_COMMENT + "\n"
        + G17_LINE + "\n"
        + "\n"
        + G18_COMMENT + "\n"
        + G18_LINE + "\n"
    )

    # Prefer insertion after G16 graph-builder attention-like smoke.
    g16_anchor = "run_target ggml-cuda8-ggml-backend-graph-builder-attnlike-smoke ggml-cuda8-ggml-backend-graph-builder-attnlike-smoke"
    pos = data.find(g16_anchor)
    if pos >= 0:
        endline = data.find("\n", pos)
        if endline >= 0:
            return data[:endline + 1] + insert_block + data[endline + 1:]
        return data + insert_block

    # Otherwise place before dispatch-all.
    dispatch_anchor = "# Legacy/host-backed dispatcher coverage"
    pos = data.find(dispatch_anchor)
    if pos >= 0:
        return data[:pos] + insert_block + "\n" + data[pos:]

    raise RuntimeError("could not find safe insertion anchor for G17/G18 targets")


def update_success_label(data):
    labels = [
        'echo "G15E regression SUCCESS"',
        'echo "CUDA8 G17B regression SUCCESS"',
        'echo "CUDA8 G17C regression SUCCESS"',
        'echo "CUDA8 G17D regression SUCCESS"',
        'echo "CUDA8 G18B regression SUCCESS"',
    ]

    found = False
    for label in labels:
        if label in data:
            data = data.replace(label, 'echo "CUDA8 G18B regression SUCCESS"')
            found = True

    if found:
        return data

    if not data.endswith("\n"):
        data += "\n"
    data += '\necho "CUDA8 G18B regression SUCCESS"\n'
    return data


def refresh_regression():
    if not os.path.exists(REG):
        raise RuntimeError("main regression script not found: {}".format(REG))

    data = read_file(REG)
    orig = data
    backup(REG, "g18b")

    if "run_target()" not in data:
        raise RuntimeError("run_g11_regression.sh does not contain run_target function")

    data = remove_existing_q8_pipeline_lines(data)
    data = insert_q8_pipeline_targets(data)
    data = update_success_label(data)

    if data != orig:
        write_file(REG, data)
        os.chmod(REG, os.stat(REG).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        print("patched", REG)
    else:
        print("main regression already current for G18B")


def refresh_readme():
    if os.path.exists(README):
        data = read_file(README)
    else:
        data = "# ggml-cuda8\n\n"

    orig = data
    backup(README, "g18b")

    start = "<!-- G18_STATUS_START -->"
    end = "<!-- G18_STATUS_END -->"

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
        print("README G18 status already current")


def main():
    refresh_regression()
    refresh_readme()
    print("G18B regression + README refresh complete.")


if __name__ == "__main__":
    main()
