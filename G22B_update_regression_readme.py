#!/usr/bin/env python3
# G22B_update_regression_readme.py
#
# G22B: refresh main CUDA8 regression script + README status for G22A.
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

G18_ADD_TARGET = "ggml-cuda8-ggml-backend-graph-builder-q8_0-add-smoke"
G18_ADD_EXE = "ggml-cuda8-ggml-backend-graph-builder-q8_0-add-smoke"

G18_MUL_ADD_TARGET = "ggml-cuda8-ggml-backend-graph-builder-q8_0-mul-add-smoke"
G18_MUL_ADD_EXE = "ggml-cuda8-ggml-backend-graph-builder-q8_0-mul-add-smoke"

G19_SOFTMAX_TARGET = "ggml-cuda8-ggml-backend-graph-builder-q8_0-mul-add-softmax-smoke"
G19_SOFTMAX_EXE = "ggml-cuda8-ggml-backend-graph-builder-q8_0-mul-add-softmax-smoke"

G20_SUMROWS_TARGET = "ggml-cuda8-ggml-backend-graph-builder-q8_0-mul-add-softmax-sumrows-smoke"
G20_SUMROWS_EXE = "ggml-cuda8-ggml-backend-graph-builder-q8_0-mul-add-softmax-sumrows-smoke"

G21_RESIDUAL_TARGET = "ggml-cuda8-ggml-backend-graph-builder-q8_0-residual-add-smoke"
G21_RESIDUAL_EXE = "ggml-cuda8-ggml-backend-graph-builder-q8_0-residual-add-smoke"

G22_RESIDUAL_SOFTMAX_TARGET = "ggml-cuda8-ggml-backend-graph-builder-q8_0-residual-add-softmax-smoke"
G22_RESIDUAL_SOFTMAX_EXE = "ggml-cuda8-ggml-backend-graph-builder-q8_0-residual-add-softmax-smoke"

G17_COMMENT = "# G17 real GGML graph-builder Q8_0 MUL_MAT graph_compute smoke coverage"
G18A_COMMENT = "# G18 real GGML graph-builder Q8_0 MUL_MAT -> ADD graph_compute smoke coverage"
G18C_COMMENT = "# G18 real GGML graph-builder Q8_0 MUL_MAT -> MUL_SCALAR -> ADD graph_compute smoke coverage"
G19A_COMMENT = "# G19 real GGML graph-builder Q8_0 MUL_MAT -> MUL_SCALAR -> ADD -> SOFTMAX graph_compute smoke coverage"
G20A_COMMENT = "# G20 real GGML graph-builder Q8_0 MUL_MAT -> MUL_SCALAR -> ADD -> SOFTMAX -> SUM_ROWS graph_compute smoke coverage"
G21A_COMMENT = "# G21 real GGML graph-builder Q8_0 MUL_MAT -> residual ADD graph_compute smoke coverage"
G22A_COMMENT = "# G22 real GGML graph-builder Q8_0 MUL_MAT -> residual ADD -> SOFTMAX graph_compute smoke coverage"

G17_LINE = "run_target {} {}".format(G17_Q8_TARGET, G17_Q8_EXE)
G18A_LINE = "run_target {} {}".format(G18_ADD_TARGET, G18_ADD_EXE)
G18C_LINE = "run_target {} {}".format(G18_MUL_ADD_TARGET, G18_MUL_ADD_EXE)
G19A_LINE = "run_target {} {}".format(G19_SOFTMAX_TARGET, G19_SOFTMAX_EXE)
G20A_LINE = "run_target {} {}".format(G20_SUMROWS_TARGET, G20_SUMROWS_EXE)
G21A_LINE = "run_target {} {}".format(G21_RESIDUAL_TARGET, G21_RESIDUAL_EXE)
G22A_LINE = "run_target {} {}".format(G22_RESIDUAL_SOFTMAX_TARGET, G22_RESIDUAL_SOFTMAX_EXE)

README_BLOCK = """<!-- G22_STATUS_START -->
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


def remove_existing_quant_pipeline_lines(data):
    remove = set([
        G17_COMMENT,
        G18A_COMMENT,
        G18C_COMMENT,
        G19A_COMMENT,
        G20A_COMMENT,
        G21A_COMMENT,
        G22A_COMMENT,
        G17_LINE,
        G18A_LINE,
        G18C_LINE,
        G19A_LINE,
        G20A_LINE,
        G21A_LINE,
        G22A_LINE,
    ])

    out = []
    for line in data.splitlines():
        if line.strip() in remove:
            continue
        out.append(line)

    return "\n".join(out) + "\n"


def insert_quant_pipeline_targets(data):
    insert_block = (
        "\n"
        + G17_COMMENT + "\n"
        + G17_LINE + "\n"
        + "\n"
        + G18A_COMMENT + "\n"
        + G18A_LINE + "\n"
        + "\n"
        + G18C_COMMENT + "\n"
        + G18C_LINE + "\n"
        + "\n"
        + G19A_COMMENT + "\n"
        + G19A_LINE + "\n"
        + "\n"
        + G20A_COMMENT + "\n"
        + G20A_LINE + "\n"
        + "\n"
        + G21A_COMMENT + "\n"
        + G21A_LINE + "\n"
        + "\n"
        + G22A_COMMENT + "\n"
        + G22A_LINE + "\n"
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

    raise RuntimeError("could not find safe insertion anchor for G17/G18/G19/G20/G21/G22 targets")


def update_success_label(data):
    labels = [
        'echo "G15E regression SUCCESS"',
        'echo "CUDA8 G17B regression SUCCESS"',
        'echo "CUDA8 G17C regression SUCCESS"',
        'echo "CUDA8 G17D regression SUCCESS"',
        'echo "CUDA8 G18B regression SUCCESS"',
        'echo "CUDA8 G18D regression SUCCESS"',
        'echo "CUDA8 G19B regression SUCCESS"',
        'echo "CUDA8 G20B regression SUCCESS"',
        'echo "CUDA8 G21B regression SUCCESS"',
        'echo "CUDA8 G22B regression SUCCESS"',
    ]

    found = False
    for label in labels:
        if label in data:
            data = data.replace(label, 'echo "CUDA8 G22B regression SUCCESS"')
            found = True

    if found:
        return data

    if not data.endswith("\n"):
        data += "\n"
    data += '\necho "CUDA8 G22B regression SUCCESS"\n'
    return data


def refresh_regression():
    if not os.path.exists(REG):
        raise RuntimeError("main regression script not found: {}".format(REG))

    data = read_file(REG)
    orig = data
    backup(REG, "g22b")

    if "run_target()" not in data:
        raise RuntimeError("run_g11_regression.sh does not contain run_target function")

    data = remove_existing_quant_pipeline_lines(data)
    data = insert_quant_pipeline_targets(data)
    data = update_success_label(data)

    if data != orig:
        write_file(REG, data)
        os.chmod(REG, os.stat(REG).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        print("patched", REG)
    else:
        print("main regression already current for G22B")


def refresh_readme():
    if os.path.exists(README):
        data = read_file(README)
    else:
        data = "# ggml-cuda8\n\n"

    orig = data
    backup(README, "g22b")

    start = "<!-- G22_STATUS_START -->"
    end = "<!-- G22_STATUS_END -->"

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
        print("README G22 status already current")


def main():
    refresh_regression()
    refresh_readme()
    print("G22B regression + README refresh complete.")


if __name__ == "__main__":
    main()
