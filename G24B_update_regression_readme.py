#!/usr/bin/env python3
# G24B_update_regression_readme.py
#
# G24B: refresh main CUDA8 regression script + README status for G24A.
#
# Safe updater:
#   - rewrites run_g11_regression.sh from a canonical list with real newlines
#   - appends/replaces README G24 status block
#   - avoids literal-\n corruption
#
# Python 3.5-compatible.

import os
import re
import stat
import time

ROOT = "/workspace/notebooks/llama.cpp-ph2"
REG = os.path.join(ROOT, "run_g11_regression.sh")
README = os.path.join(ROOT, "ggml/src/ggml-cuda8/README.md")

README_BLOCK = """<!-- G24_STATUS_START -->
## G24 status: real GGML graph-builder quantized scaled residual softmax + sum_rows coverage

Status: **PASS on GTX 560 / CUDA 8 / Fermi**.

G24 extends the residual-softmax-sumrows graph-builder path by inserting scalar scaling between the Q8_0 projection and the residual add.

Validated G24 checkpoints:

- **G24A**: real graph-builder scaled residual-softmax-sumrows quantized pipeline passes:
  - `h = ggml_mul_mat(Q8_0, x)`
  - `scaled = ggml_mul(h, scale)`
  - `biased = ggml_add(scaled, residual)`
  - `prob = ggml_soft_max(biased)`
  - `row_sum = ggml_sum_rows(prob)`
  - residual input branch isolation verified after `ggml_backend_i.graph_compute`
- **G24B**: main regression and README status refreshed for the G24A scaled residual-softmax-sumrows pipeline.

Validated G24A graph:

    Aq       : GGML_TYPE_Q8_0 matrix, shape [cols, rows]
    x        : GGML_TYPE_F32 vector, shape [cols]
    h        : GGML_TYPE_F32 vector, shape [rows]
    scale    : GGML_TYPE_F32 scalar, shape [1]
    scaled   : GGML_TYPE_F32 vector, shape [rows]
    residual : GGML_TYPE_F32 vector, shape [rows]
    biased   : GGML_TYPE_F32 vector, shape [rows]
    prob     : GGML_TYPE_F32 vector, shape [rows]
    row_sum  : GGML_TYPE_F32 scalar/vector, shape [1]

Validated G24A path:

    A_f32
      -> host pack_q8_0(A_f32)
      -> Aq Q8_0 blocks

    h       = ggml_mul_mat(Aq, x)
    scaled  = ggml_mul(h, scale)
    biased  = ggml_add(scaled, residual)
    prob    = ggml_soft_max(biased)
    row_sum = ggml_sum_rows(prob)

    ggml_build_forward_expand(row_sum)
      -> real GGML graph with five nodes
      -> node 0: GGML_OP_MUL_MAT
      -> node 1: GGML_OP_MUL
      -> node 2: GGML_OP_ADD
      -> node 3: GGML_OP_SOFT_MAX
      -> node 4: GGML_OP_SUM_ROWS
      -> ggml_backend_i.graph_compute
      -> GGML_CUDA8_OP_MUL_MAT_Q8_0_F32_VEC
      -> GGML_CUDA8_OP_MUL_SCALAR_F32
      -> GGML_CUDA8_OP_ADD_F32
      -> GGML_CUDA8_OP_SOFTMAX_ROWS_F32
      -> GGML_CUDA8_OP_REDUCE_SUM_ROWS_F32
      -> CUDA8/Fermi kernels

Implementation notes:

- G24A reuses the packed Q8_0 reference strategy validated in G17C.
- G24A combines the scalar post-projection scaling from G20A with the residual branch topology from G21A/G22A/G23A.
- G24A verifies that the residual input branch remains unchanged after graph dispatch.
- The G24A smoke uses the same standalone-GC graph-builder target pattern as G16/G17/G18/G19/G20/G21/G22/G23:

    ../ggml.c
    ../ggml-quants.c
    ../ggml-threading.cpp
    -ffunction-sections
    -fdata-sections
    -Wl,--gc-sections

- G24A focused regression passes:
  - Q8_0 MUL_MAT -> MUL_SCALAR -> residual ADD -> SOFTMAX -> SUM_ROWS graph-builder pipeline smoke,
  - Q8_0 MUL_MAT -> residual ADD -> SOFTMAX -> SUM_ROWS graph-builder pipeline smoke,
  - Q8_0 MUL_MAT -> residual ADD -> SOFTMAX graph-builder pipeline smoke,
  - Q8_0 MUL_MAT -> residual ADD graph-builder pipeline smoke,
  - Q8_0 MUL_MAT -> MUL_SCALAR -> ADD -> SOFTMAX -> SUM_ROWS graph-builder pipeline smoke,
  - packed Q8_0 graph-builder MMV smoke,
  - real graph-builder attention-like G16D smoke,
  - dispatch-all CUDA8 kernel smoke.
<!-- G24_STATUS_END -->
"""


def read_file(path):
    with open(path, "r") as f:
        return f.read()


def write_file(path, data):
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


def build_regression_script():
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "export PATH=/opt/cmake-3.22.6-linux-x86_64/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "hash -r",
        "",
        'ROOT="/workspace/notebooks/llama.cpp-ph2"',
        'BUILD="$ROOT/build-cuda8-parent"',
        "",
        'cd "$BUILD"',
        "",
        'echo "== Toolchain =="',
        'echo "cmake: $(command -v cmake)"',
        "cmake --version | head -1",
        'echo "make:  $(command -v make)"',
        "make --version | head -1",
        "echo",
        "",
        'echo "== Configure CUDA8 parent build =="',
        "cmake .. -DGGML_CUDA8=ON -DGGML_CUDA=OFF -DBUILD_SHARED_LIBS=OFF",
        "",
        "run_target() {",
        '    target="$1"',
        '    exe="$2"',
        "    echo",
        '    echo "---- build: $target ----"',
        '    make -j1 "$target"',
        '    echo "---- run: ./bin/$exe ----"',
        '    "./bin/$exe"',
        "}",
        "",
        "echo",
        'echo "== CUDA8 G11/G12/G13/G14/G15 regression =="',
        "",
        "# G11A/G11B buffer and residency foundation",
        "run_target ggml-cuda8-ggml-buffer-smoke ggml-cuda8-ggml-buffer-smoke",
        "run_target ggml-cuda8-ggml-buffer-offset-smoke ggml-cuda8-ggml-buffer-offset-smoke",
        "run_target ggml-cuda8-ggml-buffer-dispatch-smoke ggml-cuda8-ggml-buffer-dispatch-smoke",
        "run_target ggml-cuda8-ggml-buffer-residency-smoke ggml-cuda8-ggml-buffer-residency-smoke",
        "",
        "# G11 device-resident single-op paths",
        "run_target ggml-cuda8-ggml-buffer-device-add-smoke ggml-cuda8-ggml-buffer-device-add-smoke",
        "run_target ggml-cuda8-ggml-buffer-device-scalar-smoke ggml-cuda8-ggml-buffer-device-scalar-smoke",
        "run_target ggml-cuda8-ggml-buffer-device-softmax-smoke ggml-cuda8-ggml-buffer-device-softmax-smoke",
        "run_target ggml-cuda8-ggml-buffer-device-reduce-smoke ggml-cuda8-ggml-buffer-device-reduce-smoke",
        "",
        "# G11 direct device-resident graph paths",
        "run_target ggml-cuda8-ggml-buffer-device-graph-smoke ggml-cuda8-ggml-buffer-device-graph-smoke",
        "run_target ggml-cuda8-ggml-buffer-device-softmax-graph-smoke ggml-cuda8-ggml-buffer-device-softmax-graph-smoke",
        "run_target ggml-cuda8-ggml-buffer-device-attnlike-smoke ggml-cuda8-ggml-buffer-device-attnlike-smoke",
        "",
        "# G12 minimal backend object and backend-owned buffers/graphs",
        "run_target ggml-cuda8-ggml-backend-probe ggml-cuda8-ggml-backend-probe",
        "run_target ggml-cuda8-ggml-backend-buffer-graph-smoke ggml-cuda8-ggml-backend-buffer-graph-smoke",
        "run_target ggml-cuda8-ggml-backend-attnlike-smoke ggml-cuda8-ggml-backend-attnlike-smoke",
        "",
        "# G13 compute-shaped backend callback paths",
        "run_target ggml-cuda8-ggml-backend-compute-probe ggml-cuda8-ggml-backend-compute-probe",
        "run_target ggml-cuda8-ggml-backend-compute-attnlike-smoke ggml-cuda8-ggml-backend-compute-attnlike-smoke",
        "",
        "# G14 real ggml_backend_i graph API shape probe",
        "run_target ggml-cuda8-ggml-backend-graph-api-probe ggml-cuda8-ggml-backend-graph-api-probe",
        "",
        "# G15 real ggml_backend_i graph_compute execution paths",
        "run_target ggml-cuda8-ggml-backend-graph-compute-add-smoke ggml-cuda8-ggml-backend-graph-compute-add-smoke",
        "run_target ggml-cuda8-ggml-backend-graph-compute-add-mul-smoke ggml-cuda8-ggml-backend-graph-compute-add-mul-smoke",
        "run_target ggml-cuda8-ggml-backend-graph-compute-softmax-smoke ggml-cuda8-ggml-backend-graph-compute-softmax-smoke",
        "run_target ggml-cuda8-ggml-backend-graph-compute-attnlike-smoke ggml-cuda8-ggml-backend-graph-compute-attnlike-smoke",
        "",
        "# G16 real GGML graph-builder graph_compute smoke coverage",
        "run_target ggml-cuda8-ggml-backend-graph-builder-add-smoke ggml-cuda8-ggml-backend-graph-builder-add-smoke",
        "run_target ggml-cuda8-ggml-backend-graph-builder-add-mul-smoke ggml-cuda8-ggml-backend-graph-builder-add-mul-smoke",
        "run_target ggml-cuda8-ggml-backend-graph-builder-softmax-smoke ggml-cuda8-ggml-backend-graph-builder-softmax-smoke",
        "run_target ggml-cuda8-ggml-backend-graph-builder-attnlike-smoke ggml-cuda8-ggml-backend-graph-builder-attnlike-smoke",
        "",
        "# G17 real GGML graph-builder Q8_0 MUL_MAT graph_compute smoke coverage",
        "run_target ggml-cuda8-ggml-backend-graph-builder-q8_0-mmv-smoke ggml-cuda8-ggml-backend-graph-builder-q8_0-mmv-smoke",
        "",
        "# G18 real GGML graph-builder Q8_0 MUL_MAT -> ADD graph_compute smoke coverage",
        "run_target ggml-cuda8-ggml-backend-graph-builder-q8_0-add-smoke ggml-cuda8-ggml-backend-graph-builder-q8_0-add-smoke",
        "",
        "# G18 real GGML graph-builder Q8_0 MUL_MAT -> MUL_SCALAR -> ADD graph_compute smoke coverage",
        "run_target ggml-cuda8-ggml-backend-graph-builder-q8_0-mul-add-smoke ggml-cuda8-ggml-backend-graph-builder-q8_0-mul-add-smoke",
        "",
        "# G19 real GGML graph-builder Q8_0 MUL_MAT -> MUL_SCALAR -> ADD -> SOFTMAX graph_compute smoke coverage",
        "run_target ggml-cuda8-ggml-backend-graph-builder-q8_0-mul-add-softmax-smoke ggml-cuda8-ggml-backend-graph-builder-q8_0-mul-add-softmax-smoke",
        "",
        "# G20 real GGML graph-builder Q8_0 MUL_MAT -> MUL_SCALAR -> ADD -> SOFTMAX -> SUM_ROWS graph_compute smoke coverage",
        "run_target ggml-cuda8-ggml-backend-graph-builder-q8_0-mul-add-softmax-sumrows-smoke ggml-cuda8-ggml-backend-graph-builder-q8_0-mul-add-softmax-sumrows-smoke",
        "",
        "# G21 real GGML graph-builder Q8_0 MUL_MAT -> residual ADD graph_compute smoke coverage",
        "run_target ggml-cuda8-ggml-backend-graph-builder-q8_0-residual-add-smoke ggml-cuda8-ggml-backend-graph-builder-q8_0-residual-add-smoke",
        "",
        "# G22 real GGML graph-builder Q8_0 MUL_MAT -> residual ADD -> SOFTMAX graph_compute smoke coverage",
        "run_target ggml-cuda8-ggml-backend-graph-builder-q8_0-residual-add-softmax-smoke ggml-cuda8-ggml-backend-graph-builder-q8_0-residual-add-softmax-smoke",
        "",
        "# G23 real GGML graph-builder Q8_0 MUL_MAT -> residual ADD -> SOFTMAX -> SUM_ROWS graph_compute smoke coverage",
        "run_target ggml-cuda8-ggml-backend-graph-builder-q8_0-residual-add-softmax-sumrows-smoke ggml-cuda8-ggml-backend-graph-builder-q8_0-residual-add-softmax-sumrows-smoke",
        "",
        "# G24 real GGML graph-builder Q8_0 MUL_MAT -> MUL_SCALAR -> residual ADD -> SOFTMAX -> SUM_ROWS graph_compute smoke coverage",
        "run_target ggml-cuda8-ggml-backend-graph-builder-q8_0-residual-scale-add-softmax-sumrows-smoke ggml-cuda8-ggml-backend-graph-builder-q8_0-residual-scale-add-softmax-sumrows-smoke",
        "",
        "# Legacy/host-backed dispatcher coverage",
        "run_target ggml-cuda8-dispatch-all-smoke ggml-cuda8-dispatch-all-smoke",
        "",
        "echo",
        'echo "CUDA8 G24B regression SUCCESS"',
        "",
    ]
    return "\n".join(lines)


def refresh_regression():
    backup(REG, "g24b")
    data = build_regression_script()
    write_file(REG, data)
    os.chmod(REG, os.stat(REG).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    print("rewrote clean", REG)


def refresh_readme():
    if os.path.exists(README):
        data = read_file(README)
    else:
        data = "# ggml-cuda8\n\n"

    orig = data
    backup(README, "g24b")

    start = "<!-- G24_STATUS_START -->"
    end = "<!-- G24_STATUS_END -->"

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
        print("README G24 status already current")


def main():
    refresh_regression()
    refresh_readme()
    print("G24B regression + README refresh complete.")


if __name__ == "__main__":
    main()
