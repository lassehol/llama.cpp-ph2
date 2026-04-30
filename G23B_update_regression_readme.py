#!/usr/bin/env python3
# Safe G23B updater:
# - rewrites run_g11_regression.sh from a canonical list with real newlines
# - leaves README untouched if G23 status block already exists
# - avoids the previous literal-\n corruption

import os
import stat
import time

ROOT = "/workspace/notebooks/llama.cpp-ph2"
REG = os.path.join(ROOT, "run_g11_regression.sh")
README = os.path.join(ROOT, "ggml/src/ggml-cuda8/README.md")

README_BLOCK = """<!-- G23_STATUS_START -->
## G23 status: real GGML graph-builder quantized residual softmax + sum_rows coverage

Status: **PASS on GTX 560 / CUDA 8 / Fermi**.

G23 validates:

    h       = ggml_mul_mat(Aq, x)
    biased  = ggml_add(h, residual)
    prob    = ggml_soft_max(biased)
    row_sum = ggml_sum_rows(prob)

Validated dispatch:

    GGML_OP_MUL_MAT  -> GGML_CUDA8_OP_MUL_MAT_Q8_0_F32_VEC
    GGML_OP_ADD      -> GGML_CUDA8_OP_ADD_F32
    GGML_OP_SOFT_MAX -> GGML_CUDA8_OP_SOFTMAX_ROWS_F32
    GGML_OP_SUM_ROWS -> GGML_CUDA8_OP_REDUCE_SUM_ROWS_F32

G23A verifies that the residual input branch remains unchanged after graph dispatch.
G23B refreshes main regression and README status for G23A.
<!-- G23_STATUS_END -->
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
        "# Legacy/host-backed dispatcher coverage",
        "run_target ggml-cuda8-dispatch-all-smoke ggml-cuda8-dispatch-all-smoke",
        "",
        "echo",
        'echo "CUDA8 G23B regression SUCCESS"',
        "",
    ]
    return "\n".join(lines)


def refresh_regression():
    backup(REG, "g23b-repair")
    data = build_regression_script()
    write_file(REG, data)
    os.chmod(REG, os.stat(REG).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    print("rewrote clean", REG)


def refresh_readme():
    if not os.path.exists(README):
        write_file(README, "# ggml-cuda8\n\n" + README_BLOCK + "\n")
        print("created README G23 status")
        return

    data = read_file(README)
    backup(README, "g23b-repair")

    if "<!-- G23_STATUS_START -->" in data and "<!-- G23_STATUS_END -->" in data:
        print("README G23 status already present")
        return

    if not data.endswith("\n"):
        data += "\n"
    data += "\n" + README_BLOCK + "\n"
    write_file(README, data)
    print("appended README G23 status")


def main():
    refresh_regression()
    refresh_readme()
    print("G23B safe regression + README refresh complete.")


if __name__ == "__main__":
    main()
