#!/usr/bin/env python3
# G16D_fix_rowsum_1d_v2.py
#
# Fix G16D after real graph-builder attention-like smoke failed at SUM_ROWS:
#   graph node 3 op=15 REDUCE_SUM_ROWS_F32
#   ggml-cuda8/backend-dispatch: unsupported op/layout op=6
#
# The synthetic G15D attention smoke uses a compact 1D row_sum output vector.
# The first G16D writer rebound the real ggml_sum_rows output as a 2D [1, rows]
# tensor.  The CUDA8 SUM_ROWS dispatcher expects the output residency/layout used
# by the synthetic smoke: a 1D F32 vector of length rows.
#
# This patch changes only the forced backend data layout for row_sum:
#   force_2d_f32_data_layout(row_sum, 1, rows, ...)
# ->
#   force_1d_f32_data_layout(row_sum, rows, ...)
#
# The graph itself remains the real ggml_sum_rows graph-builder node.

import os
import time

ROOT = "/workspace/notebooks/llama.cpp-ph2"
SRC = os.path.join(ROOT, "ggml/src/ggml-cuda8/ggml-cuda8-ggml-backend-graph-builder-attnlike-smoke.cpp")

old = "force_2d_f32_data_layout(row_sum, 1, rows, base_u8 + off_row_sum);"
new = "force_1d_f32_data_layout(row_sum, rows, base_u8 + off_row_sum);"

with open(SRC, "r") as f:
    data = f.read()

backup = SRC + ".g16d-rowsum-1d-v2-backup-" + str(int(time.time()))
with open(backup, "w") as f:
    f.write(data)

if old in data:
    data = data.replace(old, new, 1)
    with open(SRC, "w") as f:
        f.write(data)
    print("backup", backup)
    print("patched", SRC)
    print("G16D v2 fix complete: row_sum backend layout forced to 1D vector length rows.")
elif new in data:
    print("backup", backup)
    print("row_sum layout already patched to 1D")
else:
    raise RuntimeError("could not find row_sum forced layout line to patch")
