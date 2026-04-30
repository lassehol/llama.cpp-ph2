#!/usr/bin/env python3
# Fix run_g11_regression.sh after G16/G17 insertion before run_target definition.
# Python 3.5-compatible.

import time
from pathlib import Path

p = Path("run_g11_regression.sh")
s = p.read_text()

backup = p.with_name(p.name + ".g16g17-order-backup-" + str(int(time.time())))
backup.write_text(s)
print("backup", backup)

g16g17_block = '''# G16 real GGML graph-builder graph_compute smoke coverage
run_target ggml-cuda8-ggml-backend-graph-builder-add-smoke ggml-cuda8-ggml-backend-graph-builder-add-smoke
run_target ggml-cuda8-ggml-backend-graph-builder-add-mul-smoke ggml-cuda8-ggml-backend-graph-builder-add-mul-smoke
run_target ggml-cuda8-ggml-backend-graph-builder-softmax-smoke ggml-cuda8-ggml-backend-graph-builder-softmax-smoke
run_target ggml-cuda8-ggml-backend-graph-builder-attnlike-smoke ggml-cuda8-ggml-backend-graph-builder-attnlike-smoke

# G17 real GGML graph-builder Q8_0 MUL_MAT graph_compute smoke coverage
run_target ggml-cuda8-ggml-backend-graph-builder-q8_0-mmv-smoke ggml-cuda8-ggml-backend-graph-builder-q8_0-mmv-smoke
'''

# Remove misplaced early block.
if g16g17_block in s:
    s = s.replace(g16g17_block + "\n", "", 1)
elif g16g17_block.strip() in s:
    s = s.replace(g16g17_block.strip(), "", 1)
else:
    print("warning: exact early G16/G17 block not found; will still try to avoid duplicates")

# Remove any duplicate individual G16/G17 lines before reinserting cleanly.
lines_to_remove = [
    "run_target ggml-cuda8-ggml-backend-graph-builder-add-smoke ggml-cuda8-ggml-backend-graph-builder-add-smoke",
    "run_target ggml-cuda8-ggml-backend-graph-builder-add-mul-smoke ggml-cuda8-ggml-backend-graph-builder-add-mul-smoke",
    "run_target ggml-cuda8-ggml-backend-graph-builder-softmax-smoke ggml-cuda8-ggml-backend-graph-builder-softmax-smoke",
    "run_target ggml-cuda8-ggml-backend-graph-builder-attnlike-smoke ggml-cuda8-ggml-backend-graph-builder-attnlike-smoke",
    "run_target ggml-cuda8-ggml-backend-graph-builder-q8_0-mmv-smoke ggml-cuda8-ggml-backend-graph-builder-q8_0-mmv-smoke",
]

out = []
for line in s.splitlines():
    if line.strip() in lines_to_remove:
        continue
    if line.strip() in [
        "# G16 real GGML graph-builder graph_compute smoke coverage",
        "# G17 real GGML graph-builder Q8_0 MUL_MAT graph_compute smoke coverage",
    ]:
        continue
    out.append(line)

s = "\n".join(out) + "\n"

insert_after = '''run_target ggml-cuda8-ggml-backend-graph-compute-attnlike-smoke ggml-cuda8-ggml-backend-graph-compute-attnlike-smoke
'''

insert_block = '''
# G16 real GGML graph-builder graph_compute smoke coverage
run_target ggml-cuda8-ggml-backend-graph-builder-add-smoke ggml-cuda8-ggml-backend-graph-builder-add-smoke
run_target ggml-cuda8-ggml-backend-graph-builder-add-mul-smoke ggml-cuda8-ggml-backend-graph-builder-add-mul-smoke
run_target ggml-cuda8-ggml-backend-graph-builder-softmax-smoke ggml-cuda8-ggml-backend-graph-builder-softmax-smoke
run_target ggml-cuda8-ggml-backend-graph-builder-attnlike-smoke ggml-cuda8-ggml-backend-graph-builder-attnlike-smoke

# G17 real GGML graph-builder Q8_0 MUL_MAT graph_compute smoke coverage
run_target ggml-cuda8-ggml-backend-graph-builder-q8_0-mmv-smoke ggml-cuda8-ggml-backend-graph-builder-q8_0-mmv-smoke
'''

if insert_after in s:
    s = s.replace(insert_after, insert_after + insert_block, 1)
else:
    # Fallback: insert before dispatch-all.
    anchor = "# Legacy/host-backed dispatcher coverage"
    if anchor not in s:
        raise RuntimeError("could not find insertion anchor")
    s = s.replace(anchor, insert_block + "\n" + anchor, 1)

p.write_text(s)
print("patched", p)
