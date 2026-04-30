#!/usr/bin/env python3
from pathlib import Path
import time

BASE = Path('/workspace/notebooks/llama.cpp-ph2/ggml/src/ggml-cuda8')
CMAKE = BASE / 'CMakeLists.txt'

backup = CMAKE.with_name(CMAKE.name + '.g11d-cmake-fix-backup-' + str(int(time.time())))
backup.write_text(CMAKE.read_text())
print('backup', backup)

cm = CMAKE.read_text()

target = r'''# ---------------------------------------------------------------------------
# G11D device-resident REDUCE_SUM_ROWS_F32 / REDUCE_MAX_ROWS_F32 smoke:
#   - src matrix and dst vectors are CUDA8-resident inside ggml_backend_buffer_t
#   - dispatcher reduce ops call device kernels directly
# ---------------------------------------------------------------------------

cuda_add_executable(ggml-cuda8-ggml-buffer-device-reduce-smoke
    ggml-cuda8-ggml-buffer-device-reduce-smoke.cpp
)

target_link_libraries(ggml-cuda8-ggml-buffer-device-reduce-smoke
    ggml-cuda8-kernels
    ${CUDA_LIBRARIES}
)
'''

if 'ggml-cuda8-ggml-buffer-device-reduce-smoke' not in cm:
    if not cm.endswith('\n'):
        cm += '\n'
    cm += target
    CMAKE.write_text(cm)
    print('patched', CMAKE)
else:
    print('CMake target already present')

print('G11D CMake fix complete.')
