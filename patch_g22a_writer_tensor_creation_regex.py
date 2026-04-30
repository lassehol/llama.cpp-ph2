#!/usr/bin/env python3
# Patch G22A_writer.py so tensor creation rewrite is whitespace-tolerant.
# Python 3.5-compatible.

import re
import time
from pathlib import Path

p = Path("G22A_writer.py")
s = p.read_text()

backup = p.with_name(p.name + ".tensor-regex-backup-" + str(int(time.time())))
backup.write_text(s)
print("backup", backup)

if "import re\n" not in s:
    s = s.replace("import os\nimport time\n", "import os\nimport re\nimport time\n")

start_marker = "old = '''    ggml_tensor * Aq       = ggml_new_tensor_2d"
start = s.find(start_marker)
if start < 0:
    raise RuntimeError("could not find existing tensor creation old-block marker")

end_marker = 's = replace_once(s, old, new, "tensor creation")'
end = s.find(end_marker, start)
if end < 0:
    raise RuntimeError("could not find tensor creation replace_once end marker")

end = end + len(end_marker)

replacement = r'''# Convert:
#   y = ggml_add(h, residual)
# into:
#   biased = ggml_add(h, residual)
#   y      = ggml_soft_max(biased)
#
# Use regex because the generated G21A source may have different alignment.
tensor_pat = (
    r'(    ggml_tensor \* Aq\s*=\s*ggml_new_tensor_2d\(gctx, GGML_TYPE_Q8_0, cols, rows\);\n'
    r'    ggml_tensor \* x\s*=\s*ggml_new_tensor_1d\(gctx, GGML_TYPE_F32, cols\);\n'
    r'    ggml_tensor \* h\s*=\s*ggml_mul_mat\(gctx, Aq, x\);\n'
    r'    ggml_tensor \* residual\s*=\s*ggml_new_tensor_1d\(gctx, GGML_TYPE_F32, rows\);\n)'
    r'    ggml_tensor \* y\s*=\s*ggml_add\(gctx, h, residual\);\n\n'
    r'    if \(Aq == NULL \|\| x == NULL \|\| h == NULL \|\| residual == NULL \|\| y == NULL\) \{'
)

tensor_repl = (
    r'\1'
    '    ggml_tensor * biased   = ggml_add(gctx, h, residual);\n'
    '    ggml_tensor * y        = ggml_soft_max(gctx, biased);\n\n'
    '    if (Aq == NULL || x == NULL || h == NULL || residual == NULL || biased == NULL || y == NULL) {'
)

s_new, n = re.subn(tensor_pat, tensor_repl, s, count=1)
if n != 1:
    raise RuntimeError("could not patch tensor creation regex block")

s = s_new
'''

s = s[:start] + replacement + s[end:]

p.write_text(s)
print("patched", p)
