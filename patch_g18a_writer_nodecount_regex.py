#!/usr/bin/env python3
# Patch G18A_writer.py to replace the graph node-count check with regex
# instead of exact one-line string matching.
# Python 3.5-compatible.

from pathlib import Path
import time

p = Path("G18A_writer.py")
s = p.read_text()

backup = p.with_name(p.name + ".nodecount-regex-backup-" + str(int(time.time())))
backup.write_text(s)
print("backup", backup)

# Ensure re is imported.
if "import re\n" not in s:
    s = s.replace("import os\nimport time\n", "import os\nimport re\nimport time\n")

old = '''s = replace_once(
    s,
    '    if (graph->n_nodes != 1) { std::fprintf(stderr, "expected 1 graph node, got %d\\\\n", graph->n_nodes); ggml_free(gctx); return 1; }\\n',
    '    if (graph->n_nodes != 2) { std::fprintf(stderr, "expected 2 graph nodes, got %d\\\\n", graph->n_nodes); ggml_free(gctx); return 1; }\\n',
    "graph node count"
)
'''

new = '''# Convert the G17C single-node graph assertion into a G18A two-node assertion.
# Use regex because this block may be formatted differently across generated versions.
s_new, n_subs = re.subn(
    r'if \\(graph->n_nodes != 1\\) \\{\\s*std::fprintf\\(stderr, "expected 1 graph node, got %d\\\\\\\\n", graph->n_nodes\\);\\s*ggml_free\\(gctx\\);\\s*return 1;\\s*\\}',
    'if (graph->n_nodes != 2) { std::fprintf(stderr, "expected 2 graph nodes, got %d\\\\\\\\n", graph->n_nodes); ggml_free(gctx); return 1; }',
    s,
    count=1,
    flags=re.S
)
if n_subs != 1:
    raise RuntimeError("could not replace graph node count check with regex")
s = s_new
'''

if old not in s:
    print("exact node-count replace_once block not found; trying broader replacement")

    start = s.find('s = replace_once(\n    s,\n    \'    if (graph->n_nodes != 1)')
    if start < 0:
        raise RuntimeError("could not find node-count replacement block in G18A_writer.py")

    end = s.find('\n\nold =', start)
    if end < 0:
        raise RuntimeError("could not find end of node-count replacement block")

    s = s[:start] + new + s[end:]
else:
    s = s.replace(old, new, 1)

p.write_text(s)
print("patched", p)
