#!/usr/bin/env python3
# Replace the fragile graph node-count regex logic in G18A_writer.py
# with simple literal substitutions.
# Python 3.5-compatible.

from pathlib import Path
import time

p = Path("G18A_writer.py")
s = p.read_text()

backup = p.with_name(p.name + ".nodecount-simple-backup-" + str(int(time.time())))
backup.write_text(s)
print("backup", backup)

new_block = '''# Convert the G17C single-node graph assertion into a G18A two-node assertion.
# Use simple literal substitutions to avoid depending on exact C++ formatting.
if "graph->n_nodes != 1" not in s:
    raise RuntimeError("could not find graph->n_nodes != 1 in source template")
s = s.replace("graph->n_nodes != 1", "graph->n_nodes != 2", 1)
s = s.replace("expected 1 graph node", "expected 2 graph nodes", 1)
'''

start = s.find("# Convert the G17C single-node graph assertion")
if start < 0:
    raise RuntimeError("could not find existing graph node-count conversion block")

end = s.find("\n\nold =", start)
if end < 0:
    raise RuntimeError("could not find end of existing graph node-count conversion block")

s = s[:start] + new_block + s[end:]

# Remove import re if it was only added for the failed regex patch.
# Harmless to leave, but cleaner to remove if present.
s = s.replace("import os\nimport re\nimport time\n", "import os\nimport time\n")

p.write_text(s)
print("patched", p)
