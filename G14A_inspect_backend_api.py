#!/usr/bin/env python3
# G14A_inspect_backend_api.py
#
# Inspect the actual ggml_backend_i / related backend API shape in this checkout.
# Python 3.5-compatible: no pathlib, no f-strings.

import os
import re
import time

ROOT = "/workspace/notebooks/llama.cpp-ph2"
OUT_DIR = os.path.join(ROOT, "ggml", "src", "ggml-cuda8")
REPORT = os.path.join(OUT_DIR, "G14A_backend_api_report.md")
SNAPSHOT = os.path.join(OUT_DIR, "G14A_backend_api_snapshot.txt")

CANDIDATE_NAMES = [
    "ggml-backend-impl.h",
    "ggml-backend.h",
]

STRUCTS_OF_INTEREST = [
    "ggml_backend_i",
    "ggml_backend_buffer_i",
    "ggml_backend_buffer_type_i",
    "ggml_backend_device_i",
    "ggml_backend_reg_i",
    "ggml_backend_event_i",
]

KEYWORDS = [
    "graph",
    "compute",
    "sched",
    "plan",
    "buffer",
    "tensor",
    "event",
    "device",
    "sync",
]


def read_file(path):
    with open(path, "r") as f:
        return f.read()


def write_file(path, data):
    with open(path, "w") as f:
        f.write(data)


def find_headers(root):
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        # avoid huge irrelevant dirs where possible
        parts = dirpath.split(os.sep)
        if ".git" in parts or "build" in parts or "build-cuda8-parent" in parts:
            continue
        for fn in filenames:
            if fn in CANDIDATE_NAMES:
                found.append(os.path.join(dirpath, fn))
    found.sort()
    return found


def line_number_at_offset(text, offset):
    return text.count("\n", 0, offset) + 1


def extract_struct(text, struct_name):
    # Match either "struct name {" or "typedef struct name {" style.
    patterns = [
        r"struct\s+" + re.escape(struct_name) + r"\s*\{",
        r"typedef\s+struct\s+" + re.escape(struct_name) + r"\s*\{",
    ]
    best = None
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            best = m
            break
    if not best:
        return None

    start = best.start()
    brace = text.find("{", best.start())
    if brace < 0:
        return None

    depth = 0
    i = brace
    end = None
    while i < len(text):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                # include through following semicolon if present
                semi = text.find(";", i)
                if semi >= 0:
                    end = semi + 1
                else:
                    end = i + 1
                break
        i += 1
    if end is None:
        return None

    return {
        "name": struct_name,
        "start_line": line_number_at_offset(text, start),
        "end_line": line_number_at_offset(text, end),
        "text": text[start:end],
    }


def extract_function_pointer_fields(struct_text):
    fields = []
    for raw in struct_text.splitlines():
        line = raw.strip()
        if not line or line.startswith("//"):
            continue
        # Struct callback fields usually contain (*field_name)
        m = re.search(r"\(\s*\*\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)", line)
        if m:
            fields.append({"name": m.group(1), "line": line})
    return fields


def keyword_lines(text):
    out = []
    lines = text.splitlines()
    for i, line in enumerate(lines):
        lower = line.lower()
        if any(k in lower for k in KEYWORDS):
            out.append((i + 1, line.rstrip()))
    return out


def summarize_backend_i(fields):
    names = [f["name"] for f in fields]
    graphish = [n for n in names if ("graph" in n.lower() or "compute" in n.lower())]
    bufferish = [n for n in names if "buffer" in n.lower()]
    deviceish = [n for n in names if "device" in n.lower()]

    lines = []
    lines.append("Detected `ggml_backend_i` callback fields:")
    if names:
        for n in names:
            lines.append("- `{0}`".format(n))
    else:
        lines.append("- No function-pointer fields detected by the simple parser.")

    lines.append("")
    if graphish:
        lines.append("Graph/compute-shaped fields detected: `{0}`".format("`, `".join(graphish)))
    else:
        lines.append("No graph/compute-shaped fields were detected in `ggml_backend_i` by name.")

    if bufferish:
        lines.append("Buffer-shaped fields detected: `{0}`".format("`, `".join(bufferish)))
    else:
        lines.append("No buffer-shaped fields were detected in `ggml_backend_i` by name.")

    if deviceish:
        lines.append("Device-shaped fields detected: `{0}`".format("`, `".join(deviceish)))
    else:
        lines.append("No device-shaped fields were detected in `ggml_backend_i` by name.")

    return "\n".join(lines)


def main():
    if not os.path.isdir(ROOT):
        raise SystemExit("ROOT does not exist: " + ROOT)

    if not os.path.isdir(OUT_DIR):
        os.makedirs(OUT_DIR)

    headers = find_headers(ROOT)
    if not headers:
        raise SystemExit("No backend headers found under " + ROOT)

    structs_by_file = []
    snapshot_parts = []

    for path in headers:
        text = read_file(path)
        rel = os.path.relpath(path, ROOT)
        snapshot_parts.append("=" * 100)
        snapshot_parts.append(rel)
        snapshot_parts.append("=" * 100)
        snapshot_parts.append("")

        file_entry = {"path": path, "rel": rel, "structs": {}}
        for struct_name in STRUCTS_OF_INTEREST:
            st = extract_struct(text, struct_name)
            if st:
                file_entry["structs"][struct_name] = st
                snapshot_parts.append("--- {0} lines {1}-{2} ---".format(struct_name, st["start_line"], st["end_line"]))
                snapshot_parts.append(st["text"])
                snapshot_parts.append("")

        structs_by_file.append(file_entry)

    # Determine primary backend impl/header and backend_i shape.
    backend_i_entries = []
    for fe in structs_by_file:
        if "ggml_backend_i" in fe["structs"]:
            st = fe["structs"]["ggml_backend_i"]
            backend_i_entries.append((fe, st, extract_function_pointer_fields(st["text"])))

    report = []
    report.append("# G14A Backend API Inspection Report")
    report.append("")
    report.append("Generated: `{0}`".format(time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())))
    report.append("")
    report.append("Repository root:")
    report.append("")
    report.append("```text")
    report.append(ROOT)
    report.append("```")
    report.append("")

    report.append("## Headers inspected")
    report.append("")
    for path in headers:
        report.append("- `{0}`".format(os.path.relpath(path, ROOT)))
    report.append("")

    report.append("## Structs found")
    report.append("")
    any_struct = False
    for fe in structs_by_file:
        if fe["structs"]:
            any_struct = True
            report.append("### `{0}`".format(fe["rel"]))
            report.append("")
            for name in STRUCTS_OF_INTEREST:
                if name in fe["structs"]:
                    st = fe["structs"][name]
                    report.append("- `{0}` lines {1}-{2}".format(name, st["start_line"], st["end_line"]))
            report.append("")
    if not any_struct:
        report.append("No structs of interest were found by the simple parser.")
        report.append("")

    report.append("## `ggml_backend_i` callback shape")
    report.append("")
    if backend_i_entries:
        for fe, st, fields in backend_i_entries:
            report.append("Source: `{0}` lines {1}-{2}".format(fe["rel"], st["start_line"], st["end_line"]))
            report.append("")
            report.append(summarize_backend_i(fields))
            report.append("")
            report.append("Raw struct:")
            report.append("")
            report.append("```c")
            report.append(st["text"])
            report.append("```")
            report.append("")
    else:
        report.append("`ggml_backend_i` was not found in inspected headers.")
        report.append("")

    report.append("## Keyword scan")
    report.append("")
    report.append("Lines containing one of: `{0}`".format("`, `".join(KEYWORDS)))
    report.append("")
    for path in headers:
        text = read_file(path)
        rel = os.path.relpath(path, ROOT)
        hits = keyword_lines(text)
        if hits:
            report.append("### `{0}`".format(rel))
            report.append("")
            report.append("```text")
            for ln, line in hits:
                report.append("{0}: {1}".format(ln, line))
            report.append("```")
            report.append("")

    report.append("## G14A interpretation checklist")
    report.append("")
    report.append("Use this report to decide the next step:")
    report.append("")
    report.append("```text")
    report.append("1. If ggml_backend_i has graph/compute callbacks, create a compile-only G14B probe")
    report.append("   that populates exactly those fields with stubs matching this checkout.")
    report.append("2. If graph compute lives elsewhere, e.g. backend device or scheduler structs,")
    report.append("   inspect the listed raw structs before wiring any callback.")
    report.append("3. Do not assume newer upstream fields such as get_default_buffer_type exist in")
    report.append("   ggml_backend_i; this checkout already proved that assumption false in G12A.")
    report.append("4. Keep G13 backend_dispatch_op as the stable fallback until the real interface")
    report.append("   shape has a compile-checked adapter.")
    report.append("```")
    report.append("")

    write_file(REPORT, "\n".join(report) + "\n")
    write_file(SNAPSHOT, "\n".join(snapshot_parts) + "\n")

    print("G14A backend API inspection complete")
    print("report:   " + REPORT)
    print("snapshot: " + SNAPSHOT)
    print("")
    if backend_i_entries:
        for fe, st, fields in backend_i_entries:
            print("ggml_backend_i found in {0} lines {1}-{2}".format(fe["rel"], st["start_line"], st["end_line"]))
            for field in fields:
                print("  - " + field["name"])
    else:
        print("ggml_backend_i not found")


if __name__ == "__main__":
    main()
