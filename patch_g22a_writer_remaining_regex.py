#!/usr/bin/env python3
# Patch remaining fragile G22A_writer.py replacements to whitespace-tolerant regex.
# Python 3.5-compatible.

import time
from pathlib import Path

p = Path("G22A_writer.py")
s = p.read_text()

backup = p.with_name(p.name + ".remaining-regex-backup-" + str(int(time.time())))
backup.write_text(s)
print("backup", backup)

if "import re\n" not in s:
    s = s.replace("import os\nimport time\n", "import os\nimport re\nimport time\n")


def replace_labeled_block(src, label, new_code):
    marker = 's = replace_once(s, old, new, "{}")'.format(label)
    end = src.find(marker)
    if end < 0:
        raise RuntimeError("could not find replace_once marker for label: " + label)

    start = src.rfind("old = '''", 0, end)
    if start < 0:
        raise RuntimeError("could not find old-block start for label: " + label)

    end = end + len(marker)
    return src[:start] + new_code + src[end:]


offset_code = r'''# Add an intermediate biased tensor and move y after it.
offset_pat = (
    r'    const size_t off_Aq\s*=\s*0;\n'
    r'    const size_t off_x\s*=\s*8192;\n'
    r'    const size_t off_h\s*=\s*12288;\n'
    r'    const size_t off_residual\s*=\s*16384;\n'
    r'    const size_t off_y\s*=\s*20480;\n'
    r'    const size_t total_size\s*=\s*24576;\n'
)

offset_repl = (
    '    const size_t off_Aq       = 0;\n'
    '    const size_t off_x        = 8192;\n'
    '    const size_t off_h        = 12288;\n'
    '    const size_t off_residual = 16384;\n'
    '    const size_t off_biased   = 20480;\n'
    '    const size_t off_y        = 24576;\n'
    '    const size_t total_size = 28672;\n'
)

s_new, n = re.subn(offset_pat, offset_repl, s, count=1)
if n != 1:
    raise RuntimeError("could not patch offset block with regex")
s = s_new
'''

force_code = r'''# Add explicit layout for the biased intermediate tensor.
force_pat = (
    r'    force_2d_q8_0_data_layout\(Aq, cols, rows, base_u8 \+ off_Aq\);\n'
    r'    force_1d_f32_data_layout\(x,\s*cols, base_u8 \+ off_x\);\n'
    r'    force_1d_f32_data_layout\(h,\s*rows, base_u8 \+ off_h\);\n'
    r'    force_1d_f32_data_layout\(residual,\s*rows, base_u8 \+ off_residual\);\n'
    r'    force_1d_f32_data_layout\(y,\s*rows, base_u8 \+ off_y\);\n'
)

force_repl = (
    '    force_2d_q8_0_data_layout(Aq, cols, rows, base_u8 + off_Aq);\n'
    '    force_1d_f32_data_layout(x,        cols, base_u8 + off_x);\n'
    '    force_1d_f32_data_layout(h,        rows, base_u8 + off_h);\n'
    '    force_1d_f32_data_layout(residual, rows, base_u8 + off_residual);\n'
    '    force_1d_f32_data_layout(biased,   rows, base_u8 + off_biased);\n'
    '    force_1d_f32_data_layout(y,        rows, base_u8 + off_y);\n'
)

s_new, n = re.subn(force_pat, force_repl, s, count=1)
if n != 1:
    raise RuntimeError("could not patch force layouts with regex")
s = s_new
'''

init_code = r'''# Initialize the biased intermediate tensor in the CUDA8 buffer.
init_pat = (
    r'    if \(buffer->iface.init_tensor\(buffer, Aq\)\s*!= GGML_STATUS_SUCCESS \|\|\n'
    r'        buffer->iface.init_tensor\(buffer, x\)\s*!= GGML_STATUS_SUCCESS \|\|\n'
    r'        buffer->iface.init_tensor\(buffer, h\)\s*!= GGML_STATUS_SUCCESS \|\|\n'
    r'        buffer->iface.init_tensor\(buffer, residual\)\s*!= GGML_STATUS_SUCCESS \|\|\n'
    r'        buffer->iface.init_tensor\(buffer, y\)\s*!= GGML_STATUS_SUCCESS\) \{'
)

init_repl = (
    '    if (buffer->iface.init_tensor(buffer, Aq)       != GGML_STATUS_SUCCESS ||\n'
    '        buffer->iface.init_tensor(buffer, x)        != GGML_STATUS_SUCCESS ||\n'
    '        buffer->iface.init_tensor(buffer, h)        != GGML_STATUS_SUCCESS ||\n'
    '        buffer->iface.init_tensor(buffer, residual) != GGML_STATUS_SUCCESS ||\n'
    '        buffer->iface.init_tensor(buffer, biased)   != GGML_STATUS_SUCCESS ||\n'
    '        buffer->iface.init_tensor(buffer, y)        != GGML_STATUS_SUCCESS) {'
)

s_new, n = re.subn(init_pat, init_repl, s, count=1)
if n != 1:
    raise RuntimeError("could not patch init tensors with regex")
s = s_new
'''

residency_code = r'''# Check residency of the biased intermediate tensor.
residency_pat = (
    r'    if \(!expect_resident\("Aq",\s*Aq,\s*bytes_Aq, buffer, off_Aq\)\s*\|\|\n'
    r'        !expect_resident\("x",\s*x,\s*bytes_x,\s*buffer, off_x\)\s*\|\|\n'
    r'        !expect_resident\("h",\s*h,\s*bytes_y,\s*buffer, off_h\)\s*\|\|\n'
    r'        !expect_resident\("residual",\s*residual,\s*bytes_y,\s*buffer, off_residual\)\s*\|\|\n'
    r'        !expect_resident\("y",\s*y,\s*bytes_y,\s*buffer, off_y\)\) \{'
)

residency_repl = (
    '    if (!expect_resident("Aq",       Aq,       bytes_Aq, buffer, off_Aq)       ||\n'
    '        !expect_resident("x",        x,        bytes_x,  buffer, off_x)        ||\n'
    '        !expect_resident("h",        h,        bytes_y,  buffer, off_h)        ||\n'
    '        !expect_resident("residual", residual, bytes_y,  buffer, off_residual) ||\n'
    '        !expect_resident("biased",   biased,   bytes_y,  buffer, off_biased)   ||\n'
    '        !expect_resident("y",        y,        bytes_y,  buffer, off_y)) {'
)

s_new, n = re.subn(residency_pat, residency_repl, s, count=1)
if n != 1:
    raise RuntimeError("could not patch residency checks with regex")
s = s_new
'''

host_ref_code = r'''# Change CPU reference from add-only output to add followed by softmax.
host_ref_pat = (
    r'    std::vector<float> h_ref\(rows, 0\.0f\);\n'
    r'    std::vector<float> residual_host\(rows, 0\.0f\);\n'
    r'    std::vector<float> residual_after\(rows, 0\.0f\);\n'
    r'    std::vector<float> y_ref\(rows, 0\.0f\);\n'
    r'    std::vector<float> y_out\(rows, 0\.0f\);\n\n'
    r'    fill_f32_matrix\(A_f32, rows, cols\);\n'
    r'    pack_q8_0\(A_f32, Aq_host, rows, cols\);\n'
    r'    fill_x\(x_host\);\n'
    r'    fill_residual\(residual_host\);\n'
    r'    cpu_ref_q8_mmv\(Aq_host, x_host, h_ref, rows, cols\);\n'
    r'    add_ref\(h_ref, residual_host, y_ref\);\n'
)

host_ref_repl = (
    '    std::vector<float> h_ref(rows, 0.0f);\n'
    '    std::vector<float> residual_host(rows, 0.0f);\n'
    '    std::vector<float> residual_after(rows, 0.0f);\n'
    '    std::vector<float> biased_ref(rows, 0.0f);\n'
    '    std::vector<float> y_ref(rows, 0.0f);\n'
    '    std::vector<float> y_out(rows, 0.0f);\n\n'
    '    fill_f32_matrix(A_f32, rows, cols);\n'
    '    pack_q8_0(A_f32, Aq_host, rows, cols);\n'
    '    fill_x(x_host);\n'
    '    fill_residual(residual_host);\n'
    '    cpu_ref_q8_mmv(Aq_host, x_host, h_ref, rows, cols);\n'
    '    add_ref(h_ref, residual_host, biased_ref);\n'
    '    softmax_ref(biased_ref, y_ref);\n'
)

s_new, n = re.subn(host_ref_pat, host_ref_repl, s, count=1)
if n != 1:
    raise RuntimeError("could not patch host reference setup with regex")
s = s_new
'''

for label, code in [
    ("offset block", offset_code),
    ("force layouts", force_code),
    ("init tensors", init_code),
    ("residency checks", residency_code),
    ("host reference setup", host_ref_code),
]:
    s = replace_labeled_block(s, label, code)

p.write_text(s)
print("patched", p)
