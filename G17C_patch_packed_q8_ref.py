#!/usr/bin/env python3
# G17C_patch_packed_q8_ref.py
#
# Convert the G17A graph-builder Q8_0 MMV smoke from a simplified d=1.0 Q8_0
# matrix to a packed Q8_0 quantization reference:
#
#   A_f32 -> pack_q8_0(A_f32) -> Aq
#   CPU reference uses dequantized Q8_0 blocks
#   GPU path uses real ggml_mul_mat(Q8_0, F32) through graph_compute
#
# Python 3.5-compatible: no f-strings.

import os
import time

ROOT = "/workspace/notebooks/llama.cpp-ph2"
SRC = os.path.join(ROOT, "ggml/src/ggml-cuda8/ggml-cuda8-ggml-backend-graph-builder-q8_0-mmv-smoke.cpp")

with open(SRC, "r") as f:
    s = f.read()

backup = SRC + ".g17c-packed-q8-backup-" + str(int(time.time()))
with open(backup, "w") as f:
    f.write(s)

print("backup", backup)

if "pack_q8_0(" in s and "fp32_to_fp16_bits" in s and "fp16_bits_to_fp32" in s:
    print("G17C packed Q8_0 reference appears already present")
    raise SystemExit(0)

helper_block = r'''
static uint16_t fp32_to_fp16_bits(float value) {
    union { float f; uint32_t u; } in;
    in.f = value;

    const uint32_t f = in.u;
    const uint32_t sign = (f >> 16) & 0x8000u;
    int32_t exp = (int32_t) ((f >> 23) & 0xFFu) - 127 + 15;
    uint32_t mant = f & 0x007FFFFFu;

    if (exp <= 0) {
        if (exp < -10) {
            return (uint16_t) sign;
        }

        mant |= 0x00800000u;
        const int shift = 14 - exp;
        uint32_t half_mant = mant >> shift;

        if ((mant >> (shift - 1)) & 1u) {
            half_mant += 1u;
        }

        return (uint16_t) (sign | half_mant);
    }

    if (exp >= 31) {
        return (uint16_t) (sign | 0x7C00u);
    }

    uint32_t half = sign | ((uint32_t) exp << 10) | (mant >> 13);

    if (mant & 0x00001000u) {
        half += 1u;
    }

    return (uint16_t) half;
}

static float fp16_bits_to_fp32(uint16_t h) {
    const uint32_t h_exp  = h & 0x7C00u;
    const uint32_t h_sig  = h & 0x03FFu;
    const uint32_t h_sign = h & 0x8000u;

    uint32_t f_sign = h_sign << 16;
    uint32_t f_exp;
    uint32_t f_sig;

    if (h_exp == 0) {
        if (h_sig == 0) {
            union { uint32_t u; float f; } out;
            out.u = f_sign;
            return out.f;
        }

        uint32_t sig = h_sig;
        int exp = -14;

        while ((sig & 0x0400u) == 0) {
            sig <<= 1;
            --exp;
        }

        sig &= 0x03FFu;

        f_exp = (uint32_t) (exp + 127) << 23;
        f_sig = sig << 13;

        union { uint32_t u; float f; } out;
        out.u = f_sign | f_exp | f_sig;
        return out.f;
    }

    if (h_exp == 0x7C00u) {
        f_exp = 0xFFu << 23;
        f_sig = h_sig << 13;

        union { uint32_t u; float f; } out;
        out.u = f_sign | f_exp | f_sig;
        return out.f;
    }

    const int exp = (int) (h_exp >> 10) - 15;

    f_exp = (uint32_t) (exp + 127) << 23;
    f_sig = h_sig << 13;

    union { uint32_t u; float f; } out;
    out.u = f_sign | f_exp | f_sig;
    return out.f;
}

'''

include_anchor = "#include <vector>\n"
if include_anchor not in s:
    raise RuntimeError("could not find include anchor")

s = s.replace(include_anchor, include_anchor + "\n" + helper_block, 1)

old_fill_start = s.find("static void fill_q8_matrix(")
old_fill_end = s.find("static void fill_x(", old_fill_start)

if old_fill_start < 0 or old_fill_end < 0:
    raise RuntimeError("could not find fill_q8_matrix/fill_x function range")

new_fill_block = r'''static void fill_f32_matrix(
    std::vector<float> & A,
    int rows,
    int cols
) {
    A.resize((size_t) rows * cols);

    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            const int v = (r * 17 + c * 31 + 7) % 251;
            A[(size_t) r * cols + c] = ((float) v - 125.0f) * 0.003f;
        }
    }
}

static void pack_q8_0(
    const std::vector<float> & A,
    std::vector<ggml_cuda8_q8_0_block> & Aq,
    int rows,
    int cols
) {
    const int blocks_per_row =
        (cols + GGML_CUDA8_QK8_0 - 1) / GGML_CUDA8_QK8_0;

    Aq.resize((size_t) rows * blocks_per_row);

    for (int r = 0; r < rows; ++r) {
        for (int ib = 0; ib < blocks_per_row; ++ib) {
            ggml_cuda8_q8_0_block & block =
                Aq[(size_t) r * blocks_per_row + ib];

            float max_abs = 0.0f;

            for (int k = 0; k < GGML_CUDA8_QK8_0; ++k) {
                const int c = ib * GGML_CUDA8_QK8_0 + k;

                float v = 0.0f;
                if (c < cols) {
                    v = A[(size_t) r * cols + c];
                }

                max_abs = std::max(max_abs, std::fabs(v));
            }

            float d = 0.0f;
            float id = 0.0f;

            if (max_abs > 0.0f) {
                d = max_abs / 127.0f;
                id = 127.0f / max_abs;
            }

            block.d = fp32_to_fp16_bits(d);

            for (int k = 0; k < GGML_CUDA8_QK8_0; ++k) {
                const int c = ib * GGML_CUDA8_QK8_0 + k;

                float v = 0.0f;
                if (c < cols) {
                    v = A[(size_t) r * cols + c];
                }

                int q = 0;

                if (max_abs > 0.0f) {
                    q = (int) std::floor(v * id + (v >= 0.0f ? 0.5f : -0.5f));
                }

                if (q > 127) {
                    q = 127;
                }

                if (q < -127) {
                    q = -127;
                }

                block.qs[k] = (int8_t) q;
            }
        }
    }
}

'''

s = s[:old_fill_start] + new_fill_block + s[old_fill_end:]

old_d = '''            // block.d is 1.0f by construction.
            const float d = 1.0f;
'''
new_d = '''            const float d = fp16_bits_to_fp32(block.d);
'''

if old_d not in s:
    raise RuntimeError("could not find d=1.0 CPU reference block")

s = s.replace(old_d, new_d, 1)

old_main = '''    std::vector<ggml_cuda8_q8_0_block> Aq_host;
    std::vector<float> x_host(cols);
    std::vector<float> y_ref(rows, 0.0f);
    std::vector<float> y_out(rows, 0.0f);

    fill_q8_matrix(Aq_host, rows, cols);
    fill_x(x_host);
    cpu_ref_q8_mmv(Aq_host, x_host, y_ref, rows, cols);
'''

new_main = '''    std::vector<float> A_f32;
    std::vector<ggml_cuda8_q8_0_block> Aq_host;
    std::vector<float> x_host(cols);
    std::vector<float> y_ref(rows, 0.0f);
    std::vector<float> y_out(rows, 0.0f);

    fill_f32_matrix(A_f32, rows, cols);
    pack_q8_0(A_f32, Aq_host, rows, cols);
    fill_x(x_host);
    cpu_ref_q8_mmv(Aq_host, x_host, y_ref, rows, cols);
'''

if old_main not in s:
    raise RuntimeError("could not find main input setup block")

s = s.replace(old_main, new_main, 1)

s = s.replace(
    "real graph-builder Q8_0xF32_VEC max_abs_err=",
    "real graph-builder packed Q8_0xF32_VEC max_abs_err="
)

s = s.replace(
    "real graph-builder Q8_0xF32_VEC verification PASS",
    "real graph-builder packed Q8_0xF32_VEC verification PASS"
)

s = s.replace(
    "real graph-builder Q8_0xF32_VEC verification FAIL",
    "real graph-builder packed Q8_0xF32_VEC verification FAIL"
)

with open(SRC, "w") as f:
    f.write(s)

print("patched", SRC)
print("G17C: graph-builder Q8_0 MMV smoke now uses packed Q8_0 quantization reference")
