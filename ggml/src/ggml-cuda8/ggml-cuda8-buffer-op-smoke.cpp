// ggml/src/ggml-cuda8/ggml-cuda8-buffer-op-smoke.cpp
//
// G6 backend-buffer + op path smoke test.
//
// This validates the next integration layer:
//
//   host Q8_0 matrix + host F32 vector
//       -> ggml_cuda8_backend_buffer allocation
//       -> backend-buffer upload
//       -> CUDA8 Q8_0 mul_mat using device buffer base pointers
//       -> backend-buffer download
//       -> compare CPU Q8_0 reference
//
// This still does NOT use real ggml_backend_buffer_t.
// It proves that the G5 buffer abstraction can carry the G0/G4 op path.

#include "ggml-cuda8-backend-buffer.h"
#include "ggml-cuda8-mulmat.cuh"
#include "q8_0-mmv.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdint.h>
#include <vector>

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

    const uint32_t f_sign = h_sign << 16;
    uint32_t f_exp;
    uint32_t f_sig;

    union { uint32_t u; float f; } out;

    if (h_exp == 0) {
        if (h_sig == 0) {
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

        out.u = f_sign | f_exp | f_sig;
        return out.f;
    }

    if (h_exp == 0x7C00u) {
        f_exp = 0xFFu << 23;
        f_sig = h_sig << 13;

        out.u = f_sign | f_exp | f_sig;
        return out.f;
    }

    const int exp = (int) (h_exp >> 10) - 15;
    f_exp = (uint32_t) (exp + 127) << 23;
    f_sig = h_sig << 13;

    out.u = f_sign | f_exp | f_sig;
    return out.f;
}

static void fill_inputs(std::vector<float> & A, std::vector<float> & x, int rows, int cols) {
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            const int v = (r * 17 + c * 31 + 7) % 251;
            A[(size_t) r * cols + c] = ((float) v - 125.0f) * 0.003f;
        }
    }

    for (int c = 0; c < cols; ++c) {
        const int v = (c * 13 + 5) % 197;
        x[c] = ((float) v - 98.0f) * 0.004f;
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
                const int col = ib * GGML_CUDA8_QK8_0 + k;

                float v = 0.0f;
                if (col < cols) {
                    v = A[(size_t) r * cols + col];
                }

                max_abs = std::max(max_abs, std::fabs(v));
            }

            float d = 0.0f;
            float id = 0.0f;

            if (max_abs > 0.0f) {
                d  = max_abs / 127.0f;
                id = 127.0f / max_abs;
            }

            block.d = fp32_to_fp16_bits(d);

            for (int k = 0; k < GGML_CUDA8_QK8_0; ++k) {
                const int col = ib * GGML_CUDA8_QK8_0 + k;

                float v = 0.0f;
                if (col < cols) {
                    v = A[(size_t) r * cols + col];
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

static void cpu_q8_0_ref(
    const std::vector<ggml_cuda8_q8_0_block> & Aq,
    const std::vector<float> & x,
    std::vector<float> & y,
    int rows,
    int cols
) {
    const int blocks_per_row =
        (cols + GGML_CUDA8_QK8_0 - 1) / GGML_CUDA8_QK8_0;

    for (int r = 0; r < rows; ++r) {
        const ggml_cuda8_q8_0_block * row_blocks =
            &Aq[(size_t) r * blocks_per_row];

        float sum = 0.0f;

        for (int ib = 0; ib < blocks_per_row; ++ib) {
            const ggml_cuda8_q8_0_block & block = row_blocks[ib];
            const float d = fp16_bits_to_fp32(block.d);

            const int base_col = ib * GGML_CUDA8_QK8_0;
            float block_sum = 0.0f;

            for (int k = 0; k < GGML_CUDA8_QK8_0; ++k) {
                const int col = base_col + k;

                if (col < cols) {
                    block_sum += ((float) block.qs[k]) * x[col];
                }
            }

            sum += d * block_sum;
        }

        y[r] = sum;
    }
}

static void compute_error(
    const std::vector<float> & ref,
    const std::vector<float> & got,
    double & max_abs_err,
    double & max_rel_err
) {
    max_abs_err = 0.0;
    max_rel_err = 0.0;

    for (size_t i = 0; i < ref.size(); ++i) {
        const double r = (double) ref[i];
        const double g = (double) got[i];

        const double abs_err = std::fabs(g - r);
        const double denom = std::max(1e-9, std::fabs(r));
        const double rel_err = abs_err / denom;

        max_abs_err = std::max(max_abs_err, abs_err);
        max_rel_err = std::max(max_rel_err, rel_err);
    }
}

static bool run_one_case(int rows, int cols) {
    std::printf("\nG6 backend-buffer op smoke: rows=%d cols=%d\n", rows, cols);

    const int device = 0;

    const int blocks_per_row =
        (cols + GGML_CUDA8_QK8_0 - 1) / GGML_CUDA8_QK8_0;

    const size_t n_Aq = (size_t) rows * blocks_per_row;
    const size_t n_x  = (size_t) cols;
    const size_t n_y  = (size_t) rows;

    const size_t bytes_Aq = n_Aq * sizeof(ggml_cuda8_q8_0_block);
    const size_t bytes_x  = n_x  * sizeof(float);
    const size_t bytes_y  = n_y  * sizeof(float);

    std::printf("blocks_per_row=%d | Aq=%.2f KiB | x=%.2f KiB | y=%.2f KiB\n",
        blocks_per_row,
        (double) bytes_Aq / 1024.0,
        (double) bytes_x  / 1024.0,
        (double) bytes_y  / 1024.0
    );

    std::vector<float> A((size_t) rows * cols);
    std::vector<float> x(n_x);
    std::vector<float> y_ref(n_y, 0.0f);
    std::vector<float> y_gpu(n_y, 0.0f);
    std::vector<ggml_cuda8_q8_0_block> Aq;

    fill_inputs(A, x, rows, cols);
    pack_q8_0(A, Aq, rows, cols);
    cpu_q8_0_ref(Aq, x, y_ref, rows, cols);

    ggml_cuda8_backend_buffer * b_Aq = NULL;
    ggml_cuda8_backend_buffer * b_x  = NULL;
    ggml_cuda8_backend_buffer * b_y  = NULL;

    if (ggml_cuda8_backend_buffer_alloc(device, bytes_Aq, &b_Aq) != 0) {
        return false;
    }

    if (ggml_cuda8_backend_buffer_alloc(device, bytes_x, &b_x) != 0) {
        ggml_cuda8_backend_buffer_free(b_Aq);
        return false;
    }

    if (ggml_cuda8_backend_buffer_alloc(device, bytes_y, &b_y) != 0) {
        ggml_cuda8_backend_buffer_free(b_Aq);
        ggml_cuda8_backend_buffer_free(b_x);
        return false;
    }

    bool ok = true;

    if (ggml_cuda8_backend_buffer_upload(b_Aq, 0, &Aq[0], bytes_Aq) != 0) {
        std::fprintf(stderr, "ggml-cuda8-buffer-op-smoke: Aq upload failed\n");
        ok = false;
    }

    if (ok && ggml_cuda8_backend_buffer_upload(b_x, 0, &x[0], bytes_x) != 0) {
        std::fprintf(stderr, "ggml-cuda8-buffer-op-smoke: x upload failed\n");
        ok = false;
    }

    if (ok && ggml_cuda8_backend_buffer_clear(b_y, 0) != 0) {
        std::fprintf(stderr, "ggml-cuda8-buffer-op-smoke: y clear failed\n");
        ok = false;
    }

    if (ok) {
        ggml_cuda8_q8_0_block * d_Aq =
            (ggml_cuda8_q8_0_block *) ggml_cuda8_backend_buffer_get_base(b_Aq);

        float * d_x =
            (float *) ggml_cuda8_backend_buffer_get_base(b_x);

        float * d_y =
            (float *) ggml_cuda8_backend_buffer_get_base(b_y);

        if (d_Aq == NULL || d_x == NULL || d_y == NULL) {
            std::fprintf(stderr, "ggml-cuda8-buffer-op-smoke: null device base pointer\n");
            ok = false;
        } else {
            const int rc = ggml_cuda8_mul_mat_q8_0_f32(d_Aq, d_x, d_y, rows, cols);

            if (rc != 0) {
                std::fprintf(stderr, "ggml-cuda8-buffer-op-smoke: mul_mat rc=%d\n", rc);
                ok = false;
            }
        }
    }

    if (ok && ggml_cuda8_backend_buffer_download(b_y, 0, &y_gpu[0], bytes_y) != 0) {
        std::fprintf(stderr, "ggml-cuda8-buffer-op-smoke: y download failed\n");
        ok = false;
    }

    ggml_cuda8_backend_buffer_free(b_Aq);
    ggml_cuda8_backend_buffer_free(b_x);
    ggml_cuda8_backend_buffer_free(b_y);

    if (!ok) {
        return false;
    }

    double max_abs_err = 0.0;
    double max_rel_err = 0.0;

    compute_error(y_ref, y_gpu, max_abs_err, max_rel_err);

    std::printf("max_abs_err vs CPU Q8_0 ref: %.9g\n", max_abs_err);
    std::printf("max_rel_err vs CPU Q8_0 ref: %.9g\n", max_rel_err);

    const double abs_tol = 2e-4;
    const double rel_tol = 2e-4;

    if (max_abs_err > abs_tol && max_rel_err > rel_tol) {
        std::fprintf(stderr,
            "ggml-cuda8-buffer-op-smoke: FAIL rows=%d cols=%d abs=%g rel=%g\n",
            rows, cols, max_abs_err, max_rel_err);
        return false;
    }

    std::printf("PASS\n");
    return true;
}

int main(int argc, char ** argv) {
    (void) argc;
    (void) argv;

    std::printf("ggml-cuda8-buffer-op-smoke: starting\n");
    std::printf("buffer type: %s | alignment=%zu\n",
        ggml_cuda8_backend_buffer_type_name(),
        ggml_cuda8_backend_buffer_alignment()
    );

    bool ok = true;

    ok = ok && run_one_case(4,    32);
    ok = ok && run_one_case(128,  64);
    ok = ok && run_one_case(256, 128);
    ok = ok && run_one_case(512, 512);

    if (!ok) {
        std::fprintf(stderr, "ggml-cuda8-buffer-op-smoke: FAILED\n");
        return 1;
    }

    std::printf("\n");
    std::printf("ggml-cuda8-buffer-op-smoke: SUCCESS\n");
    return 0;
}
