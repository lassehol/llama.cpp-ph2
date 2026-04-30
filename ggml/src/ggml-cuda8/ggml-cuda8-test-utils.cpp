// ggml/src/ggml-cuda8/ggml-cuda8-test-utils.cpp
//
// Local smoke/test helpers for ggml-cuda8.
// Not part of the runtime backend API.

#include "ggml-cuda8-test-utils.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <float.h>

void ggml_cuda8_test_setup_f32_vector(
    ggml_tensor & t,
    int64_t n,
    void * data
) {
    std::memset(&t, 0, sizeof(t));

    t.type = GGML_TYPE_F32;

    t.ne[0] = n;
    t.ne[1] = 1;
    t.ne[2] = 1;
    t.ne[3] = 1;

    t.nb[0] = sizeof(float);
    t.nb[1] = (size_t) n * sizeof(float);
    t.nb[2] = t.nb[1];
    t.nb[3] = t.nb[1];

    t.data = data;
}



void ggml_cuda8_test_setup_f32_matrix_2d(
    ggml_tensor & t,
    int64_t cols,
    int64_t rows,
    void * data
) {
    std::memset(&t, 0, sizeof(t));

    t.type = GGML_TYPE_F32;

    t.ne[0] = cols;
    t.ne[1] = rows;
    t.ne[2] = 1;
    t.ne[3] = 1;

    t.nb[0] = sizeof(float);
    t.nb[1] = (size_t) cols * sizeof(float);
    t.nb[2] = t.nb[1] * (size_t) rows;
    t.nb[3] = t.nb[2];

    t.data = data;
}



void ggml_cuda8_test_setup_f32_scalar(
    ggml_tensor & t,
    float * scalar
) {
    std::memset(&t, 0, sizeof(t));

    t.type = GGML_TYPE_F32;

    t.ne[0] = 1;
    t.ne[1] = 1;
    t.ne[2] = 1;
    t.ne[3] = 1;

    t.nb[0] = sizeof(float);
    t.nb[1] = sizeof(float);
    t.nb[2] = sizeof(float);
    t.nb[3] = sizeof(float);

    t.data = scalar;
}



void ggml_cuda8_test_setup_q8_0_matrix_2d(
    ggml_tensor & t,
    int64_t cols,
    int64_t rows,
    void * data
) {
    std::memset(&t, 0, sizeof(t));

    const int blocks_per_row =
        (int) ((cols + GGML_CUDA8_QK8_0 - 1) / GGML_CUDA8_QK8_0);

    const size_t block_sz = sizeof(ggml_cuda8_q8_0_block);

    t.type = GGML_TYPE_Q8_0;

    t.ne[0] = cols;
    t.ne[1] = rows;
    t.ne[2] = 1;
    t.ne[3] = 1;

    t.nb[0] = block_sz;
    t.nb[1] = (size_t) blocks_per_row * block_sz;
    t.nb[2] = t.nb[1] * (size_t) rows;
    t.nb[3] = t.nb[2];

    t.data = data;
}



bool ggml_cuda8_test_check_exact(
    const std::vector<float> & got,
    const std::vector<float> & ref,
    const char * label
) {
    if (got.size() != ref.size()) {
        std::fprintf(stderr, "%s: size mismatch\n", label);
        return false;
    }

    for (size_t i = 0; i < got.size(); ++i) {
        if (got[i] != ref[i]) {
            std::fprintf(stderr,
                "%s: mismatch i=%zu got=%f expected=%f\n",
                label, i, got[i], ref[i]);
            return false;
        }
    }

    return true;
}



bool ggml_cuda8_test_check_close(
    const std::vector<float> & got,
    const std::vector<float> & ref,
    double abs_tol,
    double rel_tol,
    const char * label
) {
    double max_abs_err = 0.0;
    double max_rel_err = 0.0;

    for (size_t i = 0; i < got.size(); ++i) {
        const double g = (double) got[i];
        const double r = (double) ref[i];
        const double diff = g - r;
        const double abs_err = diff < 0.0 ? -diff : diff;
        const double denom = std::fabs(r) > 1e-9 ? std::fabs(r) : 1e-9;
        const double rel_err = abs_err / denom;

        if (abs_err > max_abs_err) max_abs_err = abs_err;
        if (rel_err > max_rel_err) max_rel_err = rel_err;
    }

    std::printf("%s: max_abs_err=%.9g max_rel_err=%.9g\n",
        label, max_abs_err, max_rel_err);

    if (max_abs_err > abs_tol && max_rel_err > rel_tol) {
        std::fprintf(stderr,
            "%s: FAIL abs=%g rel=%g\n",
            label, max_abs_err, max_rel_err);
        return false;
    }

    return true;
}


