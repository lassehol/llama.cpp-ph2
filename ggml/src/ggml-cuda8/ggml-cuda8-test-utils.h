// ggml/src/ggml-cuda8/ggml-cuda8-test-utils.h
//
// Local smoke/test helpers for ggml-cuda8.
// Not part of the runtime backend API.

#ifndef GGML_CUDA8_TEST_UTILS_H
#define GGML_CUDA8_TEST_UTILS_H

#include "ggml.h"
#include "q8_0-mmv.cuh"

#include <stdint.h>
#include <vector>

void ggml_cuda8_test_setup_f32_vector(
    ggml_tensor & t,
    int64_t n,
    void * data
);

void ggml_cuda8_test_setup_f32_matrix_2d(
    ggml_tensor & t,
    int64_t cols,
    int64_t rows,
    void * data
);

void ggml_cuda8_test_setup_f32_scalar(
    ggml_tensor & t,
    float * scalar
);

void ggml_cuda8_test_setup_q8_0_matrix_2d(
    ggml_tensor & t,
    int64_t cols,
    int64_t rows,
    void * data
);

bool ggml_cuda8_test_check_exact(
    const std::vector<float> & got,
    const std::vector<float> & ref,
    const char * label
);

bool ggml_cuda8_test_check_close(
    const std::vector<float> & got,
    const std::vector<float> & ref,
    double abs_tol,
    double rel_tol,
    const char * label
);

#endif // GGML_CUDA8_TEST_UTILS_H
