// ggml/src/ggml-cuda8/ggml-cuda8-backend.h
//
// G3 CUDA8 backend identity skeleton.

#ifndef GGML_CUDA8_BACKEND_H
#define GGML_CUDA8_BACKEND_H

#ifdef __cplusplus
extern "C" {
#endif

#define GGML_CUDA8_BACKEND_NAME_MAX 128

struct ggml_cuda8_device_info {
    int  index;
    int  cc_major;
    int  cc_minor;
    int  warp_size;
    int  multi_processor_count;
    int  clock_rate_khz;
    int  memory_clock_rate_khz;
    int  memory_bus_width_bits;
    unsigned long long total_global_mem;
    char name[GGML_CUDA8_BACKEND_NAME_MAX];
};

const char * ggml_cuda8_backend_name(void);
int ggml_cuda8_backend_available(void);
int ggml_cuda8_backend_get_device_count(void);

int ggml_cuda8_backend_get_device_info(
    int index,
    struct ggml_cuda8_device_info * info
);

int ggml_cuda8_backend_print_devices(void);

#ifdef __cplusplus
}
#endif

#endif // GGML_CUDA8_BACKEND_H
