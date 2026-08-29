cmake_minimum_required(VERSION 3.5)

project(ggml_cuda8_smoke C CXX)

# CMake 3.5 + CUDA 8 compatible path.
# Do not use enable_language(CUDA) here.
find_package(CUDA REQUIRED)

message(STATUS "ggml-cuda8: CUDA toolkit root: ${CUDA_TOOLKIT_ROOT_DIR}")
message(STATUS "ggml-cuda8: CUDA nvcc: ${CUDA_NVCC_EXECUTABLE}")
message(STATUS "ggml-cuda8: CUDA version: ${CUDA_VERSION}")

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Prefer dynamic cudart to avoid unnecessary static runtime baggage.
# Supported by old FindCUDA.
set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)

# Fermi / GTX 560 class:
#   GTX 560 = compute capability 2.1
#
# CUDA 8 accepts this but warns that sm_21 is deprecated, which is expected.
set(CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS}
    -gencode arch=compute_20,code=sm_21
    -Wno-deprecated-gpu-targets
    -Xcompiler=-fPIC
)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CUDA_INCLUDE_DIRS}
)

# ---------------------------------------------------------------------------
# Reusable CUDA8 kernel/static support library.
#
# Keep ggml-cuda8.cu out of this library for now because it is probe/smoke
# orchestration code, not a reusable kernel layer.
# ---------------------------------------------------------------------------

cuda_add_library(ggml-cuda8-kernels STATIC
    buffer.cu
    mmv.cu
    q8_0-mmv.cu
    ggml-cuda8-mulmat.cu
    ggml-cuda8-backend.cpp
    ggml-cuda8-op-mul-mat.cpp
    ggml-cuda8-backend-buffer.cpp
    ggml-cuda8-context.cpp
    ggml-cuda8-dispatch.cpp
    ggml-cuda8-ggml-buffer.cpp
    ggml-cuda8-ggml-buffer-registry.cpp
    ggml-cuda8-ggml-backend.cpp
    ggml-cuda8-backend-reg.cpp
    ggml-cuda8-cpy.cpp
    ggml-cuda8-add.cpp
    ggml-cuda8-add.cu
    ggml-cuda8-scalar.cpp
    ggml-cuda8-scalar.cu
    ggml-cuda8-reduce.cpp
    ggml-cuda8-reduce.cu
    ggml-cuda8-softmax.cpp
    ggml-cuda8-softmax.cu
    ggml-cuda8-rms-norm.cu
    ggml-cuda8-mul.cu
    ggml-cuda8-rope.cu
    cpy.cu
    ggml-cuda8-diagmask.cu
    ggml-cuda8-getrows.cu
    ggml-cuda8-q4k.cu
    ggml-cuda8-q6k.cu
)
set_target_properties(ggml-cuda8-kernels PROPERTIES POSITION_INDEPENDENT_CODE ON)

# G36A: Define GGML_USE_CUDA8 for backend registration
target_compile_definitions(ggml-cuda8-kernels PUBLIC GGML_USE_CUDA8)

target_link_libraries(ggml-cuda8-kernels
    ${CUDA_LIBRARIES}
)

# ---------------------------------------------------------------------------
# Smoke executable:
#   - device probe
#   - buffer roundtrip
#   - F32 MMV roundtrip
#   - Q8_0 MMV roundtrip
# ---------------------------------------------------------------------------

cuda_add_executable(ggml-cuda8-smoke
    smoke-main.cpp
    ggml-cuda8.cu
)

target_link_libraries(ggml-cuda8-smoke
    ggml-cuda8-kernels
    ${CUDA_LIBRARIES}
)

# ---------------------------------------------------------------------------
# F32 MMV benchmark executable:
#   - CPU reference
#   - GPU naive
#   - GPU block
#   - GPU selected
# ---------------------------------------------------------------------------

cuda_add_executable(ggml-cuda8-mmv-bench
    mmv-bench.cpp
)

target_link_libraries(ggml-cuda8-mmv-bench
    ggml-cuda8-kernels
    ${CUDA_LIBRARIES}
)

# ---------------------------------------------------------------------------
# Q8_0 MMV benchmark executable:
#   - CPU F32 reference
#   - CPU Q8_0 reference
#   - GPU Q8_0 qblock
#   - GPU Q8_0 col_parallel
#   - GPU Q8_0 selected
# ---------------------------------------------------------------------------

cuda_add_executable(ggml-cuda8-q8_0-bench
    q8_0-bench.cpp
)

target_link_libraries(ggml-cuda8-q8_0-bench
    ggml-cuda8-kernels
    ${CUDA_LIBRARIES}
)

# ---------------------------------------------------------------------------
# G0 GGML-shaped smoke executable:
#   Q8_0 matrix [rows, cols] x F32 vector [cols] -> F32 vector [rows]
# ---------------------------------------------------------------------------

cuda_add_executable(ggml-cuda8-ggml-smoke
    ggml-cuda8-ggml-smoke.cpp
)

target_link_libraries(ggml-cuda8-ggml-smoke
    ggml-cuda8-kernels
    ${CUDA_LIBRARIES}
)

# ---------------------------------------------------------------------------
# G3 backend identity smoke executable:
#   - backend name
#   - CUDA availability
#   - device count
#   - device info
# ---------------------------------------------------------------------------

cuda_add_executable(ggml-cuda8-backend-smoke
    ggml-cuda8-backend-smoke.cpp
)

target_link_libraries(ggml-cuda8-backend-smoke
    ggml-cuda8-kernels
    ${CUDA_LIBRARIES}
)

# ---------------------------------------------------------------------------
# G4 GGML-tensor-facing op smoke executable:
#   - creates real ggml_tensor metadata objects manually
#   - validates Q8_0 x F32 vector MUL_MAT adapter
#   - intentionally does not link full ggml library yet
# ---------------------------------------------------------------------------

cuda_add_executable(ggml-cuda8-op-smoke
    ggml-cuda8-op-smoke.cpp
)

target_link_libraries(ggml-cuda8-op-smoke
    ggml-cuda8-kernels
    ${CUDA_LIBRARIES}
)

# ---------------------------------------------------------------------------
# G5 backend buffer layout smoke executable:
#   - backend buffer allocation/free
#   - upload/download
#   - offset upload/download
#   - clear
# ---------------------------------------------------------------------------

cuda_add_executable(ggml-cuda8-buffer-smoke
    ggml-cuda8-buffer-smoke.cpp
)

target_link_libraries(ggml-cuda8-buffer-smoke
    ggml-cuda8-kernels
    ${CUDA_LIBRARIES}
)

# ---------------------------------------------------------------------------
# G6 backend-buffer + op smoke executable:
#   - allocates Q8_0/F32/dst through ggml_cuda8_backend_buffer
#   - runs CUDA8 Q8_0 mul_mat from device buffer bases
# ---------------------------------------------------------------------------

cuda_add_executable(ggml-cuda8-buffer-op-smoke
    ggml-cuda8-buffer-op-smoke.cpp
)

target_link_libraries(ggml-cuda8-buffer-op-smoke
    ggml-cuda8-kernels
    ${CUDA_LIBRARIES}
)

# ---------------------------------------------------------------------------
# G7 backend context smoke executable:
#   - context create/destroy
#   - context buffer allocation/free
#   - context Q8_0 MUL_MAT using backend buffers
# ---------------------------------------------------------------------------

cuda_add_executable(ggml-cuda8-context-smoke
    ggml-cuda8-context-smoke.cpp
)

target_link_libraries(ggml-cuda8-context-smoke
    ggml-cuda8-kernels
    ${CUDA_LIBRARIES}
)

# ---------------------------------------------------------------------------
# G8B dispatcher smoke executable:
#   - uses ggml_tensor metadata (manual)
#   - dispatches supported ops via cuda8 context + backend buffers
# ---------------------------------------------------------------------------

cuda_add_executable(ggml-cuda8-dispatch-smoke
    ggml-cuda8-dispatch-smoke.cpp
)

target_link_libraries(ggml-cuda8-dispatch-smoke
    ggml-cuda8-kernels
    ${CUDA_LIBRARIES}
)

# ---------------------------------------------------------------------------
# G9B-1 CPY smoke executable:
#   - dispatches GGML_CUDA8_OP_CPY_F32 on contiguous F32 tensors
# ---------------------------------------------------------------------------

cuda_add_executable(ggml-cuda8-cpy-smoke
    ggml-cuda8-cpy-smoke.cpp
)

target_link_libraries(ggml-cuda8-cpy-smoke
    ggml-cuda8-kernels
    ${CUDA_LIBRARIES}
)


cuda_add_executable(ggml-cuda8-add-smoke
    ggml-cuda8-add-smoke.cpp
)
target_link_libraries(ggml-cuda8-add-smoke
    ggml-cuda8-kernels
    ${CUDA_LIBRARIES}
)


cuda_add_executable(ggml-cuda8-scalar-smoke
    ggml-cuda8-scalar-smoke.cpp
)

target_link_libraries(ggml-cuda8-scalar-smoke
    ggml-cuda8-kernels
    ${CUDA_LIBRARIES}
)


cuda_add_executable(ggml-cuda8-reduce-smoke
    ggml-cuda8-reduce-smoke.cpp
)

target_link_libraries(ggml-cuda8-reduce-smoke
    ggml-cuda8-kernels
    ${CUDA_LIBRARIES}
)


cuda_add_executable(ggml-cuda8-reduce-max-smoke
    ggml-cuda8-reduce-max-smoke.cpp
)

target_link_libraries(ggml-cuda8-reduce-max-smoke
    ggml-cuda8-kernels
    ${CUDA_LIBRARIES}
)


cuda_add_executable(ggml-cuda8-softmax-smoke
    ggml-cuda8-softmax-smoke.cpp
)

target_link_libraries(ggml-cuda8-softmax-smoke
    ggml-cuda8-kernels
    ${CUDA_LIBRARIES}
)


# ---------------------------------------------------------------------------
# G10 combined dispatcher smoke executable:
#   - CPY
#   - ADD
#   - scalar ADD/MUL
#   - reduce sum/max
#   - softmax
#   - Q8_0 MUL_MAT
# ---------------------------------------------------------------------------

cuda_add_executable(ggml-cuda8-dispatch-all-smoke
    ggml-cuda8-dispatch-all-smoke.cpp
    ggml-cuda8-test-utils.cpp
)

target_link_libraries(ggml-cuda8-dispatch-all-smoke
    ggml-cuda8-kernels
    ${CUDA_LIBRARIES}
)

# ---------------------------------------------------------------------------
# G11A-0 GGML backend API compile probe:
#   - includes ggml-backend.h and ggml-backend-impl.h
#   - does not link full ggml-base
# ---------------------------------------------------------------------------

cuda_add_executable(ggml-cuda8-ggml-api-probe
    ggml-cuda8-ggml-api-probe.cpp
)

target_link_libraries(ggml-cuda8-ggml-api-probe
    ggml-cuda8-kernels
    ${CUDA_LIBRARIES}
)

# ---------------------------------------------------------------------------
# G11A-1 minimal ggml_backend_buffer_t wrapper smoke:
#   - creates CUDA8 ggml backend buffer type
#   - allocates ggml_backend_buffer_t
#   - set/get/memset/clear via buffer iface
# ---------------------------------------------------------------------------

cuda_add_executable(ggml-cuda8-ggml-buffer-smoke
    ggml-cuda8-ggml-buffer-smoke.cpp
)

target_link_libraries(ggml-cuda8-ggml-buffer-smoke
    ggml-cuda8-kernels
    ${CUDA_LIBRARIES}
)

# ---------------------------------------------------------------------------
# G11A-2 tensor offset smoke for minimal ggml_backend_buffer_t wrapper:
#   - multiple tensors inside one CUDA8 ggml_backend_buffer_t
#   - set/get at tensor data offsets
#   - memset one tensor without changing another
#   - partial tensor-relative writes
# ---------------------------------------------------------------------------

cuda_add_executable(ggml-cuda8-ggml-buffer-offset-smoke
    ggml-cuda8-ggml-buffer-offset-smoke.cpp
)

target_link_libraries(ggml-cuda8-ggml-buffer-offset-smoke
    ggml-cuda8-kernels
    ${CUDA_LIBRARIES}
)

# ---------------------------------------------------------------------------
# G11A-3 GGML buffer wrapper + dispatcher interop smoke:
#   - stores tensors in one CUDA8 ggml_backend_buffer_t
#   - stages through set/get tensor
#   - runs existing dispatcher ADD_F32
#   - writes dispatcher result back into CUDA8 ggml buffer
# ---------------------------------------------------------------------------

cuda_add_executable(ggml-cuda8-ggml-buffer-dispatch-smoke
    ggml-cuda8-ggml-buffer-dispatch-smoke.cpp
)

target_link_libraries(ggml-cuda8-ggml-buffer-dispatch-smoke
    ggml-cuda8-kernels
    ${CUDA_LIBRARIES}
)

# ---------------------------------------------------------------------------
# G11A-4A CUDA8 GGML buffer residency smoke:
#   - pointer ownership lookup
#   - tensor->data device residency helper
#   - unregister-on-free behavior
# ---------------------------------------------------------------------------

cuda_add_executable(ggml-cuda8-ggml-buffer-residency-smoke
    ggml-cuda8-ggml-buffer-residency-smoke.cpp
)

target_link_libraries(ggml-cuda8-ggml-buffer-residency-smoke
    ggml-cuda8-kernels
    ${CUDA_LIBRARIES}
)

# ---------------------------------------------------------------------------
# G11A-4B device-resident ADD_F32 smoke:
#   - tensors are CUDA8-resident inside ggml_backend_buffer_t
#   - dispatcher ADD_F32 calls device kernel directly
#   - no dispatcher host staging for ADD_F32
# ---------------------------------------------------------------------------

cuda_add_executable(ggml-cuda8-ggml-buffer-device-add-smoke
    ggml-cuda8-ggml-buffer-device-add-smoke.cpp
)

target_link_libraries(ggml-cuda8-ggml-buffer-device-add-smoke
    ggml-cuda8-kernels
    ${CUDA_LIBRARIES}
)

# ---------------------------------------------------------------------------
# G11A-4D device-resident scalar ADD/MUL smoke:
#   - src/dst tensors are CUDA8-resident inside ggml_backend_buffer_t
#   - scalar tensor is CUDA8-resident
#   - dispatcher scalar ops call device kernels directly for src/dst
# ---------------------------------------------------------------------------

cuda_add_executable(ggml-cuda8-ggml-buffer-device-scalar-smoke
    ggml-cuda8-ggml-buffer-device-scalar-smoke.cpp
)

target_link_libraries(ggml-cuda8-ggml-buffer-device-scalar-smoke
    ggml-cuda8-kernels
    ${CUDA_LIBRARIES}
)

# ---------------------------------------------------------------------------
# G11B device-resident mini graph smoke:
#   - A/B/C/D/scalar inside one CUDA8 ggml_backend_buffer_t
#   - C = A + B via device-resident ADD_F32
#   - D = C * scalar via device-resident MUL_SCALAR_F32
# ---------------------------------------------------------------------------

cuda_add_executable(ggml-cuda8-ggml-buffer-device-graph-smoke
    ggml-cuda8-ggml-buffer-device-graph-smoke.cpp
)

target_link_libraries(ggml-cuda8-ggml-buffer-device-graph-smoke
    ggml-cuda8-kernels
    ${CUDA_LIBRARIES}
)

# ---------------------------------------------------------------------------
# G11A-4E device-resident SOFTMAX_ROWS_F32 smoke:
#   - src/dst matrices are CUDA8-resident inside ggml_backend_buffer_t
#   - dispatcher SOFTMAX_ROWS_F32 calls device kernel directly
# ---------------------------------------------------------------------------

cuda_add_executable(ggml-cuda8-ggml-buffer-device-softmax-smoke
    ggml-cuda8-ggml-buffer-device-softmax-smoke.cpp
)

target_link_libraries(ggml-cuda8-ggml-buffer-device-softmax-smoke
    ggml-cuda8-kernels
    ${CUDA_LIBRARIES}
)

# ---------------------------------------------------------------------------
# G11B-2 device-resident mini graph with softmax:
#   - C = A + B via device-resident ADD_F32
#   - D = C * scalar via device-resident MUL_SCALAR_F32
#   - S = softmax_rows(D) via device-resident SOFTMAX_ROWS_F32
# ---------------------------------------------------------------------------

cuda_add_executable(ggml-cuda8-ggml-buffer-device-softmax-graph-smoke
    ggml-cuda8-ggml-buffer-device-softmax-graph-smoke.cpp
)

target_link_libraries(ggml-cuda8-ggml-buffer-device-softmax-graph-smoke
    ggml-cuda8-kernels
    ${CUDA_LIBRARIES}
)
# ---------------------------------------------------------------------------
# G11D device-resident REDUCE_SUM_ROWS_F32 / REDUCE_MAX_ROWS_F32 smoke:
#   - src matrix and dst vectors are CUDA8-resident inside ggml_backend_buffer_t
#   - dispatcher reduce ops call device kernels directly
# ---------------------------------------------------------------------------

cuda_add_executable(ggml-cuda8-ggml-buffer-device-reduce-smoke
    ggml-cuda8-ggml-buffer-device-reduce-smoke.cpp
)

target_link_libraries(ggml-cuda8-ggml-buffer-device-reduce-smoke
    ggml-cuda8-kernels
    ${CUDA_LIBRARIES}
)

# ---------------------------------------------------------------------------
# G11E attention-like device-resident micrograph smoke:
#   - scores = A + B
#   - scaled = scores * scale
#   - row_max = reduce_max_rows(scaled)
#   - probs = softmax_rows(scaled)
#   - row_sum = reduce_sum_rows(probs)
# ---------------------------------------------------------------------------

cuda_add_executable(ggml-cuda8-ggml-buffer-device-attnlike-smoke
    ggml-cuda8-ggml-buffer-device-attnlike-smoke.cpp
)

target_link_libraries(ggml-cuda8-ggml-buffer-device-attnlike-smoke
    ggml-cuda8-kernels
    ${CUDA_LIBRARIES}
)

# ---------------------------------------------------------------------------
# G12A minimal ggml_backend_t registration-shaped probe:
#   - creates CUDA8 ggml_backend_t object
#   - exposes CUDA8 default buffer type
#   - allocates CUDA8 ggml_backend_buffer_t through backend default buft
# ---------------------------------------------------------------------------

cuda_add_executable(ggml-cuda8-ggml-backend-probe
    ggml-cuda8-ggml-backend-probe.cpp
)

target_link_libraries(ggml-cuda8-ggml-backend-probe
    ggml-cuda8-kernels
    ${CUDA_LIBRARIES}
)

# ---------------------------------------------------------------------------
# G12B backend-owned buffer graph smoke:
#   - allocates CUDA8 buffer through backend default buffer type helper
#   - runs device-resident ADD -> MUL_SCALAR -> SOFTMAX graph
# ---------------------------------------------------------------------------

cuda_add_executable(ggml-cuda8-ggml-backend-buffer-graph-smoke
    ggml-cuda8-ggml-backend-buffer-graph-smoke.cpp
)

target_link_libraries(ggml-cuda8-ggml-backend-buffer-graph-smoke
    ggml-cuda8-kernels
    ${CUDA_LIBRARIES}
)

# ---------------------------------------------------------------------------
# G12C backend-owned attention-like graph smoke:
#   - allocates CUDA8 buffer through backend default buffer type helper
#   - runs ADD -> MUL_SCALAR -> REDUCE_MAX -> SOFTMAX -> REDUCE_SUM graph
# ---------------------------------------------------------------------------

cuda_add_executable(ggml-cuda8-ggml-backend-attnlike-smoke
    ggml-cuda8-ggml-backend-attnlike-smoke.cpp
)

target_link_libraries(ggml-cuda8-ggml-backend-attnlike-smoke
    ggml-cuda8-kernels
    ${CUDA_LIBRARIES}
)

# ---------------------------------------------------------------------------
# G13A graph-compute-shaped backend callback probe:
#   - backend dispatch helper validates CUDA8 backend object
#   - forwards op execution to residency-aware dispatcher
#   - runs backend-owned ADD -> MUL_SCALAR -> SOFTMAX graph
# ---------------------------------------------------------------------------

cuda_add_executable(ggml-cuda8-ggml-backend-compute-probe
    ggml-cuda8-ggml-backend-compute-probe.cpp
)

target_link_libraries(ggml-cuda8-ggml-backend-compute-probe
    ggml-cuda8-kernels
    ${CUDA_LIBRARIES}
)

# ---------------------------------------------------------------------------
# G13B backend compute-shaped attention-like callback smoke:
#   - backend dispatch helper validates CUDA8 backend object
#   - forwards ops to residency-aware dispatcher
#   - runs backend-owned ADD -> MUL_SCALAR -> REDUCE_MAX -> SOFTMAX -> REDUCE_SUM graph
# ---------------------------------------------------------------------------

cuda_add_executable(ggml-cuda8-ggml-backend-compute-attnlike-smoke
    ggml-cuda8-ggml-backend-compute-attnlike-smoke.cpp
)

target_link_libraries(ggml-cuda8-ggml-backend-compute-attnlike-smoke
    ggml-cuda8-kernels
    ${CUDA_LIBRARIES}
)

# ---------------------------------------------------------------------------
# G14B real ggml_backend_i graph API shape probe:
#   - populates graph callback stubs matching this checkout's interface
#   - calls safe no-op/success graph callbacks directly
# ---------------------------------------------------------------------------

cuda_add_executable(ggml-cuda8-ggml-backend-graph-api-probe
    ggml-cuda8-ggml-backend-graph-api-probe.cpp
)

target_link_libraries(ggml-cuda8-ggml-backend-graph-api-probe
    ggml-cuda8-kernels
    ${CUDA_LIBRARIES}
)

# ---------------------------------------------------------------------------
# G15A real backend->iface.graph_compute ADD_F32 smoke:
#   - builds one tiny synthetic ggml ADD graph
#   - tensor data points into backend-owned CUDA8 buffer
#   - backend graph_compute dispatches ADD through residency-aware dispatcher
# ---------------------------------------------------------------------------

cuda_add_executable(ggml-cuda8-ggml-backend-graph-compute-add-smoke
    ggml-cuda8-ggml-backend-graph-compute-add-smoke.cpp
)

target_link_libraries(ggml-cuda8-ggml-backend-graph-compute-add-smoke
    ggml-cuda8-kernels
    ${CUDA_LIBRARIES}
)

# ---------------------------------------------------------------------------
# G15B real backend->iface.graph_compute ADD -> MUL_SCALAR smoke
# ---------------------------------------------------------------------------

cuda_add_executable(ggml-cuda8-ggml-backend-graph-compute-add-mul-smoke
    ggml-cuda8-ggml-backend-graph-compute-add-mul-smoke.cpp
)

target_link_libraries(ggml-cuda8-ggml-backend-graph-compute-add-mul-smoke
    ggml-cuda8-kernels
    ${CUDA_LIBRARIES}
)

# ---------------------------------------------------------------------------
# G15C real backend->iface.graph_compute ADD -> MUL_SCALAR -> SOFTMAX smoke
# ---------------------------------------------------------------------------

cuda_add_executable(ggml-cuda8-ggml-backend-graph-compute-softmax-smoke
    ggml-cuda8-ggml-backend-graph-compute-softmax-smoke.cpp
)

target_link_libraries(ggml-cuda8-ggml-backend-graph-compute-softmax-smoke
    ggml-cuda8-kernels
    ${CUDA_LIBRARIES}
)

# ---------------------------------------------------------------------------
# G15D real backend->iface.graph_compute attention-like smoke:
#   ADD -> MUL_SCALAR -> SOFTMAX -> SUM_ROWS
# ---------------------------------------------------------------------------

cuda_add_executable(ggml-cuda8-ggml-backend-graph-compute-attnlike-smoke
    ggml-cuda8-ggml-backend-graph-compute-attnlike-smoke.cpp
)

target_link_libraries(ggml-cuda8-ggml-backend-graph-compute-attnlike-smoke
    ggml-cuda8-kernels
    ${CUDA_LIBRARIES}
)

# ---------------------------------------------------------------------------
# G16A real GGML graph-builder ADD graph -> backend->iface.graph_compute smoke
#
# Standalone graph-builder subset for the CUDA8/Fermi legacy build:
#   - avoids full ggml target so gguf.cpp / newer C++ sources are not built
#   - avoids ggml-backend.cpp dependency chain
#   - uses function/data sections + linker GC so unused ggml.c backend helpers
#     do not require unresolved backend/allocator symbols
# ---------------------------------------------------------------------------

cuda_add_executable(ggml-cuda8-ggml-backend-graph-builder-add-smoke
    ../ggml.c
    ../ggml-quants.c
    ../ggml-threading.cpp
    ggml-cuda8-ggml-backend-graph-builder-add-smoke.cpp
)

target_include_directories(ggml-cuda8-ggml-backend-graph-builder-add-smoke PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

target_compile_definitions(ggml-cuda8-ggml-backend-graph-builder-add-smoke PRIVATE
    GGML_VERSION="${GGML_VERSION}"
    GGML_COMMIT="${GGML_COMMIT}"
)

target_compile_options(ggml-cuda8-ggml-backend-graph-builder-add-smoke PRIVATE
    $<$<COMPILE_LANGUAGE:C>:-ffunction-sections>
    $<$<COMPILE_LANGUAGE:C>:-fdata-sections>
    $<$<COMPILE_LANGUAGE:CXX>:-ffunction-sections>
    $<$<COMPILE_LANGUAGE:CXX>:-fdata-sections>
)

set_property(TARGET ggml-cuda8-ggml-backend-graph-builder-add-smoke APPEND_STRING PROPERTY LINK_FLAGS " -Wl,--gc-sections")

target_link_libraries(ggml-cuda8-ggml-backend-graph-builder-add-smoke
    ggml-cuda8-kernels
    ${CUDA_LIBRARIES}
)

# ---------------------------------------------------------------------------
# G16B real GGML graph-builder ADD -> scalar MUL graph -> graph_compute smoke
# Uses the same standalone-GC GGML graph-builder subset as G16A v10.
# ---------------------------------------------------------------------------

cuda_add_executable(ggml-cuda8-ggml-backend-graph-builder-add-mul-smoke
    ../ggml.c
    ../ggml-quants.c
    ../ggml-threading.cpp
    ggml-cuda8-ggml-backend-graph-builder-add-mul-smoke.cpp
)

target_include_directories(ggml-cuda8-ggml-backend-graph-builder-add-mul-smoke PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

target_compile_definitions(ggml-cuda8-ggml-backend-graph-builder-add-mul-smoke PRIVATE
    GGML_VERSION="${GGML_VERSION}"
    GGML_COMMIT="${GGML_COMMIT}"
)

target_compile_options(ggml-cuda8-ggml-backend-graph-builder-add-mul-smoke PRIVATE
    $<$<COMPILE_LANGUAGE:C>:-ffunction-sections>
    $<$<COMPILE_LANGUAGE:C>:-fdata-sections>
    $<$<COMPILE_LANGUAGE:CXX>:-ffunction-sections>
    $<$<COMPILE_LANGUAGE:CXX>:-fdata-sections>
)

set_property(TARGET ggml-cuda8-ggml-backend-graph-builder-add-mul-smoke APPEND_STRING PROPERTY LINK_FLAGS " -Wl,--gc-sections")

target_link_libraries(ggml-cuda8-ggml-backend-graph-builder-add-mul-smoke
    ggml-cuda8-kernels
    ${CUDA_LIBRARIES}
)

# ---------------------------------------------------------------------------
# G16C real GGML graph-builder ADD -> scalar MUL -> SOFTMAX graph smoke
# Uses the same standalone-GC GGML graph-builder subset as G16A v10.
# ---------------------------------------------------------------------------

cuda_add_executable(ggml-cuda8-ggml-backend-graph-builder-softmax-smoke
    ../ggml.c
    ../ggml-quants.c
    ../ggml-threading.cpp
    ggml-cuda8-ggml-backend-graph-builder-softmax-smoke.cpp
)

target_include_directories(ggml-cuda8-ggml-backend-graph-builder-softmax-smoke PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

target_compile_definitions(ggml-cuda8-ggml-backend-graph-builder-softmax-smoke PRIVATE
    GGML_VERSION="${GGML_VERSION}"
    GGML_COMMIT="${GGML_COMMIT}"
)

target_compile_options(ggml-cuda8-ggml-backend-graph-builder-softmax-smoke PRIVATE
    $<$<COMPILE_LANGUAGE:C>:-ffunction-sections>
    $<$<COMPILE_LANGUAGE:C>:-fdata-sections>
    $<$<COMPILE_LANGUAGE:CXX>:-ffunction-sections>
    $<$<COMPILE_LANGUAGE:CXX>:-fdata-sections>
)

set_property(TARGET ggml-cuda8-ggml-backend-graph-builder-softmax-smoke APPEND_STRING PROPERTY LINK_FLAGS " -Wl,--gc-sections")

target_link_libraries(ggml-cuda8-ggml-backend-graph-builder-softmax-smoke
    ggml-cuda8-kernels
    ${CUDA_LIBRARIES}
)

# ---------------------------------------------------------------------------
# G16D real GGML graph-builder attention-like graph smoke
# ADD -> scalar MUL -> SOFTMAX -> SUM_ROWS through graph_compute.
# Uses the same standalone-GC GGML graph-builder subset as G16A v10.
# ---------------------------------------------------------------------------

cuda_add_executable(ggml-cuda8-ggml-backend-graph-builder-attnlike-smoke
    ../ggml.c
    ../ggml-quants.c
    ../ggml-threading.cpp
    ggml-cuda8-ggml-backend-graph-builder-attnlike-smoke.cpp
)

target_include_directories(ggml-cuda8-ggml-backend-graph-builder-attnlike-smoke PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

target_compile_definitions(ggml-cuda8-ggml-backend-graph-builder-attnlike-smoke PRIVATE
    GGML_VERSION="${GGML_VERSION}"
    GGML_COMMIT="${GGML_COMMIT}"
)

target_compile_options(ggml-cuda8-ggml-backend-graph-builder-attnlike-smoke PRIVATE
    $<$<COMPILE_LANGUAGE:C>:-ffunction-sections>
    $<$<COMPILE_LANGUAGE:C>:-fdata-sections>
    $<$<COMPILE_LANGUAGE:CXX>:-ffunction-sections>
    $<$<COMPILE_LANGUAGE:CXX>:-fdata-sections>
)

set_property(TARGET ggml-cuda8-ggml-backend-graph-builder-attnlike-smoke APPEND_STRING PROPERTY LINK_FLAGS " -Wl,--gc-sections")

target_link_libraries(ggml-cuda8-ggml-backend-graph-builder-attnlike-smoke
    ggml-cuda8-kernels
    ${CUDA_LIBRARIES}
)

# ---------------------------------------------------------------------------
# G17A real GGML graph-builder Q8_0 x F32 vector smoke
# Uses standalone-GC GGML graph-builder subset and dispatches MUL_MAT Q8_0 MMV.
# ---------------------------------------------------------------------------

cuda_add_executable(ggml-cuda8-ggml-backend-graph-builder-q8_0-mmv-smoke
    ../ggml.c
    ../ggml-quants.c
    ../ggml-threading.cpp
    ggml-cuda8-ggml-backend-graph-builder-q8_0-mmv-smoke.cpp
)

target_include_directories(ggml-cuda8-ggml-backend-graph-builder-q8_0-mmv-smoke PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

target_compile_definitions(ggml-cuda8-ggml-backend-graph-builder-q8_0-mmv-smoke PRIVATE
    GGML_VERSION="${GGML_VERSION}"
    GGML_COMMIT="${GGML_COMMIT}"
)

target_compile_options(ggml-cuda8-ggml-backend-graph-builder-q8_0-mmv-smoke PRIVATE
    $<$<COMPILE_LANGUAGE:C>:-ffunction-sections>
    $<$<COMPILE_LANGUAGE:C>:-fdata-sections>
    $<$<COMPILE_LANGUAGE:CXX>:-ffunction-sections>
    $<$<COMPILE_LANGUAGE:CXX>:-fdata-sections>
)

set_property(TARGET ggml-cuda8-ggml-backend-graph-builder-q8_0-mmv-smoke APPEND_STRING PROPERTY LINK_FLAGS " -Wl,--gc-sections")

target_link_libraries(ggml-cuda8-ggml-backend-graph-builder-q8_0-mmv-smoke
    ggml-cuda8-kernels
    ${CUDA_LIBRARIES}
)

# ---------------------------------------------------------------------------
# G18A real GGML graph-builder Q8_0 MUL_MAT -> ADD pipeline smoke
# Uses standalone-GC GGML graph-builder subset.
# ---------------------------------------------------------------------------

cuda_add_executable(ggml-cuda8-ggml-backend-graph-builder-q8_0-add-smoke
    ../ggml.c
    ../ggml-quants.c
    ../ggml-threading.cpp
    ggml-cuda8-ggml-backend-graph-builder-q8_0-add-smoke.cpp
)

target_include_directories(ggml-cuda8-ggml-backend-graph-builder-q8_0-add-smoke PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

target_compile_definitions(ggml-cuda8-ggml-backend-graph-builder-q8_0-add-smoke PRIVATE
    GGML_VERSION="${GGML_VERSION}"
    GGML_COMMIT="${GGML_COMMIT}"
)

target_compile_options(ggml-cuda8-ggml-backend-graph-builder-q8_0-add-smoke PRIVATE
    $<$<COMPILE_LANGUAGE:C>:-ffunction-sections>
    $<$<COMPILE_LANGUAGE:C>:-fdata-sections>
    $<$<COMPILE_LANGUAGE:CXX>:-ffunction-sections>
    $<$<COMPILE_LANGUAGE:CXX>:-fdata-sections>
)

set_property(TARGET ggml-cuda8-ggml-backend-graph-builder-q8_0-add-smoke APPEND_STRING PROPERTY LINK_FLAGS " -Wl,--gc-sections")

target_link_libraries(ggml-cuda8-ggml-backend-graph-builder-q8_0-add-smoke
    ggml-cuda8-kernels
    ${CUDA_LIBRARIES}
)

# ---------------------------------------------------------------------------
# G18C real GGML graph-builder Q8_0 MUL_MAT -> MUL_SCALAR -> ADD pipeline smoke
# Uses standalone-GC GGML graph-builder subset.
# ---------------------------------------------------------------------------

cuda_add_executable(ggml-cuda8-ggml-backend-graph-builder-q8_0-mul-add-smoke
    ../ggml.c
    ../ggml-quants.c
    ../ggml-threading.cpp
    ggml-cuda8-ggml-backend-graph-builder-q8_0-mul-add-smoke.cpp
)

target_include_directories(ggml-cuda8-ggml-backend-graph-builder-q8_0-mul-add-smoke PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

target_compile_definitions(ggml-cuda8-ggml-backend-graph-builder-q8_0-mul-add-smoke PRIVATE
    GGML_VERSION="${GGML_VERSION}"
    GGML_COMMIT="${GGML_COMMIT}"
)

target_compile_options(ggml-cuda8-ggml-backend-graph-builder-q8_0-mul-add-smoke PRIVATE
    $<$<COMPILE_LANGUAGE:C>:-ffunction-sections>
    $<$<COMPILE_LANGUAGE:C>:-fdata-sections>
    $<$<COMPILE_LANGUAGE:CXX>:-ffunction-sections>
    $<$<COMPILE_LANGUAGE:CXX>:-fdata-sections>
)

set_property(TARGET ggml-cuda8-ggml-backend-graph-builder-q8_0-mul-add-smoke APPEND_STRING PROPERTY LINK_FLAGS " -Wl,--gc-sections")

target_link_libraries(ggml-cuda8-ggml-backend-graph-builder-q8_0-mul-add-smoke
    ggml-cuda8-kernels
    ${CUDA_LIBRARIES}
)

# ---------------------------------------------------------------------------
# G19A real GGML graph-builder Q8_0 MUL_MAT -> MUL_SCALAR -> ADD -> SOFTMAX smoke
# Uses standalone-GC GGML graph-builder subset.
# ---------------------------------------------------------------------------

cuda_add_executable(ggml-cuda8-ggml-backend-graph-builder-q8_0-mul-add-softmax-smoke
    ../ggml.c
    ../ggml-quants.c
    ../ggml-threading.cpp
    ggml-cuda8-ggml-backend-graph-builder-q8_0-mul-add-softmax-smoke.cpp
)

target_include_directories(ggml-cuda8-ggml-backend-graph-builder-q8_0-mul-add-softmax-smoke PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

target_compile_definitions(ggml-cuda8-ggml-backend-graph-builder-q8_0-mul-add-softmax-smoke PRIVATE
    GGML_VERSION="${GGML_VERSION}"
    GGML_COMMIT="${GGML_COMMIT}"
)

target_compile_options(ggml-cuda8-ggml-backend-graph-builder-q8_0-mul-add-softmax-smoke PRIVATE
    $<$<COMPILE_LANGUAGE:C>:-ffunction-sections>
    $<$<COMPILE_LANGUAGE:C>:-fdata-sections>
    $<$<COMPILE_LANGUAGE:CXX>:-ffunction-sections>
    $<$<COMPILE_LANGUAGE:CXX>:-fdata-sections>
)

set_property(TARGET ggml-cuda8-ggml-backend-graph-builder-q8_0-mul-add-softmax-smoke APPEND_STRING PROPERTY LINK_FLAGS " -Wl,--gc-sections")

target_link_libraries(ggml-cuda8-ggml-backend-graph-builder-q8_0-mul-add-softmax-smoke
    ggml-cuda8-kernels
    ${CUDA_LIBRARIES}
)

# ---------------------------------------------------------------------------
# G20A real GGML graph-builder Q8_0 MUL_MAT -> MUL_SCALAR -> ADD -> SOFTMAX -> SUM_ROWS smoke
# Uses standalone-GC GGML graph-builder subset.
# ---------------------------------------------------------------------------

cuda_add_executable(ggml-cuda8-ggml-backend-graph-builder-q8_0-mul-add-softmax-sumrows-smoke
    ../ggml.c
    ../ggml-quants.c
    ../ggml-threading.cpp
    ggml-cuda8-ggml-backend-graph-builder-q8_0-mul-add-softmax-sumrows-smoke.cpp
)

target_include_directories(ggml-cuda8-ggml-backend-graph-builder-q8_0-mul-add-softmax-sumrows-smoke PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

target_compile_definitions(ggml-cuda8-ggml-backend-graph-builder-q8_0-mul-add-softmax-sumrows-smoke PRIVATE
    GGML_VERSION="${GGML_VERSION}"
    GGML_COMMIT="${GGML_COMMIT}"
)

target_compile_options(ggml-cuda8-ggml-backend-graph-builder-q8_0-mul-add-softmax-sumrows-smoke PRIVATE
    $<$<COMPILE_LANGUAGE:C>:-ffunction-sections>
    $<$<COMPILE_LANGUAGE:C>:-fdata-sections>
    $<$<COMPILE_LANGUAGE:CXX>:-ffunction-sections>
    $<$<COMPILE_LANGUAGE:CXX>:-fdata-sections>
)

set_property(TARGET ggml-cuda8-ggml-backend-graph-builder-q8_0-mul-add-softmax-sumrows-smoke APPEND_STRING PROPERTY LINK_FLAGS " -Wl,--gc-sections")

target_link_libraries(ggml-cuda8-ggml-backend-graph-builder-q8_0-mul-add-softmax-sumrows-smoke
    ggml-cuda8-kernels
    ${CUDA_LIBRARIES}
)

# ---------------------------------------------------------------------------
# G21A real GGML graph-builder Q8_0 MUL_MAT -> residual ADD smoke
# Uses standalone-GC GGML graph-builder subset.
# ---------------------------------------------------------------------------

cuda_add_executable(ggml-cuda8-ggml-backend-graph-builder-q8_0-residual-add-smoke
    ../ggml.c
    ../ggml-quants.c
    ../ggml-threading.cpp
    ggml-cuda8-ggml-backend-graph-builder-q8_0-residual-add-smoke.cpp
)

target_include_directories(ggml-cuda8-ggml-backend-graph-builder-q8_0-residual-add-smoke PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

target_compile_definitions(ggml-cuda8-ggml-backend-graph-builder-q8_0-residual-add-smoke PRIVATE
    GGML_VERSION="${GGML_VERSION}"
    GGML_COMMIT="${GGML_COMMIT}"
)

target_compile_options(ggml-cuda8-ggml-backend-graph-builder-q8_0-residual-add-smoke PRIVATE
    $<$<COMPILE_LANGUAGE:C>:-ffunction-sections>
    $<$<COMPILE_LANGUAGE:C>:-fdata-sections>
    $<$<COMPILE_LANGUAGE:CXX>:-ffunction-sections>
    $<$<COMPILE_LANGUAGE:CXX>:-fdata-sections>
)

set_property(TARGET ggml-cuda8-ggml-backend-graph-builder-q8_0-residual-add-smoke APPEND_STRING PROPERTY LINK_FLAGS " -Wl,--gc-sections")

target_link_libraries(ggml-cuda8-ggml-backend-graph-builder-q8_0-residual-add-smoke
    ggml-cuda8-kernels
    ${CUDA_LIBRARIES}
)

# ---------------------------------------------------------------------------
# G22A real GGML graph-builder Q8_0 MUL_MAT -> residual ADD -> SOFTMAX smoke
# Uses standalone-GC GGML graph-builder subset.
# ---------------------------------------------------------------------------

cuda_add_executable(ggml-cuda8-ggml-backend-graph-builder-q8_0-residual-add-softmax-smoke
    ../ggml.c
    ../ggml-quants.c
    ../ggml-threading.cpp
    ggml-cuda8-ggml-backend-graph-builder-q8_0-residual-add-softmax-smoke.cpp
)

target_include_directories(ggml-cuda8-ggml-backend-graph-builder-q8_0-residual-add-softmax-smoke PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

target_compile_definitions(ggml-cuda8-ggml-backend-graph-builder-q8_0-residual-add-softmax-smoke PRIVATE
    GGML_VERSION="${GGML_VERSION}"
    GGML_COMMIT="${GGML_COMMIT}"
)

target_compile_options(ggml-cuda8-ggml-backend-graph-builder-q8_0-residual-add-softmax-smoke PRIVATE
    $<$<COMPILE_LANGUAGE:C>:-ffunction-sections>
    $<$<COMPILE_LANGUAGE:C>:-fdata-sections>
    $<$<COMPILE_LANGUAGE:CXX>:-ffunction-sections>
    $<$<COMPILE_LANGUAGE:CXX>:-fdata-sections>
)

set_property(TARGET ggml-cuda8-ggml-backend-graph-builder-q8_0-residual-add-softmax-smoke APPEND_STRING PROPERTY LINK_FLAGS " -Wl,--gc-sections")

target_link_libraries(ggml-cuda8-ggml-backend-graph-builder-q8_0-residual-add-softmax-smoke
    ggml-cuda8-kernels
    ${CUDA_LIBRARIES}
)

# ---------------------------------------------------------------------------
# G23A real GGML graph-builder Q8_0 MUL_MAT -> residual ADD -> SOFTMAX -> SUM_ROWS smoke
# Uses standalone-GC GGML graph-builder subset.
# ---------------------------------------------------------------------------

cuda_add_executable(ggml-cuda8-ggml-backend-graph-builder-q8_0-residual-add-softmax-sumrows-smoke
    ../ggml.c
    ../ggml-quants.c
    ../ggml-threading.cpp
    ggml-cuda8-ggml-backend-graph-builder-q8_0-residual-add-softmax-sumrows-smoke.cpp
)

target_include_directories(ggml-cuda8-ggml-backend-graph-builder-q8_0-residual-add-softmax-sumrows-smoke PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

target_compile_definitions(ggml-cuda8-ggml-backend-graph-builder-q8_0-residual-add-softmax-sumrows-smoke PRIVATE
    GGML_VERSION="${GGML_VERSION}"
    GGML_COMMIT="${GGML_COMMIT}"
)

target_compile_options(ggml-cuda8-ggml-backend-graph-builder-q8_0-residual-add-softmax-sumrows-smoke PRIVATE
    $<$<COMPILE_LANGUAGE:C>:-ffunction-sections>
    $<$<COMPILE_LANGUAGE:C>:-fdata-sections>
    $<$<COMPILE_LANGUAGE:CXX>:-ffunction-sections>
    $<$<COMPILE_LANGUAGE:CXX>:-fdata-sections>
)

set_property(TARGET ggml-cuda8-ggml-backend-graph-builder-q8_0-residual-add-softmax-sumrows-smoke APPEND_STRING PROPERTY LINK_FLAGS " -Wl,--gc-sections")

target_link_libraries(ggml-cuda8-ggml-backend-graph-builder-q8_0-residual-add-softmax-sumrows-smoke
    ggml-cuda8-kernels
    ${CUDA_LIBRARIES}
)

# ---------------------------------------------------------------------------
# G24A real GGML graph-builder Q8_0 MUL_MAT -> MUL_SCALAR -> residual ADD -> SOFTMAX -> SUM_ROWS smoke
# Uses standalone-GC GGML graph-builder subset.
# ---------------------------------------------------------------------------

cuda_add_executable(ggml-cuda8-ggml-backend-graph-builder-q8_0-residual-scale-add-softmax-sumrows-smoke
    ../ggml.c
    ../ggml-quants.c
    ../ggml-threading.cpp
    ggml-cuda8-ggml-backend-graph-builder-q8_0-residual-scale-add-softmax-sumrows-smoke.cpp
)

target_include_directories(ggml-cuda8-ggml-backend-graph-builder-q8_0-residual-scale-add-softmax-sumrows-smoke PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

target_compile_definitions(ggml-cuda8-ggml-backend-graph-builder-q8_0-residual-scale-add-softmax-sumrows-smoke PRIVATE
    GGML_VERSION="${GGML_VERSION}"
    GGML_COMMIT="${GGML_COMMIT}"
)

target_compile_options(ggml-cuda8-ggml-backend-graph-builder-q8_0-residual-scale-add-softmax-sumrows-smoke PRIVATE
    $<$<COMPILE_LANGUAGE:C>:-ffunction-sections>
    $<$<COMPILE_LANGUAGE:C>:-fdata-sections>
    $<$<COMPILE_LANGUAGE:CXX>:-ffunction-sections>
    $<$<COMPILE_LANGUAGE:CXX>:-fdata-sections>
)

set_property(TARGET ggml-cuda8-ggml-backend-graph-builder-q8_0-residual-scale-add-softmax-sumrows-smoke APPEND_STRING PROPERTY LINK_FLAGS " -Wl,--gc-sections")

target_link_libraries(ggml-cuda8-ggml-backend-graph-builder-q8_0-residual-scale-add-softmax-sumrows-smoke
    ggml-cuda8-kernels
    ${CUDA_LIBRARIES}
)

# ---------------------------------------------------------------------------
# G25A real GGML graph-builder Q8_0 MUL_MAT -> MUL_SCALAR -> residual ADD -> SOFTMAX -> SUM_ROWS
# with explicit scale and residual input branch isolation checks.
# Uses standalone-GC GGML graph-builder subset.
# ---------------------------------------------------------------------------

cuda_add_executable(ggml-cuda8-ggml-backend-graph-builder-q8_0-residual-scale-add-softmax-sumrows-isolation-smoke
    ../ggml.c
    ../ggml-quants.c
    ../ggml-threading.cpp
    ggml-cuda8-ggml-backend-graph-builder-q8_0-residual-scale-add-softmax-sumrows-isolation-smoke.cpp
)

target_include_directories(ggml-cuda8-ggml-backend-graph-builder-q8_0-residual-scale-add-softmax-sumrows-isolation-smoke PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

target_compile_definitions(ggml-cuda8-ggml-backend-graph-builder-q8_0-residual-scale-add-softmax-sumrows-isolation-smoke PRIVATE
    GGML_VERSION="${GGML_VERSION}"
    GGML_COMMIT="${GGML_COMMIT}"
)

target_compile_options(ggml-cuda8-ggml-backend-graph-builder-q8_0-residual-scale-add-softmax-sumrows-isolation-smoke PRIVATE
    $<$<COMPILE_LANGUAGE:C>:-ffunction-sections>
    $<$<COMPILE_LANGUAGE:C>:-fdata-sections>
    $<$<COMPILE_LANGUAGE:CXX>:-ffunction-sections>
    $<$<COMPILE_LANGUAGE:CXX>:-fdata-sections>
)

set_property(TARGET ggml-cuda8-ggml-backend-graph-builder-q8_0-residual-scale-add-softmax-sumrows-isolation-smoke APPEND_STRING PROPERTY LINK_FLAGS " -Wl,--gc-sections")

target_link_libraries(ggml-cuda8-ggml-backend-graph-builder-q8_0-residual-scale-add-softmax-sumrows-isolation-smoke
    ggml-cuda8-kernels
    ${CUDA_LIBRARIES}
)

# -- G24A: RMS_NORM smoke test ------------------------------------------------
cuda_add_executable(ggml-cuda8-rms-norm-smoke
    ggml-cuda8-rms-norm-smoke.cu
    ggml-cuda8-rms-norm.cu)
target_link_libraries(ggml-cuda8-rms-norm-smoke ${CUDA_LIBRARIES})
set_target_properties(ggml-cuda8-rms-norm-smoke PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# -- G26A: element-wise MUL smoke test ----------------------------------------
cuda_add_executable(ggml-cuda8-mul-smoke
    ggml-cuda8-mul-smoke.cu
    ggml-cuda8-mul.cu)
target_link_libraries(ggml-cuda8-mul-smoke ${CUDA_LIBRARIES})
set_target_properties(ggml-cuda8-mul-smoke PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")



# -- G27A: RMS_NORM -> MUL graph-builder smoke -------------------------------
cuda_add_executable(ggml-cuda8-ggml-backend-graph-builder-rmsnorm-mul-smoke
    ../ggml.c
    ../ggml-quants.c
    ../ggml-threading.cpp
    ggml-cuda8-ggml-backend-graph-builder-rmsnorm-mul-smoke.cpp
)

target_include_directories(ggml-cuda8-ggml-backend-graph-builder-rmsnorm-mul-smoke PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

target_compile_definitions(ggml-cuda8-ggml-backend-graph-builder-rmsnorm-mul-smoke PRIVATE
    GGML_VERSION="${GGML_VERSION}"
    GGML_COMMIT="${GGML_COMMIT}"
)

target_compile_options(ggml-cuda8-ggml-backend-graph-builder-rmsnorm-mul-smoke PRIVATE
    $<$<COMPILE_LANGUAGE:C>:-ffunction-sections>
    $<$<COMPILE_LANGUAGE:C>:-fdata-sections>
    $<$<COMPILE_LANGUAGE:CXX>:-ffunction-sections>
    $<$<COMPILE_LANGUAGE:CXX>:-fdata-sections>
)

set_property(TARGET ggml-cuda8-ggml-backend-graph-builder-rmsnorm-mul-smoke
    APPEND_STRING PROPERTY LINK_FLAGS " -Wl,--gc-sections")

target_link_libraries(ggml-cuda8-ggml-backend-graph-builder-rmsnorm-mul-smoke
    ggml-cuda8-kernels
    ${CUDA_LIBRARIES}
)

# -- G28A: ROPE smoke test ----------------------------------------------------
cuda_add_executable(ggml-cuda8-rope-smoke
    ggml-cuda8-rope-smoke.cu
    ggml-cuda8-rope.cu)
target_link_libraries(ggml-cuda8-rope-smoke ${CUDA_LIBRARIES})
set_target_properties(ggml-cuda8-rope-smoke PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# -- G29A: RESHAPE/VIEW/PERMUTE no-op graph-builder smoke ---------------------
cuda_add_executable(ggml-cuda8-ggml-backend-graph-builder-reshape-smoke
    ../ggml.c
    ../ggml-quants.c
    ../ggml-threading.cpp
    ggml-cuda8-ggml-backend-graph-builder-reshape-smoke.cpp
)

target_include_directories(ggml-cuda8-ggml-backend-graph-builder-reshape-smoke PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

target_compile_definitions(ggml-cuda8-ggml-backend-graph-builder-reshape-smoke PRIVATE
    GGML_VERSION="${GGML_VERSION}"
    GGML_COMMIT="${GGML_COMMIT}"
)

target_compile_options(ggml-cuda8-ggml-backend-graph-builder-reshape-smoke PRIVATE
    $<$<COMPILE_LANGUAGE:C>:-ffunction-sections>
    $<$<COMPILE_LANGUAGE:C>:-fdata-sections>
    $<$<COMPILE_LANGUAGE:CXX>:-ffunction-sections>
    $<$<COMPILE_LANGUAGE:CXX>:-fdata-sections>
)

set_property(TARGET ggml-cuda8-ggml-backend-graph-builder-reshape-smoke
    APPEND_STRING PROPERTY LINK_FLAGS " -Wl,--gc-sections")

target_link_libraries(ggml-cuda8-ggml-backend-graph-builder-reshape-smoke
    ggml-cuda8-kernels
    ${CUDA_LIBRARIES}
)

# -- G30A: CONT graph-builder smoke -------------------------------------------
cuda_add_executable(ggml-cuda8-ggml-backend-graph-builder-cont-smoke
    ../ggml.c
    ../ggml-quants.c
    ../ggml-threading.cpp
    ggml-cuda8-ggml-backend-graph-builder-cont-smoke.cpp
)

target_include_directories(ggml-cuda8-ggml-backend-graph-builder-cont-smoke PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

target_compile_definitions(ggml-cuda8-ggml-backend-graph-builder-cont-smoke PRIVATE
    GGML_VERSION="${GGML_VERSION}"
    GGML_COMMIT="${GGML_COMMIT}"
)

target_compile_options(ggml-cuda8-ggml-backend-graph-builder-cont-smoke PRIVATE
    $<$<COMPILE_LANGUAGE:C>:-ffunction-sections>
    $<$<COMPILE_LANGUAGE:C>:-fdata-sections>
    $<$<COMPILE_LANGUAGE:CXX>:-ffunction-sections>
    $<$<COMPILE_LANGUAGE:CXX>:-fdata-sections>
)

set_property(TARGET ggml-cuda8-ggml-backend-graph-builder-cont-smoke
    APPEND_STRING PROPERTY LINK_FLAGS " -Wl,--gc-sections")

target_link_libraries(ggml-cuda8-ggml-backend-graph-builder-cont-smoke
    ggml-cuda8-kernels
    ${CUDA_LIBRARIES}
)

# -- G31A: DIAG_MASK_INF smoke test -------------------------------------------
cuda_add_executable(ggml-cuda8-diagmask-smoke
    ggml-cuda8-diagmask-smoke.cu
    ggml-cuda8-diagmask.cu)
target_link_libraries(ggml-cuda8-diagmask-smoke ${CUDA_LIBRARIES})
set_target_properties(ggml-cuda8-diagmask-smoke PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# -- G32A: Full transformer block pipeline smoke ------------------------------
cuda_add_executable(ggml-cuda8-ggml-backend-graph-builder-transformer-block-smoke
    ../ggml.c
    ../ggml-quants.c
    ../ggml-threading.cpp
    ggml-cuda8-ggml-backend-graph-builder-transformer-block-smoke.cpp
)

target_include_directories(ggml-cuda8-ggml-backend-graph-builder-transformer-block-smoke PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

target_compile_definitions(ggml-cuda8-ggml-backend-graph-builder-transformer-block-smoke PRIVATE
    GGML_VERSION="${GGML_VERSION}"
    GGML_COMMIT="${GGML_COMMIT}"
)

target_compile_options(ggml-cuda8-ggml-backend-graph-builder-transformer-block-smoke PRIVATE
    $<$<COMPILE_LANGUAGE:C>:-ffunction-sections>
    $<$<COMPILE_LANGUAGE:C>:-fdata-sections>
    $<$<COMPILE_LANGUAGE:CXX>:-ffunction-sections>
    $<$<COMPILE_LANGUAGE:CXX>:-fdata-sections>
)

set_property(TARGET ggml-cuda8-ggml-backend-graph-builder-transformer-block-smoke
    APPEND_STRING PROPERTY LINK_FLAGS " -Wl,--gc-sections")

target_link_libraries(ggml-cuda8-ggml-backend-graph-builder-transformer-block-smoke
    ggml-cuda8-kernels
    ${CUDA_LIBRARIES}
)

# -- G33A: GET_ROWS smoke test ------------------------------------------------
cuda_add_executable(ggml-cuda8-getrows-smoke
    ggml-cuda8-getrows-smoke.cu
    ggml-cuda8-getrows.cu)
target_link_libraries(ggml-cuda8-getrows-smoke ${CUDA_LIBRARIES})
set_target_properties(ggml-cuda8-getrows-smoke PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# -- G34A: Full attention pipeline smoke --------------------------------------
cuda_add_executable(ggml-cuda8-ggml-backend-graph-builder-attention-smoke
    ../ggml.c
    ../ggml-quants.c
    ../ggml-threading.cpp
    ggml-cuda8-ggml-backend-graph-builder-attention-smoke.cpp
)

target_include_directories(ggml-cuda8-ggml-backend-graph-builder-attention-smoke PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

target_compile_definitions(ggml-cuda8-ggml-backend-graph-builder-attention-smoke PRIVATE
    GGML_VERSION="${GGML_VERSION}"
    GGML_COMMIT="${GGML_COMMIT}"
)

target_compile_options(ggml-cuda8-ggml-backend-graph-builder-attention-smoke PRIVATE
    $<$<COMPILE_LANGUAGE:C>:-ffunction-sections>
    $<$<COMPILE_LANGUAGE:C>:-fdata-sections>
    $<$<COMPILE_LANGUAGE:CXX>:-ffunction-sections>
    $<$<COMPILE_LANGUAGE:CXX>:-fdata-sections>
)

set_property(TARGET ggml-cuda8-ggml-backend-graph-builder-attention-smoke
    APPEND_STRING PROPERTY LINK_FLAGS " -Wl,--gc-sections")

target_link_libraries(ggml-cuda8-ggml-backend-graph-builder-attention-smoke
    ggml-cuda8-kernels
    ${CUDA_LIBRARIES}
)

# -- G35A: Full end-to-end pipeline smoke (ALL ops) ----------------------------
cuda_add_executable(ggml-cuda8-ggml-backend-graph-builder-e2e-smoke
    ../ggml.c
    ../ggml-quants.c
    ../ggml-threading.cpp
    ggml-cuda8-ggml-backend-graph-builder-e2e-smoke.cpp
)

target_include_directories(ggml-cuda8-ggml-backend-graph-builder-e2e-smoke PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

target_compile_definitions(ggml-cuda8-ggml-backend-graph-builder-e2e-smoke PRIVATE
    GGML_VERSION="${GGML_VERSION}"
    GGML_COMMIT="${GGML_COMMIT}"
)

target_compile_options(ggml-cuda8-ggml-backend-graph-builder-e2e-smoke PRIVATE
    $<$<COMPILE_LANGUAGE:C>:-ffunction-sections>
    $<$<COMPILE_LANGUAGE:C>:-fdata-sections>
    $<$<COMPILE_LANGUAGE:CXX>:-ffunction-sections>
    $<$<COMPILE_LANGUAGE:CXX>:-fdata-sections>
)

set_property(TARGET ggml-cuda8-ggml-backend-graph-builder-e2e-smoke
    APPEND_STRING PROPERTY LINK_FLAGS " -Wl,--gc-sections")

target_link_libraries(ggml-cuda8-ggml-backend-graph-builder-e2e-smoke
    ggml-cuda8-kernels
    ${CUDA_LIBRARIES}
)


cuda_add_executable(ggml-cuda8-ggml-backend-supports-op-smoke
    ../ggml.c
    ../ggml-quants.c
    ../ggml-threading.cpp
    ggml-cuda8-ggml-backend-supports-op-smoke.cpp
)

target_include_directories(ggml-cuda8-ggml-backend-supports-op-smoke PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

target_compile_definitions(ggml-cuda8-ggml-backend-supports-op-smoke PRIVATE
    GGML_VERSION="${GGML_VERSION}"
    GGML_COMMIT="${GGML_COMMIT}"
)

target_compile_options(ggml-cuda8-ggml-backend-supports-op-smoke PRIVATE
    $<$<COMPILE_LANGUAGE:C>:-ffunction-sections>
    $<$<COMPILE_LANGUAGE:C>:-fdata-sections>
    $<$<COMPILE_LANGUAGE:CXX>:-ffunction-sections>
    $<$<COMPILE_LANGUAGE:CXX>:-fdata-sections>
)

set_property(TARGET ggml-cuda8-ggml-backend-supports-op-smoke
    APPEND_STRING PROPERTY LINK_FLAGS " -Wl,--gc-sections")

target_link_libraries(ggml-cuda8-ggml-backend-supports-op-smoke
    ggml-cuda8-kernels
    ${CUDA_LIBRARIES}
)

