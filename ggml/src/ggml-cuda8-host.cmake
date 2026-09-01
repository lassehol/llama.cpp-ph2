# ggml-cuda8-host.cmake
# G36: Import pre-built CUDA8 kernel library on the host side.
# Used when GGML_CUDA8_HOST=ON (host build with GCC 11, no nvcc).
#
# Expects pre-built artifacts at:
#   ${GGML_CUDA8_LIB_DIR}/libggml-cuda8-kernels.a
#   ${GGML_CUDA8_LIB_DIR}/cuda8-libs/libcudart_static.a
#   ${GGML_CUDA8_LIB_DIR}/cuda8-libs/libcudadevrt.a
#   ${GGML_CUDA8_LIB_DIR}/cuda8-libs/libculibos.a
#   ${GGML_CUDA8_LIB_DIR}/cuda8-headers/   (CUDA 8 headers)

message(STATUS "GGML CUDA8 HOST: importing pre-built CUDA8/Fermi kernel library")

# Default path: build-cuda8-parent under the repo root
if (NOT GGML_CUDA8_LIB_DIR)
    set(GGML_CUDA8_LIB_DIR "${CMAKE_SOURCE_DIR}/build-cuda8-parent")
endif()

set(_CUDA8_LIBS    "${GGML_CUDA8_LIB_DIR}/cuda8-libs")
set(_CUDA8_HEADERS "${GGML_CUDA8_LIB_DIR}/cuda8-headers")

# Two layouts, depending on how the container built the kernels:
#   nested - configured against the repo root (-DGGML_CUDA8=ON)
#   flat   - configured standalone from ggml/src/ggml-cuda8, which is the only
#            option when the container's cmake is older than 3.14
set(_CUDA8_KERNELS_NESTED "${GGML_CUDA8_LIB_DIR}/ggml/src/ggml-cuda8/libggml-cuda8-kernels.a")
set(_CUDA8_KERNELS_FLAT   "${GGML_CUDA8_LIB_DIR}/libggml-cuda8-kernels.a")

if (EXISTS "${_CUDA8_KERNELS_NESTED}")
    set(_CUDA8_KERNELS "${_CUDA8_KERNELS_NESTED}")
elseif (EXISTS "${_CUDA8_KERNELS_FLAT}")
    set(_CUDA8_KERNELS "${_CUDA8_KERNELS_FLAT}")
else()
    message(FATAL_ERROR
        "CUDA8 kernel library not found. Looked in:\n"
        "  ${_CUDA8_KERNELS_NESTED}\n"
        "  ${_CUDA8_KERNELS_FLAT}\n"
        "Build it first inside the CUDA8 container, then point GGML_CUDA8_LIB_DIR "
        "at that build directory.")
endif()

message(STATUS "  kernel lib:  ${_CUDA8_KERNELS}")
message(STATUS "  CUDA8 libs:  ${_CUDA8_LIBS}")
message(STATUS "  CUDA8 hdrs:  ${_CUDA8_HEADERS}")

# Import the pre-built static library
add_library(ggml-cuda8-kernels STATIC IMPORTED GLOBAL)
set_target_properties(ggml-cuda8-kernels PROPERTIES
    IMPORTED_LOCATION "${_CUDA8_KERNELS}"
)

# Create interface library for CUDA8 dependencies
add_library(ggml-cuda8-deps INTERFACE)
target_link_libraries(ggml-cuda8-deps INTERFACE
    "${_CUDA8_LIBS}/libcudart_static.a"
    "${_CUDA8_LIBS}/libcudadevrt.a"
    "${_CUDA8_LIBS}/libculibos.a"
    -lcuda
    -lrt
    -ldl
    -lpthread
)

# Include paths for CUDA8 headers + our backend headers
target_include_directories(ggml-cuda8-deps INTERFACE
    "${_CUDA8_HEADERS}"
    "${CMAKE_CURRENT_SOURCE_DIR}/ggml-cuda8"
)

# The backend-reg.cpp source needs to be compiled into the host build.
# Add it to ggml-base sources.
target_sources(ggml-base PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/ggml-cuda8/ggml-cuda8-backend-reg.cpp"
)

# Link the kernel library + CUDA deps into ggml-base
target_link_libraries(ggml-base PUBLIC
    ggml-cuda8-kernels
    ggml-cuda8-deps
)

# Define GGML_USE_CUDA8 globally so ggml-backend-reg.cpp picks it up
# Define GGML_USE_CUDA8 for all relevant targets
add_compile_definitions(GGML_USE_CUDA8)
target_compile_definitions(ggml-base PUBLIC GGML_USE_CUDA8)
if(TARGET ggml)
    target_compile_definitions(ggml PUBLIC GGML_USE_CUDA8)
endif()

message(STATUS "GGML CUDA8 HOST: ready (pre-built kernel library imported)")

# Also link CUDA8 kernels into ggml target (which contains ggml-backend-reg.cpp)
if(TARGET ggml)
    target_link_libraries(ggml PUBLIC
        ggml-cuda8-kernels
        ggml-cuda8-deps
    )
endif()
