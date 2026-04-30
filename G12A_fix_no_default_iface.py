#!/usr/bin/env python3
import os
import time

PATH = "/workspace/notebooks/llama.cpp-ph2/ggml/src/ggml-cuda8/ggml-cuda8-ggml-backend.cpp"
TEXT = '// ggml/src/ggml-cuda8/ggml-cuda8-ggml-backend.cpp\n// G12A-fix: minimal ggml_backend_t wrapper/probe for CUDA8.\n//\n// This ggml version\'s ggml_backend_i does not expose get_default_buffer_type.\n// Therefore the CUDA8 default buffer type is exposed through the local helper\n// ggml_cuda8_ggml_backend_get_default_buffer_type(), not through backend->iface.\n\n#include "ggml-cuda8-ggml-backend.h"\n#include "ggml-cuda8-ggml-buffer.h"\n#include "ggml-backend-impl.h"\n\n#include <cstdio>\n#include <cstdlib>\n#include <cstring>\n\nstruct ggml_cuda8_ggml_backend_context {\n    int device;\n};\n\nstatic const char * cuda8_backend_get_name(ggml_backend_t backend) {\n    (void) backend;\n    return "CUDA8";\n}\n\nstatic void cuda8_backend_free(ggml_backend_t backend) {\n    if (backend == NULL) return;\n    ggml_cuda8_ggml_backend_context * ctx = (ggml_cuda8_ggml_backend_context *) backend->context;\n    if (ctx != NULL) std::free(ctx);\n    std::free(backend);\n}\n\nstatic struct ggml_backend_i cuda8_backend_i;\nstatic bool cuda8_backend_i_initialized = false;\n\nstatic void cuda8_backend_iface_init(void) {\n    if (cuda8_backend_i_initialized) return;\n    std::memset(&cuda8_backend_i, 0, sizeof(cuda8_backend_i));\n    cuda8_backend_i.get_name = cuda8_backend_get_name;\n    cuda8_backend_i.free = cuda8_backend_free;\n    cuda8_backend_i_initialized = true;\n}\n\nextern "C" ggml_backend_t ggml_cuda8_ggml_backend_init(int device) {\n    cuda8_backend_iface_init();\n\n    ggml_cuda8_ggml_backend_context * ctx =\n        (ggml_cuda8_ggml_backend_context *) std::malloc(sizeof(ggml_cuda8_ggml_backend_context));\n    if (ctx == NULL) return NULL;\n    ctx->device = device;\n\n    ggml_backend_t backend = (ggml_backend_t) std::malloc(sizeof(struct ggml_backend));\n    if (backend == NULL) {\n        std::free(ctx);\n        return NULL;\n    }\n\n    std::memset(backend, 0, sizeof(*backend));\n    backend->iface = cuda8_backend_i;\n    backend->context = ctx;\n    return backend;\n}\n\nextern "C" int ggml_cuda8_ggml_backend_is_cuda8(ggml_backend_t backend) {\n    if (backend == NULL || backend->iface.get_name == NULL) return 0;\n    const char * name = backend->iface.get_name(backend);\n    return name != NULL && std::strcmp(name, "CUDA8") == 0;\n}\n\nextern "C" ggml_backend_buffer_type_t ggml_cuda8_ggml_backend_get_default_buffer_type(ggml_backend_t backend) {\n    if (!ggml_cuda8_ggml_backend_is_cuda8(backend)) return NULL;\n    return ggml_cuda8_ggml_buffer_type();\n}\n'

if os.path.exists(PATH):
    backup = PATH + ".g12a-no-default-iface-backup-" + str(int(time.time()))
    with open(PATH, "r") as f:
        old = f.read()
    with open(backup, "w") as f:
        f.write(old)
    print("backup", backup)

with open(PATH, "w") as f:
    f.write(TEXT)

print("wrote", PATH)
print("G12A no-default-buffer-type iface fix complete.")
