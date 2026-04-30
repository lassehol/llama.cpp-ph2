#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""G24A_dispatch_writer.py  -  Wire RMS_NORM_F32 into CUDA8 dispatch pipeline
   Python 3.5+
"""
import os, sys, re

ROOT = "/workspace/notebooks/llama.cpp-ph2"
BASE = os.path.join(ROOT, "ggml/src/ggml-cuda8")
ENC  = {"encoding": "utf-8"}

def read_file(path):
    with open(path, **ENC) as fh:
        return fh.read()

def write_file(path, content):
    with open(path, "w", **ENC) as fh:
        fh.write(content)

ok_all = True

# ============================================================================
# 1. ggml-cuda8-rms-norm.cu -- add extern "C" dispatch wrapper
#    (avoids exposing cudaStream_t to .cpp callers)
# ============================================================================

rms_cu = os.path.join(BASE, "ggml-cuda8-rms-norm.cu")
src = read_file(rms_cu)

WRAPPER = r"""
// -- G24A: extern "C" dispatch wrapper (called from .cpp) ---------------------
extern "C" int ggml_cuda8_op_rms_norm_f32(
        const float * x, float * y,
        int nrows, int ncols, float eps) {
    ggml_cuda8_rms_norm_f32(x, y, nrows, ncols, eps, 0);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : -1;
}
"""

if "ggml_cuda8_op_rms_norm_f32" not in src:
    src = src.rstrip() + "\n" + WRAPPER
    write_file(rms_cu, src)
    print("[G24A-disp] Patched %s  +dispatch wrapper" % rms_cu)
else:
    print("[G24A-disp] %s already has dispatch wrapper, skip" % rms_cu)

# ============================================================================
# 2. ggml-cuda8-dispatch.h -- add enum value
# ============================================================================

dispatch_h = os.path.join(BASE, "ggml-cuda8-dispatch.h")
src = read_file(dispatch_h)

if "GGML_CUDA8_OP_RMS_NORM_F32" not in src:
    anchor = "GGML_CUDA8_OP_SOFTMAX_ROWS_F32,"
    if anchor in src:
        idx = src.index(anchor) + len(anchor)
        src = src[:idx] + "\n\n    GGML_CUDA8_OP_RMS_NORM_F32," + src[idx:]
    else:
        idx = src.rfind("};")
        src = src[:idx] + "    GGML_CUDA8_OP_RMS_NORM_F32,\n" + src[idx:]
    write_file(dispatch_h, src)
    print("[G24A-disp] Patched %s  +enum" % dispatch_h)
else:
    print("[G24A-disp] %s already has enum, skip" % dispatch_h)

# ============================================================================
# 3. ggml-cuda8-dispatch.cpp -- helpers + name/supported/execute cases
# ============================================================================

dispatch_cpp = os.path.join(BASE, "ggml-cuda8-dispatch.cpp")
src = read_file(dispatch_cpp)

if "GGML_CUDA8_OP_RMS_NORM_F32" not in src:

    # 3a. Insert extern decl + static supported/exec functions
    #     before the dispatch_supported function definition
    HELPERS = """
// -- G24A: RMS_NORM_F32 helpers -----------------------------------------------
extern "C" int ggml_cuda8_op_rms_norm_f32(
        const float * x, float * y,
        int nrows, int ncols, float eps);

static int ggml_cuda8_supported_rms_norm_f32(
        const struct ggml_cuda8_context * ctx,
        const struct ggml_tensor * src0,
        const struct ggml_tensor * dst) {
    (void) ctx;
    if (src0 == NULL || dst == NULL) return 0;
    if (src0->type != GGML_TYPE_F32) return 0;
    if (dst->type  != GGML_TYPE_F32) return 0;
    return 1;
}

static int ggml_cuda8_exec_rms_norm_f32(
        struct ggml_cuda8_context * ctx,
        const struct ggml_tensor * src0,
        struct ggml_tensor * dst) {
    (void) ctx;
    float eps;
    std::memcpy(&eps, dst->op_params, sizeof(float));
    const int ncols = (int) src0->ne[0];
    const int nrows = (int)(src0->ne[1] * src0->ne[2] * src0->ne[3]);
    return ggml_cuda8_op_rms_norm_f32(
        (const float *) src0->data,
        (float *)       dst->data,
        nrows, ncols, eps);
}

"""

    anchor_func = "int ggml_cuda8_dispatch_supported("
    if anchor_func in src:
        idx = src.index(anchor_func)
        src = src[:idx] + HELPERS + src[idx:]
        print("[G24A-disp] Inserted helper functions")
    else:
        print("[G24A-disp] WARNING: cannot find dispatch_supported anchor")
        ok_all = False

    # 3b. Insert op_name case after SOFTMAX line
    m = re.search(
        r'(case GGML_CUDA8_OP_SOFTMAX_ROWS_F32:\s+return "SOFTMAX_ROWS_F32";)',
        src)
    if m:
        ins = '\n        case GGML_CUDA8_OP_RMS_NORM_F32:        return "RMS_NORM_F32";'
        src = src[:m.end()] + ins + src[m.end():]
        print("[G24A-disp] Inserted op_name case")
    else:
        print("[G24A-disp] WARNING: cannot find SOFTMAX op_name line")
        ok_all = False

    # 3c. Insert supported + execute cases before respective default: blocks
    #     3 defaults expected: op_name, dispatch_supported, dispatch_execute
    defaults = [m.start() for m in re.finditer(r'^\s*default:', src, re.MULTILINE)]

    supp_case = (
        "        case GGML_CUDA8_OP_RMS_NORM_F32:\n"
        "            return ggml_cuda8_supported_rms_norm_f32(ctx, src0, dst);\n\n"
    )
    exec_case = (
        "        case GGML_CUDA8_OP_RMS_NORM_F32:\n"
        "            return ggml_cuda8_exec_rms_norm_f32(ctx, src0, dst);\n\n"
    )

    if len(defaults) >= 3:
        # 3rd default = dispatch_execute, 2nd = dispatch_supported
        # Insert in reverse order to preserve earlier positions
        src = src[:defaults[2]] + exec_case + src[defaults[2]:]
        src = src[:defaults[1]] + supp_case + src[defaults[1]:]
        print("[G24A-disp] Inserted supported + execute switch cases")
    else:
        print("[G24A-disp] ERROR: expected 3 default: blocks, found %d" % len(defaults))
        ok_all = False

    write_file(dispatch_cpp, src)
    print("[G24A-disp] Patched %s" % dispatch_cpp)
else:
    print("[G24A-disp] %s already has RMS_NORM_F32, skip" % dispatch_cpp)

# ============================================================================
# 4. ggml-cuda8-ggml-backend.cpp -- map GGML_OP_RMS_NORM -> CUDA8 op
# ============================================================================

backend_cpp = os.path.join(BASE, "ggml-cuda8-ggml-backend.cpp")
src = read_file(backend_cpp)

if "GGML_OP_RMS_NORM" not in src:
    CASE = (
        '\n'
        '            case GGML_OP_RMS_NORM: {\n'
        '                if (!cuda8_graph_is_f32(node) || !cuda8_graph_is_f32(src0)) {\n'
        '                    std::fprintf(stderr,\n'
        '                        "ggml-cuda8/backend graph_compute: RMS_NORM node %d unsupported types\\n", i);\n'
        '                    ggml_cuda8_context_destroy(ctx);\n'
        '                    return (enum ggml_status) -1;\n'
        '                }\n'
        '\n'
        '                cuda8_op = GGML_CUDA8_OP_RMS_NORM_F32;\n'
        '                opname = "RMS_NORM_F32";\n'
        '            } break;\n\n'
    )

    if "default:" in src:
        idx = src.rfind("default:")
        src = src[:idx] + CASE + "            " + src[idx:]
        write_file(backend_cpp, src)
        print("[G24A-disp] Patched %s  +GGML_OP_RMS_NORM" % backend_cpp)
    else:
        print("[G24A-disp] WARNING: no default: in backend.cpp")
        ok_all = False
else:
    print("[G24A-disp] %s already has GGML_OP_RMS_NORM, skip" % backend_cpp)

# ============================================================================
# Done
# ============================================================================

if ok_all:
    print("""
[G24A dispatch] All patches applied successfully.

Build & test:
  cd /workspace/notebooks/llama.cpp-ph2/build-cuda8-parent
  cmake .. && make -j$(nproc) ggml-cuda8-rms-norm-smoke
  ./bin/ggml-cuda8-rms-norm-smoke
""")
else:
    print("\n[G24A dispatch] Some patches had warnings -- review output above.")
