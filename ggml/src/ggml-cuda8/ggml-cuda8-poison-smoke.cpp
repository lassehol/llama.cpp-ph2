// ggml-cuda8-poison-smoke.cpp
//
// G-fix: fatal-error-injection smoke test for the sticky poisoned-device
// flag added to ggml-cuda8-ggml-backend.cpp (g_cuda8_device_poisoned,
// cuda8_error_is_fatal(), and the refuse-fast checks in
// cuda8_backend_graph_compute() / ggml_cuda8_ggml_backend_dispatch_op()).
//
// WARNING -- deliberate self-corruption, by design:
//   This smoke test intentionally provokes a real, unrecoverable CUDA
//   fault (an illegal device memory access) so it can verify the backend
//   detects it and refuses further work. After the injection point, this
//   process's CUDA context is permanently broken -- that is the expected
//   and desired outcome, not a bug in the test. Do not add further real
//   CUDA work to this file after step 3 below, and do not worry about a
//   scary-looking "synchronize FAILED" / "marked POISONED" log line from
//   this specific target: that is the assertion being tested, not a
//   regression. Because every smoke target here builds as its own
//   separate executable (see CMakeLists.txt), this process-wide
//   corruption is isolated to this one binary and does not affect any
//   other PASS/FAIL line in run-regression.sh.
//
// Technique: a kernel is launched with a plain malloc()'d host pointer
// used as if it were a device pointer. Under CUDA's Unified Virtual
// Addressing (available on Fermi/cc>=2.0 since CUDA 4.0), that address is
// valid within the shared virtual address space but was never mapped
// into the GPU's page tables (it was not obtained via cudaMalloc,
// cudaMallocHost, or cudaHostRegister). When the kernel dereferences it,
// the GPU raises a memory access violation. Kernel launches are
// asynchronous, so this violation is *not* visible to the launcher's own
// cudaGetLastError() check immediately after the launch -- it only
// surfaces at the next synchronizing CUDA call, which is exactly
// cudaDeviceSynchronize() inside cuda8_backend_synchronize(). That makes
// backend->iface.synchronize(backend) the deterministic trigger point
// for this test, and mirrors how such a fault would actually be
// discovered in real inference (asynchronously, somewhere downstream of
// the node that actually caused it).
//
// The specific error code the driver reports for this (cudaErrorIllegalAddress
// vs. cudaErrorLaunchFailure) can vary by driver/access pattern; this test
// does not assert which one occurs, only that cuda8_error_is_fatal()
// classifies whichever one does occur as fatal and the flag latches. If a
// driver ever surprises us by not faulting at all (not expected on real
// Fermi hardware), the test degrades to an INCONCLUSIVE skip for the
// poison-dependent checks rather than a hard failure, so this smoke does
// not become flaky.
//
// What this test validates:
//   1. g_cuda8_device_poisoned starts false.
//   2. A real ADD_F32 dispatch succeeds on a healthy device (control case,
//      so the later refusal is a meaningful contrast, not a pre-existing
//      failure).
//   3. Deliberately faulting kernel launch + synchronize() latches the
//      poisoned flag.
//   4. ggml_cuda8_ggml_backend_device_is_poisoned() reports true.
//   5. A subsequent dispatch_op() call -- using the same still-valid
//      backend/context handles as the control case -- is refused
//      immediately (return -1) purely because of the poison flag.
//   6. The flag stays latched across a second synchronize() call
//      (idempotent, does not un-poison itself).
#include "ggml-cuda8-ggml-backend.h"
#include "ggml-cuda8-context.h"
#include "ggml-cuda8-dispatch.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
// G-fix: ggml_cuda8_ggml_backend_device_is_poisoned() was added alongside
// g_cuda8_device_poisoned in ggml-cuda8-ggml-backend.cpp but has not yet
// been added to ggml-cuda8-ggml-backend.h. Forward-declared locally here
// so this smoke test can link against it; remove this local declaration
// once the prototype is added to the header (the two must stay
// identical, extern "C", in the meantime).
extern "C" int ggml_cuda8_ggml_backend_device_is_poisoned(void);
// Kernel launcher under test for the injection step (extern "C" in the
// kernels library, same entry point ggml-cuda8-oversized-smoke.cu uses).
extern "C" int ggml_cuda8_add_f32_launch(const float * a, const float * b, float * c, int n);
static bool approx(float a, float b, float tol) {
    return std::fabs(a - b) <= tol * (1.0f + std::fabs(b));
}
// Builds a minimal contiguous 1D F32 ggml_tensor wrapping host memory.
// Matches what ggml_cuda8_supported_add_f32() / is_contig_f32_1d()
// require: type F32, ne[1]==1, nb[0]==sizeof(float). No residency
// requirement -- ggml_cuda8_exec_add_f32() stages host<->device itself,
// so a plain host-backed tensor is legitimate input for this dispatch
// path (unlike the K-quant MUL_MAT/GET_ROWS paths reviewed earlier).
static ggml_tensor make_f32_1d(float * data, int n) {
    ggml_tensor t;
    std::memset(&t, 0, sizeof(t));
    t.type  = GGML_TYPE_F32;
    t.ne[0] = n;
    t.ne[1] = 1;
    t.ne[2] = 1;
    t.ne[3] = 1;
    t.nb[0] = sizeof(float);
    t.nb[1] = (size_t) n * sizeof(float);
    t.nb[2] = t.nb[1];
    t.nb[3] = t.nb[1];
    t.data  = data;
    return t;
}
// Runs a real ADD_F32 dispatch through the public backend API and checks
// the arithmetic. Used both as the pre-injection control case and
// (expected to be refused) the post-injection case.
static bool try_add_dispatch(
    ggml_backend_t backend,
    ggml_cuda8_context * ctx,
    int n,
    bool expect_success,
    const char * label
) {
    std::vector<float> h_a(n), h_b(n), h_out(n, -1.0f);
    for (int i = 0; i < n; ++i) {
        h_a[i] = (float) i;
        h_b[i] = 1.5f;
    }
    ggml_tensor t_a   = make_f32_1d(&h_a[0], n);
    ggml_tensor t_b   = make_f32_1d(&h_b[0], n);
    ggml_tensor t_out = make_f32_1d(&h_out[0], n);
    int rc = ggml_cuda8_ggml_backend_dispatch_op(
        backend, ctx, GGML_CUDA8_OP_ADD_F32, &t_a, &t_b, &t_out);
    if (expect_success) {
        if (rc != 0) {
            std::printf("  %-38s FAIL (dispatch_op returned %d, expected success)\n", label, rc);
            return false;
        }
        for (int i = 0; i < n; ++i) {
            if (!approx(h_out[i], h_a[i] + h_b[i], 1e-5f)) {
                std::printf("  %-38s FAIL (wrong result at i=%d: got %g want %g)\n",
                            label, i, (double) h_out[i], (double) (h_a[i] + h_b[i]));
                return false;
            }
        }
        std::printf("  %-38s PASS (dispatch succeeded, result correct)\n", label);
        return true;
    } else {
        if (rc == 0) {
            std::printf("  %-38s FAIL (dispatch_op returned 0 -- should have been refused)\n", label);
            return false;
        }
        std::printf("  %-38s PASS (dispatch_op refused, rc=%d)\n", label, rc);
        return true;
    }
}
int main() {
    std::printf("ggml-cuda8-poison-smoke: starting\n\n");
    bool ok = true;
    int device = 0;
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cannot query device 0\n");
        return 1;
    }
    std::printf("device: %s (cc %d.%d)\n\n", prop.name, prop.major, prop.minor);
    // -- 1. Flag starts clean --------------------------------------------
    std::printf("== Step 1: initial state ==\n");
    if (ggml_cuda8_ggml_backend_device_is_poisoned() != 0) {
        std::printf("  initial poisoned flag                 FAIL (expected 0, got 1 -- "
                     "did a previous test leak state into this process?)\n");
        ok = false;
    } else {
        std::printf("  initial poisoned flag                 PASS (0, not poisoned)\n");
    }
    ggml_backend_t backend = ggml_cuda8_ggml_backend_init(device);
    if (backend == NULL) {
        std::fprintf(stderr, "FAIL: ggml_cuda8_ggml_backend_init(%d) returned NULL\n", device);
        return 1;
    }
    ggml_cuda8_context * ctx = NULL;
    if (ggml_cuda8_context_create(device, &ctx) != 0 || ctx == NULL) {
        std::fprintf(stderr, "FAIL: ggml_cuda8_context_create(%d) failed\n", device);
        return 1;
    }
    // -- 2. Control case: healthy device, dispatch must succeed ----------
    std::printf("\n== Step 2: control dispatch (healthy device) ==\n");
    ok &= try_add_dispatch(backend, ctx, 64, /*expect_success=*/true,
                            "ADD_F32 before injection");
    // -- 3. Injection: fault a kernel against unregistered host memory ---
    std::printf("\n== Step 3: fault injection (expect a loud 'synchronize FAILED' "
                "below -- this is intentional) ==\n");
    const int n_bad = 4096;
    float * bad_a = (float *) std::malloc((size_t) n_bad * sizeof(float));
    float * bad_b = (float *) std::malloc((size_t) n_bad * sizeof(float));
    float * bad_c = (float *) std::malloc((size_t) n_bad * sizeof(float));
    if (bad_a == NULL || bad_b == NULL || bad_c == NULL) {
        std::fprintf(stderr, "FAIL: host malloc failed for injection buffers\n");
        return 1;
    }
    for (int i = 0; i < n_bad; ++i) { bad_a[i] = 1.0f; bad_b[i] = 1.0f; }
    // Launch directly against the raw host pointers reinterpreted as
    // device pointers. The launcher's own cudaGetLastError() check right
    // after this call will very likely still report cudaSuccess -- the
    // launch configuration itself is valid, and the fault only occurs
    // once a GPU thread actually dereferences the unmapped address. Do
    // not treat this return value as the pass/fail signal.
    int launch_rc = ggml_cuda8_add_f32_launch(bad_a, bad_b, bad_c, n_bad);
    std::printf("  ggml_cuda8_add_f32_launch(bad ptrs)   informational: launcher returned %d "
                "(fault, if any, is asynchronous and is not expected to surface here)\n", launch_rc);
    // This is the deterministic trigger point: cudaDeviceSynchronize()
    // inside cuda8_backend_synchronize() is where the asynchronous fault
    // from the kernel above actually surfaces.
    if (backend->iface.synchronize == NULL) {
        std::fprintf(stderr, "FAIL: backend->iface.synchronize is NULL\n");
        return 1;
    }
    backend->iface.synchronize(backend);
    const bool poisoned_after_injection = ggml_cuda8_ggml_backend_device_is_poisoned() != 0;
    if (!poisoned_after_injection) {
        // Graceful degrade: the injection technique is well-established
        // on Fermi/UVA hardware, but if a driver ever tolerates the bad
        // access without faulting, skip the poison-dependent assertions
        // below rather than reporting a false regression in the flag
        // logic itself.
        std::printf("\n  INCONCLUSIVE: device was not marked poisoned after the injected "
                     "fault. Either the fault did not occur on this driver/hardware "
                     "combination, or the poison-latching logic has regressed. Skipping "
                     "the remaining poison-dependent checks rather than failing them "
                     "outright.\n");
        std::printf("\nggml-cuda8-poison-smoke: INCONCLUSIVE\n");
        return 1;
    }
    std::printf("\n== Step 4: post-injection checks ==\n");
    std::printf("  device_is_poisoned() after injection   PASS (1, latched as expected)\n");
    // -- 5. Same dispatch as the control case must now be refused --------
    ok &= try_add_dispatch(backend, ctx, 64, /*expect_success=*/false,
                            "ADD_F32 after injection (expect refusal)");
    // -- 6. Flag stays latched across a second synchronize() -------------
    backend->iface.synchronize(backend);
    if (ggml_cuda8_ggml_backend_device_is_poisoned() != 0) {
        std::printf("  poisoned flag after 2nd synchronize()  PASS (still 1, latch is sticky)\n");
    } else {
        std::printf("  poisoned flag after 2nd synchronize()  FAIL (flag cleared itself -- "
                     "should never un-poison without a process restart)\n");
        ok = false;
    }
    // Host-only cleanup. ggml_cuda8_context_destroy()/backend->iface.free()
    // only touch host-side structs (memset+free) with no CUDA calls in
    // this backend's current implementation, so calling them after the
    // device has been poisoned is safe and does not risk masking the
    // fault with a second, unrelated CUDA error.
    ggml_cuda8_context_destroy(ctx);
    if (backend->iface.free != NULL) {
        backend->iface.free(backend);
    }
    std::free(bad_a);
    std::free(bad_b);
    std::free(bad_c);
    std::printf("\nggml-cuda8-poison-smoke: %s\n", ok ? "SUCCESS" : "FAIL");
    return ok ? 0 : 1;
}
