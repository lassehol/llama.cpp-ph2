
### G51 status: dispatcher robustness hardening (post-G45)

Status: **PASS on GTX 560 / CUDA 8 / Fermi** (24/24 regression, including one
new hardware-verified fault-injection target).

G51 is a correctness/robustness pass rather than a new-op checkpoint. It
closes four gaps found during a code review of the dispatcher and backend
layers, none of which were caught by the existing smoke suite because they
only manifest under partial-failure or malicious/malformed-input conditions
that the happy-path fixtures don't exercise.

#### G51A: `exec_add_f32` buffer leak on partial allocation failure

`ggml_cuda8_exec_add_f32()` (ggml-cuda8-add.cpp) allocated three CUDA8
backend buffers (`b0`, `b1`, `bd`) in sequence but only freed buffers already
allocated on a *later* allocation failure in the Q8_0 MUL_MAT and scalar
ADD/MUL paths — the plain ADD_F32 path returned `-1` on a second or third
allocation failure without freeing the first one or two buffers already
allocated. ADD_F32 is one of the most frequently dispatched ops (every
residual/bias-add node in the graph-builder pipelines goes through it), so
a leak here compounds under repeated allocation pressure rather than being
a one-off. Fixed to match the cleanup pattern already used by
`exec_mul_mat_q8_0_f32_vec()` and `exec_scalar_f32_host_staging()`: each
allocation failure now frees every buffer already allocated before
returning.

Verified via `ggml-cuda8-add-smoke` (unchanged behavior on the success
path) plus the full graph-builder regression.

#### G51B: ROPE/SWIGLU `supported_*` gates missing `op_params` validation

`ggml_cuda8_supported_rope_f32()` and `ggml_cuda8_supported_swiglu_f32()`
(ggml-cuda8-dispatch.cpp) validated only tensor types/shapes — the
`op_params`-level checks (ROPE: `mode`, `ext_factor`, `attn_factor`,
`freq_factors`/`src[2]`; SWIGLU: the `glu_op` variant) lived exclusively in
the `exec_*` functions, with an explicit comment noting this was "defence
in depth... reaching here means the scheduler bypassed supports_op." This
is the same class of bug G37 fixed for SOFT_MAX_EXT: if a scheduler's
`supports_op` hook is wired to `ggml_cuda8_dispatch_supported()` (which
`ggml_cuda8_ggml_backend_dispatch_op()` does call directly), a YaRN-scaled
ROPE node or a non-SWIGLU GLU node (GEGLU, SWIGLU_OAI, etc.) would be
accepted as "supported," assigned to this backend by the scheduler, and
only discovered unsupported once `graph_compute()` was already mid-flight
— aborting the whole graph instead of the scheduler cleanly routing it to
CPU or another backend.

Moved the `op_params` checks into the `supported_*` gates (the exec-side
checks remain as defence-in-depth, with their error messages updated to
note they are now unreachable in the normal case). No behavior change on
any currently-passing smoke case — G45's ROPE NeoX and G40's SWIGLU paths
are unaffected, since both already pass the (now duplicated) checks.

Verified via `ggml-cuda8-rope-smoke`, `ggml-cuda8-swiglu-smoke`, and full
regression.

#### G51C: K-quant `supported_*` gates missing null-data and residency checks

`supported_mul_mat_q4k_f32`, `supported_mul_mat_q6k_f32`,
`supported_get_rows_q4k`, `supported_get_rows_q6k` checked only
`!src0 || !src1 || !dst` (struct-pointer non-null), unlike the Q8_0 MUL_MAT
path's `check_tensor_ptrs()`, which additionally rejects null `->data`. The
`exec_*` counterparts for these four ops pass `src0->data`/`src1->data`/
`dst->data` straight into the kernel launcher with no host<->device
staging — the architecturally correct choice for weight/activation
throughput, since the ggml scheduler contract guarantees CUDA8-buffer
residency for any tensor assigned to this backend under normal
`graph_compute` — but it left no defence for direct-dispatch callers (e.g.
tests bypassing `graph_compute`) or a scheduler bug that assigns a
host-backed tensor here, which would fault asynchronously inside the
kernel rather than failing cleanly at the dispatch boundary.

Fixed: all four `supported_*` gates now call the shared `check_tensor_ptrs()`
helper (closing the null-data gap), plus a debug-build-only
(`#ifndef NDEBUG`) residency diagnostic using the existing
`ggml_cuda8_ggml_tensor_is_device_resident()` registry lookup. The
diagnostic logs a loud warning if a K-quant tensor isn't registered as
CUDA8-resident but does not change `supported()`'s return value, so it
cannot introduce a false rejection for any currently-passing path.

Verified via `ggml-cuda8-getrows-smoke`, `ggml-cuda8-dispatch-all-smoke`,
and full regression — the residency diagnostic was live during this run
(non-`NDEBUG` build) and produced no warnings, confirming the K-quant
smoke fixtures do allocate through the CUDA8 buffer registry as expected.

#### G51D: sticky poisoned-device flag for fatal CUDA errors — hardware verified

`cuda8_backend_synchronize()` (ggml-cuda8-ggml-backend.cpp) previously only
logged a `cudaDeviceSynchronize()` failure to stderr and returned —
`ggml_backend_i::synchronize` is `void(*)(ggml_backend_t)`, so there was no
return channel to propagate failure through in the first place. A fatal
CUDA error class (`cudaErrorIllegalAddress`, `cudaErrorLaunchFailure`,
`cudaErrorECCUncorrectable`, `cudaErrorAssert`) typically invalidates the
CUDA context — and on most driver versions, the whole per-process CUDA
state for that device — until process exit. Silently logging and
continuing meant the *next* `graph_compute()` call (a fresh
`ggml_cuda8_context_create()`) would also fail, but with a generic,
unrelated-looking message instead of one pointing back at the original
fault.

Added a process-wide `std::atomic<bool> g_cuda8_device_poisoned`, latched
by `cuda8_backend_synchronize()` on a narrow allow-list of fatal error
codes (deliberately not "anything != cudaSuccess" — recoverable
launch-configuration errors like the grid-size class G38 fixed are
excluded). Checked at the top of both `cuda8_backend_graph_compute()` and
`ggml_cuda8_ggml_backend_dispatch_op()`, so any further dispatch attempt
after a fatal fault is refused immediately with a message pointing back at
the original error, rather than failing later with an opaque, unrelated
CUDA error. Also fixed `cuda8_backend_synchronize()` to call
`cudaSetDevice(ctx->device)` before syncing — previously dead code for the
single-GPU-per-process case, but a latent multi-device bug.

New smoke target: **`ggml-cuda8-poison-smoke`**. Deliberately launches a
kernel against unregistered host memory (a `malloc()`'d pointer reinterpreted
as a device pointer under CUDA's Unified Virtual Addressing) to provoke a
real illegal-address fault, then verifies:
1. the flag starts clean,
2. a control `ADD_F32` dispatch succeeds on the healthy device,
3. the injected fault + `synchronize()` latches the flag,
4. `ggml_cuda8_ggml_backend_device_is_poisoned()` reports `1`,
5. the same dispatch call is refused immediately post-injection,
6. the flag stays latched across a second `synchronize()` call.

**Verified on real hardware** — GTX 560 / CUDA 8.0.61 / driver 390.157. The
injection technique reliably faults on this driver/hardware combination
(rather than landing in the test's `INCONCLUSIVE` degrade path), and the
full assertion chain passed: fault detected, flag latched, subsequent
dispatch refused, latch confirmed sticky across a second synchronize call.
This is the first checkpoint in this backend's history to hardware-verify
a fault-handling path rather than only a happy path.

#### Regression

Added to the `PRIMARY` target set in `run-regression.sh` (targets exercising
the most recent changes), alongside `ggml-cuda8-oversized-smoke` and the
G37/supports-op targets:

```
PRIMARY=(
    ggml-cuda8-oversized-smoke
    ggml-cuda8-ggml-backend-supports-op-smoke
    ggml-cuda8-ggml-backend-graph-compute-softmax-smoke
    ggml-cuda8-ggml-backend-graph-compute-attnlike-smoke
    ggml-cuda8-poison-smoke
)
```

Full container regression: **24/24 pass** (was 23/23 before G51D added
`ggml-cuda8-poison-smoke`).

#### Notes

- None of G51A–D add or change dispatch op coverage — the 21 dispatch ops
  + 5 no-ops inventory from G43 is unchanged. This checkpoint is purely
  about failure-mode correctness in code paths the existing fixtures
  already exercised on the success side.
- `ggml_cuda8_ggml_backend_device_is_poisoned()` is defined in
  ggml-cuda8-ggml-backend.cpp but not yet declared in
  ggml-cuda8-ggml-backend.h — `ggml-cuda8-poison-smoke.cpp` currently
  carries a local `extern "C"` forward declaration as a stopgap. Worth
  adding the real prototype to the header the next time that file is
  touched.
- The register-spill baseline from G38 (`-DGGML_CUDA8_PTXAS_VERBOSE=ON`,
  §3.3 in docs/backport-cuda8.md) is still unmeasured — unrelated to this
  checkpoint, carried over as an open item.
