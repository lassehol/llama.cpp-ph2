INSERTION 1 — new subsection, place immediately after section "3.3" (the
register-spill paragraph) and before section "### 4. What will actually fit
in 1 GB":

---

**3.4 Dispatcher/backend robustness gaps found in code review — FIXED in G51,
one hardware-verified.** (Description retained for context, same format as
3.1–3.3 above.)

Unlike 3.1–3.3, none of these were discovered by running a model — they
surfaced during a line-by-line review of the dispatcher and backend files,
because they only manifest under partial-failure or malformed-input
conditions the existing smoke fixtures don't construct.

- **Buffer leak in `exec_add_f32`.** Unlike the Q8_0 MUL_MAT and scalar
  ADD/MUL exec paths, a second or third buffer-allocation failure inside
  plain ADD_F32 returned early without freeing buffers already allocated.
  Since ADD_F32 is the single most frequently dispatched op (every
  residual/bias-add node), this would compound under repeated allocation
  pressure rather than being a one-off. Fixed to match the existing
  free-on-failure pattern.
- **ROPE/SWIGLU `supported_*` gates accepted params their kernels don't
  implement.** Same class of bug as 3.1 (SOFT_MAX_EXT): `op_params`-level
  checks (ROPE mode/ext_factor/attn_factor/freq_factors; SWIGLU glu_op
  variant) lived only in the `exec_*` functions, not the `supported_*`
  gates that a scheduler's `supports_op` hook would actually consult. A
  YaRN-scaled ROPE node or non-SWIGLU GLU node would be accepted as
  "supported" and only fail once `graph_compute()` was already running,
  instead of the scheduler cleanly falling back to CPU. Fixed by moving
  the checks into the `supported_*` gates; no behavior change on any
  currently-passing case (G45 ROPE NeoX, G40 SwiGLU).
- **K-quant `supported_*` gates missing null-data/residency checks.**
  `supported_mul_mat_q4k_f32`/`q6k_f32`/`get_rows_q4k`/`get_rows_q6k`
  checked only struct-pointer non-null, not `->data` non-null, unlike the
  Q8_0 path's shared `check_tensor_ptrs()`. The K-quant exec paths pass
  `->data` straight into the kernel launcher with no host<->device
  staging — correct for weight/activation throughput under the normal
  graph_compute residency contract, but with no defence if a direct
  dispatch call or scheduler bug ever violates that contract. Fixed with
  the shared null check plus a debug-build-only residency diagnostic
  (warns, does not reject, so it cannot introduce a false rejection).
- **`cuda8_backend_synchronize()` silently swallowed fatal CUDA errors.**
  `ggml_backend_i::synchronize` returns `void`, so a fatal
  `cudaDeviceSynchronize()` failure (illegal address, launch failure, ECC
  error, device assert — error classes that typically invalidate the CUDA
  context for the rest of the process) was only logged, not acted on. The
  *next* `graph_compute()` call would then also fail, but with a
  generic-looking error instead of one pointing back at the real cause.
  Fixed with a process-wide sticky `g_cuda8_device_poisoned` flag, latched
  on a narrow allow-list of fatal error codes and checked at the top of
  both `graph_compute()` and `ggml_cuda8_ggml_backend_dispatch_op()`, so
  further dispatch after a fatal fault is refused immediately with a
  message pointing at the original error. Also fixed a latent
  multi-device bug: `synchronize()` now calls `cudaSetDevice(ctx->device)`
  before syncing, rather than syncing whichever device happens to be
  current on the calling thread.
  **This is the one item in this list verified on real hardware**, not
  just by regression passing: a new smoke target,
  `ggml-cuda8-poison-smoke`, deliberately triggers an illegal-address
  fault (a `malloc()`'d host pointer reinterpreted as a device pointer
  under UVA) and confirms the full detect → latch → refuse chain on the
  GTX 560 / driver 390.157. Full detail in ggml-cuda8/README.md G51.

Full regression after G51: **24/24 pass** (was 23/23 — the new
`ggml-cuda8-poison-smoke` target accounts for the difference).

---

INSERTION 2 — new top-level section, place at the very end of the document,
after section "### 7. Verified: F16 conversion works on sm_21" and before the
closing italic attribution line ("_Roadmap drafted with AI assistance..._"):

---

### 8. G51: dispatcher robustness hardening (post-G45)

A code-review pass over ggml-cuda8-dispatch.cpp and
ggml-cuda8-ggml-backend.cpp found and fixed four issues unrelated to op
coverage — see §3.4 above for the technical detail and
ggml-cuda8/README.md G51 for the full writeup:

1. A buffer leak on partial allocation failure in `exec_add_f32`.
2. ROPE/SWIGLU `supported_*` gates that validated tensor types but not the
   `op_params` their kernels don't implement — the same silent-wrong-answer
   *shape* of bug as G37's SOFT_MAX_EXT fix, though here it degrades to a
   hard dispatch failure rather than a wrong answer, since the exec-side
   checks (added defensively at G40/G45 time) still catch it before any
   kernel launch.
3. K-quant `supported_*` gates missing the null-data check the Q8_0 path
   already had, plus a new debug-only residency diagnostic.
4. A sticky poisoned-device flag so a fatal CUDA error (illegal address,
   launch failure, ECC error, assert) is detected once and causes all
   further dispatch to be refused immediately, instead of degrading into
   an opaque failure on the next unrelated `graph_compute()` call.

Item 4 is verified end-to-end on real hardware via a new fault-injection
smoke target (`ggml-cuda8-poison-smoke`) — the first checkpoint in this
project to hardware-verify a failure-handling path rather than only a
success path. Full regression: 24/24.

None of these change dispatch op coverage. The G43 inventory (21 dispatch
ops + 5 no-ops) is unchanged; **G39's post-G40/G45 measurement — 2 refusals
per layer, MUL_MAT f32×f32 and SOFT_MAX soft_max_ext, i.e. exactly the
attention core — is still the current, accurate picture of what's left**
before a LLaMA-class model runs entirely on-GPU. G41/G42 (SOFT_MAX_EXT
proper + F32×F32 MUL_MAT, "a single unit of work" per §G39's
re-measurement note) remain the next highest-value checkpoint.

---

INSERTION 3 — small inline edit, section "### 1. Where the port actually
is", in the paragraph beginning "Two things exist but are not wired to the
graph path": no change needed (unrelated to G51).

INSERTION 4 — small inline edit, section "### 5. Proposed G37+ order", the
line for G38 currently reads:

    **DONE — PASS on GTX 560, 21/21.** §3.2. Register baseline (§3.3) wired
    but not yet measured.

No change needed — the register-baseline measurement is still genuinely
outstanding and unrelated to G51. Confirming here only so it isn't
mistakenly folded into this update: **§3.3's "still to do" item (running
-Xptxas -v and recording per-kernel register/spill counts, K-quant kernels
first) remains open.**
