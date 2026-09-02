# CUDA 8 / Fermi backport — state of the port and roadmap to real GGUF inference

Target hardware: GeForce GTX 560, 1 GB (GF114, compute capability 2.1).
Target toolkit: CUDA 8.0 — the last toolkit that emits sm_20/sm_21.
Host environment: Ubuntu 22.04 server, NVIDIA driver **390.157**, pinned (the 390 legacy branch is
the last one supporting Fermi).
Approach taken: **separate backend** at `ggml/src/ggml-cuda8/`, registered alongside (not replacing)
ggml-cuda, gated by `-DGGML_CUDA8=ON`.

This document supersedes nothing that came before it — the authoritative checkpoint log is
`ggml/src/ggml-cuda8/README.md` (G16–G56). This is the forward-looking half: what is done,
what is missing, and in what order the gaps should be closed.

> **Status at head of tree (G56).** A real Q4_K model (Qwen3-0.6B-Q4_K_M) runs **fully
> GPU-resident** on the GTX 560 — attention included, as a single graph segment per token —
> and generates coherent text at **CPU parity** (~5.1 tok/s vs ~6 tok/s on the host's Phenom
> II X6 1055T). Both correctness and usable performance are achieved. The narrative of how the
> port got from G36 to here is in sections 8–13; the original forward-looking roadmap (sections
> 1–7) is preserved below for context, with completed items struck through.

### 1. Where the port actually is

G36 was the last green checkpoint of the original build-out. The backend:
- registers itself into the ggml backend registry as CUDA8, filtered to compute capability 2.x–3.x so modern GPUs still get the standard CUDA backend
- implements a full `ggml_backend_device_i` (buffer type, supports_op, init_backend, offload_op)
- runs a **15-node LLaMA-shaped graph end-to-end on real hardware**, max error 1.19e-06
- solves the toolchain split via `GGML_CUDA8_HOST` (ggml/src/ggml-cuda8-host.cmake): a static kernel archive built in a CUDA 8 container is imported as IMPORTED STATIC with CUDA 8's libcudart_static/libcudadevrt/libculibos, while only ggml-cuda8-backend-reg.cpp is compiled with the modern host compiler. This was the phase-0 risk (nvcc's C11 / GCC ≤5.3 ceiling vs. llama.cpp's C17) and it is retired.

**Build mode (settled, not a stopgap).** Two-stage, split by C++ standard:

| Stage | Where | Toolchain | Configures | Produces |
|---|---|---|---|---|
| 1. Kernels | NVIDIA ML container, Ubuntu 16.04 base | CUDA **8.0.61**, GCC 5.4 / C++11, cmake 3.5.1 | ggml/src/ggml-cuda8 standalone → build-cuda8-kernels | libggml-cuda8-kernels.a + all smoke targets |
| 2. Host | Ubuntu 22.04 host | C++17 build tools, cmake ≥3.14 | repo root | llama-server / rpc-server, importing stage 1 via GGML_CUDA8_HOST |

Stage 1 configures the ggml-cuda8 subdirectory rather than the repo root: the root
requires cmake ≥3.14 and the container has 3.5.1, while ggml-cuda8/CMakeLists.txt is a
standalone project needing only 3.5 that reaches the ggml core by relative path. `ggml_abort`
and the `ggml_backend_tensor_*` symbols are handled by a separate ggml-cuda8-ggml-core
library plus --gc-sections; see G38F in ggml-cuda8/README.md for why neither can go into
the kernel archive itself.

The ggml-cuda8 public surface is C, so no C++ ABI crosses the boundary. Note that rpc-server
is in scope, which means the Fermi box can also be driven as a _remote_ ggml backend rather than
hosting the whole model — a useful escape valve given the 1 GB limit in §4.

**Implemented dispatch ops (23) + no-ops (5), at head of tree:**

```
CPY_F32   ADD_F32   ADD_SCALAR_F32   MUL_SCALAR_F32   MUL_F32   MUL_BROADCAST_F32
REDUCE_SUM_ROWS_F32   REDUCE_MAX_ROWS_F32   SOFTMAX_ROWS_F32   SOFTMAX_EXT_F32
MUL_MAT_Q8_0xF32_VEC   MUL_MAT_Q4_K_F32   MUL_MAT_Q6_K_F32   MUL_MAT_F32_F32
RMS_NORM_F32   ROPE_F32   CONT_F32   DIAG_MASK_INF_F32   SWIGLU_F32   SET_ROWS_F32
GET_ROWS_F32   GET_ROWS_Q4_K   GET_ROWS_Q6_K
no-ops: NONE  RESHAPE  VIEW  PERMUTE  TRANSPOSE
```

All Fermi-safe: shared-memory tree reductions, no warp shuffle anywhere in the directory, no
fp16 arithmetic, no dp4a, no tensor cores.

The `mmv.cu` F32×F32 matvec kernels remain present but unused by any dispatch op — see §2.1 and
§10 for why the attention matmuls needed a purpose-built batched kernel (G42) rather than the
vector one.

### 2. The gap between G36 and llama-cli

The G35 pipeline is LLaMA-_shaped_, but it is a hand-built graph with hand-chosen shapes. A real
GGUF graph differs in ways that matter. Because supports_op falls back to CPU, none of these
are _correctness_ blockers — they are **graph-split** blockers, and a graph that splits between
CPU and GPU on every other node will be slower than pure CPU.

#### 2.1 MUL_MAT — resolved

Current support, at head of tree:

| src0 type | Batching | Notes |
|---|---|---|
| **Q4_K** | true ne11 batching | ggml-cuda8-q4k.cu, one block per output element, 2D grid clamped for Fermi. Thread-utilization rewrite at G56 (§13). |
| **Q6_K** | true ne11 batching | ggml-cuda8-q6k.cu, same shape. Thread-utilization rewrite at G56. |
| **F32 × F32** | batched, GQA-broadcast, permuted views | **DONE (G42, §10).** The attention matmuls (K·Q, probs·V). |
| Q8_0 | per-token loop | Runs on GPU but with a host round-trip per token. Now largely superseded — prefer Q4_K/Q6_K, which are both better for 1 GB and properly batched. |

Remaining gaps:

| Gap | Status |
|---|---|
| ~~F32 × F32 not wired~~ | **DONE (G42).** The original claim that "the kernels already exist (mmv.cu:210) — they just need a dispatch op id" was **wrong** (see the G39 correction, §10): real attention matmuls are per-head batched and 3D, often on permuted views. G42 added a purpose-built batched kernel, not a wiring of the vector one. |
| ~~Permuted / non-contiguous src1, 3D+ shapes~~ | **Mostly addressed by G42.** The kernel takes explicit strides on dims 1-3, so permuted views with contiguous dim 0 (the common attention case) work. Only fully arbitrary dim-0 strides remain out of scope, refused to CPU — and may never matter in practice. |
| F16 src0 | token_embd / output are sometimes F16 even in quantized GGUFs. Needs G49. |

Q4_0 is no longer on the critical path: Q4_K supersedes it for the 1 GB budget and is done.
No cuBLAS is used anywhere in ggml-cuda8/, and with Q4_K/Q6_K batched natively it may not be
needed at all.

#### 2.2 Attention — resolved

- ~~**SOFT_MAX_EXT is a silent wrong-answer bug**~~ — guarded at G37 (§3.1), implemented properly at G41 (§9).
- ~~KV cache writes use GGML_OP_SET_ROWS~~ — **DONE (G43).**
- KV cache defaults to F16. Fermi has no fp16 _arithmetic_. **Workaround until G49:**
`--cache-type-k f32 --cache-type-v f32`. Costs 2× KV memory; on a small model with short context
that is affordable. F16 _conversion_ is confirmed working on sm_21 (§7), so an F16 cache with
F32 compute is reachable.
- GGML_OP_FLASH_ATTN_EXT — will never be supported; leave it unsupported so ggml decomposes it
(pass `-fa off`, since the default `auto` enables FA via the CPU backend and folds the whole
attention block into one CPU op).

#### 2.3 FFN — resolved

- ~~GGML_OP_GLU / GGML_GLU_OP_SWIGLU~~ — **DONE (G40).** Both the split and halves forms.
- GGML_OP_UNARY (SILU / GELU) for models that use the unfused form — not needed by the models tested.

#### 2.4 Smaller gaps

- ~~GET_ROWS with **quantized** src0~~ — **done** for Q4_K and Q6_K. Other quant types still fall back.
- GGML_OP_SCALE (distinct node from MUL-by-scalar) — G44, now premature (see §13).
- ADD with broadcast (bias rows) — G44.
- ~~ROPE mode NEOX (mode=2)~~ — **DONE (G45).** Frequency scaling / YaRN still refused explicitly.
- ~~RMS_NORM over multi-row 2D input~~ — verified in real-model runs (Qwen3 applies per-head QK-norm).

### 3. Latent correctness risks in the existing kernels

These issues produce **wrong numbers rather than errors**, and none of the smoke tests that
predate each fix could catch them because they use small, simple shapes. Every one of them
below was ultimately caught by a **real-model** run or a code review, not by the happy-path
fixtures — the recurring lesson of this port.

**3.1 SOFT_MAX_EXT accepted but computed incorrectly. — FIXED in G37, verified on hardware.**
supports_op claimed any F32 GGML_OP_SOFT_MAX **without inspecting src[1] (the mask) or the
scale/max_bias params**. The kernel took only (src, dst, rows, cols) and never read op_params.
So a real attention graph calling `ggml_soft_max_ext(kq, mask, scale, max_bias)` was routed to
the GPU and silently computed an **unmasked, unscaled** softmax: no crash, no fallback, just
plausible-looking wrong attention weights. Immediate mitigation was to reject SOFT_MAX when
`src[1] != NULL || scale != 1.0f || max_bias != 0.0f`, restoring correct CPU fallback; the real
fix (G41, §9) followed.

**3.2 gridDim.x maxes at 65535 on compute 2.x — FIXED in G38, verified on hardware (21/21).**
(It is 2³¹−1 only from sm_30.) Every launcher checks `cudaGetLastError()` immediately after its
launch, so an over-limit grid is rejected with cudaErrorInvalidConfiguration and surfaces as a
dispatch failure — loud, not silent. But it means any model with a tensor above ~16.7 M elements
(a 4096×4096 weight matrix, at 256 threads/block) simply cannot run. The genuinely silent variant
is _clamping without a stride loop_: the launch succeeds and quietly computes only the first
65535 blocks' worth of output. G38 converted the remaining launches to clamp + stride loop and
added an oversized-tensor smoke that checks _past_ the boundary.

**3.3 63 registers per thread on compute 2.x (255 from sm_35). — MEASURED at G55, clean.**
Spills to local memory are invisible without measurement. `-DGGML_CUDA8_PTXAS_VERBOSE=ON` was
wired at G38 but not run until G55: q4k mul_mat uses 24 registers, q6k 34 registers, **both with
zero spill stores / loads**. The long-standing spill hypothesis is disproved. (The item is
closed; the Q6_K/Q4_K speed gap that survived the G56 rewrite is _occupancy_, not spills — see
§13.)

**3.4 Dispatcher/backend robustness gaps found in code review — FIXED in G51, one hardware-verified.**
Unlike 3.1–3.3, none of these were discovered by running a model — they surfaced during a
line-by-line review of the dispatcher and backend files, because they only manifest under
partial-failure or malformed-input conditions the existing fixtures don't construct.
- **Buffer leak in `exec_add_f32`.** Unlike the Q8_0 MUL_MAT and scalar ADD/MUL exec paths, a
second or third buffer-allocation failure inside plain ADD_F32 returned early without freeing
buffers already allocated. Since ADD_F32 is the single most frequently dispatched op, this
would compound under repeated allocation pressure. Fixed to match the existing free-on-failure
pattern.
- **ROPE/SWIGLU `supported_*` gates accepted params their kernels don't implement.** Same class
as 3.1: op_params-level checks (ROPE mode/ext_factor/attn_factor/freq_factors; SWIGLU glu_op
variant) lived only in the exec_* functions, not the supported_* gates a scheduler's supports_op
hook consults. A YaRN-scaled ROPE node or non-SWIGLU GLU node would be accepted as "supported"
and only fail once graph_compute() was running, instead of cleanly falling back to CPU. Fixed by
moving the checks into the supported_* gates.
- **K-quant `supported_*` gates missing null-data/residency checks.** The four K-quant gates
checked only struct-pointer non-null, not `->data` non-null, unlike the Q8_0 path's shared
`check_tensor_ptrs()`. The K-quant exec paths pass `->data` straight into the kernel launcher
with no host↔device staging — correct under the normal residency contract, but with no defence
if a direct dispatch call or scheduler bug ever violates it. Fixed with the shared null check
plus a debug-build-only residency diagnostic (warns, does not reject).
- **`cuda8_backend_synchronize()` silently swallowed fatal CUDA errors.** `ggml_backend_i::synchronize`
returns void, so a fatal `cudaDeviceSynchronize()` failure (illegal address, launch failure, ECC
error, device assert — classes that typically invalidate the CUDA context for the rest of the
process) was only logged. The _next_ graph_compute() call would then also fail, but with a
generic-looking error instead of one pointing at the real cause. Fixed with a process-wide sticky
`g_cuda8_device_poisoned` flag, latched on a narrow allow-list of fatal error codes and checked at
the top of both graph_compute() and `ggml_cuda8_ggml_backend_dispatch_op()`. Also fixed a latent
multi-device bug: synchronize() now calls `cudaSetDevice(ctx->device)` before syncing.
**This is the one item verified on real hardware**: a new smoke target, `ggml-cuda8-poison-smoke`,
deliberately triggers an illegal-address fault and confirms the full detect → latch → refuse chain
on the GTX 560 / driver 390.157. Full detail in ggml-cuda8/README.md G51.

**3.5 Transposed dst write in K-quant MUL_MAT — FIXED in G53, verified on a real model.**
Both Q4_K and Q6_K MUL_MAT kernels wrote the output element at `dst[row*ne11 + col]` instead of
the ggml layout `dst[col*ne01 + row]`. These are equal only at ne11 == 1 (matvec), which is the
only shape every K-quant smoke and graph-builder checkpoint (G17C, G18-G23) ever used — so all of
them passed while the kernel was wrong for any ne11 > 1 (prompt prefill). Q8_0 was unaffected (its
per-token loop writes the correct layout), which is why SmolLM2-Q8_0 worked but Qwen3-Q4_K
produced structured-loop garbage. Same failure mode as 3.1/3.4: wrong numbers, graph_compute
reports SUCCESS, nothing fails loudly. Fixed (one line each in q4k.cu / q6k.cu) plus a new
ne11 > 1 regression smoke — the K-quant smokes could not have caught this because they were all
ne11 == 1. See ggml-cuda8/README.md G53.

**3.6 Flat-copy CONT ignored src0 strides — FIXED in G55, verified on a real model.**
`ggml_cuda8_exec_cont_f32` flat-copied `src0->data` as one contiguous byte run, never reading
`src0->nb[]`, and `supported_cont_f32` accepted any F32 without a contiguity check. But CONT
exists ONLY to make a non-contiguous tensor contiguous (post-permute, in attention) — ggml never
emits a CONT on already-contiguous input — so it was always handed a permuted src0 and always
copied it in the wrong element order. Invisible until attention ran on GPU (G55 config, §12),
because the attention CONTs were the only non-contiguous ones and they never executed while
attention was CPU-side. Same class as 3.5: wrong numbers, SUCCESS reported, no smoke test fed it
permuted input. Fixed with a strided-gather kernel (ggml-cuda8-cont.cu). See ggml-cuda8/README.md G55.

### 4. What will actually fit in 1 GB

GTX 560: ~1.26 TFLOPS fp32, **128 GB/s** memory bandwidth, **1 GB VRAM** (confirmed — the 1 GB
variant, not the 2 GB one).

Token generation is bandwidth-bound in principle, so 128 GB/s is the relevant number — roughly
2–3× a dual-channel DDR4 desktop. (In practice the hand-written matmuls did not approach that
bandwidth until the G56 rewrite; see §13.)

VRAM is the binding constraint, and it drives the quantization priority:

| Model | Q8_0 | Q4_K_M | Fits in ~850 MB usable? |
|---|---|---|---|
| Qwen3 0.6B | ~0.7 GB | ~0.4 GB | Q4 yes, Q8_0 marginal |
| TinyLlama 1.1B | ~1.2 GB | ~0.67 GB | **Q4 only** |
| Llama 3.2 1B | ~1.4 GB | ~0.8 GB | Q4, tight |
| 3B class | — | ~1.9 GB | partial offload only |

**Q4_K support removes the binding constraint.** With Q4_K and Q6_K implemented, TinyLlama 1.1B
and Llama 3.2 1B are both in range at Q4_K_M. Qwen3 0.6B is the roomiest option and is the model
the whole G39–G56 arc was validated against.

Note one practical wrinkle discovered at G55: with the KV cache device-resident (required for
GPU attention), the **full-context F32 cache is ~896 MiB and does not fit** alongside Q4_K
weights. Cap context with `-c 512` (cache ~11 MiB) until G49 lands an F16 cache.

### 5. Roadmap (original G37+ order, updated)

Ordered by "cost to implement ÷ graph splits eliminated", not by tidiness. Completed items struck through.

| # | Work | Status |
|---|---|---|
| ~~**G37**~~ | ~~Tighten supports_op for SOFT_MAX (reject mask / sinks / scale / max_bias)~~ | **DONE — 11/11.** §3.1. |
| ~~**G38**~~ | ~~Grid clamps on the remaining launches; oversized-tensor smoke; -Xptxas -v~~ | **DONE — 21/21.** §3.2. Register baseline measured clean at G55 (§3.3). |
| ~~**G39**~~ | ~~Load a real GGUF; log the CPU/GPU split and per-op fallback reasons~~ | **DONE.** §8. The measurement that turned the rest of the list from guesswork into data. |
| ~~**G40**~~ | ~~GLU / SWIGLU~~ | **DONE — 22/22.** 3080 in the G39 log. Both split and halves forms. |
| ~~**G41**~~ | ~~SOFT_MAX_EXT proper: mask + scale + max_bias~~ | **DONE — 25/25.** §9. Mask + scale + ALiBi; sinks and F16 masks refused. |
| ~~**G42**~~ | ~~Wire the existing F32×F32 matvec~~ Batched F32×F32 MUL_MAT, purpose-built kernel | **DONE — 26/26.** §10. The vector kernel was NOT reusable (G39 correction). GQA broadcast + dims-1-3 strides. |
| ~~**G43**~~ | ~~SET_ROWS + F32 KV cache path~~ | **DONE — hardware verified.** §11. Reordered ahead of G41/G42: -nkvo pins attention to CPU, so the attention kernels were pointless until the cache could be device-resident. F32 dst only — --cache-type f32 until G49. |
| **G44** | SCALE; broadcast ADD | Deferred — now premature (§13): GPU coverage is not the constraint. |
| ~~**G45**~~ | ~~ROPE NEOX~~ | **DONE — 21/21.** Top of the G39 log at 4312. freq_factors and attn_factor now refused explicitly. |
| ~~**G46**~~ | ~~Permuted / non-contiguous src1 for MUL_MAT~~ | **Mostly done via G42** (§2.1, §10). Only fully arbitrary dim-0 strides remain out of scope. |
| **G47** | Q8_0 batched MUL_MAT, or drop Q8_0 to CPU | Only worth doing if a Q8_0 model is actually wanted; Q4_K is the better fit for 1 GB. |
| **G49** | F16 _storage_ support (convert on load, compute in F32) | Removes the --cache-type f32 workaround and the -c 512 cap; handles F16 token_embd/output. **Unblocked** — conversion verified on sm_21, §7. |
| **G50** | Perf: occupancy, block-size retune, __launch_bounds__ | Largely subsumed by the G56 thread-utilization rewrite (§13); Q6_K occupancy is the residual. |

If any new measurement contradicts this ordering, follow the measurement — the principle G39
established and G53/G55/G56 confirmed three more times.

### 6. Housekeeping

`ggml/src/ggml-cuda8/` accumulated a large number of backup files (`*.gNN*-backup-<epoch>`,
`*.fix-*`, `*-reset-*`) not covered by .gitignore. They should be deleted — git already provides
the history they duplicate, and they make the directory listing unreadable.

A **git-working-tree-under-OneDrive hazard** was also hit: a second checkout living inside a
OneDrive-synced folder had its files silently overwritten with stale content (and CRLF-corrupted)
while git still reported "up to date". The authoritative tree is the container/Athlon checkout on
native storage; any mirror should be treated as read-only and `git reset --hard origin/main`'d,
never edited or pushed from. A `.gitattributes` with `* text=auto eol=lf` resists the CRLF half
of the problem but not the stale-content half — keep git working trees out of OneDrive entirely.

### 6b. Upstream sync to 0177dcc (RPC protocol compatibility) — DONE

**Outcome.** Merged as 2259d07. Verified afterwards: RPC_PROTO_MAJOR_VERSION 5,
`static_assert(GGML_OP_COUNT == 101)`, and both CUDA8 integration points intact
(ggml/src/CMakeLists.txt, ggml-backend-reg.cpp). No CUDA8 source changes were needed — the
backend ABI prediction held. Full container regression on the merged tree passed with no rebuild
breakage.

The merge produced ~31 conflicts (1 real content conflict resolved by taking upstream; ~30
modify/delete conflicts on workflow files the fork had deleted; .gitattributes kept from the
fork). The estimate that predicted "two files, ~15 lines" measured the CUDA8 touchpoints
correctly but missed that **the fork was created by importing a snapshot rather than forking git
history**, so git had almost no shared history to merge against. Every future upstream sync will
be similarly noisy for the same reason; re-applying the CUDA8 work onto a real clone of llama.cpp
would make subsequent merges routine.

### 7. Verified: F16 conversion works on sm_21

**Result: G49 is unblocked.** Confirmed in the CUDA 8.0.61 container two independent ways.

**(a) Header guards.** In cuda_fp16.h, `__float2half` and `__half2float` sit **outside** every
`__CUDA_ARCH__` block. The `>= 530` guards cover only the fp16 _arithmetic_ API (`__hadd`,
`__hmul`, …), which stays unavailable. Conversions are unguarded.

**(b) Generated PTX at sm_21 — decisive.** A probe kernel calling `__float2half`/`__half2float`
compiles to single `cvt.rn.f16.f32` / `cvt.f32.f16` hardware instructions at `-arch=sm_21`.

**Scope limit.** Conversion only. All F16 arithmetic still happens in F32, so G49 means "read and
write F16 buffers, compute in F32" — enough for an F16 KV cache and F16 token_embd/output.

---

### 8. G39: first real GGUF, and the CPU/GPU split log

Everything before this point was validated against hand-built graphs. A real GGUF graph differs in
ways that are hard to predict, so the point of G39 was to stop predicting and look. The result is
a measurement, not a feature.

**GGML_CUDA8_DEBUG_OPS=1** — supports_op returns a bare bool, so a model that silently falls back
to CPU leaves no record of what was refused. ggml-cuda8-backend-reg.cpp now logs each distinct
refused op signature once, and prints a frequency-ordered summary at exit. That ordering _is_ the
roadmap. It complements ggml's own `GGML_SCHED_DEBUG=2` (which shows _where_ each node ran) by
recording _why_ a backend declined a node.

**First run (Qwen3-0.6B-Q4_K_M).** The model loads on the GPU — 372 MiB weights, 306 MiB compute
buffer, 190 MiB free of 963 MiB. Refusals were dominated by **ROPE and SWIGLU by an order of
magnitude** (ROPE is the Qwen3 NeoX case; SWIGLU is the entire FFN). `MUL_MAT f32×f32` at 448 is
the attention matmuls — activations, not weights, so the Q4_K path never sees them.

Two invocation requirements surfaced:
- **`-fa off`** — the default `auto` enables FLASH_ATTN_EXT via the CPU backend, folding the whole
attention block into one CPU op, so implementing SOFT_MAX/MUL_MAT would change nothing while FA is on.
- **`-nkvo` until SET_ROWS lands (G43)** — the KV cache is written with `ggml_set_rows`; with
`-ngl 99` the cache is allocated in the CUDA8 buffer, so a refused SET_ROWS has no legal placement
and calls GGML_ABORT rather than falling back. This is the one class of unsupported op that is
fatal rather than merely slow.

**Re-measured after G45 (ROPE NeoX) and G40 (SwiGLU).** Six entries totalling ~8300 became two
totalling 168 — at 28 layers that is 2 matmuls and 1 softmax per layer, **exactly the attention
core and nothing else**. Consequence: **G41 (SOFT_MAX_EXT) and G42 (F32 MUL_MAT) are a single unit
of work** — they interleave inside the same attention block, so doing only one still leaves the
graph splitting on every layer.

**Bug found and fixed: double free during teardown.** `cuda8_free_buffer()` ended with
`std::free(buffer)`, but `ggml_backend_buffer_free()` calls that hook and then does `delete buffer`
itself — the struct was released twice, with mismatched allocators. Fixed by allocating the struct
with `new` (matching ggml's `delete`) and leaving its lifetime to ggml; the hook now owns only the
device allocation and our context. Nothing caught it earlier because the smoke tests never do a
full model teardown.

### 9. G41: SOFT_MAX_EXT proper (mask + scale + ALiBi)

Lifts the G37 restriction for real attention softmax. A new dispatch op,
`GGML_CUDA8_OP_SOFTMAX_EXT_F32`, sits alongside the unchanged plain `SOFTMAX_ROWS_F32` path and
implements the `ggml_soft_max_ext(kq, mask, scale, max_bias)` shape that real attention graphs
actually call — full technical detail and verification method in ggml-cuda8/README.md G41.

Explicitly still refused (loud, not silent — the same guard philosophy as every other gap in this
backend): attention sinks (src[2]), and non-F32 (F16) masks, the latter deferred to G49.

Per the G39 re-measurement, G41 was paired with **G42** as "a single unit of work" — both refusals
live inside the same attention block, so G41 alone does not get attention off the CPU. The G39
write-up's correction stands: the vector F32 kernel already in mmv.cu is **not** a drop-in for
G42 — real attention matmuls are per-head batched and 3D (ne02/ne03), often on permuted views.

Full regression after G41: **25/25 pass**.

### 10. G42: batched F32×F32 MUL_MAT (attention matmuls)

The second half of the G41+G42 "single unit of work". A new dispatch op,
`GGML_CUDA8_OP_MUL_MAT_F32_F32`, with a purpose-built batched, broadcast-aware kernel for the K·Q
and probs·V matmuls — full detail in ggml-cuda8/README.md G42.

**The G39 correction, now confirmed in code.** §2.1 originally described this as wiring an
already-written vector kernel (mmv.cu:210). That was wrong: real attention matmuls are per-head
batched and 3D, frequently on permuted views. The vector matvec is the wrong shape. G42 adds a new
kernel; the mmv.cu vector kernel remains unused by any dispatch op.

**Scope absorbed most of G46.** The kernel requires contiguous dim 0 (the reduction dimension) but
takes explicit strides on dims 1-3, so permuted views — attention's common case after
reshape+permute, which reorders head/token/batch but leaves dim 0 alone — are handled without a
separate CONT copy. Fully arbitrary dim-0 strides are refused to CPU.

**Two build-integration snags, both instructive** (recorded because they recur when editing the
supports_op switch):
- A duplicate `case GGML_OP_MUL_MAT` (new merged case added at the top of the switch, old
quantized-only case not removed) — GCC "duplicate case value". Fixed by deleting the stale case.
- Collateral loss of the NONE/RESHAPE/VIEW/PERMUTE/TRANSPOSE no-op group, which sat exactly where
the new MUL_MAT block was inserted. supports_op started refusing all five metadata-only ops —
worse than a test failure, it would split the graph around every reshape in a real run. Caught by
the supports-op-smoke fixture and fixed by re-adding the group. The fixture is load-bearing for
exactly this.

Full regression after G42: **26/26 pass**.

### 11. First real Q4_K model (G53), and the finding that reframed the roadmap

**Milestone.** With G41/G42 landed and the G53 transposed-dst fix (§3.5), Qwen3-0.6B-Q4_K_M
generates coherent text end-to-end on the GTX 560 via `llama-cli -ngl 99`. First real Q4_K model
on the backend; prior high-water mark was SmolLM2-135M in Q8_0.

**The finding: attention never reached the GPU.** The `GGML_CUDA8_DEBUG_OPS=1` op histogram from
the working G53 run showed `SOFTMAX_EXT_F32` (G41), `MUL_MAT_F32xF32` (G42) and `DIAG_MASK_INF_F32`
executing **zero** times on GPU, with **zero refusals logged**. Attention ran entirely on CPU and
was not being refused — it was never offered to the backend.

**Root cause: offload_op, not supports_op.** `offload_op` returned true only for MUL_MAT/GET_ROWS
whose src[0] (the weight) is GPU-resident — a G36 guard to stop the scheduler routing CPU-resident
weight layers to CUDA8. But attention's K·Q / probs·V matmuls have an _activation_ as src[0],
which lived on CPU (because `-nkvo` kept the KV cache host-side). So offload_op returned false and
the scheduler kept the whole attention subgraph CPU-side, never consulting supports_op — hence
zero GPU attention ops _and_ zero refusals.

**Resolution (G55).** The offload analysis was correct that attention was never being offered, but
the cause was the `-nkvo` flag, not a code gap. Dropping `-nkvo` (with `-c 512` to fit the F32
cache in 1 GB) put the cache — and therefore attention — on the GPU with **no code change**.
Attention then ran correctly after one more bug fix (the CONT flat-copy, §3.6 / G55). The earlier
conclusion that a residency/offload _code_ project was required is superseded — it was a
configuration change plus a kernel bug.

_This was the first checkpoint driven by a real-model measurement rather than a smoke test. The op
histogram was the authoritative input; the two prior checkpoints' assumption (that landing G41/G42
would put attention on the GPU) was wrong, and only the histogram revealed why._

### 12. Milestone: first fully GPU-resident LLM, and the definitive perf characterization

**The milestone.** Qwen3-0.6B-Q4_K_M generates coherent text with the entire transformer —
attention included — resident on the GTX 560, as a single `n_nodes=1125` graph segment per token
with zero CPU↔GPU boundaries mid-graph. This is the goal the G41 (SOFTMAX_EXT), G42 (F32 MUL_MAT),
G43 (SET_ROWS / F32 KV cache) and G55 (strided CONT) checkpoints were all built toward. Correctness
is achieved end to end on 2011 hardware.

**Getting there was configuration, then one bug.** Attention reached the GPU by dropping `-nkvo`
and capping context to `-c 512` (the full-context F32 cache is ~896 MiB and does not fit; at 512
it is ~11 MiB). That immediately exposed the CONT flat-copy bug (§3.6 / G55), fixed with a
strided-gather kernel. Working invocation:

```
llama-cli -m Qwen3-0.6B-Q4_K_M.gguf -ngl 99 -fa off \
          --cache-type-k f32 --cache-type-v f32 -c 512 \
          -no-cnv -st -p "..."
```

**The definitive perf characterization.** A per-op GPU timing probe over a real run settled the
performance question that three prior experiments had narrowed only by elimination. At the time of
G55, generation ran at ~0.8 tok/s vs ~6 tok/s CPU-only (Phenom II X6 1055T):
- Compute-bound, not overhead-bound (sum-of-op matched the ~1250 ms/token baseline). Boundary
splits, per-op host staging, and per-op device sync were each independently measured and ruled out.
- 97% of GPU time was the two K-quant WEIGHT matmuls. The entire attention core G41–G55 enabled is
<4% — essential for correctness, negligible to run.
- NOT register spilling. ptxas (§3.3): q4k 24 reg, q6k 34 reg, both 0 spill.
- Q6_K was 3.25× slower/call than Q4_K, from occupancy (34 reg + `__launch_bounds__(256,2)` ⇒ 2-3
blocks/SM vs Q4_K's ~5), not spills.

**Correction (G56).** The G55 conclusion — "bandwidth-underutilized, near hardware ceiling,
uncertain payoff, optional high-effort future project" — was **falsified** by a correctness-checked
microbenchmark. The Q4_K matmul was running at **0.24% of the card's 129 GB/s** — not near any
ceiling. Root cause: one-thread-per-256-value-block assignment left only `nb = ne00/256` threads
active (4 of 256 at ne00=1024). A mechanical rewrite (§13) delivered ~7.5× on both K-quant kernels,
taking generation from 0.8 to 5.1 tok/s (6.4×) — CPU parity, from a prior 7× deficit. The "compute-
bound" part was right; the "near ceiling" part was wrong. The lesson: the perf hypothesis, like
every other this arc, had to be measured — the timing probe said "compute-bound," but only the
microbench revealed the compute was 98% idle threads, not saturated hardware.

### 13. G56: K-quant thread-utilization rewrite — GPU reaches CPU parity

**Result.** Two contained kernel rewrites took Qwen3-0.6B-Q4_K_M generation from 0.8 to 5.1 tok/s
(6.4×), moving the GPU path from a 7× deceleration versus the Phenom II X6 1055T to essentially
parity (5.1 vs ~6 tok/s).

**Diagnosis chain (all measured).** The G55 timing probe showed 97% of GPU time in the two K-quant
weight matmuls. A new correctness-checked microbenchmark (ggml-cuda8-q4k-bench.cpp) quantified it:
0.31 GB/s of 129 peak = 0.24% bandwidth, because only `nb = ne00/256` of 256 threads/block did any
work (4 active at ne00=1024, the rest idle). ptxas had already ruled out register spills. So: not
a ceiling, not spills, not bandwidth — gross thread underutilization.

**Fix.** All 256 threads split each block's 256 values, one per thread, instead of
one-thread-per-block. Q4_K maps `tid -> (j,l)`; Q6_K maps `tid -> (half,l,k)` with a `switch(k)`
reproducing the four q1..q4 quadrants — provably the same term set as the serial loop, verified by
the microbenches and full regression.

**Numbers.**
```
Q4_K microbench matvec:  0.31 -> 2.31 GB/s  (7.4x)
Model MUL_MAT_Q4_K:      5.49 -> 0.71 ms/call (7.8x)
Model MUL_MAT_Q6_K:     17.83 -> 2.38 ms/call (7.5x)
Total GPU:               1490 -> 241 ms/graph
Generation:               0.8 -> 5.1 tok/s
```
The profile is now balanced across the three matmul families (Q4_K/Q6_K/F32 within ~3×), no
dominant hotspot.

**Guards.** ggml-cuda8-q4k-bench and ggml-cuda8-q6k-bench are permanent standalone correctness+perf
harnesses (independent CPU reference, GB/s report); 26-target regression green; real-model output
coherent.

**Remaining headroom (correctly scoped).** 2.3 GB/s is still ~2% of peak. Real, measured levers
remain: Increment 2 (weight reuse across columns, for prefill — the bench showed ne11=32 still
re-streams weights per column); Q6_K occupancy (its surviving 3.25×-vs-Q4_K gap is 34 reg +
`__launch_bounds__(256,2)` ⇒ 2-3 blocks/SM); vectorized/coalesced loads (float4, shared-mem
staging). Each is bigger/riskier with smaller marginal payoff than the two ~7× wins already taken —
optional future work, scoped by the microbenches, not guessed.

**Revised roadmap (supersedes §5's tail and §12's).** Below G56: the K-quant matmul throughput
project (Increment 2 + occupancy + vectorized loads — optional, measured headroom); G49 (F16
storage — removes the F32-cache VRAM pressure forcing the -c 512 cap, so larger contexts fit); G44
(SCALE/broadcast-ADD — still premature, coverage is not the constraint). The G38 register baseline
(§3.3) is closed (measured clean). **Correctness AND CPU-parity performance are both achieved; the
project's headline goal — a real LLM running usefully on 2011 Fermi silicon — is met.**

---

_Roadmap drafted with AI assistance against this tree at G36; extended through G56 as the work
landed. The checkpoint results it builds on are the author's own, verified on hardware. The
recurring lesson of the G39–G56 arc, proven repeatedly: do not optimize or conclude from
assumption. Every performance hypothesis — split boundaries, host staging, per-op sync, register
spills, bandwidth ceiling — was measured, and all but one were false. Follow the number._



### G57 status: warp-per-row K-quant MUL_MAT — GTX 560 overtakes the CPU

Status: **DONE, verified on GTX 560 with a real model.** Generation went from
5.1 tok/s (G56) to **14.3 tok/s** on Qwen3-0.6B-Q4_K_M — the 2011 Fermi card
is now **2.4× faster than the host's Phenom II X6 1055T** (~6 tok/s), from a 7×
deficit at the start of the optimization arc. Cumulative from the original naive
kernel: **0.8 → 14.3 tok/s (18×) on generation, 1.2 → 18.1 tok/s (15×) on prompt.**

This checkpoint extends the G56 correction. G55 called the perf ceiling
"bandwidth-underutilized, near hardware ceiling"; G56 corrected that to "thread
underutilization, 0.24% of bandwidth." G57 goes further: even after G56's
utilization fix, the matvec was still only ~2% of peak — because the **256-wide
`__syncthreads` reduction dwarfed the compute**. The card was never
bandwidth-bound; it was reduction/latency-bound.

#### The diagnosis (measured, via the microbenches)

After G56, the q4k-bench showed matvec at ~2.3 GB/s — 1.7% of the 129 GB/s
peak, a 58× gap that memory bandwidth could not explain. The G56 kernel was
one-block-per-output-row, 256 threads, each doing only `nb` values (4 at
ne00=1024), followed by an 8-step, `__syncthreads`-heavy, 256-wide tree
reduction. For nb=4 the block spent more time reducing than computing, and each
block read only ~576 bytes before stalling on the reduction — the memory
pipeline never filled.

#### The fix: warp-per-row

Restructured to fit the GF114 (7 SMs, 32-thread warps, **no `__shfl`** — that is
sm_30+, so reductions stay in shared memory; 32768 reg/SM):
- **One warp (32 threads) per output row, 8 warps (256 threads) per block → 8
  rows per block.** grid = `(ceil(ne01/8), ne11)`.
- Each thread does **8 values per superblock** (Q4_K: the `j = 0..7` loop at
  `lane = l`; Q6_K: 2 halves × 4 quadrants at `lane = l`) instead of 1 — 8× more
  compute per thread, amortizing the reduction.
- **32-wide warp-synchronous reduction** — `volatile` shared memory, no
  `__syncthreads` within a warp (valid on Fermi's lockstep warps) — replaces the
  256-wide `__syncthreads` tree. Each warp reduces only its own 32 slots.
- **`__launch_bounds__(256, 2)` dropped from Q6_K** — it had capped occupancy at
  2 blocks/SM (a G55-diagnosed limiter); the scheduler now picks freely.

Same total arithmetic per row, identical dequant math (bench-proven at G56), just
redistributed to fill the memory pipeline and gut the reduction cost.

#### Results

Microbench (matvec, ne11=1), G56 → G57:

    Q4_K  ne00=1024:  2.31 -> 9.80 GB/s   (4.2x, 0.255 -> 0.060 ms/call)
    Q4_K  ne00=2048:  3.12 -> 11.43 GB/s  (3.7x)
    Q6_K  ne00=1024:  2.61 -> 18.32 GB/s  (7.0x, 0.047 ms/call)
    Q6_K  ne00=2048:  3.26 -> 21.46 GB/s  (6.6x)

Q6_K ended up **faster than Q4_K** per GB/s (21 vs 11) and the old 3.25×
Q6_K-slower gap is erased — the denser format reads more bytes per value, so each
warp does more memory work before reducing, giving higher memory-level
parallelism. The warp structure suits Q6_K even better than Q4_K.

Prefill (ne11=32) also improved ~4-8× for free (each column runs the efficient
warp kernel): Q4_K 0.07 → 0.34 GB/s, Q6_K 0.08 → 0.65 GB/s. It remains the slow
path (weights re-streamed per column — Increment 2 territory), just less slow.

Real model (Qwen3-0.6B-Q4_K_M, -ngl 99, GPU-resident):

    Generation:  5.1 -> 14.3 tok/s   (2.8x over G56, 2.4x the CPU's ~6)
    Prompt:      7.4 -> 18.1 tok/s

Slightly exceeded the matvec-only projection (est. 9-12) because Q6_K's outsized
gain and the free prefill speedup both contributed.

#### Correctness

Both kernels' dequant math is unchanged from the bench-proven G56/G60 versions —
only the thread mapping and reduction changed, so the warp enumeration computes
exactly the same multiset of (weight × activation) products. The microbenches
(independent CPU reference, exact-value-family with scaled tolerance) pass all
shapes `OK`; the full 27-target regression is green; real-model output coherent.

`max_err` rose slightly (1e-4 → up to 3e-4) — expected and benign: the warp
reduction accumulates in a different order than the 256-wide tree, so
floating-point associativity gives marginally different rounding. Well within the
scaled `abs_tol`; not a correctness change.

#### Where the headroom is now

18-21 GB/s is ~14-16% of the 129 GB/s peak — up from 0.24% at the arc's start,
still ~6-7× from theoretical peak. Remaining measured levers, in rough value
order:
- **Vectorized loads (Increment B).** Each thread reads one `uint8` at a time;
  a `uint32` load (4 nibbles) + in-register unpack would cut memory transactions
  ~4×. The next clear lever.
- **Weight reuse across columns (Increment 2).** Prefill still re-streams the
  weight matrix per output column; helps prompt more than generation.
- **Occupancy.** With `__launch_bounds__` gone, a ptxas check on the new kernels
  would show whether register pressure limits blocks/SM.

Each is a bigger/riskier change with smaller marginal payoff than the wins
already banked — optional, scoped by the microbenches, not guessed.

#### The bottom line

The GTX 560 was never near a hardware ceiling. Across G56 and G57 the two K-quant
matmuls went from 0.24% to ~15% of the card's bandwidth — a >60× improvement in
effective throughput — taking generation from 0.8 to 14.3 tok/s (18×) and turning
a 7× GPU deceleration into a 2.4× GPU win over a period-contemporary 6-core CPU.
The project's headline goal — a real LLM running usefully on 2011 Fermi silicon —
is not just met but exceeded. Every step was measured (timing probe → microbench),
changed in one contained increment, verified against an independent reference
before touching the model, and confirmed by full regression. Follow the number.
