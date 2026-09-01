# CUDA 8 / Fermi backport — state of the port and roadmap to real GGUF inference

Target hardware: GeForce GTX 560, 1 GB (GF114, compute capability 2.1).
Target toolkit: CUDA 8.0 — the last toolkit that emits sm_20/sm_21.
Host environment: Ubuntu 22.04 server, NVIDIA driver **390.157**, pinned (the 390 legacy branch is
the last one supporting Fermi).
Approach taken: **separate backend** at `ggml/src/ggml-cuda8/`, registered alongside (not replacing) `ggml-cuda`, gated by `-DGGML_CUDA8=ON`.

This document supersedes nothing that came before it — the authoritative checkpoint log is
`ggml/src/ggml-cuda8/README.md` (G16–G36). This is the forward-looking half: what is done,
what is missing, and in what order the gaps should be closed.

---

## 1. Where the port actually is

G36 is the last green checkpoint. The backend:

- registers itself into the ggml backend registry as `CUDA8`, filtered to compute capability 2.x–3.x so modern GPUs still get the standard CUDA backend
- implements a full `ggml_backend_device_i` (buffer type, `supports_op`, `init_backend`, `offload_op`)
- runs a **15-node LLaMA-shaped graph end-to-end on real hardware**, max error 1.19e-06
- solves the toolchain split via `GGML_CUDA8_HOST` (`ggml/src/ggml-cuda8-host.cmake`): a static kernel archive built in a CUDA 8 container is imported as `IMPORTED STATIC` with CUDA 8's `libcudart_static`/`libcudadevrt`/`libculibos`, while only `ggml-cuda8-backend-reg.cpp` is compiled with the modern host compiler. This was the phase-0 risk (nvcc's C++11 / GCC ≤5.3 ceiling vs. llama.cpp's C++17) and it is retired.

**Build mode (settled, not a stopgap).** Two-stage, split by C++ standard:

| Stage | Where | Toolchain | Configures | Produces |
|---|---|---|---|---|
| 1. Kernels | NVIDIA ML container, Ubuntu 16.04 base | CUDA **8.0.61**, GCC 5.4 / C++11, cmake 3.5.1 | `ggml/src/ggml-cuda8` standalone → `build-cuda8-kernels` | `libggml-cuda8-kernels.a` + all smoke targets |
| 2. Host | Ubuntu 22.04 host | C++17 build tools, cmake ≥3.14 | repo root | `llama-server` / `rpc-server`, importing stage 1 via `GGML_CUDA8_HOST` |

Stage 1 configures the `ggml-cuda8` subdirectory rather than the repo root: the root
requires cmake ≥3.14 and the container has 3.5.1, while `ggml-cuda8/CMakeLists.txt` is a
standalone project needing only 3.5 that reaches the ggml core by relative path. `ggml_abort`
and the `ggml_backend_tensor_*` symbols are handled by a separate `ggml-cuda8-ggml-core`
library plus `--gc-sections`; see G38F in `ggml-cuda8/README.md` for why neither can go into
the kernel archive itself.

The `ggml-cuda8` public surface is C, so no C++ ABI crosses the boundary. Note that `rpc-server`
is in scope, which means the Fermi box can also be driven as a *remote* ggml backend rather than
hosting the whole model — a useful escape valve given the 1 GB limit in §4.

**Implemented dispatch ops (19) + no-ops (5):**

```
CPY_F32   ADD_F32   ADD_SCALAR_F32   MUL_SCALAR_F32   MUL_F32   MUL_BROADCAST_F32
REDUCE_SUM_ROWS_F32   REDUCE_MAX_ROWS_F32   SOFTMAX_ROWS_F32
MUL_MAT_Q8_0xF32_VEC   RMS_NORM_F32   ROPE_F32   CONT_F32
DIAG_MASK_INF_F32   GET_ROWS_F32
MUL_MAT_Q4_K_F32   MUL_MAT_Q6_K_F32   GET_ROWS_Q4_K   GET_ROWS_Q6_K
no-ops: NONE  RESHAPE  VIEW  PERMUTE  TRANSPOSE
```

All Fermi-safe: shared-memory tree reductions (`softmax.cu:13-45`, `q8_0-mmv.cu:118,203`), no warp
shuffle anywhere in the directory, no fp16 arithmetic, no dp4a, no tensor cores.

Two things exist but are not wired to the graph path: `REDUCE_MAX_ROWS_F32` has no `supports_op`
entry (reachable only via the direct dispatch API), and `mmv.cu:79,179,210` contains working
**F32×F32 matvec kernels** that no dispatch op id maps to. The latter is relevant below — F32
matmul is cheaper to add than §2.1 implies, since the kernel is already written and tested.

---

## 2. The gap between G36 and `llama-cli`

The G35 pipeline is LLaMA-*shaped*, but it is a hand-built graph with hand-chosen shapes. A real
GGUF graph differs in ways that matter. Because `supports_op` falls back to CPU, none of these
are *correctness* blockers — they are **graph-split** blockers, and a graph that splits between
CPU and GPU on every other node will be slower than pure CPU.

### 2.1 MUL_MAT — much improved, two gaps left

Current support, after the Q4_K/Q6_K work:

| src0 type | Batching | Notes |
|---|---|---|
| **Q4_K** | true `ne11` batching | `ggml-cuda8-q4k.cu:189`, one block per output element, 2D grid already clamped for Fermi |
| **Q6_K** | true `ne11` batching | `ggml-cuda8-q6k.cu:182`, same shape |
| Q8_0 | per-token loop | Runs on GPU but with a **host round-trip per token** (`dispatch.cpp:105,146-157`). Now largely superseded — prefer Q4_K/Q6_K, which are both better for 1 GB and properly batched. |

Remaining gaps:

| Gap | Why it matters |
|---|---|
| F32 × F32 not wired | Needed for the attention matmuls (K·Q, probs·V), which operate on activations, not weights. **The kernels already exist** (`mmv.cu:210`) — they just need a dispatch op id and a `supports_op` entry. Cheapest remaining item on the list. |
| Permuted / non-contiguous src1, 3D+ shapes | The attention matmuls use permuted views. Contiguous-only support means CPU fallback in the hottest loop. |
| F16 src0 | `token_embd` / `output` are sometimes F16 even in quantized GGUFs. Needs G49. |

Q4_0 is no longer on the critical path: Q4_K supersedes it for the 1 GB budget and is done.
No cuBLAS is used anywhere in `ggml-cuda8/`, and with Q4_K/Q6_K batched natively it may not be
needed at all.

### 2.2 Attention

- **`SOFT_MAX_EXT` is a silent wrong-answer bug — see §3.**
- KV cache writes use `GGML_OP_SET_ROWS` in this tree (`ggml.h:519`). Not implemented.
- KV cache defaults to F16. Fermi has no fp16 *arithmetic*. **Workaround until G49:**
  `--cache-type-k f32 --cache-type-v f32`. Costs 2× KV memory; on a small model with short context
  that is affordable. This is temporary, not permanent — F16 *conversion* is confirmed working on
  sm_21 (§7), so an F16 cache with F32 compute is reachable.
- `GGML_OP_FLASH_ATTN_EXT` — will never be supported; leave it unsupported so ggml decomposes it.

### 2.3 FFN

- `GGML_OP_GLU` / `GGML_GLU_OP_SWIGLU` (`ggml.h:578, 613`) — modern llama.cpp fuses gate+up into a
  single GLU node. Not implemented, so the entire FFN falls to CPU. This is roughly two-thirds of
  the model's FLOPs.
- `GGML_OP_UNARY` (SILU / GELU) for models that use the unfused form.

### 2.4 Smaller gaps

- ~~`GET_ROWS` with **quantized** src0~~ — **done** for Q4_K and Q6_K (`q4k.cu:174`, `q6k.cu:167`).
  Other quant types still fall back.
- `GGML_OP_SCALE` (distinct node from MUL-by-scalar).
- `ADD` with broadcast (bias rows).
- `ROPE` mode NEOX (`mode=2`) and frequency scaling / YaRN — currently `mode=0, ext_factor=0` only.
  Rules out Qwen, Phi, GPT-NeoX-derived architectures.
- `RMS_NORM` over multi-row 2D input — verify the existing kernel handles `nrows > 1` with real
  strides, not just the smoke-test shape.

---

## 3. Latent correctness risks in the existing kernels

Three issues, all of which produce **wrong numbers rather than errors**, and none of which the
current smoke tests can catch because they use small, simple shapes.

**3.1 `SOFT_MAX_EXT` is accepted but computed incorrectly. — FIXED in G37, verified on hardware.**
(Description retained for context.)
`supports_op` (`backend-reg.cpp:127-129`) claims any F32 `GGML_OP_SOFT_MAX` **without inspecting
`src[1]` (the mask) or the scale/`max_bias` params**. The kernel launcher takes only
`(src, dst, rows, cols)` (`softmax.cu:100-115`) and `op_params` is never read for SOFT_MAX anywhere
in the backend. So a real attention graph calling `ggml_soft_max_ext(kq, mask, scale, max_bias)`
will be routed to the GPU and will silently compute an **unmasked, unscaled** softmax. This is the
most dangerous item in the port: it does not fall back to CPU, it does not error, it just produces
plausible-looking wrong attention weights.
→ Immediate mitigation: tighten `supports_op` to reject SOFT_MAX when `src[1] != NULL` or
`scale != 1.0f` or `max_bias != 0.0f`. That restores correct CPU fallback in one commit. The real
fix (G40) follows.

**3.2 `gridDim.x` maxes at 65535 on compute 2.x — FIXED in G38, verified on hardware (21/21).**
(it is 2³¹−1 only from sm_30)

**Correction to an earlier draft of this document:** this was described as failing *silently with
wrong results*. That was wrong. Every launcher in `ggml-cuda8/` checks `cudaGetLastError()`
immediately after its launch, so an over-limit grid is rejected with `cudaErrorInvalidConfiguration`
and surfaces as a dispatch failure. Loud, not silent — but it still means any model with a tensor
above ~16.7 M elements (a 4096×4096 weight matrix, at 256 threads/block) simply cannot run.

The genuinely silent variant is *clamping without a stride loop*: the launch then succeeds and
quietly computes only the first 65535 blocks' worth of output. G38's smoke test checks past that
boundary specifically to catch it.

Q4_K/Q6_K MUL_MAT (`q4k.cu:195`, `q6k.cu:188`) and `q8_0-mmv.cu:178,258` already handled this with
a 2D grid before G38. G38 converted the remaining launches to clamp + stride loop: `add.cu`,
`mul.cu` (×2), `scalar.cu` (×2), `rope.cu`, `getrows.cu`, `diagmask.cu`, `reduce.cu` (×2),
`softmax.cu`, `rms-norm.cu`, `mmv.cu` (×2), and the K-quant `GET_ROWS` (`q4k.cu`, `q6k.cu`).

**3.3 63 registers per thread on compute 2.x** (255 from sm_35). Spills to local memory are
invisible without measurement — nothing fails, the kernel is just quietly several times slower.
`ggml-cuda8-q6k.cu` now uses `__launch_bounds__(256, 2)`; most kernels still do not.
→ G38 added `-DGGML_CUDA8_PTXAS_VERBOSE=ON`, which turns on `-Xptxas -v`. Still to do: run it and
record the baseline, paying particular attention to the K-quant kernels, which dequantize a whole
block into registers.

---

## 4. What will actually fit in 1 GB

GTX 560: ~1.26 TFLOPS fp32, **128 GB/s** memory bandwidth, **1 GB VRAM** (confirmed — the 1 GB
variant, not the 2 GB one).

Token generation is bandwidth-bound, so 128 GB/s is the relevant number — roughly 2–3× a
dual-channel DDR4 desktop. Real, but modest.

VRAM is the binding constraint, and it drives the quantization priority:

| Model | Q8_0 | Q4_K_M | Fits in ~850 MB usable? |
|---|---|---|---|
| Qwen3 0.6B | ~0.7 GB | ~0.4 GB | Q4 yes, Q8_0 marginal |
| TinyLlama 1.1B | ~1.2 GB | ~0.67 GB | **Q4 only** |
| Llama 3.2 1B | ~1.4 GB | ~0.8 GB | Q4, tight |
| 3B class | — | ~1.9 GB | partial offload only |

**Q4_K support removes the binding constraint.** With Q4_K and Q6_K now implemented, TinyLlama 1.1B
and Llama 3.2 1B are both in range at Q4_K_M, where Q8_0-only support capped the port at well under
1B. Qwen3 0.6B is the roomiest option but still needs ROPE NeoX (G45), so Llama 3.2 1B or TinyLlama
is the more likely first real load.

---

## 5. Proposed G37+ order

Ordered by "cost to implement ÷ graph splits eliminated", not by tidiness.

| # | Work | Rationale |
|---|---|---|
| ~~**G37**~~ | ~~Tighten `supports_op` for SOFT_MAX (reject mask / sinks / scale / max_bias)~~ | **DONE — PASS on GTX 560, 11/11.** §3.1. See `ggml-cuda8/README.md` G37. |
Already landed out of order (see `ggml-cuda8/README.md`): **Q4_K / Q6_K MUL_MAT with true `ne11`
batching**, and **`GET_ROWS` for Q4_K / Q6_K**. Those covered what were G40, G44 (for K-quants),
G46 and G48 in the original ordering, and they moved the critical path considerably. Renumbered
accordingly:

| # | Work | Rationale |
|---|---|---|
| ~~**G38**~~ | ~~Grid clamps on the remaining launches; oversized-tensor smoke; `-Xptxas -v`~~ | **DONE — PASS on GTX 560, 21/21.** §3.2. Register baseline (§3.3) wired but not yet measured. |
| **G39** | Load a real GGUF through `llama-server` (or `rpc-server`) with `GGML_CUDA8_HOST`; log the CPU/GPU split count and per-op fallback reasons | **Measure before optimizing.** With K-quants working, a Q4_K model is now a realistic first load. The split log turns the rest of this list from guesswork into data. |
| ~~**G40**~~ | ~~`GLU` / `SWIGLU`~~ | **DONE — PASS on GTX 560, 22/22.** 3080 in the G39 log. Both the split and halves forms; row strides honoured rather than assuming contiguity. |
| **G41** | `SOFT_MAX_EXT` proper: mask tensor + scale + `max_bias` | Reverts G37's restriction. Real attention softmax currently runs on CPU. |
| **G42** | Wire the existing F32×F32 matvec (`mmv.cu:210`) into dispatch + `supports_op` | Kernel already written and benched — near-free win for the attention matmuls. |
| ~~**G43**~~ | ~~`SET_ROWS` + F32 KV cache path~~ | **Code done, pending hardware regression.** Reordered ahead of G41/G42: `-nkvo` makes llama.cpp pin attention to the CPU, so the attention kernels are pointless until the cache is device-resident. F32 dst only — `--cache-type-k/v f32` still required until G49. |
| **G44** | `SCALE`; broadcast `ADD` | Cheap leaf-node splits. |
| ~~**G45**~~ | ~~ROPE NEOX~~ | **DONE — PASS on GTX 560, 21/21.** Top of the G39 log at 4312. freq_factors and attn_factor now explicitly refused rather than silently dropped. |
| **G46** | Permuted / non-contiguous src1 for MUL_MAT | The attention matmuls use permuted views. |
| **G47** | Q8_0 batched MUL_MAT, or drop Q8_0 to CPU | Only worth doing if a Q8_0 model is actually wanted; Q4_K is the better fit for 1 GB. |
| **G49** | F16 *storage* support (convert on load, compute in F32) | Removes the `--cache-type f32` workaround and handles F16 `token_embd`/`output`. **Unblocked** — conversion verified on sm_21, §7. |
| **G50** | Perf: occupancy, block-size retune for 48 KB shared / 63 registers, `__launch_bounds__` | Only meaningful once the graph stops splitting. |

G37–G39 are prerequisites for sensible prioritisation of everything below them. If the G39 split
log contradicts this ordering, follow the log.

---

## 6. Housekeeping

`ggml/src/ggml-cuda8/` currently holds **141 backup files** (`*.gNN*-backup-<epoch>`,
`*.fix-*`, `*-reset-*`) against ~121 real source files, and `.gitignore` does not cover them —
two more arrived with the G17A3 synchronize fix, so they are being committed to git as they
accumulate.
They should be deleted or moved out of the source tree — git already provides the history they
duplicate, and they make the directory listing unreadable.

---

## 6b. Upstream sync to `0177dcc` (RPC protocol compatibility) — DONE

**Outcome.** Merged as `2259d07`. Verified afterwards: `RPC_PROTO_MAJOR_VERSION 5`,
`static_assert(GGML_OP_COUNT == 101)`, and both CUDA8 integration points intact
(`ggml/src/CMakeLists.txt` 11 lines, now at 522-546; `ggml-backend-reg.cpp` 4 lines,
now at 33-129). No CUDA8 source changes were needed - the backend ABI prediction held.

Full container regression on the merged tree: **21/21 pass**, no rebuild breakage. The
"expected breakage" list below predicted the standalone build would lose symbols to
upstream restructuring; it did not - `--gc-sections` absorbed whatever moved, and the
five new ops fell through `supports_op`'s `default: return false` as designed.

**Where the estimate below was wrong.** It predicted "two files, ~15 lines" of conflict.
The CUDA8-specific part of that was right - `CMakeLists.txt` and `ggml-backend-reg.cpp`
both auto-merged - but the merge as a whole produced ~31 conflicts:

- 1 real content conflict, `ggml/src/ggml-backend-meta.cpp`, resolved by taking
  upstream (the fork's version had no CUDA8 content; it only appeared modified because
  the fork's initial commit imported every file at once)
- ~30 modify/delete conflicts on `.github/workflows/*.yml` and `build-xcframework.sh`,
  which the fork had deleted and upstream had edited - all resolved by keeping them
  deleted
- `.gitattributes`, deleted upstream, kept from the fork for `*.sh text eol=lf`

The estimate measured the CUDA8 touchpoints and missed two things: the fork's own
unrelated divergence, and the fact that **the fork was created by importing a snapshot
rather than forking git history**, so git had almost no shared history to merge
against. Every future upstream sync will be similarly noisy for the same reason.
Re-applying the CUDA8 work onto a real clone of llama.cpp would make subsequent merges
routine, and is worth considering if upstream tracking becomes a recurring need.

The original analysis follows.

### Original analysis

**Why.** The Fermi box is to run `rpc-server` against a `llama-server` built from
upstream llama.cpp at `0177dcc7300bad8914bb838baabce87899812491`. RPC requires an
exact major-version match (`ggml-rpc.cpp:356`), and:

| | this fork | upstream `0177dcc` |
|---|---|---|
| `RPC_PROTO_MAJOR_VERSION` | 4 | **5** |
| `GGML_OP_COUNT` | 96 | **101** |

So a v4 server and a v5 client refuse to connect. Note `0177dcc` itself is unrelated
to RPC — it is the `--mmap`/`--no-mmap` → `--load-mode` migration (#26934). It is
simply the revision the peer will be built from.

**Copying the RPC files across is not an option.** `ggml-rpc.h` carries
`static_assert(GGML_OP_COUNT == 101)`, which fails against this tree's 96, and the RPC
wire format serialises the ggml op enum - a v5 protocol on a 96-op ggml would claim a
compatibility it does not have. The whole fork has to move.

**The good news: no backend ABI drift.** `ggml-backend-impl.h` is identical between
this tree and `0177dcc` - `GGML_BACKEND_API_VERSION 2` on both, and
`ggml_backend_buffer_i` has the same eleven members in the same order (including
`set_tensor_2d`/`get_tensor_2d`). The CUDA8 backend implements those vtables
positionally, so it should carry forward untouched.

**Conflict surface.** Everything else the port owns is new files, which cannot
conflict:

| File | Change | Conflict risk |
|---|---|---|
| `ggml/src/CMakeLists.txt` | 11 lines (`GGML_CUDA8`, `GGML_CUDA8_HOST` options) | real, small |
| `ggml/src/ggml-backend-reg.cpp` | 4 lines (registration under `#ifdef GGML_USE_CUDA8`) | real, small |
| `.gitattributes` | 1 line (`*.sh text eol=lf`) | only if upstream touched it |
| `ggml/src/ggml-cuda8/**` | all new | none |
| `ggml/src/ggml-cuda8-host.cmake` | new | none |
| `scripts/fix_cuda8_synchronize.sh`, `docs/backport-cuda8.md` | new | none |

### Procedure

    git branch pre-0177dcc-backup                      # cheap insurance
    git remote add upstream https://github.com/ggml-org/llama.cpp
    git fetch upstream 0177dcc7300bad8914bb838baabce87899812491
    git merge 0177dcc7300bad8914bb838baabce87899812491

Resolve the two conflicts by keeping *both* sides: upstream's changes plus the CUDA8
option block and the registration `#ifdef`.

### Verify after merging

    grep RPC_PROTO_MAJOR_VERSION ggml/include/ggml-rpc.h     # expect 5
    grep -c cuda8 ggml/src/CMakeLists.txt                    # expect 11
    grep -c CUDA8 ggml/src/ggml-backend-reg.cpp              # expect 4

then the full container regression, then the host build.

### Expected breakage, in order of likelihood

1. **The standalone container build losing symbols again.**
   `ggml-cuda8-ggml-core` compiles `../ggml.c`, `../ggml-quants.c` and
   `../ggml-threading.cpp` only. If upstream moved code into new translation units,
   or added calls that `--gc-sections` cannot discard, this reappears as undefined
   references - exactly the `ggml_abort` / `ggml_backend_tensor_set` episode from G38F.
   The fix is the same: add the source, or confirm the section is unreachable.

2. **The five new ops.** `supports_op` ends in `default: return false`, so unknown ops
   are refused safely. No action expected, but the G39 rejection log will show them if
   Qwen3 uses any.

3. **`--load-mode`.** `0177dcc` removes `--mmap`/`--no-mmap`/`--direct-io`. The
   documented G39 invocation uses `-ngl`, `-nkvo`, `-fa`, `--cache-type-k/v`, none of
   which are affected - but re-check the command after merging.

4. **CMake integration points moving.** `ggml/src/CMakeLists.txt` is the file most
   likely to have been restructured upstream; the `GGML_CUDA8` block may need
   repositioning rather than a straight conflict resolution.

Do this **before** further kernel work (G40/G45), so the op implementations target the
final ggml API and the G39 measurement is not repeated against a moving base.

---

## 7. Verified: F16 conversion works on sm_21

**Result: G49 is unblocked.** Confirmed in the CUDA 8.0.61 container, two independent ways.

**(a) Header guards.** In `/usr/local/cuda-8.0/include/cuda_fp16.h`, `__float2half` (decl 128,
def 2305) and `__half2float` (decl 167, def 2329) sit **outside** every `__CUDA_ARCH__` block. The
`>= 530` guards cover lines 1010–1929 and 2598–3370 — that is the fp16 *arithmetic* API
(`__hadd`, `__hmul`, …), which stays unavailable. Conversions are unguarded.

**(b) Generated PTX at sm_21 — decisive.**

```cuda
// fp16probe.cu
#include <cuda_fp16.h>
__global__ void probe(const float * in, half * hout, float * fout) {
    half h  = __float2half(in[0]);   // f32 -> f16
    hout[0] = h;
    fout[0] = __half2float(h);       // f16 -> f32
}
```

```console
$ nvcc -arch=sm_21 -std=c++11 -ptx fp16probe.cu -o - | grep -n "cvt.*f16"
34:     {  cvt.rn.f16.f32 %rs1, %f1;}
39:     {  cvt.f32.f16   %f2, %rs1;}
```

Both directions compile to single hardware instructions. (The `sm_21 is deprecated` warning is
expected and harmless — suppress with `-Wno-deprecated-gpu-targets`.)

**Not yet done — (c) on-card confirmation.** A round-trip of `1.5f` checking the raw half bits are
`0x3E00`, to rule out the 390.157 driver mis-JITing the PTX. Unlikely, but cheap; fold it into G49
as the first smoke test rather than running it separately.

**Scope limit.** Conversion only. All F16 arithmetic still happens in F32, so G49 means "read and
write F16 buffers, compute in F32" — enough for an F16 KV cache and F16 `token_embd`/`output`, and
nothing beyond that.

No open questions remain.

---

*Roadmap drafted with AI assistance against this tree at G36. The G16–G36 checkpoint results it
builds on are the author's own, verified on hardware.*

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

G41 UPDATE — supersedes the "next steps" framing in backport-cuda8-append.md
(§8's closing paragraph and the original document's §5 roadmap table entry
for G41). Apply these edits IN ADDITION TO backport-cuda8-append.md's
INSERTION 1/2 - this file does not repeat that content, only updates the
status of what it left open.

---

EDIT 1 — in backport-cuda8-append.md's "INSERTION 2" content (the new
top-level "### 8. G51: dispatcher robustness hardening" section), the
closing paragraph currently ends:

    G41/G42 (SOFT_MAX_EXT proper + F32×F32 MUL_MAT, "a single unit of
    work" per §G39's re-measurement note) remain the next highest-value
    checkpoint.

Replace with:

    G41 (SOFT_MAX_EXT proper) has since landed - see §9 below. G42
    (F32×F32 MUL_MAT) is the remaining half of that "single unit of
    work" pairing and is now the next highest-value checkpoint.

---

EDIT 2 — in the ORIGINAL document (not the G51 append), section
"### 5. Proposed G37+ order", the table row for G41 currently reads:

| **G41** | SOFT_MAX_EXT proper: mask tensor + scale + max_bias | Reverts G37's restriction. Real attention softmax currently runs on CPU. |

Replace with:

| ~~**G41**~~ | ~~SOFT_MAX_EXT proper: mask tensor + scale + max_bias~~ | **DONE — PASS on GTX 560, 25/25.** Mask + scale + ALiBi implemented; attention sinks and non-F32 masks still refused (deferred to a future checkpoint / G49 respectively). See ggml-cuda8/README.md G41 and §9 below. |

---

EDIT 3 — new top-level section, place at the end of the document, after
section "### 8. G51: dispatcher robustness hardening" (from
backport-cuda8-append.md's INSERTION 2) and before the closing italic
attribution line:

---

### 9. G41: SOFT_MAX_EXT proper (mask + scale + ALiBi)

Lifts the G37 restriction for real attention softmax. A new dispatch op,
`GGML_CUDA8_OP_SOFTMAX_EXT_F32`, sits alongside the unchanged plain
`SOFTMAX_ROWS_F32` path and implements the `ggml_soft_max_ext(kq, mask,
scale, max_bias)` shape that real attention graphs actually call - full
technical detail and verification method in ggml-cuda8/README.md G41.

Explicitly still refused (loud, not silent - the same guard philosophy as
every other gap in this backend): attention sinks (`src[2]`), and
non-F32 (F16) masks, the latter deferred to the general F16-storage work
already tracked as G49.

Per §8/G51's closing note and the G39 re-measurement, G41 was paired with
**G42** as "a single unit of work" - both refusals (`SOFT_MAX
soft_max_ext` and `MUL_MAT f32xf32`) live inside the same attention
block, so G41 alone does not get attention off the CPU. **G42 is now the
next highest-value checkpoint.** The G39 write-up's correction stands and
is worth restating: the vector F32 kernel already in `mmv.cu` is not a
drop-in for G42 - real attention matmuls are per-head batched and 3D
(`ne02`/`ne03`), often on permuted views, so G42 will likely need to land
together with **G46** (permuted/non-contiguous src1 for MUL_MAT) rather
than in isolation. A fresh `GGML_CUDA8_DEBUG_OPS=1` rejection-log run
against a real model (now that G41 has landed) is the recommended first
step before committing to G42's exact scope, following the same
"measure before optimizing" principle G39 established.

Full regression after G41: **25/25 pass** (was 24/24 - the new
`ggml-cuda8-softmax-ext-smoke` target accounts for the difference).

---
G42 UPDATE — supersedes the "next steps" framing in
backport-cuda8-append-2.md (its section 9 and its EDIT 1). Apply IN ADDITION
TO backport-cuda8-append.md (INSERTION 1/2) and backport-cuda8-append-2.md.
This file updates the status of what those left open; it does not repeat
their content.

---

EDIT 1 — in backport-cuda8-append-2.md's "INSERTION 3" content (the new
"### 9. G41" section), its closing paragraph currently ends:

    **G42 is now the next highest-value checkpoint.** ... G42 will likely
    need to land together with **G46** ... A fresh GGML_CUDA8_DEBUG_OPS=1
    rejection-log run against a real model (now that G41 has landed) is the
    recommended first step before committing to G42's exact scope ...

Replace that closing paragraph with:

    G42 has since landed - see section 10 below. It absorbed most of G46
    (permuted/non-contiguous src1) by taking explicit strides on dims 1-3
    rather than assuming packed layout, so G46 is no longer a separate
    blocker for the common attention permuted-view case. With G41+G42 both
    done, the attention core no longer forces a per-layer CPU fallback in
    principle. The recommended next step is the GGML_CUDA8_DEBUG_OPS=1
    re-measurement (now that BOTH have landed), not another kernel - see
    section 10.

---

EDIT 2 — in the ORIGINAL document, section "### 5. Proposed G37+ order",
the table row for G42 currently reads:

| **G42** | Wire the existing F32xF32 matvec (mmv.cu:210) into dispatch + supports_op | Kernel already written and benched - near-free win for the attention matmuls. |

Replace with:

| ~~**G42**~~ | ~~Wire the existing F32xF32 matvec (mmv.cu:210)~~ Batched F32xF32 MUL_MAT, purpose-built kernel | **DONE — PASS on GTX 560, 26/26.** The vector kernel was NOT reusable (see G39 correction / section 10): real attention matmuls are batched, 3D, permuted. New kernel handles GQA broadcast + dims-1-3 strides. See ggml-cuda8/README.md G42 and section 10. |

---

EDIT 3 — in the ORIGINAL document, section "### 2.1 MUL_MAT — much improved,
two gaps left", the remaining-gaps table. The "F32 x F32 not wired" row and
the "Permuted / non-contiguous src1, 3D+ shapes" row are both now largely
addressed. Update them:

- "F32 x F32 not wired" row: strike through and mark **DONE (G42)**. Note
  that the claim in its "Why it matters" cell ("The kernels already exist
  (mmv.cu:210) - they just need a dispatch op id") was wrong; see the G39
  correction. G42 added a purpose-built batched kernel.
- "Permuted / non-contiguous src1, 3D+ shapes" row: mark **mostly addressed
  by G42**. G42 takes explicit strides on dims 1-3, so permuted views with
  contiguous dim 0 (the common attention case) work. Only fully arbitrary
  dim-0 strides remain out of scope, refused to CPU.

The F16 src0 row is unchanged (still needs G49).

---

EDIT 4 — new top-level section, place at the end of the document, after
section "### 9. G41" (from backport-cuda8-append-2.md) and before the
closing italic attribution line:

---

### 10. G42: batched F32xF32 MUL_MAT (attention matmuls)

The second half of the G41+G42 "single unit of work". A new dispatch op,
`GGML_CUDA8_OP_MUL_MAT_F32_F32`, with a purpose-built batched, broadcast-aware
kernel for the K.Q and probs.V matmuls - full detail in ggml-cuda8/README.md
G42.

**The G39 correction, now confirmed in code.** Section 2.1 originally
described this as wiring an already-written vector kernel (mmv.cu:210). That
was wrong, as the G39 re-measurement first flagged: real attention matmuls
are per-head batched and 3D (ne02/ne03), frequently on permuted views. The
vector matvec is the wrong shape. G42 adds a new kernel instead. The
mmv.cu:210 vector kernel remains unused by any dispatch op.

**Scope absorbed most of G46.** The kernel requires contiguous dim 0 (the
reduction dimension) but takes explicit strides on dims 1-3, so permuted
views - attention's common case after reshape+permute, which reorders
head/token/batch but leaves dim 0 alone - are handled without a separate
CONT copy. Fully arbitrary dim-0 strides are refused to CPU. This means G46
(permuted/non-contiguous src1) is largely closed as a side effect; only the
dim-0-strided remainder is still open, and it may never matter in practice.

**Two build-integration snags, both instructive** (recorded because they
are the kind of thing that recurs when editing the supports_op switch):
1. A duplicate `case GGML_OP_MUL_MAT` (new merged case added at the top of
   the switch, old quantized-only case not removed) - GCC "duplicate case
   value". Fixed by deleting the stale case.
2. Collateral loss of the NONE/RESHAPE/VIEW/PERMUTE/TRANSPOSE no-op group,
   which sat exactly where the new MUL_MAT block was inserted. supports_op
   started refusing all five metadata-only ops - worse than a test failure,
   it would split the graph around every reshape in a real run. Caught by
   the supports-op-smoke fixture and fixed by re-adding the group. The
   fixture is load-bearing for exactly this.

Full regression after G42: **26/26 pass** (was 25/25 - the new
`ggml-cuda8-mulmat-f32-smoke` target accounts for the difference).

**Next step is measurement, not code.** G41+G42 close the attention core in
principle. Before the next kernel, re-run the G39-style rejection log now
that both have landed:

```
GGML_CUDA8_DEBUG_OPS=1 <llama-cli / rpc-server invocation, Q4_K model>
```

The log confirms what the smoke suite cannot: whether a real attention graph
actually takes the G42 path or hits a dim-0-strided layout it refuses;
whether GGML_OP_SCALE / broadcast-ADD (G44) are now the top refusals; and
whether any new op surfaces once the graph stops splitting at attention. If
the log is clean or close, priority order is G44 (SCALE, broadcast ADD),
then G49 (F16 storage - unblocked per section 7), then G50 (perf, only once
the graph stops splitting). The G38 register-spill baseline (section 3.3)
remains open and independent.

---
G53 UPDATE — records the transposed-dst K-quant MUL_MAT fix and, more
importantly, the first real-model measurement finding: attention never
reaches the GPU. Apply IN ADDITION TO backport-cuda8-append.md,
-append-2.md, -append-3.md. This is the most consequential update since
G39 because it changes what the next phase should be.

---

EDIT 1 — in the ORIGINAL document, section "### 3. Latent correctness risks
in the existing kernels", add a new subsection after 3.4 (the G51
robustness subsection from backport-cuda8-append.md):

    **3.5 Transposed dst write in K-quant MUL_MAT — FIXED in G53, verified
    on a real model.** Both Q4_K and Q6_K MUL_MAT kernels wrote the output
    element at `dst[row*ne11 + col]` instead of the ggml layout
    `dst[col*ne01 + row]`. These are equal only at ne11 == 1 (matvec), which
    is the only shape every K-quant smoke and graph-builder checkpoint
    (G17C, G18-G23) ever used - so all of them passed while the kernel was
    wrong for any ne11 > 1 (prompt prefill). Q8_0 was unaffected (its
    per-token loop writes the correct layout), which is why SmolLM2-Q8_0
    worked but Qwen3-Q4_K produced structured-loop garbage. Same failure
    mode as 3.1/3.4: wrong numbers, graph_compute reports SUCCESS, nothing
    fails loudly. Fixed (one line each in q4k.cu:142 / q6k.cu:160) plus a
    new ne11 > 1 regression smoke - the K-quant smokes could not have caught
    this because they were all ne11 == 1. See ggml-cuda8/README.md G53.

---

EDIT 2 — new top-level section, place at the end of the document, after
section "### 10. G42" (from backport-cuda8-append-3.md) and before the
closing italic attribution line:

---

### 11. First real Q4_K model, and the finding that reframes the roadmap

**Milestone.** With G41/G42 landed and the G53 transposed-dst fix,
Qwen3-0.6B-Q4_K_M generates coherent text end-to-end on the GTX 560 via
llama-cli -ngl 99. This is the first real Q4_K model on the backend; the
prior high-water mark was SmolLM2-135M in Q8_0. Correctness is achieved.

**The finding: attention never reaches the GPU.** The GGML_CUDA8_DEBUG_OPS=1
op histogram from the working run is decisive. GPU-executed ops, by
frequency:

    5712  MUL_MAT_Q4_K       (weights)
    3842  RMS_NORM_F32
    2069  MUL_BROADCAST_F32
    1904  ROPE_F32
    1904  ADD_F32
    1773  MUL_F32
     986  MUL_MAT_Q6_K       (weights)
     952  SWIGLU_F32
      68  GET_ROWS_F32

Conspicuously ABSENT, zero occurrences each: SOFTMAX_EXT_F32 (G41),
MUL_MAT_F32xF32 (G42), DIAG_MASK_INF_F32. The entire attention inner block
- scores, causal mask, softmax, context - runs on CPU. And the rejection
log is EMPTY: attention is not being refused by supports_op, it is never
offered to the backend at all.

**Root cause: offload_op, not supports_op.** offload_op
(backend-reg.cpp) returns true only for MUL_MAT/GET_ROWS whose src[0] (the
weight) is already GPU-resident - a deliberate G36 guard to stop the
scheduler routing CPU-resident weight layers to CUDA8. But attention's K.Q
and probs.V matmuls have an *activation* as src[0] (K, or the attention
probs), which lives on CPU. So offload_op returns false, the scheduler
keeps the whole attention subgraph CPU-side, and supports_op is never even
consulted on the softmax/matmul/mask nodes - hence zero GPU attention ops
AND zero refusals. G41/G42 are correct and smoke-proven but are currently
dead code on real models: the kernels exist, nothing calls them.

**This reframes the roadmap.** The prior assumption (sections 8-10) was
that landing G41/G42 would get attention onto the GPU. It did not, because
the blocker was never op coverage - it was the offload policy. The two
high-value levers now are:

1. **Offload policy for activation-src0 matmuls.** Make attention's F32
   matmuls actually reach G42. Subtle: naively returning true for F32
   MUL_MAT would route activation-only matmuls to a GPU that must then
   upload both operands every call - possibly slower, given the host
   round-trip. Only worthwhile coupled with (2).

2. **Device-resident execution across the graph (the real G50).** Kill the
   Q8_0-era per-op host staging (alloc -> upload -> launch -> download ->
   free every dispatch) and keep tensors resident on the GPU across nodes.
   This is the prerequisite that turns (1) from a wash into a speedup, and
   is the dominant cost behind the current -ngl 99 numbers.

**Performance context.** -ngl 99 runs at ~1.2/0.8 t/s vs -ngl 0 at ~9.0/6.0
t/s - GPU offload is currently a ~7x DECELERATION. This is fully explained
by the two costs above (attention CPU round-trip every layer + per-op host
staging on every GPU dispatch), not by slow kernels. It is not a coverage
gap and not a G42-scope problem - G42's kernel is correct, it is simply
never called. The honest status: correctness is done; making the GPU path
faster than CPU is a structural residency/offload project, not another
kernel checkpoint.

**Revised priority below G53:** the residency/offload work above supersedes
G44 (SCALE, broadcast-ADD) and G49 (F16 storage) as the highest-value next
step, because without it every GPU op is host-staging-bound and adding more
GPU ops makes the graph slower, not faster. G44/G49 remain worthwhile but
are premature until the graph stops round-tripping. The G38 register-spill
baseline (section 3.3) also remains open and independent.

_Note: this is the first checkpoint driven by a real-model measurement
rather than a smoke test. The op histogram above is the authoritative
input; if future work contradicts this ordering, re-measure and follow the
histogram - the same principle G39 established._

---
