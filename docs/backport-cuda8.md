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
| **G40** | `GLU` / `SWIGLU` (+ `UNARY` SILU) | Recovers the FFN — the largest single block of FLOPs, and now the biggest hole. |
| **G41** | `SOFT_MAX_EXT` proper: mask tensor + scale + `max_bias` | Reverts G37's restriction. Real attention softmax currently runs on CPU. |
| **G42** | Wire the existing F32×F32 matvec (`mmv.cu:210`) into dispatch + `supports_op` | Kernel already written and benched — near-free win for the attention matmuls. |
| **G43** | `SET_ROWS` + F32 KV cache path | Keeps the KV cache on-device. |
| **G44** | `SCALE`; broadcast `ADD` | Cheap leaf-node splits. |
| ~~**G45**~~ | ~~ROPE NEOX~~ | **Code done, pending hardware regression.** Top of the G39 log at 4312. freq_factors and attn_factor now explicitly refused rather than silently dropped. |
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
