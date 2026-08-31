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

| Stage | Where | Toolchain | Produces |
|---|---|---|---|
| 1. Kernels | NVIDIA ML container, Ubuntu 16.04 base | CUDA **8.0.61**, C++11 build tools | `libggml-cuda8-kernels.a` |
| 2. Host | Ubuntu 22.04 host | C++17 build tools | `llama-server` / `rpc-server`, importing stage 1 via `GGML_CUDA8_HOST` |

The `ggml-cuda8` public surface is C, so no C++ ABI crosses the boundary. Note that `rpc-server`
is in scope, which means the Fermi box can also be driven as a *remote* ggml backend rather than
hosting the whole model — a useful escape valve given the 1 GB limit in §4.

**Implemented dispatch ops (15) + no-ops (5):**

```
CPY_F32   ADD_F32   ADD_SCALAR_F32   MUL_SCALAR_F32   MUL_F32   MUL_BROADCAST_F32
REDUCE_SUM_ROWS_F32   REDUCE_MAX_ROWS_F32   SOFTMAX_ROWS_F32
MUL_MAT_Q8_0xF32_VEC   RMS_NORM_F32   ROPE_F32   CONT_F32
DIAG_MASK_INF_F32   GET_ROWS_F32
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

### 2.1 MUL_MAT — the dominant gap

Current support: `Q8_0 [cols,rows] × F32 vector [cols] → F32 [rows]`. That is matrix-vector only.

| Gap | Why it matters |
|---|---|
| `ne11 > 1` is *emulated*, not batched | It does run on GPU, but as a per-token loop of MMV launches with a **host round-trip per token** (`dispatch.cpp:105,146-157`). Prompt eval will be latency-bound on PCIe, not compute-bound. Needs a real batched path. |
| **Q4_0 / Q4_K / Q6_K** | See §4 — Q8_0 does not fit in 1 GB for any useful model. This is the hard blocker on usefulness. |
| F16 src0 | `token_embd` / `output` are commonly F16 even in quantized GGUFs. |
| F32 × F32 not wired | Needed for the attention matmuls (K·Q, probs·V), which operate on activations, not weights. **The kernels already exist** (`mmv.cu:210`) — they just need a dispatch op id and a `supports_op` entry. |
| Permuted / non-contiguous src1, 3D+ shapes | The attention matmuls use permuted views. Contiguous-only support means CPU fallback in the hottest loop. |

No tensor-core substitute is needed: dequantize-to-F32 + `cublasSgemm` covers batched matmul, and
cuBLAS SGEMM works fine on CUDA 8 / Fermi. There is currently no cuBLAS usage anywhere in
`ggml-cuda8/`.

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

- `GET_ROWS` with **quantized** src0 — in a quantized GGUF the embedding table is quantized, so the
  current F32-only path never fires on a real model.
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

**3.1 `SOFT_MAX_EXT` is accepted but computed incorrectly. — FIXED in G37, pending hardware
regression.** (Description retained for context.)
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

**3.2 `gridDim.x` maxes at 65535 on compute 2.x** (it is 2³¹−1 only from sm_30). Grep for `65535`,
`gridDim` or a grid-stride loop across `ggml-cuda8/` returns **zero hits** — all 17 kernel launches
are unclamped: `add.cu:25`, `mul.cu:29,74`, `scalar.cu:49,84`, `mmv.cu:99,199`, `rope.cu:80`,
`getrows.cu:41`, `diagmask.cu:37`, and the row-indexed launches `reduce.cu:137,176`,
`softmax.cu:113`, `rms-norm.cu:47`, **`q8_0-mmv.cu:177,256`** (the MUL_MAT path). At 256
threads/block an element-wise kernel exceeds the limit above ~16.7 M elements — exactly a 4096×4096
tensor. Smoke tests use small shapes, so this has never fired. On a real model it will.
→ Add a grid-stride loop + `min(blocks, 65535)` clamp to every kernel, plus a deliberate
oversized-tensor smoke test. Before the first real model run, not after.

**3.3 63 registers per thread on compute 2.x** (255 from sm_35). Nothing in the backend uses
`__launch_bounds__` or `-maxrregcount` (zero hits, including the 53 KB `CMakeLists.txt`). Spills to
local memory are invisible without measurement.
→ Build with `-Xptxas -v` and record per-kernel register/spill counts as a baseline.

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

**This is why Q4_0/Q4_K matters more than batched matmul.** Q8_0-only support means the largest
fully-offloaded model is under 1B parameters. Note also that ROPE NeoX is required for Qwen3, so
the two most attractive small models each need one currently-missing feature.

---

## 5. Proposed G37+ order

Ordered by "cost to implement ÷ graph splits eliminated", not by tidiness.

| # | Work | Rationale |
|---|---|---|
| ~~**G37**~~ | ~~Tighten `supports_op` for SOFT_MAX (reject mask / sinks / scale / max_bias)~~ | **Done, pending hardware regression.** §3.1. See `ggml-cuda8/README.md` G37. |
| **G38** | Grid-dim clamp + grid-stride loops across all 17 launches; oversized-tensor smoke; `-Xptxas -v` register baseline | §3.2, §3.3. Must precede any real model run. |
| **G39** | Load a real GGUF through `llama-server` (or `rpc-server`) with `GGML_CUDA8_HOST`; log the CPU/GPU split count and per-op fallback reasons | **Measure before optimizing.** The split log turns the rest of this list from guesswork into data. |
| **G40** | Q4_0 MMV (matrix-vector) | Unlocks models that fit in 1 GB. Highest value per line of code. |
| **G41** | `SOFT_MAX_EXT` proper: mask tensor + scale + `max_bias` | Small delta on an existing kernel; removes a split from the hottest loop and reverts G37's restriction. |
| **G42** | Wire the existing F32×F32 matvec (`mmv.cu:210`) into dispatch + `supports_op` | Kernel already written and benched — near-free win for the attention matmuls. |
| **G43** | `GLU` / `SWIGLU` (+ `UNARY` SILU) | Recovers the FFN — the largest single block of FLOPs. |
| **G44** | True batched MUL_MAT (`ne11 > 1`) via dequant + `cublasSgemm`, replacing the per-token host round-trip loop | Makes prompt processing compute-bound instead of PCIe-bound. |
| **G45** | `SET_ROWS` + F32 KV cache path | Keeps the KV cache on-device. |
| **G46** | `GET_ROWS` quantized src0; `SCALE`; broadcast `ADD` | Cheap leaf-node splits. |
| **G47** | ROPE NEOX + freq scaling | Opens up Qwen/Phi-class models. |
| **G48** | Q4_K / Q6_K MMV | Better quality per byte than Q4_0. |
| **G49** | F16 *storage* support (convert on load, compute in F32) | Removes the `--cache-type f32` workaround and handles F16 `token_embd`/`output`. **Unblocked** — conversion verified on sm_21, §7. |
| **G50** | Perf: occupancy, block-size retune for 48 KB shared / 63 registers, `__launch_bounds__` | Only meaningful once the graph stops splitting. |

G37–G39 are prerequisites for sensible prioritisation of everything below them. If the G39 split
log contradicts this ordering, follow the log.

---

## 6. Housekeeping

`ggml/src/ggml-cuda8/` currently holds **139 backup files** (`*.gNN*-backup-<epoch>`,
`*.fix-*`, `*-reset-*`) against 119 real source files, and `.gitignore` does not cover them.
They should be deleted or moved out of the source tree — git already provides the history they
duplicate, and they make the directory listing unreadable.

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
