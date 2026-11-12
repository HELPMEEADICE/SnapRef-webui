# SnapRef-webui architecture

## Source audit (2026-11-12)

Host: Haoming02/sd-webui-forge-classic, branch neo,
commit 4f4284b02a5c814b81c328c64ed7e99a139f21ef. Host working tree was clean.
Read backend/nn/anima.py, diffusion_engine/anima.py, patcher/{unet,base,lora}.py,
attention.py, sampling/{sampling_function,condition}.py, modules/{scripts,
script_callbacks,processing,img2img,sd_models}.py and built-in ImageStitch/LoRA.
Upstream research clones live outside the extension in the OS temporary directory.
SnapRef train/asym.py, comfyui_nodes/snapref/{__init__,simple}.py,
web/snapref.js, configs/train_v2a.json, deploy/quantize_adapter_fp8.py and tests
define the algorithm. Existing LeonWGal and eduardoabreu81 integrations were
examined for Forge interfaces. No host code is copied into this extension.

## Model and weights

Forge Anima engine wraps backend.nn.anima.Anima in KModel/UnetPatcher. Context
is already processed by Qwen06Engine (including the LLM adapter); do not run a
second text preprocessing pass. VAE.encode returns VAE-space samples;
first_stage_model.process_in converts to model space. Using VAE directly avoids
Anima.encode_first_stage's native edit reference side effects.

SnapRef metadata format anima_ref_kv/1 specifies ref_t and modes. Its ref/tgt
matrices are independent deltas scaled by alpha/rank, not Forge LoRA weights.
FP8 E4M3FN storage requires per-row FP32 scales before BF16 computation.
Base v1.0: 28 blocks. Upstream 28→40 map skips inserted blocks
{2,5,8,11,14,17,21,24,27,30,33,36}; these blocks still execute and record K/V.
Neo's ordinary LoRA BLOCK_MAPPINGS duplicates weights and is inappropriate here.
52-block support stays rejected: upstream provides no verified SnapRef map.

## Reference and target

Reference: aspect-preserving downscale only, sides divisible by 16, VAE model
latent [1,16,1,H,W]. Independent RoPE at t=10, spatially centered on target,
sigma=0. Add selected mode embedding after timestep normalization. Run ref
LoRA, record normalized/rotated K and V at every self-attention block. FP16
compute uses FP32 residuals. The reference never sees the target.

Target: original Anima forward and native sampler, tgt LoRA in ALL CFG rows.
Positive rows read reference; negative rows optionally read it, with
log(1-fidelity) bias unless explicit uncond/reference guidance is enabled.
Simple strength multiplies reference weights (1=trained); advanced log bias
adds to logits. Separate references concatenate K/V, never average latents.
Weighted attention uses fused group attention plus LSE merge, with local SDPA
fallback. Zero-bias all-reference attention uses Forge's configured backend.

## Request lifecycle

Always-on img2img Script binds init/sample/close on the processing INSTANCE.
This bypasses native img2img init entirely in reference generation mode;
sampler.sample starts from ImageRNG noise with requested dimensions. No class
or global processing function is patched. Inpaint is rejected explicitly.
Disabled requests are untouched. Script callbacks swallow exceptions in this
host, so all critical validation runs inside bound init/sample and aborts the
request instead of falling back to ordinary img2img.

Clone the final UnetPatcher after normal LoRA activation and other sampling
callbacks. Add an object patch for diffusion_model.forward only. Inside that
forward, scoped hooks apply ref/tgt deltas and attention; finally removes all
handles and restores instance attributes, even on exception/cancellation.
The sampling finally restores object patches and the previous patcher. No
permanent GPU closure is attached to the model. A process lock serializes
SnapRef transactions. Model-loaded/script-unloaded callbacks clear caches.

## Cache and conflicts

One bounded CPU reference cache across runs, GPU copy per sampling session.
Key includes model identity/checkpoint/patch UUID, adapter stat identity,
image bytes, actual sizes, mode/caption/context, target size, dtype/device,
VAE/CLIP identities and patches. Zero-weight references skip encoding.
Memory estimates reserve K/V and adapter space through UnetPatcher.
Native anima_do_reference, nonempty dynamic_args.ref_latents and foreign
reference/attention patches are rejected; no other extension settings changed.
Reference guidance's extra prediction rejects Spectrum/model wrappers because
forecasting state cannot safely be advanced twice. Hires/inpaint integration
is not advertised as supported by this img2img-only extension.

## License boundary

Original extension code: GPL-3.0-only. SnapRef-derived attention and reference
math retain Apache-2.0 attribution and bundled license. Forge is an external
AGPL-3.0 application accessed through its extension APIs; no Forge source is
vendored. Combined distribution with Forge must respect Forge's AGPL terms;
this repository does not relicense Forge. Weights are separately licensed
under CircleStone Labs Non-Commercial License and never bundled.

## Verification policy

CPU mathematical tests and Forge smoke tests are distinct from real checkpoint
GPU tests. 40-block mapping requires executable structural tests before being
enabled. 52 blocks remain blocked. Test reports record actual commands/results,
not assumed quality or compatibility.
