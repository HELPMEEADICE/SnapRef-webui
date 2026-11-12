# SnapRef-webui

[简体中文](README.zh-CN.md) · [日本語](README.ja-JP.md)

Native **Character Reference Generation** in Forge Neo's **img2img** tab.
The main image becomes a character reference; sampling starts from fresh noise.
SnapRef v2a-pre2's asymmetric attention, independent reference/target LoRAs,
mode embedding and reference K/V cache actually run inside Anima.

Status: tested local release candidate, not a guarantee of identity preservation
for every reference/model. See [measured results and limitations](docs/testing.md).

## Installation

Requires [Forge Neo](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo),
an Anima checkpoint, the matching Qwen text encoder and Qwen image VAE. Tested
with Python 3.13.11, PyTorch 2.9.1+cu130, Gradio **4.40.0**, SageAttention 2.2,
and an RTX 5080 Laptop GPU (16 GB). Use the host's Python environment.

In Extensions → Install from URL, enter `https://github.com/HELPMEEADICE/SnapRef-webui.git`
and restart WebUI. Alternatively, run from the WebUI root:

```sh
git clone https://github.com/HELPMEEADICE/SnapRef-webui.git extensions/SnapRef-webui
```

No host source patch is required. Missing
Python dependencies are installed by `install.py`; Torch/Gradio are not replaced.

## Use

1. Select Anima base v1.0 (recommended starting point), its VAE and text encoder.
2. Open **img2img**, upload the character image to its normal main canvas.
3. Select **Reference Generation**, beside img2img / Sketch / Inpaint / Batch.
   SnapRef enables on entering this sub-tab for a compatible model.
4. Leave **Use current img2img image as reference** checked to reuse the main
   canvas, or upload directly in this new sub-tab. A direct upload automatically
   unchecks that option and never overwrites the original img2img canvas.
5. Write the new scene/action/pose prompt; choose output width/height in
   **Resize to**. Start with Euler / Simple, Shift 3, CFG 4.5, 28 steps.
6. Generate. **Denoising strength has no effect in this mode.** The reference
   is not used as the noisy initial image. Switching back to any original sub-tab
   automatically disables SnapRef and restores native controls. No
   `<lora:snapref...>` tag is needed; do not add one.

Example scene: `1girl, solo, sitting on a bench, crossed legs, night, street`.
Character & Style transfers appearance and art style; Character uses the
character embedding while leaving more style control to the model/prompt.
Fine details and outfits can still change; this is a generative adapter.

## Parameters and multiple images

| Setting | Meaning / default |
|---|---|
| Strength | 1 = trained attention weight, 0 = disabled, 2 = twice; never scales LoRA matrices |
| Fidelity | 1 = trained default; lower values add reference to negative CFG rows with weight `1 − Fidelity` |
| Reference resolution | 1 MP; downscale only, preserve aspect, dimensions aligned to 16 pixels |
| Separate | Encode each reference independently, concatenate each block's K/V; useful for views/details of one character |
| Stitch | Resize to first image's height, concatenate left to right, then apply the area cap; different characters can mix identities |
| Image weights | 0–2, main first (when used), then gallery order; missing entries = 1, zero skips encoding |
| Additional log bias | `0`; adds to `log(Strength × image weight)` |
| Reference in Unconditional | Off; enables full reference in negative rows and ignores Fidelity |
| Reference Guidance | Off; when enabled, defaults to 4.5 and puts reference in both branches |
| Correction / End | APG / 1.0; one extra reference-free prediction until this fraction; correction follows upstream |
| Caption | Empty; optional reference appearance tags, not a scene prompt |

Up to **8 total images**. Expand Additional references, drop/paste/upload an
image, then Add. Select a thumbnail to replace, delete or move it left/right.
Weights follow positions: update the comma-separated weights after reordering.
Stitch ignores nonzero individual weights; zero still removes an image.
Strength/Fidelity and explicit unconditional/reference guidance follow upstream
semantics; Fidelity is ignored with either explicit guidance option enabled.

Language can be switched between 简体中文 / 日本語 / English immediately, without
reloading the model. The preference is remembered. Initial language follows
Forge localization, then system locale; other languages use English.

## Weights and automatic download

Recursively scans Forge's `--lora-dir`, its `models_path/Lora` (including
`--model-ref`/`--data-dir`) and already registered matching LoRA paths for:

* `snapref_v2a_pre2.safetensors` (BF16 preferred by default)
* `snapref_v2a_pre2_fp8.safetensors` (optional storage preference)

Metadata and all declared branch targets are checked. FP8 E4M3FN uses the
author's per-row FP32 scales, then computes in the model's dtype. FP8 storage
does not imply FP8 attention/LoRA computation. If just one valid format exists,
it is reused. A damaged installed file produces an error, not silent replacement.

Only the **first enabled generation on a compatible Anima model**, when no
valid file exists, downloads from the author's [official repository](https://huggingface.co/0ppxnhximxr/snapref-v2a-pre2).
It uses `huggingface_hub.hf_hub_download`, a pinned resolved revision, progress,
three attempts, bounded network timeouts, a file lock, size/SHA256 and full
tensor checks, and atomic publication. Incomplete publication files are removed;
HF may retain its resumable cache. Unwritable custom folders fall back to Forge's
local model folder; the actual location is reported. No startup download.

**Weights have the CircleStone Labs Non-Commercial License v1.2, not GPL.**
Read the model's license before downloading/using it; weights are never bundled.
The license notice is visible before generation. A network/download failure
aborts that request and leaves the WebUI running. Configure your existing proxy
for Hugging Face if necessary, or install the official weights manually and rescan.
Live download success could not be verified on the development machine because
its configured proxy broke TLS; failure, cancellation, corruption and successful
atomic publication were tested with controlled transfers.

## Compatibility

| Model / feature | Status |
|---|---|
| Anima base v1.0, 28 blocks | Real BF16-weight GPU inference tested |
| Anima 2.9B preview, 40 blocks | Upstream inserted-block mapping, real GPU generation/cache tested |
| Anima finetunes / Turbo, same validated structure | Structurally accepted; each finetune/Turbo checkpoint is not quality-certified |
| Anima 3.8B, 52 blocks | Rejected; no verified upstream SnapRef mapping |
| SDXL / FLUX / Z-Image / Qwen / MOSS | Rejected by actual engine/transformer types; MOSS rejection tested after real model load |
| Ordinary Anima LoRA | Tested with a character LoRA; native Forge deltas remain active |
| Spectrum | Tested without Reference Guidance; extra-call guidance is explicitly rejected |
| Native Anima Reference/Edit, ImageStitch references | Conflict blocked; their global options are never modified |
| Hires.fix, refiner, inpaint, reference + normal img2img | Unsupported in enabled mode; disable SnapRef to use native features |
| Batch 2, CFG 1, repeat/cancel/model switch | Tested; see test report |
| SageAttention / weighted attention | Native backend for unbiased cases, fused PyTorch LSE/SDPA locally for biased cases |

Other extensions that replace reference/self-attention forwards are rejected.
Compiled models, foreign block modulation hooks and untested third-party
reference combinations are not advertised as compatible. No global attention
backend is changed. Different samplers retain their native schedules: the tested
12-step DPM++ 2M/Karras output was very dark, so prefer Euler/Simple for this setup.

## Cache, performance and memory

Reference VAE/text encoding and K/V are reused when content and model conditions
match. The K/V entry is bounded at 2 GiB on **CPU** across requests; a per-sampling
GPU copy is released in `finally`. One reference at 1 MP is roughly 0.9 GB BF16
K/V; memory and compute grow with separate image count. Reducing reference
resolution is the first response to VRAM pressure. CPU offload uses Forge's
normal weight operations; low-VRAM combinations have not all been benchmarked.

At 512², 12 Euler/Simple steps, 0.25 MP reference: single request 9.98 s,
repeat/cache hit 4.62 s; sampling peak allocated VRAM 6726 / 6195 MiB. These
measurements are hardware/settings-specific, not general promises. Clear cache
or disabling SnapRef releases the retained reference resources. Model-load and
script-unload callbacks invalidate caches. NaN/Inf and OOM stop the request.

## PNG Info and API

PNG infotext records enabled state, adapter/version/identity, actual model,
mode, Strength, Fidelity, resolution, image weights and guidance configuration
in `SnapRef Config`; it never embeds reference image bytes. PNG Info restores
recoverable settings and asks you to select reference images again.

The standard Forge **`/sdapi/v1/img2img`** API calls the exact same runtime.
Use `alwayson_scripts["snapref-webui"]["args"]`. Argument order is documented in
[API.md](docs/API.md), with runnable test clients in `tools/`.

## Test / update / uninstall

From this extension directory, with the host Python:

```text
python -m unittest discover -s tests -v
python tools/gpu_suite.py --reference /absolute/reference.png --output /absolute/test-results
python tools/compatibility_probe.py --reference /absolute/reference.png --output /absolute/compat-results
```

Start an isolated Forge `--api --port 7862` instance first for GPU tests. See
[testing.md](docs/testing.md) for exact environment, cases, results and limitations.
The GPU clients make requests and can change **that test server's** selected
checkpoint. Do not point them at another user's active generation session.

Update with `git pull` inside this independent extension repository or use Forge's
Extensions update controls. To uninstall, stop WebUI and remove only `extensions/SnapRef-webui`.
Optional model files remain in your model folder and can be removed separately.
No host source, shared model, or user setting is deleted by the extension.

## License / sources

Original extension: **GPL-3.0-only**, [LICENSE](LICENSE). SnapRef-derived math:
Apache-2.0 notices retained in source, [NOTICE](NOTICE), [Apache license](licenses/Apache-2.0.txt).
Forge is an external **AGPL-3.0** host; this repository does not relicense Forge
or claim an exception for a combined distribution. Model weights remain
non-commercial under their own license. Do not bundle them with this repository.
See [architecture.md](docs/architecture.md) for the source audit and license boundary.
