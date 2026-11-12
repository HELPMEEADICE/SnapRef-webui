# Verification record — 2026-11-12

This is an execution record, not a claim that all model/sampler combinations
are supported. No pretrained/test image is committed to this repository.

## Environment

* Host Neo commit: `4f4284b02a5c814b81c328c64ed7e99a139f21ef`.
* Windows, Python 3.13.11 from the host distribution.
* PyTorch 2.9.1+cu130, Gradio 4.40.0, SageAttention 2.2.
* NVIDIA GeForce RTX 5080 Laptop GPU, 16 GB, 32 GB system RAM.
* Anima base v1.0 SHA256 `bd43b7cffe1ed1153d9c41e7beb2f18cb1273eafbaa3af3edd6a173dc90a006e`.
* Local author-format BF16 `snapref_v2a_pre2.safetensors`, 381,821,960 bytes,
  metadata format `anima_ref_kv/1`, step 15000, ref r128 / tgt r64, modes cs/char.
* Qwen 0.6B encoder and Qwen Image VAE. 2.9B preview separately tested.

The initial host Git worktree was clean. Host source files were not changed.
An isolated `--data-dir` held the test config, ui-config, outputs and an extension
junction; it used existing models through `--model-ref`. The test process had
only this external extension plus the host's built-in extensions enabled.

## CPU unit tests

```text
<host-python> -m unittest discover -s tests -v
```

21 CPU tests passed in the final audit. Coverage includes metadata/declared coverage,
BF16 and row-scale FP8 parity, damaged files and scales, absent streams,
28/40/52 mapping acceptance/rejection, exact ref/tgt switching, inserted layers,
attention recording and target reading, repeated/reentrant hooks and exception
cleanup, explicit softmax versus weighted SDPA, LSE merge equivalence,
mixed CFG rows and exact CPU Fidelity, correction finiteness, CPU cache budget,
key invalidation, translation key/placeholder parity and context switching,
reference/zero-weight/stitch validation, actual-type gate, download failure,
cancellation/corruption/success/reuse and request method restoration.

## Real checkpoint GPU tests

512×512; seed 782341; Euler / Simple; 12 steps; CFG 4.5; Shift 3;
reference cap 0.25 MP unless stated. Prompt:
`1girl, solo, sitting on a bench, crossed legs, night, street, full body, detailed anime illustration`.
Negative: `low quality, worst quality, blurry`.

References were the public example image `ref_betterdays_input.png` from the
researched LeonWGal repository, and its `output_rdbtanima_yoga.png` for changed
reference/multi-reference probes. They are not distributed here.

| Case | Wall seconds | Result |
|---|---:|---|
| Native base txt2img baseline | 14.446 | Finite, nonuniform RGB |
| Single reference | 9.982 | Real reference branch, cache miss |
| Exact repeat | 4.618 | Cache hit; identical RGB output |
| Changed reference | 5.397 | Cache miss; changed output |
| Character mode | 5.397 | Finite, different mode result |
| Character & Style | 5.404 | Reproduces original cs result |
| Two references, weights 1 / 0.7 | 6.556 | Finite weighted attention |
| Fidelity 0.6 | 5.588 | Finite, changed result |
| Reference Guidance + APG, end 0.6 | 5.598 | Finite extra-call path |
| Disabled, native baseline again | 3.010 | Max RGB difference from baseline **0** |
| Ordinary character LoRA + SnapRef | 8.763 | Native LoRA reported loaded; finite output |
| Stitch | 4.977 | Finite output from stitched reference |
| Three references, weights 1 / 0.5 / 0.2 | 7.000 | Finite multi-cache output |
| Reference in Unconditional | 4.231 | Finite output |
| Native img2img before / after SnapRef | 8.920 / 0.879 | Max RGB difference **0** |
| 2.9B preview single / repeat | 25.353 / 5.972 | Real 40-block map, repeat identical, hit |
| Denoising strength 0.1 versus 0.95 (2.9B) | 6.139 / 5.883 | Max RGB difference **0** |
| Batch size 2 (2.9B) | 9.537 | Two images returned |
| CFG 1 (2.9B) | 3.056 | Finite, nonuniform output |
| DPM++ 2M / Karras, 12 steps (2.9B) | 6.118 | Finite but **very dark/weak composition**; not recommended |
| Spectrum, reference guidance off (2.9B) | 11.077 | Finite output |
| Guidance + Spectrum | 0.073 | Explicit conflict error, no sampling |
| Native Anima Reference on | 0.052 | Explicit conflict error |
| Mask supplied | 0.081 | Explicit inpaint error |
| Cancel 200-step run after sampling starts | 2.268 | Native cancellation completed, partial output returned |
| Same reference after cancel/errors | 6.576 | Exact same RGB as before cancellation |
| Switch to real MOSS engine with matching 9B encoder/Flux2 VAE | 9.902 | Explicit actual-type rejection before SnapRef inference |
| Base v1.0, 1024², 28 steps, 1 MP single / repeat | 42.379 / 31.765 | Finite, repeat identical, miss / hit |

Base 512² sampling peaks: 6726.2 MiB (miss), 6195.0 MiB (hit).
Base 1024² sampling peaks: 8182.1 / 7123.8 MiB. Values are Torch **allocated**
peak memory during the measured sampling interval, not total OS VRAM usage.
Cold startup/model-loading time is included in wall times where applicable.

Visual inspection of the base single-reference result found orange braids,
red eyes/scarf and the white/black outfit preserved, with the requested new
crossed-leg bench pose and night scene. It was not a copy of the standing
reference. These examples do not establish broad identity/pose quality metrics.
Numeric guards reject degenerate uniform images; visual assessment remains necessary.

## UI and integration

Real WebUI startup and `/sdapi/v1/scripts`, `/sdapi/v1/script-info`, `/config`
confirmed extension discovery and correct always-on argument layout.
Browser checks verified Chinese/Japanese/English runtime relabeling, primary
canvas upload and preview, and no page horizontal overflow at desktop and narrow
viewports. The initial bottom accordion was replaced after user clarification
with a native **Reference Generation / 参考生图 / 参照生成** sub-tab beside
img2img/Sketch/Inpaint/Batch. No bottom SnapRef accordion remains.

Native sub-tab transport and direct image upload are regression-tested through
the same request runtime. A real browser Generate click from the new sub-tab,
with a dedicated upload and an empty original canvas, completed in 13.0 seconds
(512², 12 Euler/Simple steps, 1 MP cap). Its infotext confirmed use_main=false,
one reference, actual SnapRef branches and 7803.4 MiB allocated sampling peak.
Switching back to the original tabs disables reference generation and restores
their controls. Backend checks still apply to API calls.
The final browser regression also verified that a prior native Resize by
selection survives native-tab switches, is hidden during Reference Generation,
and is restored with the denoising slider when returning to Sketch.

The final automated run passed **24 tests**: 21 CPU tests and 3 read-only live
Forge integration tests. Live checks verify the new TabItem shares the exact
native parent with all six old sub-tabs, every old sub-tab disables SnapRef,
the old bottom accordion is absent, transport uses valid native mode components,
and the always-on API exposes 20 arguments. Set `SNAPREF_FORGE_URL` to the test
server URL to include these integration checks; otherwise they are explicitly skipped.

Final API regression (after moving to the sub-tab): baseline 7.242 s, single
reference 4.117 s, cache-hit repeat 2.743 s, disabled baseline 1.950 s. Repeat
pixels match and the disabled baseline maximum RGB difference is **0**.
After tightening Anima geometry checks to 2048 hidden channels, 16×128 heads
and 1024 context channels, another real base-model generation passed (12.555 s)
with identical RGB output to the earlier base single-reference test.

Issues found and fixed during testing:

* Windows safetensors mmap held BF16 files open: cloned CPU matrices after read.
* Forge hashes img2img init_latent during text cache setup: create a blank target
  canvas solely for this host contract; never call sample_img2img for SnapRef.
* Raw callback exceptions are swallowed by ScriptRunner: critical validation is
  inside bound processing methods, where it aborts the request.
* Isolated test directory lacked its configured temp directory / JS allowed path:
  created that test temp directory and supplied the host path to the test server.
  No plugin/host source workaround was used for these test setup issues.

## Downloads, formats and unverified paths

The real first-generation auto-download was attempted and failed after retries:
the machine's configured proxy `127.0.0.1:7897` broke TLS handshakes to Hugging Face
and GitHub. The user then supplied the official BF16 file locally. No false
download success or placeholder file was used. Controlled tests cover successful
atomic publication/reuse, SHA mismatch, HTML corruption, timeout and cancellation.
**Actual official-network download success remains unverified in this environment.**

FP8 matrices and required per-row FP32 scales are exercised by CPU fixtures;
the author's actual FP8 checkpoint and FP16/FlashAttention/low-VRAM inference
matrices were not provided/tested. SageAttention BF16 and local fused LSE paths
were actually exercised. Anima 52-block models remain rejected. Turbo-specific
quality, Hires/refiner/inpaint mixed usage and foreign reference/modulation
extensions are not certified. The code explicitly rejects known incompatible paths.

## Reproduction

Use a disposable config with the correct model/modules and `anima_do_reference=false`.
Start Forge with `--api --port 7862`; a separate `--data-dir` and `--model-ref`
avoid changing normal user configuration. With an isolated data-dir, create its
configured temporary folder and allow the host JS path via `--gradio-allowed-path`.

```text
<host-python> tools/gpu_suite.py --reference <absolute.png> --reference2 <other.png> --output <outside-repo>
<host-python> tools/gpu_suite.py --reference <absolute.png> --output <outside-repo> --cases lora --lora <installed-name>
<host-python> tools/gpu_suite.py --reference <absolute.png> --output <outside-repo> --checkpoint Anima-2.9B-preview-v1.safetensors --cases single,repeat
<host-python> tools/gpu_suite.py --reference <absolute.png> --output <outside-repo> --size 1024 --steps 28 --megapixels 1 --cases single,repeat
<host-python> tools/compatibility_probe.py --reference <absolute.png> --output <outside-repo>
```

Clients save actual images and JSON responses, times, memory and infotext outside
Git. The extended client optionally accepts `--non-anima` and
`--non-anima-modules` for a locally installed incompatible model.
