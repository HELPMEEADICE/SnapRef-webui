# Forge API

The standard `POST /sdapi/v1/img2img` endpoint and the UI share the request-local
runtime. Obtain current script information from `/sdapi/v1/script-info`; script
name is `snapref-webui`, not the translated tab caption.

`alwayson_scripts["snapref-webui"]["args"]` is positional:

| Index | Parameter | Default |
|---:|---|---|
| 0 | enabled | false |
| 1 | use_main (use init_images[0]) | true |
| 2 | mode | `cs` (`char` also accepted) |
| 3 | strength | 1.0 |
| 4 | fidelity | 1.0 |
| 5 | megapixels | 1.0 |
| 6 | layout | `separate` (`stitch` also accepted) |
| 7 | weights | `1,1,1,1,1,1,1,1` |
| 8 | in_uncond | false |
| 9 | guidance_on | false |
| 10 | guidance | 4.5 |
| 11 | correction | `apg` (`rescale`, `none` also accepted) |
| 12 | guidance_end | 1.0 |
| 13 | caption | empty string |
| 14 | log_bias | 0.0 |
| 15 | preferred storage precision | `bf16` (`fp8` also accepted) |
| 16 | language | `en` (`zh-CN`, `ja-JP` also accepted) |
| 17 | extra references | list of base64 image strings |
| 18 | dedicated primary reference | optional base64 image; used when use_main=false |
| 19 | UI sub-tab state | false; UI-only Gradio State, leave default in API |

First 18 arguments remain sufficient. API clients normally supply the primary
reference using `init_images` and use_main=true. The dedicated primary input is
for the new native Reference Generation sub-tab and does not change existing
API behavior. Do not pass server-local filesystem paths as image strings.
The first active primary image's weight comes first, then gallery weights.

```python
import base64
from pathlib import Path
import requests

ref = base64.b64encode(Path("reference.png").read_bytes()).decode()
response = requests.post("http://127.0.0.1:7862/sdapi/v1/img2img", json={
    "init_images": [ref],
    "prompt": "1girl, solo, sitting on a bench, crossed legs, night, street",
    "negative_prompt": "low quality, worst quality",
    "seed": 782341,
    "width": 1024, "height": 1024,
    "steps": 28, "cfg_scale": 4.5, "distilled_cfg_scale": 3.0,
    "sampler_name": "Euler", "scheduler": "Simple",
    "denoising_strength": 0.1,
    "alwayson_scripts": {"snapref-webui": {"args": [
        True, True, "cs", 1.0, 1.0, 1.0, "separate",
        "1,1,1,1,1,1,1,1", False, False, 4.5, "apg", 1.0,
        "", 0.0, "bf16", "en", []
    ]}},
    "save_images": False
}, timeout=1200)
response.raise_for_status()
Path("output.png").write_bytes(base64.b64decode(response.json()["images"][0]))
```

The low denoising_strength above deliberately demonstrates independence from
ordinary img2img strength. There is no reference image data in `SnapRef Config`.
PNG Info restores the JSON settings; clients must supply images again.
Errors abort the request with Forge's normal error response, including a
localized SnapRef message. Non-Anima calls do not start adapter downloads.

## Native sub-tab integration

The extension adds a Gradio TabItem to the captured `mode_img2img` Tabs object,
beside img2img, Sketch, Inpaint and Batch. It does not replace Forge's global
submit function or add a seventh undocumented integer branch to img2img.py.
The new sub-tab selects native transport mode 0, while the request's own
`init`/`sample` methods implement fresh-noise character generation.

Selecting an existing sub-tab turns SnapRef off for that browser session and
restores the native resize/denoise controls. Entering Reference Generation
selects Resize to and hides Resize by, remembering its previous selection for
return to a native tab. Directly uploading a primary reference unchecks use_main,
so the original img2img canvas is never overwritten. With use_main checked,
its current image is previewed and reused without a second upload.

The backend validates both Anima type and the enabled state. If a user disables
SnapRef while staying inside Reference Generation, Generate returns a clear
message instead of silently performing ordinary img2img.
