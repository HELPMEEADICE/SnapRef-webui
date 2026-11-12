# SPDX-License-Identifier: GPL-3.0-only
"""Additional real Forge API checks. Use a disposable server configuration."""
import argparse
import base64
import concurrent.futures
import io
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import requests
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from snapref_webui.settings import Config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:7862")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--non-anima")
    parser.add_argument("--non-anima-modules", nargs="*")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    original_options = requests.get(args.url + "/sdapi/v1/options").json()
    report = []
    ref = base64.b64encode(args.reference.read_bytes()).decode()
    payload = dict(prompt="1girl, solo, sitting on a bench, crossed legs, night, street, full body, detailed anime illustration",
                   negative_prompt="low quality, worst quality, blurry", seed=782341, steps=12, cfg_scale=4.5,
                   distilled_cfg_scale=3.0, width=512, height=512, sampler_name="Euler", scheduler="Simple",
                   save_images=False, send_images=True, init_images=[ref], denoising_strength=.1)

    def run(name, cfg=None, expect_error=None, **overrides):
        config = asdict(cfg or Config(enabled=True, megapixels=.25))
        request = {**payload, **overrides}
        scripts = {"snapref-webui": {"args": list(config.values()) + [[]]}}
        scripts.update(request.pop("extra_scripts", {}))
        request["alwayson_scripts"] = scripts
        started = time.perf_counter()
        response = requests.post(args.url + "/sdapi/v1/img2img", json=request, timeout=1200)
        item = dict(case=name, status=response.status_code, seconds=round(time.perf_counter() - started, 3))
        if expect_error:
            if response.status_code == 200 or expect_error not in response.text:
                raise AssertionError((name, response.status_code, response.text[:1000]))
            item["error"] = response.text[:2000]
            pixels = None
        else:
            response.raise_for_status()
            result = response.json()
            image = Image.open(io.BytesIO(base64.b64decode(result["images"][0]))).convert("RGB")
            image.save(args.output / f"{name}.png")
            pixels = np.asarray(image).astype(np.float32)
            assert pixels.std() > 3 and 3 < pixels.mean() < 252
            item.update(images=len(result["images"]), mean=float(pixels.mean()), std=float(pixels.std()), info=json.loads(result["info"]))
        report.append(item)
        (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps({k: v for k, v in item.items() if k != "info"}), flush=True)
        return pixels

    try:
        baseline = run("reference_low_denoise")
        high = run("reference_high_denoise", denoising_strength=.95)
        assert np.array_equal(baseline, high), "Reference mode incorrectly depends on denoising strength"
        run("batch2", batch_size=2)
        run("cfg1", cfg_scale=1.)
        run("dpmpp2m_karras", sampler_name="DPM++ 2M", scheduler="Karras")
        spectrum = {"spectrum integrated": {"args": [True, .25, 6, .5, 2, 0., 6, .9]}}
        run("spectrum", extra_scripts=spectrum)
        run("guidance_spectrum_rejected", Config(enabled=True, megapixels=.25, guidance_on=True),
            expect_error="Reference Guidance", extra_scripts=spectrum)
        run("native_reference_rejected", expect_error="native Anima", override_settings={"anima_do_reference": True})
        run("inpaint_rejected", expect_error="plain img2img", mask=ref)

        # Cancel after the request has actually entered sampling, then verify restoration.
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(run, "cancelled", steps=200)
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline:
                progress = requests.get(args.url + "/sdapi/v1/progress", params={"skip_current_image": True}).json()
                if progress.get("state", {}).get("sampling_step", 0) >= 2:
                    requests.post(args.url + "/sdapi/v1/interrupt", timeout=15).raise_for_status()
                    break
                if future.done():
                    raise AssertionError("Generation finished before cancellation")
                time.sleep(.2)
            future.result(timeout=120)
        restored = run("after_cancel")
        assert np.array_equal(baseline, restored), "Reference changed after cancellation/error"
        if args.non_anima:
            modules = args.non_anima_modules or original_options["forge_additional_modules"]
            requests.post(args.url + "/sdapi/v1/options", json={"sd_model_checkpoint": args.non_anima,
                          "forge_additional_modules": modules}, timeout=120).raise_for_status()
            run("non_anima_rejected", expect_error="Anima engine")
    finally:
        requests.post(args.url + "/sdapi/v1/options", json={key: original_options[key] for key in
            ("sd_model_checkpoint", "forge_additional_modules", "anima_do_reference")}, timeout=120).raise_for_status()


if __name__ == "__main__":
    main()
