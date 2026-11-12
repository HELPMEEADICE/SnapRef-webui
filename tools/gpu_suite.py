# SPDX-License-Identifier: GPL-3.0-only
"""Run against an isolated Forge --api server. Images/reports stay outside Git."""
import argparse
import base64
import io
import json
import time
from dataclasses import asdict
from pathlib import Path
import sys

import numpy as np
import requests
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from snapref_webui.settings import Config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:7862")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--reference2", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", default="baseline,single,repeat,switch,char,cs,multi,restored")
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--lora")
    parser.add_argument("--non-anima")
    parser.add_argument("--checkpoint")
    parser.add_argument("--cfg", type=float, default=4.5)
    parser.add_argument("--sampler", default="Euler")
    parser.add_argument("--scheduler", default="Simple")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--denoising", type=float, default=.1)
    parser.add_argument("--precision", default="bf16", choices=["bf16", "fp8"])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.checkpoint:
        response = requests.post(args.url + "/sdapi/v1/options", json={"sd_model_checkpoint": args.checkpoint}, timeout=120)
        response.raise_for_status()
    def encode(path):
        return base64.b64encode(path.read_bytes()).decode()
    ref = encode(args.reference)
    ref2 = encode(args.reference2) if args.reference2 else ref
    report = []
    baseline = None
    for name in args.cases.split(","):
        config = asdict(Config(enabled=True, megapixels=.25, precision=args.precision))
        payload = dict(prompt="1girl, solo, sitting on a bench, crossed legs, night, street, full body, detailed anime illustration",
                       negative_prompt="low quality, worst quality, blurry", seed=782341, steps=args.steps,
                       cfg_scale=args.cfg, distilled_cfg_scale=3.0, width=args.size, height=args.size,
                       sampler_name=args.sampler, scheduler=args.scheduler, batch_size=args.batch_size, n_iter=1,
                       save_images=False, send_images=True)
        endpoint = "/sdapi/v1/img2img"
        extras = []
        if name in ("baseline", "restored"):
            endpoint = "/sdapi/v1/txt2img"
        else:
            payload.update(init_images=[ref], denoising_strength=args.denoising)
            if name in ("native_before", "native_after"):
                config["enabled"] = False
            if name == "strength_zero":
                config["strength"] = 0.
            if name == "stitch":
                extras = [ref2]
                config["layout"] = "stitch"
            if name == "three":
                extras = [ref2, ref]
                config["weights"] = "1,0.5,0.2"
            if name == "in_uncond":
                config["in_uncond"] = True
            if name == "switch":
                payload["init_images"] = [ref2]
            if name == "char":
                config["mode"] = "char"
            if name == "multi":
                extras = [ref2]
                config["weights"] = "1,0.7"
            if name == "fidelity":
                config["fidelity"] = .6
            if name == "guidance":
                config["guidance_on"] = True
                config["guidance_end"] = .6
            if name == "lora":
                if not args.lora:
                    raise ValueError("--lora is required for lora case")
                payload["prompt"] += f" <lora:{args.lora}:0.5>"
            if name == "nonanima":
                if not args.non_anima:
                    raise ValueError("--non-anima is required")
                payload["override_settings"] = {"sd_model_checkpoint": args.non_anima}
            payload["alwayson_scripts"] = {"snapref-webui": {"args": list(config.values()) + [extras]}}
        print(f"Starting {name}", flush=True)
        started = time.perf_counter()
        response = requests.post(args.url + endpoint, json=payload, timeout=2400)
        entry = dict(case=name, status=response.status_code, seconds=round(time.perf_counter() - started, 3))
        if response.status_code != 200:
            entry["error"] = response.text[:4000]
            report.append(entry)
            (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(json.dumps(entry), flush=True)
            if name == "nonanima" and "Anima" in response.text:
                continue
            raise RuntimeError(response.text)
        result = response.json()
        if len(result["images"]) < args.batch_size:
            raise AssertionError("Missing batch outputs")
        data = base64.b64decode(result["images"][0])
        image = Image.open(io.BytesIO(data)).convert("RGB")
        image.save(args.output / f"{name}.png")
        pixels = np.asarray(image, dtype=np.float32)
        entry.update(mean=float(pixels.mean()), std=float(pixels.std()), min=float(pixels.min()), max=float(pixels.max()),
                     info=json.loads(result["info"]), memory=requests.get(args.url + "/sdapi/v1/memory").json())
        if pixels.std() < 3 or pixels.mean() < 3 or pixels.mean() > 252:
            raise AssertionError(f"Degenerate output: {entry}")
        if name == "baseline":
            baseline = pixels
        elif name == "restored" and baseline is not None:
            entry["baseline_max_difference"] = float(np.max(np.abs(pixels - baseline)))
            if entry["baseline_max_difference"] != 0:
                raise AssertionError("Baseline changed after SnapRef")
        ref_pixels = np.asarray(Image.open(args.reference).convert("RGB").resize(image.size), dtype=np.float32)
        entry["reference_mean_absolute_difference"] = float(np.abs(pixels - ref_pixels).mean())
        report.append(entry)
        (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps({k: v for k, v in entry.items() if k not in ("info", "memory")}), flush=True)


if __name__ == "__main__":
    main()
