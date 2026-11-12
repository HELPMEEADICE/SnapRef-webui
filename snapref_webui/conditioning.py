# SPDX-License-Identifier: GPL-3.0-only
# Guidance equations adapted from ArcAnima/SnapRef (Apache-2.0).
import torch
import torch.nn.functional as F


def row_biases(options, batch, config, device):
    if options.get("snapref_off"):
        return torch.full((batch,), -float("inf"), device=device)
    flags = options.get("cond_or_uncond") or [0]
    if batch % len(flags):
        from .i18n import SnapRefError
        raise SnapRefError("unsupported_structure")
    negative = 0.0 if config.in_uncond or config.guidance_on else -float("inf") if config.fidelity >= 1 else __import__("math").log(max(1 - config.fidelity, 1e-6))
    return torch.tensor([0.0 if f == 0 else negative for f in flags], device=device).repeat_interleave(batch // len(flags))


def correct(guided, positive, mode):
    dims = tuple(range(1, positive.ndim))
    if mode == "apg":
        update = (guided - positive).double()
        unit = F.normalize(positive.double().flatten(1), dim=1).view_as(positive)
        return positive + (update - (update * unit).sum(dims, keepdim=True) * unit).to(positive.dtype)
    if mode == "rescale":
        ratio = positive.std(dims, keepdim=True) / guided.std(dims, keepdim=True).clamp_min(1e-8)
        return .5 * guided * ratio + .5 * guided
    return guided


def reference_guidance(config):
    def post_cfg(args):
        from backend.sampling.sampling_function import calc_cond_uncond_batch
        guided, positive = args["denoised"], args["cond_denoised"]
        if config.guidance_end >= 1 or float(args["sigma"].flatten()[0]) >= args["model"].predictor.percent_to_sigma(config.guidance_end) - 1e-5:
            options = {**args["model_options"], "transformer_options": {**args["model_options"].get("transformer_options", {}), "snapref_off": True}}
            cond = args["uncond"] if args["uncond"] is not None else args["cond"]
            ref = args["uncond_denoised"] if args["uncond"] is not None else positive
            base, _ = calc_cond_uncond_batch(args["model"], cond, None, args["input"], args["sigma"], options)
            guided = guided + (config.guidance - 1) * (ref - base)
        return correct(guided, positive, config.correction)
    return post_cfg
