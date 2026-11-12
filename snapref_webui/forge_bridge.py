# SPDX-License-Identifier: GPL-3.0-only
"""Transactional Forge adapter: no class monkeypatches or persistent attention hooks."""
import logging
import math
import threading
from contextlib import contextmanager
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .attention import weighted_attention
from .cache import CACHE
from .conditioning import reference_guidance, row_biases
from .i18n import SnapRefError
from .reference import encode_reference

LOCK = threading.RLock()
log = logging.getLogger("snapref")


@dataclass
class ForwardState:
    stream: str | None = None
    record: list | None = None
    kv: list | None = None
    sizes: list | None = None
    rows: torch.Tensor | None = None


@contextmanager
def hooks(dm, adapter, state, config, device_weights):
    """Resolve before mutating, remove every handle and restore exact attribute ownership."""
    targets = adapter.resolve(dm)
    saved, handles = [], []
    try:
        for module, streams in targets:
            def delta(mod, args, out, _streams=streams):
                if state.stream not in _streams:
                    return out
                down, up, scale = _streams[state.stream]
                x = args[0]
                key = (id(down), x.device, x.dtype)
                if key not in device_weights:
                    device_weights[key] = down.to(x), up.to(x)
                d, u = device_weights[key]
                return out + F.linear(F.linear(x, d), u) * scale
            handles.append(module.register_forward_hook(delta))
        for index, block in enumerate(dm.blocks):
            attn = block.self_attn
            saved.append((attn, "forward", "forward" in attn.__dict__, attn.__dict__.get("forward")))
            original = attn.forward
            def forward(x, context=None, rope_emb=None, transformer_options=None, _attn=attn, _i=index, _orig=original):
                options = transformer_options or {}
                if state.record is None and state.kv is None:
                    return _orig(x, context, rope_emb=rope_emb, transformer_options=options)
                q, k, v = _attn.compute_qkv(x, context, rope_emb=rope_emb)
                if state.record is not None:
                    state.record[_i] = (k.detach(), v.detach())
                    return _attn.compute_attention(q, k, v, transformer_options=options)
                biases = [config.log_bias + math.log(config.strength * weight) for weight in state.weights]
                out = weighted_attention(_attn, q, k, v, state.kv[_i], state.sizes, biases, state.rows, options)
                return _attn.output_dropout(_attn.output_proj(out))
            attn.forward = forward
        yield
    finally:
        state.stream = state.record = state.kv = state.rows = None
        for handle in reversed(handles):
            handle.remove()
        for obj, attr, owned, value in reversed(saved):
            if owned:
                setattr(obj, attr, value)
            else:
                delattr(obj, attr)


def conflict_check(engine, unet, config):
    from backend.args import dynamic_args
    from modules import shared
    if shared.opts.anima_do_reference or dynamic_args.ref_latents or getattr(engine, "ref_latents", []):
        raise SnapRefError("native_reference_conflict")
    foreign = [name for name in unet.object_patches if name.startswith("diffusion_model") and
               ("self_attn" in name or name.endswith("forward"))]
    if foreign:
        raise SnapRefError("extension_conflict", detail=", ".join(foreign))
    dm = unet.get_model_object("diffusion_model")
    if any("forward" in b.self_attn.__dict__ or "compute_attention" in b.self_attn.__dict__ for b in dm.blocks):
        raise SnapRefError("extension_conflict", detail="self_attn")
    if config.guidance_on and "model_function_wrapper" in unet.model_options:
        raise SnapRefError("guidance_conflict")


@contextmanager
def sampling_transaction(engine, adapter, latents, context, weights, config, key, interrupted):
    with LOCK:
        unet = engine.forge_objects.unet
        conflict_check(engine, unet, config)
        dm = unet.get_model_object("diffusion_model")
        adapter.resolve(dm)
        clone = unet.clone()
        original = clone.get_model_object("diffusion_model.forward")
        state, device_weights, gpu_cache = ForwardState(), {}, {}
        state.weights = weights
        cached = CACHE.get(key)
        log.info("SnapRef K/V cache %s", "hit" if cached is not None else "miss")

        def wrapped(x, timesteps, context, padding_mask=None, **kwargs):
            options = kwargs.get("transformer_options") or {}
            if interrupted():
                raise SnapRefError("cancelled")
            from backend.args import dynamic_args
            if dynamic_args.ref_latents:
                raise SnapRefError("native_reference_conflict")
            with hooks(dm, adapter, state, config, device_weights):
                if "kv" not in gpu_cache:
                    if cached is None:
                        records = [encode_reference(dm, ref.to(x), reference_context.to(x), tuple(x.shape[-2:]), adapter,
                                                    config.mode, state, interrupted) for ref in latents]
                        sizes = [record[0][0].shape[1] for record in records]
                        kv = [(torch.cat([r[i][0] for r in records], 1), torch.cat([r[i][1] for r in records], 1)) for i in range(len(dm.blocks))]
                        CACHE.put(key, (kv, sizes))
                    else:
                        kv, sizes = cached
                        kv = [(k.to(x), v.to(x)) for k, v in kv]
                    gpu_cache["kv"] = kv, sizes
                state.kv, state.sizes = gpu_cache["kv"]
                state.stream = "tgt"
                state.rows = row_biases(options, x.shape[0], config, x.device)
                result = original(x, timesteps, context=context, padding_mask=padding_mask, **kwargs)
                if not torch.isfinite(result).all():
                    raise SnapRefError("nonfinite")
                return result

        reference_context = context.detach().cpu()
        clone.add_object_patch("diffusion_model.forward", wrapped)
        tokens = sum((ref.shape[-2] // dm.patch_spatial) * (ref.shape[-1] // dm.patch_spatial) for ref in latents)
        itemsize = torch.empty((), dtype=unet.model.computation_dtype).element_size()
        lora_bytes = sum((d.numel() + u.numel()) * itemsize for streams in adapter.weights.values() for d, u, _ in streams.values())
        clone.add_extra_preserved_memory_during_sampling(tokens * dm.blocks[0].x_dim * 2 * len(dm.blocks) * itemsize + lora_bytes)
        if config.guidance_on:
            clone.set_model_sampler_post_cfg_function(reference_guidance(config), disable_cfg1_optimization=True)
        engine.forge_objects.unet = clone
        try:
            yield
        finally:
            # Forge shares object_patches_backup between clones; restore through its API.
            clone.unpatch_model(unpatch_weights=False)
            engine.forge_objects.unet = unet
            clone.object_patches.pop("diffusion_model.forward", None)
            device_weights.clear()
            gpu_cache.clear()
