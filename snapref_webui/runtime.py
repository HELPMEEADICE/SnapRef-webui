# SPDX-License-Identifier: GPL-3.0-only
import json
import logging
import uuid
import weakref
from dataclasses import fields
from types import MethodType

import torch

from .cache import CACHE, digest
from .forge_bridge import LOCK, conflict_check, sampling_transaction
from .i18n import CURRENT, SnapRefError
from .model_detection import detect_engine
from .model_manager import ensure_adapter, release
from .reference import encode_image, resized, select_references
from .settings import Config

log = logging.getLogger("snapref")
CONFIG_FIELDS = [f.name for f in fields(Config)]
_identities = weakref.WeakKeyDictionary()
_prepared = {}


def clear():
    with LOCK:
        CACHE.clear()
        _prepared.clear()
        release()


def identity(obj):
    if obj not in _identities:
        _identities[obj] = uuid.uuid4().hex
    return _identities[obj]


def interrupted():
    from modules import shared
    return shared.state.interrupted or shared.state.stopping_generation or shared.state.skipped


def status(text):
    from modules import shared
    shared.state.textinfo = text
    log.info(text)


def prepare(p, config, refs, adapter):
    from modules.prompt_parser import SdConditioning
    engine = p.sd_model
    forge = engine.forge_objects
    dm = detect_engine(engine)
    images = [(resized(image, config.megapixels), weight) for image, weight in refs]
    key = digest([identity(dm), getattr(engine, "sd_model_hash", None), adapter.identity,
                  identity(forge.vae.first_stage_model), identity(forge.clip.cond_stage_model),
                  str(forge.clip.patcher.patches_uuid), str(forge.unet.patches_uuid), config.mode,
                  config.caption, config.layout, [(im.size, w) for im, w in images],
                  *[im.tobytes() for im, _ in images]])
    if _prepared.get("key") != key:
        _prepared.clear()
        latents = [encode_image(image, forge.vae) for image, _ in images]
        # Negative flag avoids native Anima reference mutations; caption text is unchanged.
        context = engine.get_learned_conditioning(SdConditioning([config.caption], is_negative_prompt=True))[0].detach().cpu()
        _prepared.update(key=key, latents=latents, context=context)
    cache_key = digest([key, p.width, p.height, str(forge.unet.load_device),
                        str(forge.unet.model.computation_dtype), _prepared["context"].float().numpy().tobytes()])
    return _prepared["latents"], _prepared["context"], [w for _, w in images], cache_key


def bind_request(p, values):
    """Called by Script.process; all fallible work is deferred into p.init/sample."""
    if hasattr(p, "_snapref_restore"):
        p._snapref_restore()
    if not values or not values[0]:
        clear()
        return
    originals = {name: (name in p.__dict__, p.__dict__.get(name)) for name in ("init", "sample", "close")}
    old_close = p.close
    request = {}

    def restore():
        for name, (owned, value) in originals.items():
            if owned:
                setattr(p, name, value)
            else:
                p.__dict__.pop(name, None)
        p.__dict__.pop("_snapref_restore", None)

    def init(this, all_prompts, all_seeds, all_subseeds):
        from modules import sd_samplers
        cfg = Config(**dict(zip(CONFIG_FIELDS, values[:len(CONFIG_FIELDS)])))
        token = CURRENT.set(cfg.language)
        try:
            cfg.validate()
            detect_engine(this.sd_model)
            if this.image_mask is not None or this.latent_mask is not None:
                raise SnapRefError("inpaint_unsupported")
            if getattr(this, "enable_hr", False) or this.refiner_checkpoint_info is not None:
                raise SnapRefError("hires_unsupported")
            conflict_check(this.sd_model, this.sd_model.forge_objects.unet, cfg)
            main = (this.init_images or [None])[0]
            refs = select_references(main, values[len(CONFIG_FIELDS)] if len(values) > len(CONFIG_FIELDS) else [], cfg)
            request.update(config=cfg, refs=refs)
            this.sampler = sd_samplers.create_sampler(this.sampler_name, this.sd_model)
            this.image_cfg_scale = None
            this.init_latent = None
            this.mask = this.nmask = None
            this.overlay_images = this.color_corrections = None
            this.paste_to = None
            # No VAE encoding of init_images, no denoising strength, no resize-to-input.
        finally:
            CURRENT.reset(token)

    def sample(this, conditioning, unconditional_conditioning, seeds, subseeds, subseed_strength, prompts):
        from modules import processing
        cfg = request["config"]
        token = CURRENT.set(cfg.language)
        try:
            detect_engine(this.sd_model)
            this.sd_model.set_shift(shift=this.distilled_cfg_scale, width=this.width, height=this.height)
            noise = this.rng.next()
            this.sd_model.forge_objects = this.sd_model.forge_objects_after_applying_lora.shallow_copy()
            processing.apply_token_merging(this.sd_model, this.get_token_merging_ratio())
            if this.scripts is not None:
                this.scripts.process_before_every_sampling(this, x=noise, noise=noise, c=conditioning, uc=unconditional_conditioning)
            if this.modified_noise is not None:
                noise, this.modified_noise = this.modified_noise, None
            conflict_check(this.sd_model, this.sd_model.forge_objects.unet, cfg)
            if cfg.strength == 0:
                return this.sampler.sample(this, noise, conditioning, unconditional_conditioning, image_conditioning=this.txt2img_image_conditioning(noise))
            adapter = ensure_adapter(cfg.precision, status, interrupted)
            latents, context, weights, key = prepare(this, cfg, request["refs"], adapter)
            this.extra_generation_params.update({"SnapRef": True, "SnapRef Config": json.dumps(cfg.metadata(), ensure_ascii=True, separators=(",", ":")),
                                                "SnapRef Adapter": adapter.path.name, "SnapRef Adapter Step": adapter.meta.get("step", "unknown"),
                                                "SnapRef Adapter Identity": str(adapter.identity), "SnapRef Model": this.sd_model_name,
                                                "SnapRef References": len(request["refs"])})
            with sampling_transaction(this.sd_model, adapter, latents, context, weights, cfg, key, interrupted):
                return this.sampler.sample(this, noise, conditioning, unconditional_conditioning, image_conditioning=this.txt2img_image_conditioning(noise))
        except torch.cuda.OutOfMemoryError as exc:
            clear()
            raise SnapRefError("oom") from exc
        except Exception:
            clear()
            raise
        finally:
            CURRENT.reset(token)

    def close(this):
        try:
            return old_close()
        finally:
            request.clear()
            restore()

    p._snapref_restore = restore
    p.init, p.sample, p.close = MethodType(init, p), MethodType(sample, p), MethodType(close, p)
