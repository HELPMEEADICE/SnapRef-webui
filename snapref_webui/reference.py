# SPDX-License-Identifier: GPL-3.0-only
# Explicit reference RoPE/sigma logic adapted from ArcAnima/SnapRef (Apache-2.0).
# Modified 2026-11-12 for Neo's prepare_embedded_sequence signature and scoped hooks.
import base64
import io
import math

import numpy as np
import torch
from PIL import Image, ImageOps

from .i18n import SnapRefError


def image_from_input(value):
    if isinstance(value, (tuple, list)):
        value = value[0]
    if isinstance(value, str):
        try:
            raw = value.split(",", 1)[1] if value.startswith("data:") else value
            value = Image.open(io.BytesIO(base64.b64decode(raw, validate=True)))
            value.load()
        except Exception as exc:
            raise SnapRefError("bad_image", detail=str(exc)) from exc
    if not isinstance(value, Image.Image):
        raise SnapRefError("bad_image", detail=type(value).__name__)
    value = ImageOps.exif_transpose(value).convert("RGBA")
    bg = Image.new("RGBA", value.size, "white")
    return Image.alpha_composite(bg, value).convert("RGB")


def select_references(main, extra, config):
    items = ([main] if config.use_main else []) + list(extra or [])
    if len(items) > 8:
        raise SnapRefError("max_references")
    try:
        weights = [float(x.strip()) for x in config.weights.split(",") if x.strip()]
    except ValueError as exc:
        raise SnapRefError("bad_weights") from exc
    if not weights or any(not math.isfinite(w) or not 0 <= w <= 2 for w in weights):
        raise SnapRefError("bad_weights")
    weights += [1.0] * max(0, len(items) - len(weights))
    refs = [(image_from_input(image), weight) for image, weight in zip(items, weights) if image is not None and weight > 0]
    if not refs:
        raise SnapRefError("no_reference")
    if config.layout == "stitch" and len(refs) > 1:
        h = refs[0][0].height
        pieces = [img.resize((max(1, round(img.width * h / img.height)), h), Image.Resampling.LANCZOS) for img, _ in refs]
        sheet = Image.new("RGB", (sum(im.width for im in pieces), h), "white")
        offset = 0
        for image in pieces:
            sheet.paste(image, (offset, 0))
            offset += image.width
        return [(sheet, 1.0)]
    return refs


def resized(image, megapixels):
    scale = min(1.0, math.sqrt(megapixels * 1024**2 / (image.width * image.height)))
    size = tuple(max(16, int(side * scale) // 16 * 16) for side in image.size)
    return image.resize(size, Image.Resampling.LANCZOS)


@torch.inference_mode()
def encode_image(image, vae):
    pixels = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255).unsqueeze(0)
    ref = vae.first_stage_model.process_in(vae.encode(pixels)).detach().cpu()
    return ref.unsqueeze(2) if ref.ndim == 4 else ref


def explicit_rope(pe, t, h, w):
    def rot(pos, freq):
        angle = torch.outer(pos.float(), freq)
        return torch.stack((angle.cos(), -angle.sin(), angle.sin(), angle.cos()), -1)
    emb = torch.cat([rot(t, 1 / ((10000 * pe.t_ntk_factor) ** pe.dim_temporal_range.to(t.device))),
                     rot(h, 1 / ((10000 * pe.h_ntk_factor) ** pe.dim_spatial_range.to(t.device))),
                     rot(w, 1 / ((10000 * pe.w_ntk_factor) ** pe.dim_spatial_range.to(t.device)))], -2)
    return emb.reshape(*emb.shape[:-1], 2, 2).float()


@torch.inference_mode()
def encode_reference(dm, ref, context, target_hw, adapter, mode, state, interrupted=lambda: False):
    x, _, extra = dm.prepare_embedded_sequence(ref, padding_mask=None)
    if extra is not None:
        raise SnapRefError("unsupported_structure")
    _, _, hp, wp, _ = x.shape
    h0 = round((target_hw[0] // dm.patch_spatial) / 2 - hp / 2)
    w0 = round((target_hw[1] // dm.patch_spatial) / 2 - wp / 2)
    hh, ww = torch.meshgrid(torch.arange(hp, device=ref.device) + h0, torch.arange(wp, device=ref.device) + w0, indexing="ij")
    rope = explicit_rope(dm.pos_embedder, torch.full_like(hh, adapter.ref_t).flatten(), hh.flatten(), ww.flatten())[None, :, None]
    temb, adaln = dm.t_embedder[1](dm.t_embedder[0](torch.zeros(ref.shape[0], 1, device=ref.device)).to(x.dtype))
    temb = dm.t_embedding_norm(temb) + adapter.mode_embed[adapter.meta["modes"].index(mode)].to(temb)[None, None]
    state.stream, state.record = "ref", [None] * len(dm.blocks)
    try:
        x = x.float() if x.dtype == torch.float16 else x
        for block in dm.blocks:
            if interrupted():
                raise SnapRefError("cancelled")
            x = block(x, temb, context, rope_emb_L_1_1_D=rope, adaln_lora_B_T_3D=adaln,
                      extra_per_block_pos_emb=None, transformer_options={})
        result = state.record
        if any(pair is None or not all(torch.isfinite(t).all() for t in pair) for pair in result):
            raise SnapRefError("nonfinite")
        return result
    finally:
        state.stream = state.record = None
