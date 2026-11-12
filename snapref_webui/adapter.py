# SPDX-License-Identifier: GPL-3.0-only
# Storage rules adapted from ArcAnima/SnapRef (Apache-2.0); modified for strict
# branch coverage, dimension/finite validation and Windows mmap release.
"""SnapRef anima_ref_kv/1 storage, deliberately independent of Forge's LoRA loader."""
import json
import math
import re
from pathlib import Path

import torch
from safetensors import safe_open

from .i18n import SnapRefError
from .model_detection import block_map

PREFIX = "anima_ref_kv."
TARGET = re.compile(r"^blocks\.(\d+)\.(self_attn\.(q_proj|k_proj|v_proj|output_proj)|cross_attn\.(q_proj|output_proj)|mlp\.(layer1|layer2))$")


def metadata(handle):
    md = handle.metadata() or {}
    try:
        result = json.loads(md.get("snapref") or md.get("anima_ref_kv") or "{}")
    except (TypeError, json.JSONDecodeError) as exc:
        raise SnapRefError("bad_adapter", detail=str(exc)) from exc
    if result.get("format") != "anima_ref_kv/1":
        raise SnapRefError("bad_adapter", detail="format != anima_ref_kv/1")
    return result


def read_matrix(handle, key):
    value = handle.get_tensor(key)
    if value.ndim != 2:
        raise SnapRefError("bad_adapter", detail=key)
    if value.dtype == torch.float8_e4m3fn:
        scale = handle.get_tensor(key + "_scale")
        if scale.dtype != torch.float32 or tuple(scale.shape) != (value.shape[0], 1) or not torch.isfinite(scale).all():
            raise SnapRefError("bad_adapter", detail=key + "_scale")
        value = value.float() * scale
    elif value.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise SnapRefError("bad_adapter", detail=str(value.dtype))
    if not torch.isfinite(value).all():
        raise SnapRefError("bad_adapter", detail=key)
    # Detach from safetensors mmap: Windows must allow adapter replacement/uninstall.
    return value.to(torch.bfloat16).clone()


def inspect_file(path):
    """Header-level scan; safe_open also validates the tensor extent against file length."""
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            meta = metadata(handle)
            keys = set(handle.keys())
            declared = meta.get("lora", {})
            expected = set()
            for stream in ("ref", "tgt"):
                spec = declared.get(stream, {})
                if not isinstance(spec.get("rank"), int) or spec["rank"] < 1 or not spec.get("targets"):
                    raise SnapRefError("bad_adapter", detail=f"metadata lora.{stream}")
                for index in range(28):
                    for target in spec["targets"]:
                        name = f"blocks.{index}.{target}"
                        if not TARGET.fullmatch(name):
                            raise SnapRefError("bad_adapter", detail=name)
                        expected.add(f"{PREFIX}{stream}.{name}.lora_down")
            actual = {key for key in keys if key.endswith(".lora_down")}
            if expected != actual:
                raise SnapRefError("bad_adapter", detail="metadata/tensor target coverage mismatch")
            branches = {s: set() for s in ("ref", "tgt")}
            for key in keys:
                if not key.endswith(".lora_down"):
                    continue
                _, stream, name = key[:-len(".lora_down")].split(".", 2)
                if stream not in branches or not TARGET.fullmatch(name):
                    raise SnapRefError("bad_adapter", detail=key)
                stem = key[:-len("lora_down")]
                if not {stem + "lora_up", stem + "alpha"} <= keys:
                    raise SnapRefError("bad_adapter", detail=stem)
                down_shape = handle.get_slice(key).get_shape()
                up_shape = handle.get_slice(stem + "lora_up").get_shape()
                rank = declared[stream]["rank"]
                if len(down_shape) != 2 or len(up_shape) != 2 or down_shape[0] != rank or up_shape[1] != rank:
                    raise SnapRefError("bad_adapter", detail=stem + " rank")
                branches[stream].add(int(name.split(".")[1]))
            if any(indices != set(range(28)) for indices in branches.values()):
                raise SnapRefError("bad_adapter", detail="ref/tgt blocks != 0..27")
            if "anima_ref_kv.mode_embed" not in keys or not {"cs", "char"} <= set(meta.get("modes", [])):
                raise SnapRefError("bad_adapter", detail="mode_embed / modes")
            return meta
    except SnapRefError:
        raise
    except Exception as exc:
        raise SnapRefError("bad_adapter", detail=f"{Path(path).name}: {exc}") from exc


class Adapter:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.meta = inspect_file(path)
        stat = self.path.stat()
        self.identity = (str(self.path), stat.st_mtime_ns, stat.st_size)
        self.weights = {}
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            self.mode_embed = handle.get_tensor(PREFIX + "mode_embed").float().clone()
            if self.mode_embed.ndim != 2 or self.mode_embed.shape[0] != len(self.meta["modes"]) or not torch.isfinite(self.mode_embed).all():
                raise SnapRefError("bad_adapter", detail="mode_embed shape / finite")
            for key in handle.keys():
                if not key.endswith(".lora_down"):
                    continue
                stem = key[:-len(".lora_down")]
                _, stream, name = stem.split(".", 2)
                down, up = read_matrix(handle, key), read_matrix(handle, stem + ".lora_up")
                alpha = float(handle.get_tensor(stem + ".alpha"))
                if down.shape[0] != up.shape[1] or down.shape[0] < 1 or not math.isfinite(alpha):
                    raise SnapRefError("bad_adapter", detail=stem)
                self.weights.setdefault(name, {})[stream] = (down, up, alpha / down.shape[0])
        self.indices = {int(name.split(".")[1]) for name in self.weights}
        self.ref_t = int(self.meta.get("ref_t", 10))

    def resolve(self, dm):
        mapping = block_map(self.indices, len(dm.blocks))
        targets = []
        for name, streams in self.weights.items():
            _, index, suffix = name.split(".", 2)
            module = dm.get_submodule(f"blocks.{mapping[int(index)]}.{suffix}")
            for down, up, _ in streams.values():
                if (down.shape[1], up.shape[0]) != (module.in_features, module.out_features):
                    raise SnapRefError("bad_adapter", detail=name + " dimensions")
            targets.append((module, streams))
        if self.mode_embed.shape[1] != dm.blocks[0].x_dim:
            raise SnapRefError("bad_adapter", detail="mode_embed width")
        return targets
