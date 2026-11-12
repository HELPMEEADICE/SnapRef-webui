# SPDX-License-Identifier: GPL-3.0-only
import json
import math
import os
import threading
from dataclasses import asdict, dataclass
from pathlib import Path

PREFERENCES = Path(__file__).resolve().parent.parent / "settings.json"
_lock = threading.RLock()


def read_preferences():
    with _lock:
        try:
            return json.loads(PREFERENCES.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}


def save_preferences(**values):
    with _lock:
        data = {**read_preferences(), **values}
        temp = PREFERENCES.with_suffix(".part")
        temp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        os.replace(temp, PREFERENCES)


@dataclass(frozen=True)
class Config:
    enabled: bool = False
    use_main: bool = True
    mode: str = "cs"
    strength: float = 1.0
    fidelity: float = 1.0
    megapixels: float = 1.0
    layout: str = "separate"
    weights: str = "1,1,1,1,1,1,1,1"
    in_uncond: bool = False
    guidance_on: bool = False
    guidance: float = 4.5
    correction: str = "apg"
    guidance_end: float = 1.0
    caption: str = ""
    log_bias: float = 0.0
    precision: str = "bf16"
    language: str = "en"

    def validate(self):
        from .i18n import SnapRefError
        enums = {"mode": ("cs", "char"), "layout": ("separate", "stitch"),
                 "correction": ("apg", "rescale", "none"), "precision": ("bf16", "fp8"),
                 "language": ("zh-CN", "ja-JP", "en")}
        for name, choices in enums.items():
            if getattr(self, name) not in choices:
                raise SnapRefError("invalid_parameter", name=name, value=getattr(self, name))
        for name, lo, hi in (("strength", 0, 2), ("fidelity", 0, 1), ("megapixels", .25, 2),
                             ("guidance", 0, 15), ("guidance_end", 0, 1), ("log_bias", -3, 3)):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or not lo <= value <= hi:
                raise SnapRefError("invalid_parameter", name=name, value=value)
        return self

    def metadata(self):
        return asdict(self)
