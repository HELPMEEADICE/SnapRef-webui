# SPDX-License-Identifier: GPL-3.0-only
"""Request-local translations; canonical parameter values never depend on language."""
import json
import locale
from contextvars import ContextVar
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LANGUAGES = ("zh-CN", "ja-JP", "en")
CATALOGS = {lang: json.loads((ROOT / "locales" / f"{lang}.json").read_text(encoding="utf-8")) for lang in LANGUAGES}
CURRENT = ContextVar("snapref_language", default="en")


def normalize(value):
    value = str(value or "").lower().replace("_", "-")
    return "zh-CN" if value.startswith("zh") else "ja-JP" if value.startswith("ja") else "en"


def detect():
    from .settings import read_preferences
    selected = read_preferences().get("language", "auto")
    if selected in LANGUAGES:
        return selected
    try:
        from modules import shared
        value = shared.opts.localization
        if value and value not in ("None", "Automatic"):
            return normalize(value)
    except (ImportError, AttributeError):
        return normalize(locale.getlocale()[0])
    return normalize(locale.getlocale()[0])


def tr(key, lang=None, **kwargs):
    return CATALOGS[lang or CURRENT.get()][key].format(**kwargs)


class SnapRefError(RuntimeError):
    def __init__(self, key, **kwargs):
        self.key, self.params = key, kwargs
        super().__init__(tr(key, **kwargs))
