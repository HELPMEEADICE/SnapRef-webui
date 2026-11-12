# SPDX-License-Identifier: GPL-3.0-only
import hashlib
import logging
import os
import shutil
import tempfile
import time
from pathlib import Path

from filelock import FileLock

from .adapter import Adapter, inspect_file
from .i18n import SnapRefError, tr

REPO = "0ppxnhximxr/snapref-v2a-pre2"
FILES = {"bf16": "snapref_v2a_pre2.safetensors", "fp8": "snapref_v2a_pre2_fp8.safetensors"}
log = logging.getLogger("snapref")
_loaded = None


def model_roots():
    from modules import paths, shared
    paths_list = [getattr(shared.cmd_opts, "lora_dir", None), Path(paths.models_path) / "Lora",
                  Path(paths.data_path) / "models" / "Lora"]
    # Forge's discovered networks also cover additional model paths exposed by extensions.
    try:
        import networks
        paths_list.extend(Path(net.filename).parent for net in networks.available_networks.values()
                          if Path(net.filename).name in FILES.values())
    except ImportError:
        log.debug("LoRA network catalogue not initialized")
    return list(dict.fromkeys(Path(p).resolve() for p in paths_list if p))


def scan(roots=None):
    found, invalid = [], []
    for root in model_roots() if roots is None else roots:
        if not root.exists():
            continue
        for path in root.rglob("*.safetensors"):
            if path.name not in FILES.values():
                continue
            try:
                inspect_file(path)
                if path.resolve() not in found:
                    found.append(path.resolve())
            except SnapRefError as exc:
                invalid.append((path, str(exc)))
    return found, invalid


def writable_root(roots):
    for root in roots:
        try:
            root.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryFile(dir=root):
                return root
        except OSError:
            log.info(tr("unwritable", path=root))
    raise SnapRefError("no_model_directory")


def download(root, precision, status=lambda text: None, cancelled=lambda: False, attempts=3):
    """HF transfer cache has resumable .incomplete files; only validated copies become final files."""
    from huggingface_hub import HfApi, hf_hub_download
    from tqdm.auto import tqdm
    filename = FILES[precision]
    root.mkdir(parents=True, exist_ok=True)
    dest = root / filename
    class Progress(tqdm):
        def update(self, n=1):
            if cancelled():
                raise SnapRefError("cancelled")
            result = super().update(n)
            status(tr("downloading", percent=round(100 * self.n / max(self.total or 1, 1), 1)))
            return result
    with FileLock(str(dest) + ".lock", timeout=600):
        if dest.exists():
            inspect_file(dest)
            return dest
        log.warning(tr("license_notice"))
        status(tr("license_notice"))
        for attempt in range(attempts):
            part = root / (filename + f".{os.getpid()}.part")
            try:
                if cancelled():
                    raise SnapRefError("cancelled")
                info = HfApi().model_info(REPO, files_metadata=True, timeout=30)
                sibling = next(s for s in info.siblings if s.rfilename == filename)
                source = Path(hf_hub_download(REPO, filename, revision=info.sha, etag_timeout=30, tqdm_class=Progress))
                if sibling.size and source.stat().st_size != sibling.size:
                    raise SnapRefError("download_integrity")
                lfs = sibling.lfs
                expected = (lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)) if lfs else None
                h = hashlib.sha256()
                with source.open("rb") as src, part.open("wb") as dst:
                    while chunk := src.read(8 * 1024**2):
                        if cancelled():
                            raise SnapRefError("cancelled")
                        h.update(chunk)
                        dst.write(chunk)
                    dst.flush()
                    os.fsync(dst.fileno())
                if expected and h.hexdigest() != expected:
                    raise SnapRefError("download_integrity")
                # Full tensor and scale verification before atomic publication.
                Adapter(part)
                os.replace(part, dest)
                status(tr("downloaded", path=dest))
                return dest
            except SnapRefError as exc:
                if exc.key == "cancelled":
                    raise
                error = exc
            except Exception as exc:
                error = exc
            finally:
                part.unlink(missing_ok=True)
            if attempt + 1 < attempts:
                status(tr("download_retry", attempt=attempt + 2, detail=error))
                time.sleep(min(2**attempt, 4))
        raise SnapRefError("download_failed", detail=str(error)) from error


def ensure_adapter(precision, status=lambda text: None, cancelled=lambda: False):
    global _loaded
    roots = model_roots()
    found, invalid = scan(roots)
    preferred = sorted(found, key=lambda p: (p.name != FILES[precision], str(p)))
    if preferred:
        path = preferred[0]
    else:
        if invalid:
            raise SnapRefError("bad_adapter", detail="; ".join(reason for _, reason in invalid))
        path = download(writable_root(roots), precision, status, cancelled)
    stat = path.stat()
    identity = (str(path), stat.st_mtime_ns, stat.st_size)
    if _loaded is None or _loaded.identity != identity:
        _loaded = Adapter(path)
    status(tr("adapter_ready", path=path))
    return _loaded


def release():
    global _loaded
    _loaded = None
