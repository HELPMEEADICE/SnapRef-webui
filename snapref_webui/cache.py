# SPDX-License-Identifier: GPL-3.0-only
import hashlib
import json
import threading


def digest(parts):
    h = hashlib.sha256()
    for part in parts:
        data = part if isinstance(part, bytes) else json.dumps(part, sort_keys=True, default=str).encode()
        h.update(len(data).to_bytes(8, "little"))
        h.update(data)
    return h.hexdigest()


class ReferenceCache:
    """One CPU entry with a byte budget. Never retain a GPU tensor across requests."""
    def __init__(self, max_bytes=2 * 1024**3):
        self.max_bytes = max_bytes
        self.lock = threading.RLock()
        self.clear()
        self.hits = self.misses = 0

    def clear(self):
        with self.lock:
            self.key = self.value = None
            self.bytes = 0

    def get(self, key):
        with self.lock:
            if key != self.key:
                self.misses += 1
                return None
            self.hits += 1
            return self.value

    def put(self, key, value):
        def cpu(obj):
            if isinstance(obj, tuple):
                return tuple(cpu(x) for x in obj)
            if isinstance(obj, list):
                return [cpu(x) for x in obj]
            return obj.detach().cpu().clone() if hasattr(obj, "detach") else obj
        def size(obj):
            return sum(size(x) for x in obj) if isinstance(obj, (list, tuple)) else obj.numel() * obj.element_size() if hasattr(obj, "numel") else 0
        amount = size(value)
        with self.lock:
            self.clear()
            if amount <= self.max_bytes:
                self.key, self.value, self.bytes = key, cpu(value), amount


CACHE = ReferenceCache()
