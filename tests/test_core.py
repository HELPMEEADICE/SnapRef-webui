# SPDX-License-Identifier: GPL-3.0-only
import json
import math
import string
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import save_file

from snapref_webui.adapter import Adapter, inspect_file
from snapref_webui.attention import masked_attention, merge_lse, weighted_attention
from snapref_webui.cache import ReferenceCache, digest
from snapref_webui.conditioning import correct, row_biases
from snapref_webui.forge_bridge import ForwardState, hooks
from snapref_webui.i18n import CATALOGS, CURRENT, SnapRefError, normalize, tr
from snapref_webui.model_detection import BASE_BLOCKS_29B, block_map
from snapref_webui.reference import resized, select_references
from snapref_webui.settings import Config


def checkpoint(path, fp8=False, bad_scale=False):
    weights = {"anima_ref_kv.mode_embed": torch.zeros(2, 4)}
    for i in range(28):
        for stream, value in (("ref", 1.), ("tgt", 2.)):
            root = f"anima_ref_kv.{stream}.blocks.{i}.self_attn.q_proj."
            for name, tensor in (("lora_down", torch.ones(2, 4)), ("lora_up", torch.full((4, 2), value))):
                if fp8:
                    weights[root + name] = tensor.to(torch.float8_e4m3fn)
                    weights[root + name + "_scale"] = torch.ones((1 if bad_scale else tensor.shape[0], 1), dtype=torch.float32)
                else:
                    weights[root + name] = tensor.to(torch.bfloat16)
            weights[root + "alpha"] = torch.tensor(2.)
    save_file(weights, str(path), metadata={"snapref": json.dumps({"format": "anima_ref_kv/1", "modes": ["cs", "char"], "ref_t": 10,
        "lora": {stream: {"rank": 2, "targets": ["self_attn.q_proj"]} for stream in ("ref", "tgt")}})})


class TinyAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = torch.nn.Linear(4, 4, bias=False)
        torch.nn.init.zeros_(self.q_proj.weight)
        self.output_proj = torch.nn.Identity()
        self.output_dropout = torch.nn.Identity()

    def compute_qkv(self, x, context=None, rope_emb=None):
        return self.q_proj(x).reshape(*x.shape[:-1], 1, 4), x[..., None, :], x[..., None, :]

    @staticmethod
    def torch_attention_op(q, k, v, transformer_options=None):
        return F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2).flatten(2)

    def compute_attention(self, q, k, v, transformer_options=None):
        return self.torch_attention_op(q, k, v)

    def forward(self, x, context=None, rope_emb=None, transformer_options=None):
        return self.compute_attention(*self.compute_qkv(x, context, rope_emb))


class TinyModel(torch.nn.Module):
    def __init__(self, blocks=28):
        super().__init__()
        self.blocks = torch.nn.ModuleList()
        for _ in range(blocks):
            b = torch.nn.Module()
            b.x_dim, b.self_attn = 4, TinyAttention()
            self.blocks.append(b)


class AdapterTests(unittest.TestCase):
    def test_bf16_fp8(self):
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory) / "a.safetensors", Path(directory) / "b.safetensors"
            checkpoint(a)
            checkpoint(b, fp8=True)
            aa, bb = Adapter(a), Adapter(b)
            for name in aa.weights:
                for stream in ("ref", "tgt"):
                    for lhs, rhs in zip(aa.weights[name][stream][:2], bb.weights[name][stream][:2]):
                        torch.testing.assert_close(lhs, rhs)

    def test_bad_scale_and_corrupt(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "a.safetensors"
            checkpoint(path, fp8=True, bad_scale=True)
            with self.assertRaises(SnapRefError):
                Adapter(path)
            path.write_bytes(b"<html>not weights</html>")
            with self.assertRaises(SnapRefError):
                inspect_file(path)

    def test_incomplete_stream(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "a.safetensors"
            save_file({"x": torch.ones(1)}, str(path), metadata={"snapref": '{"format":"anima_ref_kv/1"}'})
            with self.assertRaises(SnapRefError):
                inspect_file(path)

    def test_layouts(self):
        self.assertEqual(block_map(range(28), 28), tuple(range(28)))
        expected = tuple(i for i in range(40) if i not in (2, 5, 8, 11, 14, 17, 21, 24, 27, 30, 33, 36))
        self.assertEqual(block_map(range(28), 40), expected)
        self.assertEqual(len(set(BASE_BLOCKS_29B)), 28)
        for indices, count in ((range(28), 52), (range(27), 28), (range(40), 40)):
            with self.assertRaises(SnapRefError):
                block_map(indices, count)

    def test_streams_mapping_and_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "a.safetensors"
            checkpoint(path)
            ad = Adapter(path)
            for count in (28, 40):
                dm, state = TinyModel(count), ForwardState()
                x = torch.ones(1, 2, 4)
                for repeat in range(2):
                    with hooks(dm, ad, state, Config(), {}):
                        state.stream = "ref"
                        torch.testing.assert_close(dm.blocks[0].self_attn.q_proj(x), torch.full_like(x, 8.))
                        state.stream = "tgt"
                        torch.testing.assert_close(dm.blocks[0].self_attn.q_proj(x), torch.full_like(x, 16.))
                        if count == 40:
                            torch.testing.assert_close(dm.blocks[2].self_attn.q_proj(x), torch.zeros_like(x))
                    self.assertEqual(len(dm.blocks[0].self_attn.q_proj._forward_hooks), 0)
                    self.assertNotIn("forward", dm.blocks[0].self_attn.__dict__)
                    torch.testing.assert_close(dm.blocks[0].self_attn.q_proj(x), torch.zeros_like(x))
                with self.assertRaisesRegex(RuntimeError, "injected"):
                    with hooks(dm, ad, state, Config(), {}):
                        raise RuntimeError("injected")
                self.assertEqual(len(dm.blocks[0].self_attn.q_proj._forward_hooks), 0)
                with hooks(dm, ad, state, Config(), {}):
                    with self.assertRaises(SnapRefError):
                        with hooks(dm, ad, state, Config(), {}):
                            self.fail("Reentrant installation accepted")

    def test_record_then_target_reads_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "a.safetensors"
            checkpoint(path)
            dm, state = TinyModel(), ForwardState()
            state.weights = [1.]
            x = torch.ones(1, 2, 4)
            with hooks(dm, Adapter(path), state, Config(), {}):
                state.stream, state.record = "ref", [None] * 28
                dm.blocks[0].self_attn(3 * x)
                kv = state.record[0]
                torch.testing.assert_close(kv[0], (3 * x)[..., None, :])
                state.record, state.stream = None, "tgt"
                state.kv, state.sizes, state.rows = [kv] * 28, [2], torch.zeros(1)
                out = dm.blocks[0].self_attn(x)
                self.assertTrue((out > 1).all())


class MathTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)

    def test_weighted_sdpa_matches_explicit_softmax(self):
        q, k, v = [torch.randn(2, n, 2, 4) for n in (3, 4, 4)]
        refs = (torch.randn(1, 5, 2, 4), torch.randn(1, 5, 2, 4))
        rows = torch.tensor([0., math.log(.3)])
        out = masked_attention(q, k, v, refs, [2, 3], [math.log(.7), math.log(1.5)], rows)
        kk = torch.cat((refs[0].expand(2, -1, -1, -1), k), 1).transpose(1, 2)
        vv = torch.cat((refs[1].expand(2, -1, -1, -1), v), 1).transpose(1, 2)
        bias = torch.tensor([math.log(.7)] * 2 + [math.log(1.5)] * 3 + [0.] * 4).expand(2, -1).clone()
        bias[:, :5] += rows[:, None]
        scores = (q.transpose(1, 2) @ kk.transpose(-2, -1)) / 2 + bias[:, None, None]
        expected = (scores.softmax(-1) @ vv).transpose(1, 2).flatten(2)
        torch.testing.assert_close(out, expected)

    def test_lse_merge_matches_concat(self):
        q = torch.randn(1, 2, 3, 4)
        parts, keys, values = [], [], []
        for n, bias in ((2, 0.), (3, .7), (1, -1.2)):
            k, v = torch.randn(1, 2, n, 4), torch.randn(1, 2, n, 4)
            logits = q @ k.transpose(-2, -1) / 2
            parts.append((bias, logits.softmax(-1) @ v, logits.logsumexp(-1, keepdim=True)))
            keys.append(logits + bias)
            values.append(v)
        expected = torch.cat(keys, -1).softmax(-1) @ torch.cat(values, -2)
        torch.testing.assert_close(merge_lse(parts), expected)

    def test_cfg_rows_and_fidelity_cpu(self):
        cfg = Config(fidelity=.75)
        rows = row_biases({"cond_or_uncond": [1, 0]}, 4, cfg, "cpu")
        torch.testing.assert_close(rows, torch.tensor([math.log(.25)] * 2 + [0.] * 2))
        q, k, v = (torch.randn(2, 3, 1, 4) for _ in range(3))
        refs = (torch.randn(1, 2, 1, 4), torch.randn(1, 2, 1, 4))
        out = weighted_attention(TinyAttention(), q, k, v, refs, [2], [0.], torch.tensor([0., -float("inf")]), {})
        plain = TinyAttention.torch_attention_op(q[1:], k[1:], v[1:])
        torch.testing.assert_close(out[1:], plain)

    def test_guidance_corrections_finite(self):
        for mode in ("apg", "rescale", "none"):
            self.assertTrue(torch.isfinite(correct(torch.zeros(1, 4, 2, 2), torch.zeros(1, 4, 2, 2), mode)).all())

    def test_cache_identity_budget_and_cpu(self):
        cache = ReferenceCache(100)
        a = digest(["model", b"pixels", "caption", (32, 32)])
        b = digest(["model", b"pixels", "caption", (32, 48)])
        cache.put(a, [torch.ones(2)])
        self.assertEqual(cache.get(a)[0].device.type, "cpu")
        self.assertIsNone(cache.get(b))
        cache.put(b, [torch.ones(101)])
        self.assertIsNone(cache.get(a))
        cache.clear()
        self.assertEqual(cache.bytes, 0)


class InputsAndTranslationTests(unittest.TestCase):
    def test_keys_placeholders_and_language_switch(self):
        expected = set(CATALOGS["en"])
        formatter = string.Formatter()
        for lang, catalog in CATALOGS.items():
            self.assertEqual(set(catalog), expected)
            for key, value in catalog.items():
                extract = lambda s: {name for _, name, _, _ in formatter.parse(s) if name is not None}
                self.assertEqual(extract(value), extract(CATALOGS["en"][key]))
            token = CURRENT.set(lang)
            self.assertEqual(tr("title"), catalog["title"])
            CURRENT.reset(token)
        self.assertEqual([normalize(x) for x in ("zh_Hans", "ja", "fr")], ["zh-CN", "ja-JP", "en"])
        with self.assertRaises(KeyError):
            tr("missing_translation")

    def test_reference_validation_and_zero_skip(self):
        image = Image.new("RGB", (64, 96), "red")
        refs = select_references(image, [(image, None), (None, None)], Config(weights="1,0,0"))
        self.assertEqual(len(refs), 1)
        with self.assertRaises(SnapRefError):
            select_references(image, [], Config(weights="nan"))
        with self.assertRaises(SnapRefError):
            select_references(image, [image] * 8, Config())
        sheet = select_references(image, [image], Config(layout="stitch"))[0][0]
        self.assertEqual(sheet.size, (128, 96))
        self.assertEqual(resized(image, 1).size, (64, 96))
        for kw in ({"fidelity": float("nan")}, {"mode": "style"}, {"precision": "int8"}):
            with self.assertRaises(SnapRefError):
                replace(Config(), **kw).validate()

    def test_non_anima_blocked(self):
        # Type checks must precede any access to model internals.
        from snapref_webui.model_detection import detect_engine
        class HostEngine: ...
        fake_engine = SimpleNamespace(Anima=HostEngine)
        fake_transformer = SimpleNamespace(Anima=TinyModel)
        with patch.dict("sys.modules", {"backend.diffusion_engine.anima": fake_engine, "backend.nn.anima": fake_transformer}):
            with self.assertRaises(SnapRefError):
                detect_engine(SimpleNamespace())


class DownloadTests(unittest.TestCase):
    def test_cancel_and_corrupt_transfer(self):
        from snapref_webui.model_manager import download
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "models"
            with self.assertRaises(SnapRefError) as caught:
                download(root, "bf16", cancelled=lambda: True, attempts=1)
            self.assertEqual(caught.exception.key, "cancelled")
            source = Path(directory) / "bad.safetensors"
            source.write_bytes(b"<html>error</html>")
            info = SimpleNamespace(sha="revision", siblings=[SimpleNamespace(rfilename="snapref_v2a_pre2.safetensors", size=source.stat().st_size, lfs={"sha256": "0" * 64})])
            with patch("huggingface_hub.HfApi") as api, patch("huggingface_hub.hf_hub_download", return_value=str(source)):
                api.return_value.model_info.return_value = info
                with self.assertRaises(SnapRefError):
                    download(root, "bf16", attempts=1)
            self.assertFalse(list(root.glob("*.safetensors")))
            self.assertFalse(list(root.glob("*.part")))

    def test_failure_no_final_file_or_partial(self):
        from snapref_webui.model_manager import download
        with tempfile.TemporaryDirectory() as directory, patch("huggingface_hub.HfApi") as api:
            api.return_value.model_info.side_effect = TimeoutError("injected timeout")
            with self.assertRaises(SnapRefError) as caught:
                download(Path(directory), "bf16", attempts=1)
            self.assertEqual(caught.exception.key, "download_failed")
            self.assertFalse(list(Path(directory).glob("*.part")))
            self.assertFalse(list(Path(directory).glob("*.safetensors")))

    def test_success_and_reuse(self):
        from snapref_webui.model_manager import download
        with tempfile.TemporaryDirectory() as directory:
            source, root = Path(directory) / "source.safetensors", Path(directory) / "models"
            checkpoint(source)
            info = SimpleNamespace(sha="fixed-revision", siblings=[SimpleNamespace(rfilename="snapref_v2a_pre2.safetensors", size=source.stat().st_size, lfs=None)])
            with patch("huggingface_hub.HfApi") as api, patch("huggingface_hub.hf_hub_download", return_value=str(source)) as transfer:
                api.return_value.model_info.return_value = info
                dest = download(root, "bf16")
                self.assertEqual(dest.read_bytes(), source.read_bytes())
                self.assertEqual(download(root, "bf16"), dest)
                self.assertEqual(transfer.call_count, 1)


class RequestLifecycleTests(unittest.TestCase):
    class Processing:
        def init(self, *args):
            return "native_init"
        def sample(self, *args):
            return "native_sample"
        def close(self):
            self.closed = True

    def test_disabled_does_not_patch_processing(self):
        from snapref_webui.runtime import bind_request, CONFIG_FIELDS
        p = self.Processing()
        before = dict(p.__dict__)
        bind_request(p, [False])
        self.assertEqual(p.__dict__, before)
        self.assertEqual(p.init(), "native_init")

    def test_error_is_not_swallowed_and_close_restores(self):
        from dataclasses import asdict
        from snapref_webui.runtime import bind_request
        p = self.Processing()
        values = list(asdict(Config(enabled=True)).values()) + [[]]
        bind_request(p, values)
        with patch.dict("sys.modules", {"modules": SimpleNamespace(sd_samplers=SimpleNamespace())}), \
             patch("snapref_webui.runtime.detect_engine", side_effect=SnapRefError("anima_only")):
            p.sd_model = SimpleNamespace()
            with self.assertRaises(SnapRefError):
                p.init([], [], [])
        p.close()
        self.assertTrue(p.closed)
        self.assertNotIn("init", p.__dict__)
        self.assertNotIn("sample", p.__dict__)
        self.assertNotIn("close", p.__dict__)
        self.assertEqual(p.init(), "native_init")

    def test_rebinding_does_not_stack_wrappers(self):
        from dataclasses import asdict
        from snapref_webui.runtime import bind_request
        p = self.Processing()
        values = list(asdict(Config(enabled=True)).values()) + [[]]
        bind_request(p, values)
        bind_request(p, values)
        p.close()
        self.assertNotIn("close", p.__dict__)
        self.assertNotIn("_snapref_restore", p.__dict__)

    def test_disabled_reference_tab_cannot_fall_back_to_img2img(self):
        from dataclasses import asdict
        from snapref_webui.runtime import bind_request
        p = self.Processing()
        p.sd_model = SimpleNamespace()
        values = list(asdict(Config(enabled=False)).values()) + [[], None, True]
        bind_request(p, values)
        with patch.dict("sys.modules", {"modules": SimpleNamespace(sd_samplers=SimpleNamespace())}), \
             patch("snapref_webui.runtime.detect_engine"):
            with self.assertRaises(SnapRefError) as caught:
                p.init([], [], [])
            self.assertEqual(caught.exception.key, "reference_tab_disabled")
        p.close()


if __name__ == "__main__":
    unittest.main()
