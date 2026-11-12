# SPDX-License-Identifier: GPL-3.0-only
"""Read-only integration checks against a running, isolated Forge instance."""
import os
import unittest

import requests


@unittest.skipUnless(os.environ.get("SNAPREF_FORGE_URL"), "Set SNAPREF_FORGE_URL for live Forge integration checks")
class ForgeLiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.url = os.environ["SNAPREF_FORGE_URL"].rstrip("/")
        response = requests.get(cls.url + "/config", timeout=30)
        response.raise_for_status()
        cls.config = response.json()
        cls.components = {c["id"]: c for c in cls.config["components"]}
        cls.by_elem = {c["props"].get("elem_id"): c for c in cls.config["components"] if c["props"].get("elem_id")}

    def test_native_sibling_subtab_and_no_bottom_accordion(self):
        parents = {}
        def visit(node):
            for child in node.get("children", []):
                parents[child["id"]] = node["id"]
                visit(child)
        visit(self.config["layout"])
        snapref, tabs = self.by_elem["img2img_snapref_tab"], self.by_elem["mode_img2img"]
        self.assertEqual(snapref["type"], "tabitem")
        self.assertEqual(parents[snapref["id"]], tabs["id"])
        for elem_id in ("img2img_img2img_tab", "img2img_img2img_sketch_tab", "img2img_inpaint_tab",
                        "img2img_inpaint_sketch_tab", "img2img_inpaint_upload_tab", "img2img_batch_tab"):
            native = self.by_elem[elem_id]["id"]
            self.assertEqual(parents[native], tabs["id"])
            self.assertTrue(any([native, "select"] in dep["targets"] and self.by_elem["snapref_enabled"]["id"] in dep["outputs"]
                                for dep in self.config["dependencies"]))
        self.assertNotIn("snapref_panel", self.by_elem)

    def test_subtab_uses_valid_native_transport_mode(self):
        snapref = self.by_elem["img2img_snapref_tab"]
        enter = next(d for d in self.config["dependencies"] if [snapref["id"], "select"] in d["targets"])
        enabled = self.by_elem["snapref_enabled"]["id"]
        self.assertIn(enabled, enter["outputs"])
        outputs = [self.components[index] for index in enter["outputs"]]
        self.assertGreaterEqual(sum(c["type"] == "number" for c in outputs), 2)
        self.assertTrue(any(c["type"] == "state" for c in outputs))
        self.assertTrue(any(c["type"] == "tabs" for c in outputs))

    def test_alwayson_api_discovery_and_image_inputs(self):
        response = requests.get(self.url + "/sdapi/v1/script-info", timeout=30)
        response.raise_for_status()
        script = next(s for s in response.json() if s["name"] == "snapref-webui")
        self.assertTrue(script["is_alwayson"])
        self.assertTrue(script["is_img2img"])
        self.assertEqual(len(script["args"]), 20)
        self.assertEqual(script["args"][2]["value"], "cs")
        self.assertEqual(self.by_elem["snapref_reference_image"]["type"], "image")
