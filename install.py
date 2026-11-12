# SPDX-License-Identifier: GPL-3.0-only
"""Only install missing dependencies; never download weights during WebUI startup."""
import importlib.util
import launch

for module, package in (("huggingface_hub", "huggingface_hub>=0.24,<2"),
                        ("safetensors", "safetensors>=0.4.5"), ("filelock", "filelock>=3.12")):
    if importlib.util.find_spec(module) is None:
        launch.run_pip(f'install "{package}"', "SnapRef-webui dependency")
