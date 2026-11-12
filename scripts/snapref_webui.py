# SPDX-License-Identifier: GPL-3.0-only
"""Native Forge Neo always-on img2img extension."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gradio as gr
from modules import script_callbacks, scripts

from snapref_webui import runtime
from snapref_webui.i18n import CURRENT, SnapRefError, detect, tr
from snapref_webui.model_detection import detect_engine
from snapref_webui.model_manager import scan
from snapref_webui.settings import Config, save_preferences

_components = {}
_main_pending = False
_last_number = None


def capture(component, **kwargs):
    global _main_pending, _last_number
    elem_id = getattr(component, "elem_id", None)
    if isinstance(component, gr.Number):
        _last_number = component
        if getattr(component, "parent", None) is _components.get("img_tabs"):
            _components["mode_index"] = component
    if elem_id == "mode_img2img":
        _components["img_tabs"] = component
        _components["native_tabs"] = []
    if elem_id in {"img2img_img2img_tab", "img2img_img2img_sketch_tab", "img2img_inpaint_tab",
                    "img2img_inpaint_sketch_tab", "img2img_inpaint_upload_tab", "img2img_batch_tab"}:
        _components.setdefault("native_tabs", []).append(component)
    if elem_id == "img2img_tabs_resize":
        _components["resize_tabs"] = component
        _components["scale_index"] = _last_number
    if elem_id == "img2img_tab_resize_by":
        _components["resize_by_tab"] = component
    if elem_id in ("resize_mode", "img2img_denoising_strength"):
        _components[elem_id] = component
    if elem_id == "setting_sd_model_checkpoint":
        _components["checkpoint"] = component
    if elem_id == "img2img_image":
        _main_pending = True
    if _main_pending and "logical_image_background" in (getattr(component, "elem_classes", None) or []):
        _components["main"] = component
        _main_pending = False


def compatibility():
    from modules import sd_models, shared
    from safetensors import safe_open
    engine = sd_models.model_data.sd_model
    # A newly selected checkpoint is loaded only when Generate is pressed. Inspect its
    # tensor structure for UI availability; actual engine isinstance is mandatory later.
    selected = sd_models.get_closet_checkpoint_match(shared.opts.sd_model_checkpoint)
    if selected and str(getattr(engine, "filename", "")) != str(selected.filename):
        try:
            with safe_open(selected.filename, framework="pt", device="cpu") as handle:
                keys = set(handle.keys())
                prefix = next((p for p in ("net.", "diffusion_model.", "") if p + "blocks.0.self_attn.q_proj.weight" in keys and p + "t_embedder.1.linear_2.weight" in keys), None)
                if prefix is None:
                    raise SnapRefError("anima_only")
                count = max(int(k[len(prefix):].split(".")[1]) for k in keys if k.startswith(prefix + "blocks.")) + 1
                from snapref_webui.model_detection import block_map
                block_map(range(28), count)
                return count
        except SnapRefError:
            raise
        except Exception as exc:
            raise SnapRefError("anima_only") from exc
    return len(detect_engine(engine).blocks)


def model_status(language):
    token = CURRENT.set(language)
    try:
        try:
            blocks = compatibility()
            available, message = True, tr("ready", blocks=blocks)
        except SnapRefError as exc:
            available, message = False, str(exc)
        found, invalid = scan()
        message += "\n\n" + (tr("adapter_status", path=", ".join(str(p) for p in found)) if found else tr("adapter_missing"))
        message += "\n\n" + "\n".join(reason for _, reason in invalid)
        return message, gr.update(interactive=available, **({} if available else {"value": False}))
    finally:
        CURRENT.reset(token)


class SnapRefScript(scripts.Script):
    sorting_priority = 530
    create_group = False

    def title(self):
        return tr("plugin_name", "en")

    def show(self, is_img2img):
        return scripts.AlwaysVisible if is_img2img else False

    def ui(self, is_img2img):
        language = detect()
        labels = []
        def register(component, key, info=None, choices=None, prop="label"):
            labels.append((component, key, info, choices, prop))
            return component
        def text(key):
            return tr(key, language)
        def choice(keys):
            return [(text(key), key) for key in keys]
        required = {"img_tabs", "mode_index", "main", "resize_tabs", "scale_index"}
        if not required <= _components.keys():
            raise SnapRefError("ui_entry_missing")
        with _components["img_tabs"], gr.TabItem(text("tab_title"), id="snapref", elem_id="img2img_snapref_tab") as panel:
            register(panel, "tab_title")
            register(gr.Markdown("### " + text("title")), "title", prop="value")
            register(gr.Markdown(text("intro")), "intro", prop="value")
            lang = register(gr.Dropdown(choices=[("简体中文", "zh-CN"), ("日本語", "ja-JP"), ("English", "en")], value=language, label=text("language")), "language")
            state = gr.Markdown("")
            enabled = register(gr.Checkbox(False, label=text("enable"), elem_id="snapref_enabled"), "enable")
            use_main = register(gr.Checkbox(True, label=text("use_main")), "use_main")
            preview = register(gr.Image(type="pil", sources=["upload", "clipboard"], label=text("preview"), interactive=True,
                                        height=320, elem_id="snapref_reference_image"), "preview")
            tab_active = gr.State(False)
            previous_scale = gr.State(0)
            mode = register(gr.Radio(choice(["cs", "char"]), value="cs", label=text("mode")), "mode", choices=["cs", "char"])
            with gr.Row():
                strength = register(gr.Slider(0, 2, value=1, step=.05, min_width=240, label=text("strength"), info=text("strength_info")), "strength", "strength_info")
                fidelity = register(gr.Slider(0, 1, value=1, step=.05, min_width=240, label=text("fidelity"), info=text("fidelity_info")), "fidelity", "fidelity_info")
            resolution = register(gr.Slider(.25, 2, value=1, step=.05, label=text("resolution"), info=text("resolution_info")), "resolution", "resolution_info")
            register(gr.Markdown(text("license_notice")), "license_notice", prop="value")
            with gr.Accordion(text("extras"), open=False) as extra_panel:
                register(extra_panel, "extras")
                gallery = register(gr.Gallery(type="pil", columns=4, rows=2, height=300, label=text("extras"), allow_preview=True, object_fit="contain"), "extras")
                selected = gr.State(-1)
                upload = register(gr.Image(type="pil", sources=["upload", "clipboard"], label=text("upload"), height=200), "upload")
                with gr.Row():
                    add, replace, delete, left, right = [register(gr.Button(text(key)), key, prop="value") for key in ("add", "replace", "delete", "left", "right")]
            with gr.Accordion(text("advanced"), open=False) as advanced:
                register(advanced, "advanced")
                layout = register(gr.Radio(choice(["separate", "stitch"]), value="separate", label=text("layout"), info=text("layout_info")), "layout", "layout_info", ["separate", "stitch"])
                weights = register(gr.Textbox(value=Config.weights, label=text("weights"), info=text("weights_info")), "weights", "weights_info")
                in_uncond = register(gr.Checkbox(False, label=text("in_uncond")), "in_uncond")
                guidance_on = register(gr.Checkbox(False, label=text("guidance_on")), "guidance_on")
                guidance = register(gr.Slider(0, 15, value=4.5, step=.1, label=text("guidance")), "guidance")
                correction = register(gr.Radio(choice(["apg", "rescale", "none"]), value="apg", label=text("correction")), "correction", choices=["apg", "rescale", "none"])
                guidance_end = register(gr.Slider(0, 1, value=1, step=.01, label=text("guidance_end"), info=text("guidance_info")), "guidance_end", "guidance_info")
                caption = register(gr.Textbox(value="", lines=2, label=text("caption")), "caption")
                log_bias = register(gr.Slider(-3, 3, value=0, step=.1, label=text("log_bias"), info=text("log_bias_info")), "log_bias", "log_bias_info")
                precision = register(gr.Radio(["bf16", "fp8"], value="bf16", label=text("precision")), "precision")
                with gr.Row():
                    clear_cache = register(gr.Button(text("clear_cache")), "clear_cache", prop="value")
                    rescan = register(gr.Button(text("rescan")), "rescan", prop="value")

        def on_select(evt: gr.SelectData):
            return evt.index
        gallery.select(on_select, outputs=[selected], queue=False)
        def edit(items, index, image, operation, use_current):
            from snapref_webui.reference import image_from_input
            items = list(items or [])
            if operation == "add" and image is not None:
                if len(items) >= (7 if use_current else 8):
                    raise gr.Error(text("max_references"))
                items.append((image_from_input(image), None))
                index = len(items) - 1
            elif 0 <= index < len(items):
                if operation == "replace" and image is not None:
                    items[index] = (image_from_input(image), None)
                elif operation == "delete":
                    items.pop(index)
                    index = min(index, len(items) - 1)
                elif operation in ("left", "right"):
                    target = max(0, min(len(items) - 1, index + (-1 if operation == "left" else 1)))
                    items[index], items[target] = items[target], items[index]
                    index = target
            return items, index, None
        for button, operation in ((add, "add"), (replace, "replace"), (delete, "delete"), (left, "left"), (right, "right")):
            button.click(lambda items, index, image, current, op=operation: edit(items, index, image, op, current),
                         inputs=[gallery, selected, upload, use_main], outputs=[gallery, selected, upload], queue=False)
        # Main canvas is read-only from this extension. A local upload becomes an
        # independent primary reference, without replacing the user's img2img image.
        _components["main"].change(lambda image, current: image if current else gr.skip(),
                                    inputs=[_components["main"], use_main], outputs=[preview], queue=False)
        preview.input(lambda _image: False, inputs=[preview], outputs=[use_main], queue=False)
        use_main.change(lambda current, image: image if current else gr.skip(),
                        inputs=[use_main, _components["main"]], outputs=[preview], queue=False)

        native_only = [_components[key] for key in ("resize_mode", "img2img_denoising_strength", "resize_by_tab") if key in _components]
        def enter_tab(value, current, main_image, own_image, scale, previous, active):
            message, update = model_status(value)
            return [gr.update(value=update.get("interactive", False), interactive=update.get("interactive", False)), True, 0,
                    gr.update(selected="to"), 0, main_image if current else own_image, message,
                    previous if active else scale,
                    *[gr.update(visible=False) for _ in native_only]]
        panel.select(enter_tab, inputs=[lang, use_main, _components["main"], preview, _components["scale_index"], previous_scale, tab_active],
                     outputs=[enabled, tab_active, _components["mode_index"], _components["resize_tabs"],
                              _components["scale_index"], preview, state, previous_scale, *native_only], queue=False)
        def leave_tab(previous, active):
            if not active:
                return [False, False, gr.skip(), gr.skip(), *[gr.skip() for _ in native_only]]
            return [False, False, gr.update(selected="by" if previous else "to"), previous,
                    *[gr.update(visible=True) for _ in native_only]]
        for native_tab in _components["native_tabs"]:
            native_tab.select(leave_tab, inputs=[previous_scale, tab_active],
                              outputs=[enabled, tab_active, _components["resize_tabs"], _components["scale_index"], *native_only], queue=False)

        def switch_language(value):
            nonlocal language
            language = value
            save_preferences(language=value)
            updates = []
            for component, key, info, choices, prop in labels:
                kw = {prop: tr(key, value)}
                if key == "title" and prop == "value":
                    kw[prop] = "### " + kw[prop]
                if info:
                    kw["info"] = tr(info, value)
                if choices:
                    kw["choices"] = [(tr(key, value), key) for key in choices]
                updates.append(gr.update(**kw))
            return updates + list(model_status(value))
        lang.change(switch_language, inputs=[lang], outputs=[item[0] for item in labels] + [state, enabled], queue=False)
        rescan.click(model_status, inputs=[lang], outputs=[state, enabled], queue=False)
        def clear_cache_fn(value):
            runtime.clear()
            return tr("cache_cleared", value)
        clear_cache.click(clear_cache_fn, inputs=[lang], outputs=[state], queue=False)
        def enabled_changed(value):
            if not value:
                runtime.clear()
        enabled.change(enabled_changed, inputs=[enabled], queue=False, show_progress="hidden")
        if "checkpoint" in _components:
            _components["checkpoint"].change(lambda _checkpoint, value: model_status(value), inputs=[_components["checkpoint"], lang], outputs=[state, enabled], queue=False)
        # Gradio root load handles initial selected checkpoint before model is loaded.
        from gradio.context import Context
        Context.root_block.load(model_status, inputs=[lang], outputs=[state, enabled], queue=False)

        controls = [enabled, use_main, mode, strength, fidelity, resolution, layout, weights, in_uncond,
                    guidance_on, guidance, correction, guidance_end, caption, log_bias, precision, lang, gallery, preview, tab_active]
        def restore_field(key):
            def get_value(params):
                try:
                    return json.loads(params.get("SnapRef Config", "{}"))[key]
                except (ValueError, KeyError, TypeError):
                    return gr.skip()
            return get_value
        self.infotext_fields = [(component, restore_field(key)) for component, key in zip(controls, runtime.CONFIG_FIELDS)]
        self.infotext_fields.append((_components["img_tabs"], lambda params:
            gr.update(selected="snapref") if params.get("SnapRef") in (True, "True") else gr.skip()))
        self.paste_field_names = ["SnapRef", "SnapRef Config", "SnapRef Adapter"]
        return controls

    def process(self, p, *values):
        runtime.bind_request(p, values)


def pasted(infotext, params):
    if params.get("SnapRef"):
        gr.Info(tr("reload_images", detect()))


script_callbacks.on_after_component(capture)
script_callbacks.on_model_loaded(lambda model: runtime.clear())
script_callbacks.on_script_unloaded(runtime.clear)
script_callbacks.on_infotext_pasted(pasted)
