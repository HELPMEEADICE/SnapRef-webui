# SnapRef-webui

[English](README.md) · [日本語](README.ja-JP.md)

在 Forge Neo 原生 **图生图** 页提供 **人设参考生图（SnapRef）**。
主图只作为角色参考，目标从新噪声开始生成。实际执行 SnapRef v2a-pre2 的
非对称注意力、Reference / Target 双分支 LoRA、模式嵌入及逐层 K/V 缓存。

当前是经过本地实测的候选版本，不保证任意参考图和模型都能完全保留人设。
完整的已测与未测范围见 [测试报告](docs/testing.md)。

## 安装

需要 [Forge Neo](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo)、
Anima 模型及匹配的 Qwen 文本编码器、Qwen Image VAE。实测环境为 Python 3.13.11、
PyTorch 2.9.1+cu130、Gradio **4.40.0**、SageAttention 2.2、RTX 5080 Laptop 16GB。
请使用 WebUI 自带的 Python 环境。

在扩展页「从网址安装」中填写 `https://github.com/HELPMEEADICE/SnapRef-webui.git`，
安装后重启 WebUI。也可在 WebUI 根目录执行：

```sh
git clone https://github.com/HELPMEEADICE/SnapRef-webui.git extensions/SnapRef-webui
```

`install.py` 只补装缺失依赖，不替换
Torch/Gradio，不修改 WebUI 源码，启动时不下载权重。

## 最简使用路径

1. 选择 Anima base v1.0 和对应 VAE、文本编码器。
2. 在 **图生图主画布** 上传人物立绘。
3. 点击与 img2img / Sketch / Inpaint / Batch **同级的「参考生图」子选项卡**。
   兼容模型会在进入此子页时自动启用 SnapRef，不再把入口放在底部折叠区。
4. 保留 **使用当前 img2img 图片作为参考** 可复用原主图；也可以直接在新子页上传。
   直接上传会自动取消此选项，并且不会覆盖原来的 img2img 画布。
5. 输入新动作、姿势和场景；在 **Resize to** 设置输出尺寸。建议从 Euler / Simple、
   Shift 3、CFG 4.5、28 步开始。
6. 点击生成。**此模式不使用去噪强度**，参考图不会作为普通 img2img 初始图。
   无需添加 `<lora:snapref...>` 标签，也不要手动重复加载 SnapRef。

例如提示词：`1girl, solo, sitting on a bench, crossed legs, night, street`。
**角色与画风** 参考人物和绘画风格；**仅角色** 使用角色模式嵌入，让模型和提示词
对画风有更多控制。耳饰、细纹和服装等细节仍可能变化。

切回原有任意子页时自动关闭 SnapRef，恢复原生图生图与局部重绘控件。开启模式不支持蒙版、局部重绘、
高清修复、精炼模型，也不提供“额外人设参考 + 普通图生图”混合模式。

## 参数与多参考图

| 参数 | 含义与默认值 |
|---|---|
| 参考强度 | 1 为训练注意力权重；0 关闭；2 为两倍；不会缩放 LoRA 矩阵 |
| 保真度 | 默认 1；降低后，负向分支以 `1 − 保真度` 的权重读取参考 |
| 参考分辨率 | 默认 1 MP；保持长宽比，只缩小，边长对齐 16 像素 |
| 独立编码 | 各图独立编码，各层拼接 K/V，适合同一角色的视角与细节 |
| 横向拼接 | 先按第一张高度从左向右拼接，再应用总面积上限；不同角色可能身份串扰 |
| 单图权重 | 0–2，顺序为主图（如使用）及图库图片；缺省 1，0 跳过编码 |
| 对数偏置 | 默认 0，加到 `log（参考强度 × 单图权重）` 上 |
| 负向分支读取参考 | 默认关闭；开启后负向使用完整参考，忽略保真度 |
| 独立参考引导 | 默认关闭；开启时强度默认 4.5，参考进入两个分支 |
| 引导校正／结束 | APG／1.0；结束位置前每步增加一次无参考预测 |
| 参考图描述 | 默认空；可填写外观标签，不是目标场景提示词 |

总计最多 **8 张**。展开额外参考图片，拖入、粘贴或上传图片后点击添加。
选中缩略图可以替换、删除、左移或右移。权重按位置对应，图片排序后请同步调整
逗号分隔的权重。拼接模式忽略非零单图权重，但权重 0 仍会移除对应图片。

Strength 与 Fidelity 保留上游数学含义；显式开启负向参考或独立参考引导时，
保真度不生效。中日英可即时切换且记住选择，不会因此重新加载模型。初始语言优先
采用 Forge 本地化设置，再读取系统语言，其他语言回退英语。

## 权重扫描和自动下载

通过 Forge 的 `--lora-dir`、`models_path/Lora`（含 `--model-ref`、`--data-dir`）
以及已注册的匹配模型路径，递归查找：

* `snapref_v2a_pre2.safetensors`：默认优先 BF16。
* `snapref_v2a_pre2_fp8.safetensors`：可选择优先 FP8 存储。

校验 metadata、双分支及声明的全部目标矩阵。FP8 E4M3FN 按作者逐行 FP32 scale
反量化，再用模型计算类型运行；FP8 存储不等于 FP8 计算。只有一种合法格式时直接
复用；已安装文件损坏时明确报错，不静默覆盖。

仅在 **Anima 上首次启用并实际生成**、且没有合法文件时，通过
`huggingface_hub.hf_hub_download` 从 [作者官方仓库](https://huggingface.co/0ppxnhximxr/snapref-v2a-pre2)
下载。支持解析并固定版本、进度、三次重试、超时、下载锁、文件大小与 SHA256 校验、
完整张量校验及原子写入。临时发布文件会清理；HF 可能保留可续传缓存。
自定义模型目录不可写时回退到 Forge 本地模型目录，并报告实际位置。

**预训练权重为 CircleStone Labs 非商业许可证 v1.2，并非 GPL。**
生成前可看到许可证提示，使用前请阅读作者模型许可证，插件不打包权重。
网络失败只中止当前请求，不会回退普通图生图或导致 WebUI 退出。可修复现有代理，
或手动放入官方文件后重新扫描。本机代理 TLS 故障导致真实下载成功路径未验证；
已用受控传输测试失败、取消、损坏与成功原子发布路径。

## 模型和生态兼容

* **Anima base v1.0（28 层）**：真实 BF16 权重 GPU 推理通过。
* **Anima 2.9B preview（40 层）**：使用上游跳过新增层的映射，真实 GPU 生成及缓存通过。
* **相同结构的微调和 Turbo**：结构检查通过可用，但未逐一验证其人设质量。
* **Anima 3.8B（52 层）**：没有经过验证的上游 SnapRef 映射，明确拒绝。
* 非 Anima 通过真实引擎／Transformer 类型拦截；已实测加载 MOSS 后拒绝。
* 普通角色 LoRA 共用 Forge 原有加载器，已实测与 SnapRef 同开。
* Spectrum 普通模式已测；独立参考引导与 Spectrum 同开会被阻止。
* 原生 Anima Reference/Edit、ImageStitch 引用通路与 SnapRef 冲突，明确阻止，
  不修改其他扩展的全局设置。替换参考注意力的第三方 Hook 也会被拒绝。
* 批次 2、CFG=1、连续生成、取消及模型切换已测。编译模型及其他第三方块调制机制
  不声明兼容。测试详情见英文测试报告。

无偏置情况复用当前 Forge 注意力后端；加权融合局部使用 PyTorch 融合 LSE 或 SDPA，
不全局关闭 SageAttention、FlashAttention 等加速。采样器和调度保持原生行为；
本机 DPM++ 2M/Karras 的 12 步结果很暗，建议优先 Euler/Simple。

## 缓存、性能与显存

相同图片和模型条件会复用参考 VAE、文字编码和 K/V。跨请求仅保留一项、最多 2 GiB
的 **CPU K/V 缓存**，单次采样的 GPU 副本在 `finally` 释放。每张 1 MP 参考约需
0.9 GB BF16 K/V，多图独立编码会增加显存和计算量。显存不足时优先降低参考分辨率。
使用 Forge 原有权重卸载机制；没有穷举所有低显存组合。清理缓存、关闭 SnapRef、
模型加载或脚本卸载时会失效或释放相应资源。NaN/Inf 或 OOM 会终止当前请求。

512²、12 步 Euler/Simple、0.25 MP 参考实测：单图请求 9.98 秒，重复命中 4.62 秒；
采样峰值已分配显存约 6726／6195 MiB。仅代表本机设置，不是所有硬件的速度承诺。

## PNG 信息、API、测试与卸载

PNG 参数保存启用状态、实际模型、适配器版本和身份、模式、强度、保真度、参考
分辨率、单图权重及引导设置。`SnapRef Config` 不含参考图片数据，恢复后需重选图片。
标准 `/sdapi/v1/img2img` 的 `alwayson_scripts["snapref-webui"]["args"]`
与 UI 共用同一推理核心；参数顺序和示例见 [API 文档](docs/API.md)。

在扩展目录用 WebUI 的 Python 运行：

```text
python -m unittest discover -s tests -v
python tools/gpu_suite.py --reference 参考图绝对路径 --output 测试结果绝对路径
python tools/compatibility_probe.py --reference 参考图绝对路径 --output 测试结果绝对路径
```

GPU 测试需先启动隔离配置的 Forge `--api --port 7862`；测试客户端可能改变该测试实例
的模型选择，请勿指向他人正在使用的实例。完整命令与结果见 [测试报告](docs/testing.md)。
可在扩展独立仓库内 `git pull` 或通过 Forge 扩展管理页更新。卸载时先关闭 WebUI，再只删除
`extensions/SnapRef-webui`。模型文件保留，可由用户另行删除。插件不会删除共享模型、
用户配置或主程序文件。

## 授权与来源

本扩展原创代码：**GPL-3.0-only**，见 [LICENSE](LICENSE)。SnapRef 衍生算法代码
保留 Apache-2.0 来源声明、[NOTICE](NOTICE) 和 [许可证全文](licenses/Apache-2.0.txt)。
Forge 主程序为外部 **AGPL-3.0** 宿主，本仓库不重新授权 Forge，合并发行仍需遵守
其 AGPL 条款。预训练权重保持独立的非商业许可证。源码研究记录与边界分析见
[架构文档](docs/architecture.md)。
