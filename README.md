# AgnesAgent

> 免费全能 AI 桌面助手 · 对话问答 / 联网搜索 / 代码执行 / 多模态理解 / 图像生成 / 语音交互，一个窗口全搞定

AgnesAgent 是一款基于 **PySide6 (Qt for Python)** 构建的 Windows 桌面 AI 助手。它把在线大模型 API、免费零 Key 通道、本地 GGUF 模型、离线语音识别与合成、OCR、文档解析、联网搜索整合进同一个原生桌面应用，无需浏览器、无需 Docker，开箱即用。

---

## 功能特性

- **多 Provider 自动兜底**：主 API 不可用时自动切换备用通道（智谱 / 百度 / Ollama 本地 / g4f 零 Key 通道 / freellmapi 网关）
- **本地模型直连**：内置 llama.cpp 推理引擎，加载 GGUF 权重即可完全离线对话，不依赖外部 Ollama 服务
- **多模态**：图像理解（mmproj 视觉投影）、图像生成、文档图片 OCR
- **视频生成（预留接口）**：文生视频入口已预留三级链路（AgnesAI 视频 API → HuggingFace 免费推理 → 本地 ComfyUI + Wan2.1），默认未启用——本地显存/内存不足以运行本地视频模型，可直接接入 MiniMax H3 等云端视频模型，接口留有足够扩展余地
- **联网搜索卡片化**：网页搜索结果与图片缩略图直接渲染在聊天区，不再是纯文本
- **语音条**：TTS 生成后内嵌聊天气泡语音条，点击即播
- **离线语音能力**：SenseVoice 语音识别（ASR）+ VITS 中文语音合成（TTS），断网可用
- **文档解析**：直接读取 PDF / Word / Excel 内容进行问答（默认关闭，开启方式见「配置说明」）
- **Fluent 风格 UI**：基于 QFluentWidgets 的 Win11 Fluent Design 界面
- **代码执行**：聊天内直接运行 Python 代码并回显结果

## 快速开始

### 环境要求

- Windows 10/11
- Python 3.10+（推荐 3.12）
- 可选：NVIDIA GPU（本地模型推理可加速）

### 安装

```bash
git clone https://github.com/你的用户名/AgnesAgent.git
cd AgnesAgent
pip install -r requirements.txt
```

### 配置

1. 复制 `config.example.json` 为 `config.json`
2. 填入你申请的 API Key（AgnesAI / NVIDIA NIM 等，均可免费申请）
3. 若使用本地模型，参考下文"本地模型"下载权重放入 `dl/` 目录

```bash
cp config.example.json config.json
```

### 运行

```bash
python main.py
```

### 打包为 exe（可选）

```bash
pip install pyinstaller
pyinstaller AgnesAgent.spec
```

打包产物位于 `dist/AgnesAgent/`，模型与引擎会自动随包收集。

## 本地模型

### 为什么要保留 dl/ 本地模型库

无论国内还是国外，AI 本质上都是被主权政府与资本双重控制的实体。很多人沾沾自喜地以为"付了钱东西就是你的"，实际上每一家科技公司的算力与商业地位，都建立在所在主权政府的政策支持之上。在线模型算力固然好，但科技永远是一个巨大的泡沫——你训练进去的素材，会成为别人的创意素材；你为模型贡献的数据，不会被归还；这不是预言，而是已经在发生的事实。

在线 AI 还会因各种原因被限制：地域、合规、内容审查、成本、封号。对普通用户来说这未必是坏事，但对设计者、创作者个人而言，影响非常巨大——你的创作工具、你的数据主权、你的交付能力，都不该被单点控制。

因此本项目坚持内置 `dl/` 本地模型库：本地权重属于你自己，断网可用、数据不出本机、创意不被采集、服务不被任何单一实体卡住。这也是 AgnesAgent 与纯在线 AI 产品最根本的区别。

### 默认接入模型

默认接入 **Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive**（GGUF 量化版），配合视觉投影 `mmproj` 实现多模态。权重体积较大（约 11GB），不随仓库分发，请自行下载后放入 `dl/`：

| 文件 | 说明 |
|------|------|
| `Qwen3.6-35B-A3B-IQ2_M.gguf` | 主模型权重（IQ2_M 量化） |
| `mmproj-Qwen3.6-35B-A3B-*.gguf` | 多模态视觉投影 |
| `llama/` | llama.cpp Windows 推理引擎（server 等工具） |

- 模型获取与推荐：见 [freedidi 相关说明](https://freedidi.com/24284.html)，也可从魔搭 / HuggingFace 搜索同名权重
- 离线 ASR / TTS 模型由 `dl/dl_models.py` 从 hf-mirror 自动下载

## 配置说明

`config.example.json` 中的 Provider 结构：

| Provider | 类型 | 说明 |
|----------|------|------|
| agnes | openai | AgnesAI 主通道（免费额度，含图像/视频模型） |
| nvidia | openai | NVIDIA NIM 免费模型通道 |
| zhipu | openai | 智谱 GLM-4-Flash 免费通道 |
| baidu | openai | 百度 ERNIE-Speed 免费通道 |
| ollama | openai | 本地模型（包内 Qwen3.6-35B，完全离线，llama-server 提供） |
| g4f | g4f | gpt4free 零 Key 保底通道 |
| freellmapi | openai | 本地免费 API 聚合网关 |
| github | openai | GitHub Models 免费通道（需 GitHub PAT） |

本地与在线模式在同一窗口内二选一，切换聊天记录互不干扰。

### 功能部门开关（默认关闭的模块）

部分功能模块（部门）默认关闭，对应工具不会传给模型。当前默认关闭的部门：

| 部门 | 对应能力 | 默认状态 |
|------|----------|----------|
| 读取文件部 | 文档解析：直接读取 PDF / Word / Excel 内容进行问答 | 关闭 |

**开启方式（二选一）：**

1. **界面开启（推荐）**：启动 AgnesAgent 后，在左侧边栏「能力部门」面板中找到「读取文件部」，将开关拨到开启，状态会自动写入 `config.json` 并即时生效。
2. **手动编辑配置**：打开 `config.json`，将 `dept_disabled` 中的 `"读取文件部"` 改为 `false`，保存后重启应用。

```json
{
  "dept_disabled": {
    "读取文件部": false
  }
}
```

> 说明：文档解析依赖 `pypdf` / `python-docx` / `openpyxl` / `PyMuPDF`，已在 `requirements.txt` 与致谢中列出；不需要该功能时保持关闭即可，不影响其他能力。

## 如何改造成你自己的项目

- **换模型**：替换 `dl/` 下的 GGUF 权重即可，无需改代码；或把 Provider 指向任意 OpenAI 兼容接口
- **接入视频生成（以 MiniMax H3 为例）**：视频入口 `generate_video_blocking` 已预留三级链路，在对应层级填入 MiniMax H3 的 API 地址与 Key 即可启用；本地内存不足时可只保留云端链路（如 `_generate_video_api` 指向 MiniMax H3），关闭本地 ComfyUI 兜底
- **换 UI 主题**：基于 QFluentWidgets 的样式表集中管理，修改配色与圆角参数即可
- **加功能**：主入口在 `main.py`，聊天流、工具调用、语音管线均为独立模块结构，按需扩展
- **分发**：修改 `AgnesAgent.spec` 中的名称与图标后重新打包

## 后续方向（Roadmap）

- [ ] 插件系统：支持第三方工具链动态挂载
- [ ] 视频生成默认通道：接入 MiniMax H3 等云端视频模型（接口已预留，本地内存不足时默认走云端）
- [ ] 更多本地模型模板：一键下载与管理多套权重
- [ ] 多模态本地化：完整离线图像理解链路
- [ ] 移动端 / Web 远程访问
- [ ] 会话云同步与多设备迁移
- [ ] 适合大学生创业/需要借壳生蛋套现跑路的黑心老板

## 开源声明：本项目基于以下开源项目构建

AgnesAgent **本身不包含闭源核心**——它的界面、推理、语音、OCR、搜索、文档解析等全部能力，均构建在以下开源项目与免费服务之上。本项目代码是这些开源组件的整合封装层，任何功能离开对应的上游开源项目都无法独立工作。

| 能力模块 | 依赖的开源项目 | 用途 |
|----------|----------------|------|
| GUI 框架 | PySide6 (Qt for Python) | 全部窗口与控件 |
| UI 主题 | QFluentWidgets | Win11 Fluent 风格界面 |
| 本地推理 | llama.cpp | GGUF 模型推理引擎 |
| 基础模型 | Qwen 系列（阿里） | 对话/多模态模型基座 |
| 离线语音 | sherpa-onnx / SenseVoice / VITS | ASR 与 TTS |
| 联网搜索 | ddgs (duckduckgo-search) | 免 Key 网页搜索 |
| 零 Key 通道 | g4f (gpt4free) | 免费模型兜底 |
| OCR | RapidOCR | 离线文字识别 |
| 文档解析 | pypdf / python-docx / openpyxl / PyMuPDF | PDF/Word/Excel 读取 |
| API 网关 | AgnesAI / NVIDIA NIM / freellmapi / GitHub Models | 在线模型通道 |
| 打包分发 | PyInstaller | exe 打包 |

**因此，本项目遵守各上游开源项目的许可协议（MIT / Apache-2.0 / 模型各自许可），并在下文逐一列出致谢。若你是上游项目作者，认为本项目使用方式不妥，欢迎提交 Issue 沟通。**

---

## 致谢

本项目站在众多优秀开源项目与免费服务的肩膀上，特此致谢：

**核心框架与 UI**

- [Qt for Python (PySide6)](https://wiki.qt.io/Qt_for_Python) — 跨平台 GUI 框架
- [QFluentWidgets](https://github.com/zhiyiYo/PyQt-Fluent-Widgets) — Win11 Fluent Design 风格组件库
- [PyInstaller](https://github.com/pyinstaller/pyinstaller) — Python 应用打包

**本地推理引擎与模型**

- [llama.cpp](https://github.com/ggerganov/llama.cpp) — GGUF 本地推理引擎（llama-server 等）
- [Qwen 系列](https://github.com/QwenLM) — 阿里巴巴通义千问开源模型基座
- HauhauCS — Qwen3.6-35B-A3B Uncensored 微调与量化版本作者
- [freedidi](https://freedidi.com/24284.html) — 模型获取与推荐来源

**语音能力（离线 ASR / TTS）**

- [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) — 离线语音推理框架
- [SenseVoice](https://github.com/FunAudioLLM/SenseVoice) — 阿里 FunAudioLLM 语音识别模型
- [VITS](https://github.com/jaywalnut310/vits) — 中文语音合成模型（zh-aishell3）
- csukuangfj — sherpa-onnx 相关模型打包与发布
- [edge-tts](https://github.com/rany2/edge-tts) — 微软 Edge 在线语音合成通道
- [SpeechRecognition](https://github.com/Uberi/speech_recognition)、sounddevice、soundfile — 录音与音频处理

**视觉与文档**

- [RapidOCR](https://github.com/RapidAI/RapidOCR) — 轻量离线 OCR
- [python-docx](https://github.com/python-openxml/python-docx)、[openpyxl](https://foss.heptapod.net/openpyxl/openpyxl)、[pypdf](https://github.com/py-pdf/pypdf)、[PyMuPDF](https://github.com/pymupdf/PyMuPDF) — 文档解析

**联网与免费通道**

- [g4f (gpt4free)](https://github.com/xtekky/gpt4free) — 零 Key 免费模型兜底通道
- [ddgs (DuckDuckGo Search)](https://github.com/deedy5/duckduckgo_search) — 免 Key 联网搜索
- [freellmapi](https://github.com/freellmapi/freellmapi) — 本地免费 API 网关
- [NumPy](https://github.com/numpy/numpy)、requests、jaraco.text 等 — 基础依赖

**API 平台**

- [AgnesAI](https://apihub.agnes-ai.com) — 免费多模态 API 主通道
- [NVIDIA NIM](https://integrate.api.nvidia.com) — 免费模型推理服务

再次向以上所有项目与团队致敬。若遗漏了任何贡献者，欢迎提交 Issue 补充。

## 开源协议

本项目采用 [MIT License](LICENSE) 开源。