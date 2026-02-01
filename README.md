
# ai-chatbot

---

# 🎙️ Jarvis - 本地化全栈语音助手

这是一个基于 Python 构建的轻量级、低延迟、完全本地运行的语音助手。它集成了唤醒词检测、语音识别 (STT)、大语言模型 (LLM) 和语音合成 (TTS) 四大核心模块，实现了类似 Iron Man 中 J.A.R.V.I.S 的交互体验。

## ✨ 核心特性

*   **完全本地化**：所有处理（听、想、说）均在本地运行，保护隐私，无需联网（除首次下载模型外）。
*   **多线程架构**：音频采集与逻辑处理分离，互不阻塞，杜绝录音卡顿。
*   **智能 VAD (语音活动检测)**：
    *   **预录音缓冲 (Pre-roll)**：防止说话过快导致的“吞字”现象。
    *   **动态静音检测**：允许说话时短暂亦停顿，不会轻易打断。
*   **防误触机制**：休眠时自动清空音频队列与模型状态，防止环境噪音导致的“连环唤醒”。
*   **模块化设计**：
    *   🧠 **大脑**: Ollama (支持 Llama3, Gemma3, DeepSeek 等)
    *   👂 **耳朵**: Faster-Whisper (OpenAI Whisper 的加速版)
    *   🗣️ **嘴巴**: Piper TTS (高质量、低延迟的神经网络语音合成)
    *   ⚡ **唤醒**: OpenWakeWord (精准的唤醒词检测)

## 🛠️ 环境依赖

### 1. 软件要求
*   Python 3.10 或更高版本
*   [Ollama](https://ollama.com/) (必须安装并运行服务)

### 2. Python 库安装
在终端中运行以下命令安装所需依赖：

```bash
pip install sounddevice numpy soundfile ollama openwakeword faster-whisper piper-tts
```

*注意：如果你有 NVIDIA 显卡，建议安装 CUDA 版本的 PyTorch 以获得更快的响应速度。*

## 📥 模型下载与配置 (关键步骤)

项目运行需要准备以下三个部分的模型文件，请按照目录结构放置：

### 1. 🧠 Ollama 模型 (大脑)
确保安装了 Ollama，并在终端运行过你想要使用的模型（代码默认为 `gemma3`，也可以换成 `qwen2` 或 `llama3`）：

```bash
ollama run gemma3
```

### 2. 🗣️ Piper TTS 模型 (嘴巴)
代码默认使用 `zh_CN-huayan-medium` 声音。你需要下载 **.onnx** 和 **.onnx.json** 两个文件，并放在项目根目录下。

*   **下载地址**: [HuggingFace - Piper Voices (zh_CN)](https://huggingface.co/rhasspy/piper-voices/tree/main/zh/zh_CN/huayan/medium)
*   **需要下载的文件**:
    1.  `zh_CN-huayan-medium.onnx`
    2.  `zh_CN-huayan-medium.onnx.json`

### 3. ⚡ OpenWakeWord (唤醒)
首次运行代码时，`openwakeword` 会自动下载模型，无需手动操作。

---

## 📂 推荐目录结构

确保你的文件夹长这个样子：

```text
MyJarvis/
├── main.py                     # 主程序代码
├── voice.onnx    # TTS 模型文件
├── voice.onnx.json # TTS 配置文件
└── README.md
```

---

## 🚀 运行指南

1.  **启动 Ollama 服务**（通常安装后会自动后台运行）。
2.  **运行脚本**：

```bash
python main.py
```

3.  **开始交互**：
    *   等待控制台出现 `[系统就绪] 请说: hey_jarvis`。
    *   对着麦克风说 **"Hey Jarvis"**。
    *   听到（或看到日志）回应 "我在" 后，直接说出你的指令（例如："帮我写一个 Python 的 Hello World"）。
    *   说完后自动检测结束，AI 会开始回答。

---

## ⚙️ 配置参数说明

在 `main.py` 顶部的 `CONFIG` 字典中，你可以调整以下参数以适应你的硬件：

| 参数名 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `WAKE_WORD` | `hey_jarvis` | 唤醒词，openwakeword 支持多种预设 (如 `alexa`, `hey_mycroft`) |
| `WHISPER_MODEL` | `small` | 语音识别模型大小。电脑慢可用 `tiny` 或 `base`，快可用 `medium` |
| `OLLAMA_MODEL` | `gemma3` | 指定 Ollama 使用的模型名称 |
| `VAD_THRESHOLD` | `0.03` | **灵敏度调节**。环境吵杂请调大 (0.05-0.1)，环境安静可调小 |
| `SILENCE_TIMEOUT`| `1.2` | 说话停顿多久算结束。如果你说话慢，建议调大到 `1.5` 或 `2.0` |
| `AUTO_SLEEP_TIMEOUT`| `10` | 多少秒无对话后自动进入休眠模式 |

---

## ❓ 常见问题 (Q&A)

**Q1: 报错 `FileNotFoundError: [Errno 2] No such file or directory: 'zh_CN-huayan-medium.onnx'`**
> **A:** 你没有下载 TTS 模型文件。请参考上方“模型下载”部分，下载两个必要文件到脚本同级目录。

**Q2: 唤醒很灵敏，但一直不停止录音 / 一直显示“正在录音...”**
> **A:** 你的环境可能有底噪（风扇声、空调声）。请增大 `VAD_THRESHOLD` 的值（例如改为 `0.05` 或 `0.08`）。

**Q3: AI 回答速度很慢**
> **A:** 这取决于你的硬件。
> *   **STT慢**: 将 `WHISPER_MODEL` 改为 `tiny`。
> *   **LLM慢**: 换用更小的模型（如 `qwen2:0.5b` 或 `gemma2:2b`）。
> *   **TTS慢**: 确保使用了 `low` 或 `medium` 质量的 Piper 模型。

**Q4: 休眠后立刻被唤醒？**
> **A:** 最新版代码已修复此问题。代码会在休眠时自动清空音频缓冲区和重置模型状态。

---

## 📝 待办 / 改进计划
- [ ] 增加 GUI 悬浮球界面
- [ ] 支持打断功能 (Barge-in)，需要回声消除 (AEC)
- [ ] 接入工具调用 (Function Calling)，例如控制智能家居

## 📜 许可证
MIT License


