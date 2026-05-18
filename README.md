# 视频翻译配音工具

中文视频自动翻译配音与字幕生成工具。支持硬字幕 OCR / 语音识别 → 翻译 → TTS 配音，并支持说话人分离与性别识别。

---

## 目录

- [功能概览](#功能概览)
- [环境要求](#环境要求)
- [安装](#安装)
- [部署文档](#部署文档)
- [首次配置](#首次配置)
- [快速开始](#快速开始)
- [命令行参数详解](#命令行参数详解)
- [可用声音列表](#可用声音列表)
- [说话人感知配音](#说话人感知配音)
- [XTTS-v2 音色克隆](#xtts-v2-音色克隆)
- [常见错误与解决](#常见错误与解决)

---

## 功能概览

| 脚本 | 用途 |
|------|------|
| `video_dubbing.py` | 完整配音：OCR/语音识别 → 翻译 → TTS 合成 → 字幕叠加 → 输出 |
| `video_subtitles_only.py` | 仅字幕：OCR/语音识别 → 翻译 → 字幕叠加（不替换原声） |

两个脚本均支持单文件、批量处理、批量处理后合并三种模式。

---

## 环境要求

- Windows 10/11（64 位）
- Python **3.10** 或 **3.11**（推荐 3.10，wheel 覆盖最全）
- 磁盘空间：安装约 8 GB，运行时模型缓存约 4 GB
- 内存：最低 8 GB，推荐 16 GB
- CPU-only：为降低 Windows 依赖冲突，当前安装和运行均固定使用 CPU

---

## 安装

### 方式一：一键脚本（推荐）

将项目文件放在同一目录下，双击运行 `install_windows.bat`。

脚本会自动完成：
1. 检测 Python 版本
2. 检查部署包是否完整（包括 `voice-gender-classifier/model.py`）
3. 创建主虚拟环境 `trans_env` 并选择 CPU-only torch wheel
4. 按正确顺序安装所有依赖（numpy 版本锁定、torch 专用源）
5. 创建独立人声分离环境 `separation_env`，安装 `audio-separator[cpu]`
6. 创建独立硬字幕 OCR 环境 `ocr_env`，安装 `PaddleOCR`
7. 验证每个关键包是否能正常导入，并检查 `video_dubbing.py --help`

脚本会在项目根目录生成 `install_windows.log`。安装失败时先看这个日志，不要只看命令行最后一行。

> **注意**：CPU 版 torch 下载仍可能较慢，请耐心等待。如下载超时，可在脚本运行前设置代理或使用国内镜像（脚本内有说明）。

> **espeak-ng（英语 TTS 必需）**：使用英语声音（`en_vctk_vits_*` / `en_ljspeech_vits`）时，必须安装 espeak-ng。安装脚本会自动检测并给出下载链接。XTTS-v2 多语言声音（韩/日/中/法等）不需要 espeak-ng。

### 方式二：手动安装

```bat
# 1. 升级基础工具
pip install --upgrade pip setuptools wheel

# 2. 先锁定 numpy（必须在 torch 之前）
pip install "numpy>=1.24.0,<2.0.0"

# 3. 安装 CPU 版 torch
pip install torch>=2.3.0,<2.4.0 torchaudio>=2.3.0,<2.4.0 ^
    --index-url https://download.pytorch.org/whl/cpu

# 4. 安装其余依赖
pip install -r requirements.txt

# 5. 创建独立人声分离环境（不要装进主环境）
python -m venv separation_env
separation_env\Scripts\python -m pip install --upgrade pip setuptools wheel
separation_env\Scripts\python -m pip install imageio-ffmpeg
separation_env\Scripts\python -m pip install "audio-separator[cpu]"

# 6. 创建独立硬字幕 OCR 环境（不要装进主环境）
python -m venv ocr_env
ocr_env\Scripts\python -m pip install --upgrade pip setuptools wheel
ocr_env\Scripts\python -m pip install paddleocr paddlepaddle opencv-python-headless

# 7. ASR 兜底使用主环境中的 faster-whisper，无需创建独立 ASR 环境
```

---

## 部署文档

完整部署说明见 [DEPLOYMENT.md](/Users/liangrn/Downloads/Trans/DEPLOYMENT.md)。

这里先强调当前版本的几个硬要求：

1. 当前版本固定为 CPU-only 运行，不再提供 GPU/CUDA 安装路径。
2. 部署目录必须完整包含 `voice-gender-classifier/` 子目录，至少要有 `voice-gender-classifier/model.py`。
3. 所有视频处理都会先做人声分离；部署目录必须有可用的 `separation_env` 或 PATH 中可用的 `audio-separator`。
4. 中文文本识别优先使用硬字幕 OCR；只要 OCR 结果可用，就会跳过 ASR。部署目录必须有可用的 `ocr_env`，或用环境变量 `OCR_PYTHON` 指定 Python。
5. 无可用硬字幕时，才使用主环境里的 `faster-whisper` 兜底识别，默认模型为 `medium`，可用环境变量 `WHISPER_MODEL_SIZE` 调整。
6. 如果缺少 `voice-gender-classifier/model.py`，程序会记录
   `本地性别模型文件不存在: .../voice-gender-classifier/model.py`，随后退回 embedding 统计，男女声识别准确率会明显下降。
7. Windows 推荐直接运行 `install_windows.bat`。
8. 在线说话人分离依赖 HuggingFace token；未设置 `HF_TOKEN` 时会跳过说话人分离，整段视频使用同一个声音。

---

## 首次配置

### 1. HuggingFace Token（说话人分离功能必需）

说话人分离（区分不同说话人、按性别分配声音）依赖 pyannote 模型，需要 HuggingFace 账号授权。

**步骤：**

1. 注册 [huggingface.co](https://huggingface.co)
2. 访问以下两个页面，分别点击 **Agree and access repository**：
   - [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
   - [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
3. 在 [Settings → Access Tokens](https://huggingface.co/settings/tokens) 创建一个 **Read** 权限的 Token
4. 设置系统环境变量：

```bat
# 方式A：永久设置（推荐，重启后生效）
setx HF_TOKEN "hf_xxxxxxxxxxxxxxxxxxxxxxxx"

# 方式B：临时设置（仅当前命令行窗口有效）
set HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxx
```

> 未设置 HF_TOKEN 时程序仍可运行，但跳过说话人分离，全程使用同一个声音。

### 2. ffmpeg（已自动处理）

`imageio-ffmpeg` 会在首次运行时自动下载 ffmpeg 到 Python 环境内，无需手动安装或配置 PATH。

如需使用自定义 ffmpeg：

```bat
# 方式A：环境变量
set FFMPEG_BIN=C:\tools\ffmpeg\bin\ffmpeg.exe

# 方式B：命令行参数
python video_dubbing.py ... --ffmpeg_bin C:\tools\ffmpeg\bin\ffmpeg.exe
```

---

## 快速开始

### 单文件配音（最常用）

```bat
python video_dubbing.py ^
    --mode single ^
    --input_video  input.mp4 ^
    --target_lang  en ^
    --voice        en_vctk_vits_m001 ^
    --output_video output_en.mp4
```

### 单文件仅生成字幕

```bat
python video_subtitles_only.py ^
    --mode single ^
    --input_video  input.mp4 ^
    --target_lang  en ^
    --output_video output_subtitled.mp4
```

### 批量处理一个目录

```bat
python video_dubbing.py ^
    --mode      batch ^
    --input_dir  ./videos/ ^
    --output_dir ./dubbed/ ^
    --target_lang ja ^
    --voice       ja_male_001
```

### 批量处理后自动合并为一个视频

```bat
python video_dubbing.py ^
    --mode           batch_merge ^
    --input_dir       ./episodes/ ^
    --output_dir      ./output/ ^
    --target_lang     ko ^
    --voice           ko_female_001 ^
    --merged_filename all_episodes_ko.mp4
```

### 仅合并已处理好的视频

```bat
python video_dubbing.py ^
    --mode           merge_only ^
    --input_dir       ./dubbed/ ^
    --output_dir      ./final/ ^
    --merged_filename final.mp4
```

---

## 命令行参数详解

### `video_dubbing.py`

| 参数 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| `--mode` | ✓ | — | `single` / `batch` / `batch_merge` / `merge_only` |
| `--input_video` | 单文件必填 | — | 输入视频路径 |
| `--input_dir` | 批量必填 | — | 输入视频目录 |
| `--output_video` | | `dubbed_output.mp4` | 单文件模式输出路径 |
| `--output_dir` | | `./output_videos/` | 批量/合并模式输出目录 |
| `--target_lang` | 配音必填 | — | 目标语言代码，见下表 |
| `--voice` | | `en_vctk_vits_m001` | 声音 key，见[可用声音列表](#可用声音列表) |
| `--max_speed` | | `1.5` | TTS 超长时最大加速倍数，超过此值听感失真 |
| `--min_speed` | | `1.0` | TTS 短于原时长时不拉伸，原速播放 |
| `--merged_filename` | | `merged_output.mp4` | `batch_merge` / `merge_only` 合并后文件名 |
| `--parallel` | | `true` | 并行翻译开关，设为 `false` 禁用 |
| `--workers` | | `10` | 翻译并行线程数，推荐 5~20 |
| `--tts_workers` | | `3` | TTS 并行线程数，CPU 模式推荐 2~3 |
| `--ffmpeg_bin` | | 自动 | 自定义 ffmpeg 可执行路径 |
| `--font_path` | | 自动 | 自定义字幕字体文件路径（.ttf / .ttc） |

### `video_subtitles_only.py`

| 参数 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| `--mode` | ✓ | — | `single` / `batch` / `batch_merge` / `merge_only` |
| `--input_video` | 单文件必填 | — | 输入视频路径 |
| `--input_dir` | 批量必填 | — | 输入视频目录 |
| `--output_video` | | `subtitled_output.mp4` | 单文件输出路径 |
| `--output_dir` | | `./output_subtitles/` | 批量输出目录 |
| `--target_lang` | ✓ | — | 目标语言代码 |
| `--font_path` | | 自动 | 自定义字幕字体路径 |
| `--parallel` | | `true` | 并行翻译开关 |
| `--workers` | | `10` | 翻译线程数 |
| `--merged_filename` | | `merged_output.mp4` | 合并后文件名 |

### 翻译模式

当前版本仅保留在线翻译流程（`deep_translator` / Google Translate）。旧的 `本地翻译版本/` NLLB 离线翻译目录已明确移除，以减少 Windows 新环境的安装体积和依赖冲突面。

### 部署时必须一并带走的文件

最小部署目录建议至少包含：

```text
video_dubbing.py
video_subtitles_only.py
speaker_aware_dubbing.py
gender_classifier.py
requirements.txt
install_windows.bat          # Windows 推荐保留
audio_separation.py
asr_recognition.py
ocr_recognition.py
ocr_subtitle_probe.py
separation_env/              # 可重新创建；用于 audio-separator
ocr_env/                     # 可重新创建；用于 PaddleOCR
voice-gender-classifier/
  ├── model.py
  ├── README.md
  └── requirements.txt
```

如果只复制主脚本，不复制 `voice-gender-classifier/`，当前版本不会崩溃，但会降级为弱识别路径，日志里会显示“完整模型失败(本地性别模型文件不存在...)，使用 embedding 统计”。

### 目标语言代码

| 代码 | 语言 | 代码 | 语言 |
|------|------|------|------|
| `en` | 英语 | `ko` | 韩语 |
| `ja` | 日语 | `zh` | 中文 |
| `es` | 西班牙语 | `fr` | 法语 |
| `de` | 德语 | `pt` | 葡萄牙语 |
| `it` | 意大利语 | `ru` | 俄语 |
| `ar` | 阿拉伯语 | `hi` | 印地语 |
| `tr` | 土耳其语 | `nl` | 荷兰语 |
| `pl` | 波兰语 | | |

---

## 可用声音列表

### 英语（tts_models/en/vctk/vits — 英国英语，无需参考音频）

| Key | 说明 |
|-----|------|
| `en_vctk_vits_m001` | 男声1，深沉 |
| `en_vctk_vits_m002` | 男声2，温和 |
| `en_vctk_vits_m003` | 男声3，年轻 |
| `en_vctk_vits_m004` | 男声4，沉稳 |
| `en_vctk_vits_m005` | 男声5，磁性 |
| `en_vctk_vits_m006` | 男声6，浑厚 |
| `en_vctk_vits_m007` | 男声7，年轻 |
| `en_vctk_vits_m008` | 男声8，中性 |
| `en_vctk_vits_m009` | 男声9，温和 |
| `en_vctk_vits_m010` | 男声10，沉稳 |
| `en_vctk_vits_m011` | 男声11，浑厚 |
| `en_vctk_vits_m012` | 男声12，磁性 |
| `en_vctk_vits_m013` | 男声13，年轻 |
| `en_vctk_vits_m014` | 男声14，沉稳 |
| `en_vctk_vits_m015` | 男声15，温和 |
| `en_vctk_vits_m016` | 男声16，磁性 |
| `en_vctk_vits_f001` | 女声1，甜美清晰 |
| `en_vctk_vits_f002` | 女声2，明亮活泼 |
| `en_vctk_vits_f003` | 女声3，成熟稳重 |
| `en_vctk_vits_f004` | 女声4，清脆悦耳 |
| `en_vctk_vits_f005` | 女声5，明亮自信 |
| `en_vctk_vits_f006` | 女声6，清新活泼 |
| `en_vctk_vits_f007` | 女声7，温柔细腻 |
| `en_vctk_vits_f008` | 女声8，优雅知性 |
| `en_vctk_vits_f009` | 女声9，开朗热情 |
| `en_vctk_vits_f010` | 女声10，柔和亲切 |
| `en_ljspeech_vits` | 女声，标准美式英语（LJSpeech） |

### 德语（单语种 VITS，无需参考音频）

| Key | 说明 |
|-----|------|
| `de_male_001` | Thorsten，男声，标准 |
| `de_male_002` | Thorsten-Emotion，表情丰富 |

### 多语言（XTTS-v2，可选音色克隆）

以下语言均使用 XTTS-v2 模型。不提供参考音频时使用内置默认音色；提供 3~6 秒参考音频可克隆任意音色（见[音色克隆](#xtts-v2-音色克隆)）。

| Key | 语言 | 说明 |
|-----|------|------|
| `ko_male_001` / `ko_male_002` | 韩语 | 男声 |
| `ko_female_001` / `ko_female_002` | 韩语 | 女声 |
| `ja_male_001` / `ja_male_002` | 日语 | 男声 |
| `ja_female_001` / `ja_female_002` | 日语 | 女声 |
| `zh_male_001` / `zh_male_002` | 中文 | 男声 |
| `zh_female_001` / `zh_female_002` | 中文 | 女声 |
| `es_male_001` / `es_male_002` | 西班牙语 | 男声 |
| `es_female_001` / `es_female_002` | 西班牙语 | 女声 |
| `fr_male_001` | 法语 | 男声 |
| `fr_female_001` | 法语 | 女声 |
| `pt_male_001` | 葡萄牙语 | 男声 |
| `pt_female_001` | 葡萄牙语 | 女声 |
| `it_male_001` | 意大利语 | 男声 |
| `it_female_001` | 意大利语 | 女声 |
| `ru_male_001` | 俄语 | 男声 |
| `ru_female_001` | 俄语 | 女声 |
| `ar_male_001` | 阿拉伯语 | 男声 |
| `ar_female_001` | 阿拉伯语 | 女声 |
| `hi_male_001` | 印地语 | 男声 |
| `hi_female_001` | 印地语 | 女声 |
| `tr_male_001` | 土耳其语 | 男声 |
| `tr_female_001` | 土耳其语 | 女声 |
| `nl_male_001` | 荷兰语 | 男声 |
| `nl_female_001` | 荷兰语 | 女声 |
| `pl_male_001` | 波兰语 | 男声 |
| `pl_female_001` | 波兰语 | 女声 |

---

## 说话人感知配音

设置 `HF_TOKEN` 后，程序自动启用说话人分离：

- 识别视频中的不同说话人
- 对每位说话人进行性别识别
- 自动为不同说话人分配不同声音（男声池 / 女声池轮换）
- 说话人分离与 ASR 并行运行，不增加总处理时间

**流程示意：**

```
视频 ──→ [说话人分离，后台并行] ──→ 说话人A（男）→ en_vctk_vits_m001
     ↘→ [ASR + 翻译，主线程]  ──→ 说话人B（女）→ en_vctk_vits_f001
                                 说话人C（男）→ en_vctk_vits_m002
```

未设置 `HF_TOKEN` 时跳过此步骤，所有片段使用 `--voice` 指定的单一声音。

---

## XTTS-v2 音色克隆

XTTS-v2 支持声音克隆：提供一段 **3~6 秒** 的参考音频，输出音色会贴近该说话人。

**使用方式：** 在 `get_available_coqui_voices()` 中为对应的声音条目添加 `speaker_wav` 字段，指向参考音频文件路径。

```python
# video_dubbing.py 中 get_available_coqui_voices() 的条目示例
"ko_male_001": {
    "model_name": "tts_models/multilingual/multi-dataset/xtts_v2",
    "language": "ko",
    "speaker_wav": "C:/refs/my_voice_sample.wav",   # ← 添加此行
    "description": "XTTS-v2 (韩语, 男声1, 标准)"
},
```

**参考音频要求：**
- 格式：WAV，16 kHz 或 22 kHz，单声道
- 时长：3~6 秒（过短效果差，过长无额外收益）
- 内容：安静环境录制，无背景音乐，发音清晰

不添加 `speaker_wav` 时，使用 XTTS-v2 内置默认说话人 `Claribel Dervla`（中性偏女声）。

---

## 常见错误与解决

### 安装阶段

**`ERROR: Could not find a version that satisfies the requirement torch>=2.3.0,<2.4.0`**

pip 从 PyPI 主源找不到对应 wheel。必须从 pytorch.org 专用源安装：
```bat
pip install torch==2.3.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cpu
```

**`[!] No espeak backend found`（英语 TTS 无法合成）**

英语 VCTK / LJSpeech VITS 模型底层依赖 espeak-ng 做文本正则化，没有它报此错：
1. 下载安装包：[espeak-ng Releases](https://github.com/espeak-ng/espeak-ng/releases)，找最新的 `.msi` 文件
2. 双击安装，保持默认路径（`C:\Program Files\eSpeak NG`）
3. 重新打开命令行窗口（PATH 需要刷新）后再运行程序

如果只使用非英语声音（XTTS-v2 系列），不需要安装 espeak-ng。

**`coqui-tts` 安装失败，报 `error: Microsoft Visual C++ 14.0 or greater is required`**

系统缺少 C++ 编译环境。解决方案：
1. 安装 [Visual Studio Build Tools 2022](https://visualstudio.microsoft.com/downloads/)，勾选"使用 C++ 的桌面开发"
2. 安装 [espeak-ng](https://github.com/espeak-ng/espeak-ng/releases) 并加入 PATH
3. 重新运行 `install_windows.bat`

或者直接指定版本跳过编译：
```bat
pip install coqui-tts==0.24.1
```

**`ImportError: numpy 2.x detected`** 或 torch 导入时 ABI 错误

numpy 被其他包升级到 2.x，强制降级：
```bat
pip install --force-reinstall "numpy>=1.24.0,<2.0.0"
```

**`完整模型失败(本地性别模型文件不存在: .../voice-gender-classifier/model.py)`**

部署目录缺少 `voice-gender-classifier/` 子目录，或只复制了主脚本没有复制模型源码目录。修复：
1. 确认部署目录下存在 `voice-gender-classifier/model.py`
2. 确认它和 `gender_classifier.py` 位于同一个项目根目录下
3. 重新运行程序

如果你打算重新打包发布，直接按 [DEPLOYMENT.md](/Users/liangrn/Downloads/Trans/DEPLOYMENT.md) 的目录清单准备即可。

---

### 运行阶段

**`TypeError: Pipeline.from_pretrained() got an unexpected keyword argument 'use_auth_token'`**

huggingface_hub 版本过旧或补丁未生效。升级：
```bat
pip install --upgrade huggingface_hub
```

**`401 Unauthorized` / `Repository not found`**

HF_TOKEN 无效，或未对 pyannote 模型点击 "Agree and access repository"。检查：
1. Token 是否正确复制（以 `hf_` 开头）
2. 是否已登录 HuggingFace 并同意两个模型的协议（链接见[首次配置](#首次配置)）
3. 环境变量是否已在当前命令行窗口生效（用 `echo %HF_TOKEN%` 验证）

**`AttributeError: module 'torchaudio' has no attribute 'AudioMetaData'`**

pyannote.audio 3.1.x 的已知 bug，3.3.0 已修复。升级：
```bat
pip install "pyannote.audio>=3.3.0" --upgrade
```

**`RuntimeError: operator torchvision::nms does not exist`**

torchvision 版本与 torch 不匹配（常见于 pyannote 链式依赖拉了最新 torchvision）。修复：
```bat
pip install "torchvision>=0.18.0,<0.19.0" --force-reinstall
```

**`FileNotFoundError: [WinError 2] 系统找不到指定的文件: 'ffmpeg'`**

imageio-ffmpeg 未安装或未触发。检查：
```bat
python -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())"
```
如果报错，重新安装：
```bat
pip install imageio-ffmpeg>=0.4.9
```

**`PermissionError: [WinError 32] 另一个程序正在使用此文件`**

Windows 文件句柄延迟释放，程序已内置重试机制（最多重试 5 次）。若仍报错，可能是杀毒软件锁定了临时文件，将项目目录加入杀毒软件白名单后重试。

**输出视频无声音 / 只有部分片段有声音**

TTS 生成某些片段静音后被过滤。可尝试：
1. 降低 `--max_speed`（默认 1.5，调为 2.0 可保留更多片段）
2. 换用语速更快的声音（VCTK 系列比 XTTS-v2 快约 3 倍）

**XTTS-v2 生成速度很慢**

XTTS-v2 是大模型，CPU 下每句约 5~15 秒。解决方法：
1. 降低 `--tts_workers`（减少并行避免 CPU 负载过高）
2. 短视频内容改用英语 VCTK 系列（速度快 5~10 倍）
3. 分批处理长视频，避免一次性处理过多片段

**字幕乱码（方框或问号）**

系统未安装中文/目标语言字体，或字体路径检测失败。指定字体：
```bat
python video_dubbing.py ... --font_path C:\Windows\Fonts\msyh.ttc
```
常用字体路径：
- 中文：`C:\Windows\Fonts\msyh.ttc`（微软雅黑）
- 韩文：`C:\Windows\Fonts\malgun.ttf`
- 日文：`C:\Windows\Fonts\msgothic.ttc`
- 全语言：`C:\Windows\Fonts\arialuni.ttf`（需单独安装 Arial Unicode MS）
