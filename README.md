# 视频翻译配音工具

将视频中的语音自动识别、翻译并生成配音或字幕。

## 功能特点

- **语音识别**: 使用 faster-whisper 进行高精度语音识别
- **时间戳优化**: 启用 VAD 和词级时间戳，解决配音对齐问题
- **翻译**: 支持在线翻译 (Google) 和离线翻译 (NLLB-200)
- **配音合成**: 使用 Coqui TTS 生成目标语言配音
- **字幕渲染**: 自动生成双语字幕

## 安装

### 1. 系统依赖

需要安装 **FFmpeg**：

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg

# Windows
# 从 https://ffmpeg.org/download.html 下载并添加到 PATH
```

### 2. Python 依赖

```bash
pip install faster-whisper torch deep_translator moviepy Pillow TTS onnxruntime
```

**离线翻译版本额外依赖**：
```bash
pip install transformers sentencepiece
```

### 3. 下载离线翻译模型（可选）

如果使用离线翻译版本：
```bash
# 首次运行会自动下载 NLLB-200 模型（约 3GB）
python 本地翻译版本/video_dubbing.py --help
```

## 使用方法

### 在线翻译版本（推荐）

```bash
# 单视频配音（含语音合成）
python video_dubbing.py --mode single --input_video input/video.mp4 --target_lang en --output_video output/result.mp4

# 单视频仅字幕
python video_subtitles_only.py --mode single --input_video input/video.mp4 --target_lang en --output_video output/result.mp4

# 批量处理（仅字幕）
python video_subtitles_only.py --mode batch --input_dir ./input --output_dir ./output --target_lang en

# 批量处理并合并
python video_subtitles_only.py --mode batch_merge --input_dir ./input --output_dir ./output --target_lang en --merged_filename final.mp4
```

### 离线翻译版本

```bash
# 进入离线翻译目录
cd 本地翻译版本

# 使用方式同上
python video_dubbing.py --mode single --input_video ../input/video.mp4 --target_lang en --output_video ../output/result.mp4
```

## 支持的目标语言

| 代码 | 语言 | 代码 | 语言 |
|------|------|------|------|
| `en` | 英语 | `ja` | 日语 |
| `ko` | 韩语 | `zh` | 中文 |
| `fr` | 法语 | `de` | 德语 |
| `es` | 西班牙语 | `pt` | 葡萄牙语 |
| `ru` | 俄语 | `it` | 意大利语 |
| `tr` | 土耳其语 | `ar` | 阿拉伯语 |
| `hi` | 印地语 | `th` | 泰语 |
| `vi` | 越南语 | `id` | 印尼语 |

## 配音声音选项

配音模式下可指定声音：

```bash
python video_dubbing.py --mode single --input_video input/video.mp4 --target_lang en --voice en_sam_tacotron --output_video output/result.mp4
```

常用声音：
- `en_sam_tacotron` - 英语男声
- `zh_CN-baker` - 中文女声

更多声音请运行 `python video_dubbing.py --help` 查看。

## 目录结构

```
Trans/
├── input/                    # 源视频目录
├── output/                   # 输出视频目录
├── video_dubbing.py          # 配音脚本（在线翻译）
├── video_subtitles_only.py   # 字幕脚本（在线翻译）
├── 本地翻译版本/              # 离线翻译版本
│   ├── video_dubbing.py
│   ├── video_subtitles_only.py
│   └── local_translator.py   # NLLB-200 翻译模块
└── README.md
```

## 更新日志

### 2026-03-11 - 时间戳对齐优化

**问题**: 部分视频的配音与原视频对不上，ASR 时间戳比实际语音提前约 3 秒。

**解决**: 在所有 ASR 调用中启用：
- `word_timestamps=True` - 词级时间戳，提高精度
- `vad_filter=True` - VAD 过滤非语音片段
- `vad_parameters` - 优化静音检测参数

**新增依赖**: `onnxruntime`（VAD 支持）

## 环境变量

- `FFMPEG_BIN`: 指定 ffmpeg 路径
- `SUBTITLE_FONT_PATH`: 指定字幕字体文件

## 常见问题

**Q: 提示找不到 ffmpeg**
A: 安装 ffmpeg 并确保在 PATH 中，或设置 `FFMPEG_BIN` 环境变量

**Q: 配音对不上时间**
A: 已在最新版本修复，请确保使用更新后的脚本

**Q: 离线翻译模型下载慢**
A: 模型约 3GB，需要耐心等待；也可手动下载后放到 `~/.cache/huggingface/`