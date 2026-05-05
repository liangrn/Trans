# 部署文档

本文档描述当前版本在 Windows 和 macOS 上的推荐部署方式。目标是降低新环境安装失败率，因此当前版本统一采用 CPU-only 方案。

## 1. 当前部署原则

1. 统一 CPU-only，不再提供 CUDA / GPU 安装分支。
2. 部署目录必须完整携带 `voice-gender-classifier/`。
3. 在线翻译保留，旧的 `本地翻译版本/` 已移除。
4. 默认配音行为是“静音背景 + TTS 配音”，不会保留原视频人声。

## 2. 最小部署目录

建议将下面这些文件和目录一起打包：

```text
Trans/
├── video_dubbing.py
├── video_subtitles_only.py
├── speaker_aware_dubbing.py
├── gender_classifier.py
├── test_diarization.py
├── requirements.txt
├── README.md
├── install_windows.bat
└── voice-gender-classifier/
    ├── model.py
    ├── README.md
    └── requirements.txt
```

其中 `voice-gender-classifier/model.py` 是当前本地性别模型入口。缺少它时，程序会降级为 embedding 统计路径，日志类似：

```text
[模型] 完整模型失败(本地性别模型文件不存在: .../voice-gender-classifier/model.py)，使用 embedding 统计
```

这不会直接导致程序崩溃，但男女声识别准确率会明显下降。

## 3. Windows 部署

### 3.1 环境要求

- Windows 10/11 64-bit
- Python 3.10 或 3.11
- 可访问 PyPI 和 HuggingFace
- 推荐磁盘空间至少 8 GB

### 3.2 推荐安装方式

在项目根目录双击运行：

```bat
install_windows.bat
```

脚本会完成：

1. 检查部署目录关键文件是否齐全
2. 创建 `trans_env` 虚拟环境
3. 固定安装 `numpy<2`
4. 从 `https://download.pytorch.org/whl/cpu` 安装 CPU 版 `torch/torchaudio/torchvision`
5. 安装 `onnxruntime`
6. 安装 `coqui-tts` 与其余依赖
7. 验证关键模块可导入，并检查主脚本 `--help`

脚本会在项目根目录生成 `install_windows.log`。安装失败时，优先查看这个日志。

### 3.3 Windows 额外要求

- 使用英语 VITS 声音时，需要先安装 `espeak-ng`
- 说话人分离功能需要设置 `HF_TOKEN`
- 不再需要 NVIDIA/CUDA/显卡驱动匹配

## 4. macOS 部署

### 4.1 环境要求

- macOS 13+
- Python 3.10 或 3.11
- `ffmpeg` 可用

### 4.2 推荐安装方式

建议使用 conda：

```bash
conda create -n iai python=3.10 -y
conda activate iai
python -m pip install --upgrade pip setuptools wheel
python -m pip install "numpy>=1.24.0,<2.0.0"
python -m pip install "torch>=2.3.0,<2.4.0" "torchaudio>=2.3.0,<2.4.0"
python -m pip install -r requirements.txt
```

如果系统没有 `ffmpeg`，先安装：

```bash
brew install ffmpeg
```

## 5. HuggingFace 配置

说话人分离使用 `pyannote/speaker-diarization-3.1`，首次使用前需要：

1. 注册 HuggingFace 账号
2. 同意以下模型协议：
   - `pyannote/speaker-diarization-3.1`
   - `pyannote/segmentation-3.0`
3. 创建只读 token
4. 设置环境变量：

```bash
export HF_TOKEN="hf_xxx"
```

Windows：

```bat
setx HF_TOKEN "hf_xxx"
```

## 6. 验证部署是否完整

### 6.1 基础导入检查

```bash
python -m py_compile video_dubbing.py video_subtitles_only.py speaker_aware_dubbing.py gender_classifier.py test_diarization.py
```

### 6.2 查看 help

```bash
python video_dubbing.py --help
python video_subtitles_only.py --help
```

### 6.3 性别模型目录检查

确认以下文件存在：

```text
voice-gender-classifier/model.py
voice-gender-classifier/README.md
```

## 7. 常见部署问题

### 7.1 缺少 `voice-gender-classifier/model.py`

现象：

```text
[模型] 完整模型失败(本地性别模型文件不存在: ...)，使用 embedding 统计
```

原因：部署包不完整。  
修复：把整个 `voice-gender-classifier/` 目录一起复制。

### 7.2 `numpy 2.x detected`

原因：某些包把 numpy 升到了 2.x。  
修复：

```bash
pip install --force-reinstall "numpy>=1.24.0,<2.0.0"
```

### 7.3 `No espeak backend found`

原因：使用英语 VITS 声音，但未安装 `espeak-ng`。  
修复：安装 `espeak-ng`，然后重新打开终端。

### 7.4 `401 Unauthorized` / `Repository not found`

原因：`HF_TOKEN` 未设置、无效，或未先同意 pyannote 模型协议。  
修复：重新配置 token，并确认已同意模型协议。
