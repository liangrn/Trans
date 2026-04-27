# Windows 部署指引（视频配音翻译工具）

## 前置要求

- Windows 10/11 64位
- Python 3.10（**必须是 3.10**，不要用 3.11/3.12，TTS 与新版不兼容）
  - 下载：https://www.python.org/downloads/release/python-31011/
  - 安装时勾选 **"Add Python to PATH"**
- FFmpeg
  - 下载：https://github.com/BtbN/FFmpeg-Builds/releases → ffmpeg-master-latest-win64-gpl.zip
  - 解压后把 `bin` 目录加入系统 PATH
- Git（用于克隆性别分类模型）
  - 下载：https://git-scm.com/download/win

---

## 第一步：创建虚拟环境

```cmd
cd D:\video_trans
python -m venv trans_env
trans_env\Scripts\activate
pip install --upgrade pip
```

---

## 第二步：按顺序安装依赖（顺序很重要）

### 2.1 先装 numpy 和 pandas（锁定版本）
```cmd
pip install "numpy==1.26.4"
pip install "pandas==1.5.3"
```

### 2.2 装 PyTorch（CPU版，无显卡用这个）
```cmd
pip install torch==2.1.0+cpu torchaudio==2.1.0+cpu --index-url https://download.pytorch.org/whl/cpu
```

如果有 NVIDIA 显卡（CUDA 11.8）：
```cmd
pip install torch==2.1.0+cu118 torchaudio==2.1.0+cu118 --index-url https://download.pytorch.org/whl/cu118
```

### 2.3 装 TTS（Coqui）
```cmd
pip install "TTS==0.22.0" --no-deps
pip install "coqpit==0.0.16"
pip install "trainer" "anyascii" "bangla" "bnnumerizer" "bnunicodenormalizer"
pip install "gruut==2.2.3"
```

### 2.4 装 faster-whisper（语音识别）
```cmd
pip install "faster-whisper==1.0.3"
```

### 2.5 装 pyannote（说话人分离）
```cmd
pip install "pyannote.audio==3.3.2"
```

> 注意：用 3.3.2 而不是 4.x，3.3.2 对 numpy 1.x 兼容性更好

### 2.6 装 speechbrain（性别识别）
```cmd
pip install "speechbrain==0.5.16"
```

> 用 0.5.16，这是最后一个不依赖新版 torchaudio API 的稳定版本

### 2.7 装其他依赖
```cmd
pip install "deep-translator==1.11.4"
pip install "moviepy==1.0.3"
pip install "Pillow>=9.0"
pip install "librosa==0.10.1"
pip install "soundfile==0.12.1"
pip install "zhconv==1.4.3"
pip install "huggingface-hub==0.19.4"
pip install "transformers==4.36.0"
```

---

## 第三步：下载性别分类模型

```cmd
cd D:\video_trans
git clone https://huggingface.co/JaesungHuh/ecapa-gender voice-gender-classifier
```

如果 git clone 慢，也可以直接从 HuggingFace 下载压缩包：
https://huggingface.co/JaesungHuh/ecapa-gender/tree/main

---

## 第四步：复制程序文件

把以下文件复制到 `D:\video_trans\`：
- `video_dubbing.py`
- `video_subtitles_only.py`
- `speaker_aware_dubbing.py`
- `gender_classifier.py`

---

## 第五步：设置 HuggingFace Token

说话人分离需要 HuggingFace Token，并且需要在以下两个模型页面点击同意协议：
- https://huggingface.co/pyannote/speaker-diarization-3.1
- https://huggingface.co/pyannote/segmentation-3.0

设置 Token（命令行临时设置）：
```cmd
set HF_TOKEN=你的token
```

永久设置（推荐）：
- 右键"此电脑" → 属性 → 高级系统设置 → 环境变量
- 新建系统变量：变量名 `HF_TOKEN`，变量值填你的 token

---

## 第六步：验证安装

```cmd
python -c "
import torch, torchaudio, TTS, faster_whisper
import pyannote.audio, speechbrain
print('torch:', torch.__version__)
print('TTS OK')
print('pyannote OK')
print('speechbrain OK')
print('全部OK')
"
```

```cmd
python -c "
from TTS.api import TTS
t = TTS('tts_models/en/vctk/vits')
print('speakers:', t.speakers[:3])
"
```

---

## 第七步：运行

```cmd
cd D:\video_trans
trans_env\Scripts\activate

python video_dubbing.py --mode single ^
  --input_video D:\input\video.mp4 ^
  --target_lang en ^
  --voice en_vctk_vits_m001 ^
  --output_video D:\output\video_dubbed.mp4
```

---

## 常见问题

### Q: `'grep' 不是内部命令`
Windows 用 `findstr` 代替：
```cmd
pip show TTS | findstr Version
```

### Q: `UnicodeEncodeError: gbk`
Windows 控制台编码问题，不影响程序运行，忽略即可。或在运行前设置：
```cmd
chcp 65001
```

### Q: `KMP_DUPLICATE_LIB_OK` 相关崩溃
程序代码里已自动处理，无需手动设置。

### Q: 首次运行很慢
第一次运行会自动下载模型文件（约 2-3 GB），需要网络连接，下载完后会缓存本地。

### Q: 说话人分离报错
确认已设置 `HF_TOKEN` 且在 HuggingFace 上接受了模型协议。
未设置 Token 时程序会自动降级为单一声音模式，仍然可以运行。

---

## 目录结构

```
D:\video_trans\
├── trans_env\              ← 虚拟环境
├── voice-gender-classifier\ ← 性别分类模型
├── pretrained_models\      ← 自动下载的模型缓存
├── video_dubbing.py
├── video_subtitles_only.py
├── speaker_aware_dubbing.py
└── gender_classifier.py
```
