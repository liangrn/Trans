# Windows 增量升级指南

本文档面向已经部署过旧版主环境的用户。目标是在不重装整套环境的前提下，覆盖新代码，并补齐当前版本需要的独立运行环境。

当前用户环境前提：

```text
myenv           主环境：翻译、TTS、说话人分离、ASR 兜底
pretrained_models
voice-gender-classifier
separation_env  新增环境：人声分离：audio-separator[cpu]
ocr_env         新增环境：硬字幕 OCR：PaddleOCR
```

其中 `myenv`、`pretrained_models`、`voice-gender-classifier` 已经存在；本次增量升级只需要覆盖新代码，并新增 `separation_env` 和 `ocr_env`。

## 1. 升级前备份

在项目目录外备份旧目录。下面这些目录应当保留：

```text
myenv\
pretrained_models\
voice-gender-classifier\
```

如果用户项目目录里已有自己的媒体文件或输出文件，也一并保留。输入和输出路径不要求叫 `input` 或 `output`。

## 2. 覆盖新代码

你拿到的升级包应包含全量代码和本文档。把升级包里的代码文件覆盖到旧部署目录即可。用户可用任意输入文件和任意输出路径运行命令。

覆盖时保留用户已有目录：

```text
myenv\
pretrained_models\
voice-gender-classifier\
```

如果旧部署目录里还有用户自己的媒体目录或输出目录，也需要保留。不要用升级包里的空目录覆盖这些目录。

同时确认下面目录存在：

```text
voice-gender-classifier\model.py
voice-gender-classifier\README.md
```

缺少 `voice-gender-classifier\model.py` 时程序仍可运行，但男女声识别会降级，准确率会变差。

## 3. 检查现有主环境 myenv

打开命令行，进入项目目录（举例）：

```bat
cd /d D:\your\Trans
```

先检查现有 `myenv` 是否能直接运行新代码（只是用来校验当前环境是否正常）：

```bat
myenv\Scripts\python -m py_compile video_dubbing.py video_subtitles_only.py audio_separation.py ocr_recognition.py asr_recognition.py speaker_aware_dubbing.py gender_classifier.py
myenv\Scripts\python video_dubbing.py --help
myenv\Scripts\python video_subtitles_only.py --help
myenv\Scripts\python -c "from TTS.api import TTS; from faster_whisper import WhisperModel; print('main ok')"
```

如果这些检查通过，`myenv` 不需要升级。

只有在这些检查失败时，才补装或升级主环境依赖：

```bat
myenv\Scripts\python -m pip install --upgrade pip setuptools wheel
myenv\Scripts\python -m pip install --prefer-binary "numpy>=1.24.0,<2.0.0"
myenv\Scripts\python -m pip install --prefer-binary "torch>=2.3.0,<2.4.0" "torchaudio>=2.3.0,<2.4.0" "torchvision>=0.18.0,<0.19.0" --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
myenv\Scripts\python -m pip install --prefer-binary -r requirements.txt
```

## 4. 创建或更新人声分离环境 separation_env

用户当前目录里还没有 `separation_env`，直接创建（在主目录CMD命令行执行）：

```bat
python -m venv separation_env
```

安装或更新依赖：

```bat
separation_env\Scripts\python -m pip install --upgrade pip setuptools wheel
separation_env\Scripts\python -m pip install --prefer-binary imageio-ffmpeg
separation_env\Scripts\python -m pip install --prefer-binary "audio-separator[cpu]"
```

验证：

```bat
separation_env\Scripts\audio-separator.exe --env_info
```

如果验证失败，不要改主环境。删除 `separation_env` 后只重建这个环境：

```bat
rmdir /s /q separation_env
python -m venv separation_env
separation_env\Scripts\python -m pip install --upgrade pip setuptools wheel
separation_env\Scripts\python -m pip install --prefer-binary imageio-ffmpeg
separation_env\Scripts\python -m pip install --prefer-binary "audio-separator[cpu]"
```

## 5. 创建或更新 OCR 环境 ocr_env

用户当前目录里还没有 `ocr_env`，直接创建（在主目录CMD命令行执行）：

```bat
python -m venv ocr_env
```

安装或更新依赖：

```bat
ocr_env\Scripts\python -m pip install --upgrade pip setuptools wheel
ocr_env\Scripts\python -m pip install --prefer-binary "paddlepaddle==3.3.0" "paddleocr>=3.3.0,<3.4.0" opencv-python-headless
```

验证：

```bat
ocr_env\Scripts\python -c "import os; os.environ['FLAGS_use_mkldnn']='0'; import paddle; paddle.utils.run_check(); from paddleocr import PaddleOCR; import cv2; print('PaddleOCR OK')"
```

如果验证失败，或运行时出现 `ConvertPirAttribute2RuntimeAttribute` / `onednn_instruction.cc`，不要改主环境。删除 `ocr_env` 后只重建这个环境：

```bat
rmdir /s /q ocr_env
python -m venv ocr_env
ocr_env\Scripts\python -m pip install --upgrade pip setuptools wheel
ocr_env\Scripts\python -m pip install --prefer-binary "paddlepaddle==3.3.0" "paddleocr>=3.3.0,<3.4.0" opencv-python-headless
```

说明：程序运行 OCR 时会自动禁用 PaddleOCR oneDNN/MKLDNN 推理路径，优先保证 Windows CPU 稳定性。OCR 不可用时会自动退回主环境里的 faster-whisper ASR。

## 5.1 可选：预热 HuggingFace 性别模型缓存

首次运行说话人/男女声识别时，程序会从 HuggingFace 下载模型。网络慢时可能看到 timeout/retry 日志，这是首次缓存过程，不代表程序错误。可以提前执行：

```bat
myenv\Scripts\python -m pip install "hf_xet>=1.1.0"
myenv\Scripts\python -c "from speechbrain.inference.classifiers import EncoderClassifier; EncoderClassifier.from_hparams(source='speechbrain/spkrec-ecapa-voxceleb', savedir='pretrained_models/spkrec-ecapa-voxceleb', run_opts={'device':'cpu'}); print('ECAPA cache OK')"
myenv\Scripts\python -c "import importlib.util, pathlib, torch; p=pathlib.Path('voice-gender-classifier/model.py'); spec=importlib.util.spec_from_file_location('vgc_model', p); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); model=m.ECAPA_gender.from_pretrained('JaesungHuh/voice-gender-classifier'); model.to(torch.device('cpu')); print('gender model cache OK')"
```

## 6. 验证主环境

```bat
myenv\Scripts\python -m py_compile video_dubbing.py video_subtitles_only.py audio_separation.py ocr_recognition.py asr_recognition.py speaker_aware_dubbing.py gender_classifier.py
myenv\Scripts\python video_dubbing.py --help
myenv\Scripts\python video_subtitles_only.py --help
myenv\Scripts\python -c "from faster_whisper import WhisperModel; print('faster-whisper OK')"
myenv\Scripts\python -c "from TTS.api import TTS; print('coqui-tts OK')"
```

## 7. 清理旧阶段缓存

新版流程引入人声分离、OCR 优先、TTS 缓存修正。覆盖代码后，建议清理旧输出任务的阶段缓存，避免复用旧中间产物。

阶段目录总是：

```text
<输出视频所在目录>\<输出文件名不含扩展名>\
```

单文件示例：

```bat
myenv\Scripts\python video_dubbing.py --mode single --input_video D:\media\movie.mp4 --target_lang en --voice en_vctk_vits_m001 --output_video D:\result\movie_en.mp4
```

上面命令的阶段目录是：

```text
D:\result\movie_en\
```

批处理示例：

```bat
myenv\Scripts\python video_dubbing.py --mode batch --input_dir D:\media --output_dir D:\result --target_lang en --voice en_vctk_vits_m001
```

批处理会给每个实际输出视频生成独立阶段目录：

```text
D:\result\<输出文件名不含扩展名>\
```

如果需要清理某个输出任务的旧阶段缓存，进入对应阶段目录后删除：

```bat
rmdir /s /q 01_audio
rmdir /s /q 02_recognition
rmdir /s /q 03_speaker_gender
rmdir /s /q 05_tts
rmdir /s /q 06_composition
```

如果你想保留翻译缓存，可以不删 `04_translation`。如果怀疑翻译内容也需要重跑，再删除 `04_translation`。

如果你刚调整了原视频 clone 策略，或者想强制让旧 clone 参考音频重新采样，再额外删除：

```bat
rmdir /s /q 05_tts\speaker_refs
```

## 8. 运行升级后的验证视频

先用一个你自己的短视频验证：

```bat
myenv\Scripts\python video_dubbing.py --mode single --input_video D:\media\short.mp4 --target_lang en --voice en_vctk_vits_m001 --output_video D:\result\short_en.mp4
```

如果你想先关闭原视频 speaker clone 再验证，可以加上：

```bat
myenv\Scripts\python video_dubbing.py --mode single --input_video D:\media\short.mp4 --target_lang en --voice en_vctk_vits_m001 --output_video D:\result\short_en.mp4 --clone_voice false
```

再验证目标视频：

```bat
myenv\Scripts\python video_dubbing.py --mode single --input_video D:\media\movie.mp4 --target_lang en --voice en_vctk_vits_m001 --output_video D:\result\movie_en.mp4
```

## 9. 什么时候改用全量安装脚本

以下情况建议直接运行 `install_windows.bat`，比手工排查更快：

- 主环境里 numpy 被升级到 2.x。
- torch / torchvision 版本不匹配。
- `coqui-tts` 无法 import。
- 增量步骤执行后，`myenv` 仍有多个 import 失败。

运行全量脚本不会主动删除用户媒体文件或输出文件，但会继续使用或补齐当前目录下的环境。

## 10. 常见问题

**是否需要 asr_env？**  
不需要。ASR 兜底使用主环境中的 `faster-whisper`。

**是否能把 separation_env 和 ocr_env 合并？**  
不建议。`audio-separator[cpu]`、PaddleOCR、OpenCV 属于不同二进制栈，合并会增加 Windows DLL 和 wheel 冲突概率。

**是否必须使用 input 和 output 目录？**  
不需要。输入和输出路径完全由命令行参数决定。

**是否要删除旧输出目录？**  
不需要。只清理对应输出文件名下面的阶段目录即可。

**是否要重新设置 HF_TOKEN？**  
如果 Windows 系统环境变量已经设置过，不需要。可用下面命令检查：

```bat
echo %HF_TOKEN%
```

为空时重新设置：

```bat
setx HF_TOKEN "hf_xxxxxxxxxxxxxxxxxxxxxxxx"
```
