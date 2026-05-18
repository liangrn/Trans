@echo off
setlocal EnableDelayedExpansion

if "%1" == "--child" goto :main
cmd /k "%~f0" --child
exit /b

:main
set "SCRIPT_DIR=%~dp0"
set LOG_FILE=%~dp0install_windows.log
set "VENV_DIR=%~dp0trans_env"
set "VENV_PY=%~dp0trans_env\Scripts\python.exe"
set "SEP_ENV_DIR=%~dp0separation_env"
set "SEP_PY=%~dp0separation_env\Scripts\python.exe"
set "SEP_BIN=%~dp0separation_env\Scripts\audio-separator.exe"
set "OCR_ENV_DIR=%~dp0ocr_env"
set "OCR_PY=%~dp0ocr_env\Scripts\python.exe"
set "TORCH_INDEX=https://download.pytorch.org/whl/cpu"
set "ONNX_PKG=onnxruntime"
set "LAST_STEP="

> "%LOG_FILE%" echo ==============================================================
>>"%LOG_FILE%" echo Video Dubbing Tool - Windows Install Log
>>"%LOG_FILE%" echo Started: %date% %time%
>>"%LOG_FILE%" echo Script Dir: %SCRIPT_DIR%
>>"%LOG_FILE%" echo ==============================================================

echo.
echo ==============================================================
echo   Video Dubbing Tool  --  Windows Install Script
echo ==============================================================
echo.
echo Log file: %LOG_FILE%
echo.

call :check_required_file "requirements.txt"
call :check_required_file "video_dubbing.py"
call :check_required_file "video_subtitles_only.py"
call :check_required_file "speaker_aware_dubbing.py"
call :check_required_file "gender_classifier.py"
call :check_required_file "audio_separation.py"
call :check_required_file "asr_recognition.py"
call :check_required_file "ocr_recognition.py"
call :check_required_file "ocr_subtitle_probe.py"
call :check_required_file "voice-gender-classifier\model.py"
call :check_required_file "voice-gender-classifier\README.md"

echo [Step 0/11] Checking Python environment...
python --version >nul 2>&1
if errorlevel 1 call :fail "Python not found. Install Python 3.10 or 3.11 and add it to PATH."

for /f "tokens=2" %%v in ('python --version 2^>^&1') do set "PY_VER=%%v"
for /f "tokens=1,2 delims=." %%a in ("!PY_VER!") do (
    set "PY_MAJOR=%%a"
    set "PY_MINOR=%%b"
)
echo  [OK] Python !PY_VER! found
>>"%LOG_FILE%" echo Python Version: !PY_VER!

if not "!PY_MAJOR!"=="3" call :fail "Python 3.x required, found !PY_VER!."
if !PY_MINOR! LSS 10 call :fail "Python 3.10 or 3.11 required, found !PY_VER!."
if !PY_MINOR! GTR 11 call :fail "Python 3.10 or 3.11 required, found !PY_VER!."

python -m pip --version >nul 2>&1
if errorlevel 1 (
    echo  [WARN] pip not available, trying ensurepip...
    >>"%LOG_FILE%" echo Running: python -m ensurepip --upgrade
    python -m ensurepip --upgrade >>"%LOG_FILE%" 2>&1
    if errorlevel 1 call :fail "pip bootstrap failed. Install pip manually and retry."
)
echo  [OK] pip available

echo.
echo [Step 1/11] Checking deployment package...
echo  [OK] Required project files exist

echo.
echo [Step 2/11] Setting up main virtual environment...
if exist "%VENV_PY%" (
    echo  [OK] Virtual environment already exists: %VENV_DIR%
) else (
    echo  Creating virtual environment at: %VENV_DIR%
    >>"%LOG_FILE%" echo Running: python -m venv "%VENV_DIR%"
    python -m venv "%VENV_DIR%" >>"%LOG_FILE%" 2>&1
    if errorlevel 1 call :fail "Failed to create virtual environment."
)

call "%VENV_DIR%\Scripts\activate.bat" >>"%LOG_FILE%" 2>&1
if not exist "%VENV_PY%" call :fail "Virtual environment python not found at %VENV_PY%."
echo  [OK] Activated: %VENV_DIR%

echo.
echo [Step 3/11] Selecting CPU-only dependency mode...
echo  [OK] CPU-only mode selected
echo    TORCH_INDEX = !TORCH_INDEX!
echo    ONNX_PKG    = !ONNX_PKG!
>>"%LOG_FILE%" echo TORCH_INDEX=!TORCH_INDEX!
>>"%LOG_FILE%" echo ONNX_PKG=!ONNX_PKG!

echo.
echo [Step 4/11] Checking optional runtime requirements...
set "ESPEAK_OK=0"
where espeak-ng >nul 2>&1 && set "ESPEAK_OK=1"
if !ESPEAK_OK! EQU 0 if exist "C:\Program Files\eSpeak NG\espeak-ng.exe" (
    set "ESPEAK_OK=1"
    set "PATH=%PATH%;C:\Program Files\eSpeak NG"
)
if !ESPEAK_OK! EQU 0 if exist "C:\Program Files (x86)\eSpeak NG\espeak-ng.exe" (
    set "ESPEAK_OK=1"
    set "PATH=%PATH%;C:\Program Files (x86)\eSpeak NG"
)

if !ESPEAK_OK! EQU 1 (
    echo  [OK] espeak-ng available
) else (
    echo  [WARN] espeak-ng not found. English TTS voices will not work until it is installed.
    echo         Download: https://github.com/espeak-ng/espeak-ng/releases
)

if "%HF_TOKEN%"=="" (
    echo  [WARN] HF_TOKEN is not set. Speaker diarization will be skipped until you configure it.
    >>"%LOG_FILE%" echo HF_TOKEN is not set
) else (
    echo  [OK] HF_TOKEN detected
    >>"%LOG_FILE%" echo HF_TOKEN detected
)

where ffmpeg >nul 2>&1
if errorlevel 1 (
    echo  [WARN] ffmpeg not found in PATH. This is acceptable; imageio-ffmpeg will manage it automatically.
    >>"%LOG_FILE%" echo ffmpeg not found in PATH; relying on imageio-ffmpeg
) else (
    echo  [OK] ffmpeg found in PATH
    >>"%LOG_FILE%" echo ffmpeg found in PATH
)

echo.
echo [Step 5/11] Upgrading packaging tools...
call :run_pip_install "Upgrade pip/setuptools/wheel" --upgrade pip setuptools wheel

echo.
echo [Step 6/11] Installing core numeric and torch packages...
call :run_pip_install "Install numpy < 2.0" "numpy>=1.24.0,^<2.0.0"
call :run_python_check "Verify numpy stays on 1.x" "import numpy as np; v=np.__version__; major=int(v.split('.')[0]); print('  [OK] numpy ' + v) if major < 2 else (_ for _ in ()).throw(SystemExit(1))"
call :run_pip_install "Install CPU torch stack" "torch>=2.3.0,^<2.4.0" "torchaudio>=2.3.0,^<2.4.0" "torchvision>=0.18.0,^<0.19.0" --index-url !TORCH_INDEX! --extra-index-url https://pypi.org/simple
call :run_python_check "Verify torch import" "import torch; print('  [OK] torch ' + torch.__version__)"
call :run_pip_install "Install onnxruntime" "!ONNX_PKG!>=1.17.0,^<2.0.0"

echo.
echo [Step 7/11] Installing application dependencies...
call :run_pip_install "Pin transformers 4.x" "transformers>=4.40.0,^<5.0.0"
call :run_pip_install "Install faster-whisper" "faster-whisper>=1.1.0,^<2.0.0"
call :run_pip_install "Install coqui-tts" "coqui-tts>=0.24.0,^<1.0.0"
call :run_pip_install "Install soundfile" "soundfile>=0.12.0,^<0.13.0"
call :run_pip_install "Install librosa" "librosa>=0.10.0,^<1.0.0"
call :run_pip_install "Install imageio + imageio-ffmpeg" "imageio>=2.9.0,^<3.0.0" "imageio-ffmpeg>=0.4.9"
call :run_pip_install "Install moviepy 1.0.3" "moviepy==1.0.3"
call :run_pip_install "Install deep_translator" "deep_translator>=1.11.0,^<2.0.0"
call :run_pip_install "Re-pin transformers 4.x" "transformers>=4.40.0,^<5.0.0"
call :run_pip_install "Install huggingface_hub" "huggingface_hub>=0.23.0,^<2.0.0"
call :run_pip_install "Install speechbrain" "speechbrain>=1.0.0,^<2.0.0"
call :run_pip_install "Install pyannote.audio" "pyannote.audio>=3.3.0,^<4.0.0"
call :run_pip_install "Install Pillow" "Pillow>=9.0.0,^<11.0.0"
call :run_pip_install "Install scipy" "scipy>=1.11.0,^<2.0.0"
call :run_pip_install "Install zhconv" "zhconv>=1.4.3"

echo.
echo [Step 8/11] Re-checking locked versions...
call :run_python_check "Verify numpy did not upgrade to 2.x" "import numpy as np; v=np.__version__; major=int(v.split('.')[0]); print('  [OK] numpy ' + v) if major < 2 else (_ for _ in ()).throw(SystemExit(1))"
call :run_python_check "Verify transformers remains 4.x" "import transformers; v=transformers.__version__; major=int(v.split('.')[0]); print('  [OK] transformers ' + v) if major < 5 else (_ for _ in ()).throw(SystemExit(1))"

echo.
echo [Step 9/11] Setting up vocal separation environment...
if exist "%SEP_PY%" (
    echo  [OK] Separation environment already exists: %SEP_ENV_DIR%
) else (
    echo  Creating separation environment at: %SEP_ENV_DIR%
    >>"%LOG_FILE%" echo Running: python -m venv "%SEP_ENV_DIR%"
    python -m venv "%SEP_ENV_DIR%" >>"%LOG_FILE%" 2>&1
    if errorlevel 1 call :fail "Failed to create separation_env."
)

if not exist "%SEP_PY%" call :fail "Separation environment python not found at %SEP_PY%."
>>"%LOG_FILE%" echo.
>>"%LOG_FILE%" echo [PIP] Install audio-separator[cpu] in separation_env
echo  [RUN] Install audio-separator[cpu]
"%SEP_PY%" -m pip install --upgrade pip setuptools wheel >>"%LOG_FILE%" 2>&1
if errorlevel 1 call :fail "Failed to upgrade pip in separation_env."
"%SEP_PY%" -m pip install --prefer-binary imageio-ffmpeg >>"%LOG_FILE%" 2>&1
if errorlevel 1 call :fail "Failed to install imageio-ffmpeg in separation_env."
"%SEP_PY%" -m pip install --prefer-binary "audio-separator[cpu]" >>"%LOG_FILE%" 2>&1
if errorlevel 1 (
    echo  [WARN] audio-separator[cpu] failed on first attempt, retrying once...
    >>"%LOG_FILE%" echo [RETRY] Install audio-separator[cpu]
    "%SEP_PY%" -m pip install --prefer-binary "audio-separator[cpu]" >>"%LOG_FILE%" 2>&1
    if errorlevel 1 call :fail "Failed to install audio-separator[cpu] in separation_env."
)
if not exist "%SEP_BIN%" call :fail "audio-separator executable not found at %SEP_BIN%."
echo  [OK] audio-separator[cpu] installed in separation_env

echo.
echo.
echo [Step 10/11] Setting up OCR environment...
if exist "%OCR_PY%" (
    echo  [OK] OCR environment already exists: %OCR_ENV_DIR%
) else (
    echo  Creating OCR environment at: %OCR_ENV_DIR%
    >>"%LOG_FILE%" echo Running: python -m venv "%OCR_ENV_DIR%"
    python -m venv "%OCR_ENV_DIR%" >>"%LOG_FILE%" 2>&1
    if errorlevel 1 call :fail "Failed to create ocr_env."
)

if not exist "%OCR_PY%" call :fail "OCR environment python not found at %OCR_PY%."
>>"%LOG_FILE%" echo.
>>"%LOG_FILE%" echo [PIP] Install PaddleOCR in ocr_env
echo  [RUN] Install PaddleOCR
"%OCR_PY%" -m pip install --upgrade pip setuptools wheel >>"%LOG_FILE%" 2>&1
if errorlevel 1 call :fail "Failed to upgrade pip in ocr_env."
"%OCR_PY%" -m pip install --prefer-binary paddleocr paddlepaddle opencv-python-headless >>"%LOG_FILE%" 2>&1
if errorlevel 1 (
    echo  [WARN] PaddleOCR install failed on first attempt, retrying once...
    >>"%LOG_FILE%" echo [RETRY] Install PaddleOCR
    "%OCR_PY%" -m pip install --prefer-binary paddleocr paddlepaddle opencv-python-headless >>"%LOG_FILE%" 2>&1
    if errorlevel 1 call :fail "Failed to install PaddleOCR in ocr_env."
)
echo  [OK] PaddleOCR installed in ocr_env

echo.
echo [Step 11/11] Running post-install verification...
set "IMPORT_FAIL=0"
call :run_python_check "Check torch" "import torch; print('  [OK] torch ' + torch.__version__)" || set "IMPORT_FAIL=1"
call :run_python_check "Check torchaudio" "import torchaudio; print('  [OK] torchaudio ' + torchaudio.__version__)" || set "IMPORT_FAIL=1"
call :run_python_check "Check transformers" "import transformers; print('  [OK] transformers ' + transformers.__version__)" || set "IMPORT_FAIL=1"
call :run_python_check "Check coqui-tts" "from TTS.api import TTS; print('  [OK] coqui-tts')" || set "IMPORT_FAIL=1"
call :run_python_check "Check pyannote.audio" "from pyannote.audio import Pipeline; print('  [OK] pyannote.audio')" || set "IMPORT_FAIL=1"
call :run_python_check "Check moviepy 1.x" "from moviepy.editor import VideoFileClip; print('  [OK] moviepy 1.x')" || set "IMPORT_FAIL=1"
call :run_python_check "Check deep_translator" "from deep_translator import GoogleTranslator; print('  [OK] deep_translator')" || set "IMPORT_FAIL=1"
call :run_python_check "Check speechbrain" "import speechbrain; print('  [OK] speechbrain')" || set "IMPORT_FAIL=1"
call :run_python_check "Check faster-whisper" "from faster_whisper import WhisperModel; print('  [OK] faster-whisper')" || set "IMPORT_FAIL=1"
call :run_python_check "Check librosa" "import librosa; print('  [OK] librosa')" || set "IMPORT_FAIL=1"
call :run_python_check "Check soundfile" "import soundfile; print('  [OK] soundfile')" || set "IMPORT_FAIL=1"
call :run_python_check "Check imageio-ffmpeg" "import imageio_ffmpeg; print('  [OK] imageio-ffmpeg => ' + imageio_ffmpeg.get_ffmpeg_exe())" || set "IMPORT_FAIL=1"
call :run_python_check "Check dubbing help" "import subprocess, sys; subprocess.run([sys.executable, 'video_dubbing.py', '--help'], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL); print('  [OK] python video_dubbing.py --help')" || set "IMPORT_FAIL=1"
call :run_python_check "Check subtitles help" "import subprocess, sys; subprocess.run([sys.executable, 'video_subtitles_only.py', '--help'], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL); print('  [OK] python video_subtitles_only.py --help')" || set "IMPORT_FAIL=1"
echo  [RUN] Check PaddleOCR import
>>"%LOG_FILE%" echo.
>>"%LOG_FILE%" echo [CHECK] PaddleOCR import
"%OCR_PY%" -c "from paddleocr import PaddleOCR; import cv2; print('  [OK] PaddleOCR import')" >>"%LOG_FILE%" 2>&1
if errorlevel 1 (
    echo  [ERROR] PaddleOCR import
    set "IMPORT_FAIL=1"
) else (
    echo  [OK] PaddleOCR import
)
echo  [RUN] Check audio-separator --env_info
>>"%LOG_FILE%" echo.
>>"%LOG_FILE%" echo [CHECK] audio-separator --env_info
"%SEP_PY%" -c "import os, subprocess, imageio_ffmpeg; ffmpeg=imageio_ffmpeg.get_ffmpeg_exe(); os.environ['PATH']=os.path.dirname(ffmpeg)+os.pathsep+os.environ.get('PATH',''); subprocess.run([r'%SEP_BIN%', '--env_info'], check=True)" >>"%LOG_FILE%" 2>&1
if errorlevel 1 (
    echo  [ERROR] audio-separator --env_info
    set "IMPORT_FAIL=1"
) else (
    echo  [OK] audio-separator --env_info
)

echo.
if !IMPORT_FAIL! EQU 1 (
    call :fail "Post-install verification failed. See %LOG_FILE% for details."
)

echo ==============================================================
echo   [SUCCESS] All dependencies installed and verified.
echo ==============================================================
echo.
echo Next steps:
echo   1. Set HF_TOKEN if you need speaker diarization:
echo      setx HF_TOKEN "hf_xxxxxxxxxxxxxxxxxxxxxxxx"
echo   2. Activate the environment before use:
echo      trans_env\Scripts\activate
echo   3. Vocal separation runs through:
echo      separation_env\Scripts\audio-separator.exe
echo   4. OCR runs through:
echo      ocr_env\Scripts\python.exe
echo   5. ASR fallback uses faster-whisper in trans_env.
echo   6. First run example:
echo      python video_dubbing.py --mode single --input_video input.mp4 --target_lang en --voice en_vctk_vits_m001 --output_video output.mp4
echo.
echo Installation log saved to:
echo   %LOG_FILE%
echo.
pause
endlocal
exit /b 0

:check_required_file
if exist "%SCRIPT_DIR%%~1" (
    echo  [OK] Found %~1
    >>"%LOG_FILE%" echo Found required file: %~1
    exit /b 0
)
call :fail "Deployment package is incomplete. Missing required file: %~1"
exit /b 1

:run_pip_install
set "LAST_STEP=%~1"
echo  [RUN] !LAST_STEP!
>>"%LOG_FILE%" echo.
>>"%LOG_FILE%" echo [PIP] !LAST_STEP!
shift
"%VENV_PY%" -m pip install --prefer-binary %* >>"%LOG_FILE%" 2>&1
if errorlevel 1 (
    echo  [WARN] !LAST_STEP! failed on first attempt, retrying once...
    >>"%LOG_FILE%" echo [RETRY] !LAST_STEP!
    "%VENV_PY%" -m pip install --prefer-binary %* >>"%LOG_FILE%" 2>&1
    if errorlevel 1 call :fail "!LAST_STEP! failed. See %LOG_FILE%."
)
echo  [OK] !LAST_STEP!
exit /b 0

:run_python_check
set "LAST_STEP=%~1"
echo  [RUN] !LAST_STEP!
>>"%LOG_FILE%" echo.
>>"%LOG_FILE%" echo [PYTHON] !LAST_STEP!
"%VENV_PY%" -c "%~2" >>"%LOG_FILE%" 2>&1
if errorlevel 1 (
    echo  [ERROR] !LAST_STEP!
    exit /b 1
)
echo  [OK] !LAST_STEP!
exit /b 0

:fail
echo.
echo ==============================================================
echo   [FAILED] %~1
if not "%LAST_STEP%"=="" echo   Last step: %LAST_STEP%
echo   Log file: %LOG_FILE%
echo ==============================================================
echo.
>>"%LOG_FILE%" echo.
>>"%LOG_FILE%" echo [FAILED] %~1
if not "%LAST_STEP%"=="" >>"%LOG_FILE%" echo Last step: %LAST_STEP%
pause
endlocal
exit /b 1
