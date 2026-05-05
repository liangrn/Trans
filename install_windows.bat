@echo off
setlocal EnableDelayedExpansion

:: Safety net: if the script exits unexpectedly, pause so the window stays open
:: (only effective when launched by double-click, not from an existing cmd window)
if "%1" == "--child" goto :main
cmd /k "%~f0" --child
exit /b

:main

:: ============================================================
::  Video Dubbing Tool - Windows Install Script
::  Supports: Python 3.9/3.10/3.11, CPU-only deployment
:: ============================================================

echo.
echo ==============================================================
echo   Video Dubbing Tool  --  Windows Install Script
echo ==============================================================
echo.

:: ------------------------------------------------------------
:: Step 0: Basic environment check
:: ------------------------------------------------------------
echo [Step 0/8] Checking basic environment...

python --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo  [ERROR] Python not found!
    echo    Please install Python 3.10 or 3.11:
    echo    https://www.python.org/downloads/
    echo    Make sure to check "Add Python to PATH" during install.
    echo.
    pause
    exit /b 1
)

for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PY_VER=%%v
for /f "tokens=1,2 delims=." %%a in ("!PY_VER!") do (
    set PY_MAJOR=%%a
    set PY_MINOR=%%b
)
echo  [OK] Python !PY_VER! found

if !PY_MAJOR! NEQ 3 (
    echo  [ERROR] Python 3.x required, found !PY_VER!
    pause & exit /b 1
)
if !PY_MINOR! LSS 9 (
    echo  [ERROR] Python 3.9+ required, found !PY_VER!
    pause & exit /b 1
)
if !PY_MINOR! GTR 11 (
    echo  [WARN] Python !PY_VER! is outside tested range (3.10/3.11 recommended)
    echo    Some pre-built wheels may not be available.
    echo    Press any key to continue, or close window to exit.
    pause
)

python -m pip --version >nul 2>&1
if errorlevel 1 (
    echo  [ERROR] pip not available, trying to fix...
    python -m ensurepip --upgrade
    if errorlevel 1 (
        echo  [ERROR] pip install failed, please install pip manually.
        pause & exit /b 1
    )
)
echo  [OK] pip available

:: ------------------------------------------------------------
:: Step 0b: Create and activate virtual environment
:: ------------------------------------------------------------
echo.
echo [Step 0b] Setting up virtual environment...

:: %~dp0 always ends with backslash; use it directly as the base path
:: No string trimming needed - just append the venv folder name
set VENV_DIR=%~dp0trans_env
set VENV_PY=%~dp0trans_env\Scripts\python.exe

if exist "%VENV_DIR%\Scripts\python.exe" (
    echo  [OK] Virtual environment already exists: %VENV_DIR%
) else (
    echo  Creating virtual environment at: %VENV_DIR%
    python -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo  [ERROR] Failed to create virtual environment!
        pause & exit /b 1
    )
    echo  [OK] Virtual environment created
)

call "%VENV_DIR%\Scripts\activate.bat"

:: Verify activation by checking VENV_PY directly (errorlevel from activate.bat is unreliable)
if not exist "%VENV_PY%" (
    echo  [ERROR] Virtual environment python not found at: %VENV_PY%
    echo  Try deleting the trans_env folder and re-running this script.
    pause & exit /b 1
)
echo  [OK] Activated: %VENV_DIR%
echo.

:: ------------------------------------------------------------
:: Step 1: Select CPU-only dependency mode
:: ------------------------------------------------------------
echo.
echo [Step 1/8] Selecting CPU-only dependency mode...

set TORCH_INDEX=https://download.pytorch.org/whl/cpu
set ONNX_PKG=onnxruntime
echo  [OK] CPU-only mode selected for simpler Windows installs
echo    TORCH_INDEX = !TORCH_INDEX!
echo    ONNX_PKG    = !ONNX_PKG!

:: ------------------------------------------------------------
:: Step 1b: Check espeak-ng (required for English TTS voices)
:: ------------------------------------------------------------
echo.
echo [Step 1b/8] Checking espeak-ng...
set ESPEAK_OK=0

where espeak-ng >nul 2>&1
if not errorlevel 1 (
    set ESPEAK_OK=1
    echo  [OK] espeak-ng found in PATH
) else (
    if exist "C:\Program Files\eSpeak NG\espeak-ng.exe" (
        set ESPEAK_OK=1
        echo  [OK] espeak-ng found at default path
        set "PATH=%PATH%;C:\Program Files\eSpeak NG"
    ) else if exist "C:\Program Files (x86)\eSpeak NG\espeak-ng.exe" (
        set ESPEAK_OK=1
        echo  [OK] espeak-ng found (x86 path)
        set "PATH=%PATH%;C:\Program Files (x86)\eSpeak NG"
    )
)

if !ESPEAK_OK! EQU 0 (
    echo.
    echo  [WARN] espeak-ng not found!
    echo.
    echo  espeak-ng is REQUIRED for English TTS voices (en_vctk_vits_*, en_ljspeech_vits).
    echo  Without it, English speech synthesis will fail.
    echo.
    echo  Installation steps:
    echo    1. Go to: https://github.com/espeak-ng/espeak-ng/releases
    echo    2. Download the latest Windows .msi installer
    echo    3. Run the installer (keep default install path)
    echo    4. Re-open this command window and run this script again
    echo.
    echo  NOTE: If you only plan to use non-English voices (Korean/Japanese/Chinese
    echo        etc. via XTTS-v2), you can skip espeak-ng.
    echo.
    choice /C YN /M "Skip espeak-ng and continue? (only XTTS-v2 multilingual voices will work)"
    if errorlevel 2 (
        echo  Please install espeak-ng first, then re-run this script.
        pause & exit /b 1
    )
    echo  [WARN] Skipping espeak-ng. English en_vctk_vits_* voices will NOT work.
)

:: ------------------------------------------------------------
:: Step 2: Upgrade pip / setuptools / wheel
:: ------------------------------------------------------------
echo.
echo [Step 2/8] Upgrading pip / setuptools / wheel...
!VENV_PY! -m pip install --upgrade pip setuptools wheel
if errorlevel 1 (
    echo  [ERROR] pip upgrade failed, check network connection
    pause & exit /b 1
)
echo  [OK] pip upgraded

:: ------------------------------------------------------------
:: Step 3: Install numpy <2.0 FIRST (before torch)
:: ------------------------------------------------------------
echo.
echo [Step 3/8] Installing numpy less than 2.0 (numpy 2.x breaks torch/onnxruntime ABI)...
!VENV_PY! -m pip install "numpy>=1.24.0,^<2.0.0"
if errorlevel 1 (
    echo  [ERROR] numpy install failed
    pause & exit /b 1
)

!VENV_PY! -c "import numpy as np; v=np.__version__; ok=int(v.split('.')[0])^<2; print('  [OK] numpy ' + v) if ok else print('  [ERROR] numpy ' + v + ' >= 2.0')"
if errorlevel 1 (
    echo  numpy version check failed, force reinstalling...
    !VENV_PY! -m pip install --force-reinstall "numpy>=1.24.0,^<2.0.0"
)

:: ------------------------------------------------------------
:: Step 4: Install torch (from pytorch.org, MUST be before other packages)
:: ------------------------------------------------------------
echo.
echo [Step 4/8] Installing CPU torch (this may take a while)...
echo.

!VENV_PY! -m pip install "torch>=2.3.0,^<2.4.0" "torchaudio>=2.3.0,^<2.4.0" "torchvision>=0.18.0,^<0.19.0" --index-url !TORCH_INDEX! --extra-index-url https://pypi.org/simple

if errorlevel 1 (
    echo.
    echo  [ERROR] torch install failed! Common causes:
    echo    1. Network issue (pytorch.org can be slow)
    echo    2. Not enough disk space (need ~5 GB free)
    echo    3. Unsupported Python version (need 3.9/3.10/3.11)
    echo.
    echo  For China users:
    echo    pip install torch==2.3.1 torchaudio==2.3.1 -i https://mirrors.cloud.tencent.com/pypi/simple/
    pause & exit /b 1
)

python -c "import torch; v=torch.__version__; ok=v.startswith('2.3'); print('  [OK] torch ' + v) if ok else exit(1)"
if errorlevel 1 (
    echo  [ERROR] torch import or version check failed
    pause & exit /b 1
)
echo  [OK] torch verified

:: ------------------------------------------------------------
:: Step 5: Install onnxruntime
:: ------------------------------------------------------------
echo.
echo [Step 5/8] Installing !ONNX_PKG!...
!VENV_PY! -m pip install "!ONNX_PKG!>=1.17.0,^<2.0.0"
if errorlevel 1 (
    echo  [WARN] !ONNX_PKG! failed, falling back to CPU onnxruntime...
    !VENV_PY! -m pip install "onnxruntime>=1.17.0,^<2.0.0"
    if errorlevel 1 (
        echo  [ERROR] onnxruntime install failed
        pause & exit /b 1
    )
)
echo  [OK] onnxruntime installed

:: ------------------------------------------------------------
:: Step 6: Install coqui-tts
:: ------------------------------------------------------------
echo.
echo [Step 6/8] Installing coqui-tts (TTS engine)...

:: Pin transformers to <5.0 BEFORE coqui-tts.
:: transformers 5.x removed isin_mps_friendly which coqui-tts imports at startup.
:: If transformers 5.x is installed first, coqui-tts will fail to import even if
:: the package itself installed correctly.
echo  - Pinning transformers to 4.x first (5.x breaks coqui-tts)...
!VENV_PY! -m pip install "transformers>=4.40.0,^<5.0.0"

!VENV_PY! -m pip install "coqui-tts>=0.24.0,^<1.0.0"
if errorlevel 1 (
    echo.
    echo  [ERROR] coqui-tts install failed!
    echo.
    echo  Option A: Try a specific version (recommended for Python 3.10):
    echo    pip install coqui-tts==0.24.1
    echo.
    echo  Option B: If compilation is needed, install first:
    echo    1. Visual Studio Build Tools 2022 (with C++ workload)
    echo       https://visualstudio.microsoft.com/downloads/
    echo    2. espeak-ng (added to PATH)
    echo       https://github.com/espeak-ng/espeak-ng/releases
    echo    3. cmake:  pip install cmake
    echo    Then re-run this script.
    echo.
    pause & exit /b 1
)

!VENV_PY! -c "from TTS.api import TTS; print('  [OK] coqui-tts import verified')"
if errorlevel 1 (
    echo  [ERROR] coqui-tts installed but import failed
    echo.
    echo  Most likely cause: transformers was upgraded to 5.x by another package.
    echo  Fix: run the following and try again:
    echo    trans_env\Scripts\activate
    echo    pip install "transformers>=4.40.0,^<5.0.0" --force-reinstall
    echo.
    !VENV_PY! -c "import transformers; print('  Current transformers: ' + transformers.__version__)"
    pause & exit /b 1
)

:: ------------------------------------------------------------
:: Step 7: Install remaining dependencies (ordered to avoid numpy upgrade)
:: ------------------------------------------------------------
echo.
echo [Step 7/8] Installing remaining packages...

echo  - soundfile (includes libsndfile DLL for Windows)...
!VENV_PY! -m pip install "soundfile>=0.12.0,^<0.13.0"

echo  - librosa...
!VENV_PY! -m pip install "librosa>=0.10.0,^<1.0.0"

echo  - imageio-ffmpeg (auto-manages ffmpeg binary, no PATH config needed)...
!VENV_PY! -m pip install "imageio>=2.9.0,^<3.0.0" "imageio-ffmpeg>=0.4.9"

echo  - moviepy 1.0.3 (locked version, 2.x is incompatible)...
!VENV_PY! -m pip install "moviepy==1.0.3"

echo  - faster-whisper...
!VENV_PY! -m pip install "faster-whisper>=1.0.0,^<2.0.0"

echo  - deep_translator...
!VENV_PY! -m pip install "deep_translator>=1.11.0,^<2.0.0"

echo  - transformers (re-pin to ensure 4.x not overridden)...
!VENV_PY! -m pip install "transformers>=4.40.0,^<5.0.0"

echo  - huggingface_hub...
!VENV_PY! -m pip install "huggingface_hub>=0.23.0,^<2.0.0"

echo  - speechbrain (gender detection)...
!VENV_PY! -m pip install "speechbrain>=1.0.0,^<2.0.0"

echo  - pyannote.audio 3.3+ (3.1.x has torchaudio.AudioMetaData bug)...
!VENV_PY! -m pip install "pyannote.audio>=3.3.0,^<4.0.0"

echo  - Pillow...
!VENV_PY! -m pip install "Pillow>=9.0.0,^<11.0.0"

echo  - scipy...
!VENV_PY! -m pip install "scipy>=1.11.0,^<2.0.0"

echo  - zhconv (Chinese traditional/simplified conversion)...
!VENV_PY! -m pip install "zhconv>=1.4.3"

:: Final numpy check - some packages silently upgrade it to 2.x
echo.
echo  Checking numpy was not silently upgraded to 2.x...
!VENV_PY! -c "import numpy as np; v=np.__version__; ok=int(v.split('.')[0])^<2; print('  [OK] numpy ' + v) if ok else (print('  [ERROR] numpy ' + v + ' >= 2.0, force reinstalling...'), exit(1))"
if errorlevel 1 (
    !VENV_PY! -m pip install --force-reinstall "numpy>=1.24.0,^<2.0.0"
    echo  [OK] numpy reinstalled
)

:: ------------------------------------------------------------
:: Step 8: Verify all key imports
:: ------------------------------------------------------------
echo.
echo [Step 8/8] Verifying all key package imports...
echo.

set IMPORT_FAIL=0
!VENV_PY! -c "import torch; print('  [OK] torch ' + torch.__version__)" || set IMPORT_FAIL=1
!VENV_PY! -c "import torchaudio; print('  [OK] torchaudio ' + torchaudio.__version__)" || set IMPORT_FAIL=1
!VENV_PY! -c "import faster_whisper; print('  [OK] faster-whisper')" || set IMPORT_FAIL=1
!VENV_PY! -c "import transformers; v=transformers.__version__; ok=int(v.split('.')[0])^<5; print('  [OK] transformers ' + v) if ok else (print('  [ERROR] transformers ' + v + ' is 5.x, must be 4.x'), exit(1))" || set IMPORT_FAIL=1
!VENV_PY! -c "from TTS.api import TTS; print('  [OK] coqui-tts')" || set IMPORT_FAIL=1
!VENV_PY! -c "from pyannote.audio import Pipeline; print('  [OK] pyannote.audio')" || set IMPORT_FAIL=1
!VENV_PY! -c "from moviepy.editor import VideoFileClip; print('  [OK] moviepy 1.x')" || set IMPORT_FAIL=1
!VENV_PY! -c "from deep_translator import GoogleTranslator; print('  [OK] deep_translator')" || set IMPORT_FAIL=1
!VENV_PY! -c "import speechbrain; print('  [OK] speechbrain')" || set IMPORT_FAIL=1
!VENV_PY! -c "import librosa; print('  [OK] librosa')" || set IMPORT_FAIL=1
!VENV_PY! -c "import soundfile; print('  [OK] soundfile')" || set IMPORT_FAIL=1
!VENV_PY! -c "import imageio_ffmpeg; p=imageio_ffmpeg.get_ffmpeg_exe(); print('  [OK] imageio-ffmpeg => ' + p)" || set IMPORT_FAIL=1
!VENV_PY! -c "import numpy as np; print('  [OK] numpy ' + np.__version__)" || set IMPORT_FAIL=1

echo.
if !IMPORT_FAIL! EQU 1 (
    echo  ==============================================================
    echo   [FAILED] Some packages failed to import (see [ERROR] above)
    echo   Fix the errors above and re-run this script.
    echo  ==============================================================
) else (
    echo  ==============================================================
    echo   [SUCCESS] All dependencies installed and verified!
    echo  ==============================================================
    echo.
    echo  Next steps (required before first use):
    echo.
    echo  1. HuggingFace Token (needed for speaker diarization):
    echo     a) Register at https://huggingface.co
    echo     b) Accept model agreements at:
    echo        https://huggingface.co/pyannote/speaker-diarization-3.1
    echo        https://huggingface.co/pyannote/segmentation-3.0
    echo     c) Create a token at https://huggingface.co/settings/tokens
    echo     d) Set environment variable:
    echo        setx HF_TOKEN "hf_xxxxxxxxxxxxxxxxxxxxxxxx"
    echo.
    echo  2. ffmpeg is managed automatically by imageio-ffmpeg.
    echo     No manual install needed.
    echo.
    echo  3. Activate virtual environment before each use:
    echo     Open a command prompt in this folder, then run:
    echo       trans_env\Scripts\activate
    echo     Your prompt will show: (trans_env)
    echo.
    echo  4. Run the program (after activation):
    echo     python video_dubbing.py --mode single --input_video input.mp4 --target_lang en --voice en_vctk_vits_m001 --output_video output.mp4
    echo.
    echo  5. To exit the virtual environment when done:
    echo     deactivate
)

echo.
pause
endlocal
