"""Mandatory vocal/background separation helpers.

The main application keeps its stable CPU-only dependency set. Vocal separation
runs through a separate `separation_env` with the audio-separator CLI installed.
"""

from dataclasses import dataclass
from pathlib import Path
import os
import shutil
import subprocess
import tempfile
import time
from typing import Optional


DEFAULT_SEPARATOR_MODEL = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"


@dataclass
class SeparationResult:
    vocals_path: str
    dialogue_path: str
    background_path: str
    work_dir: str
    source_audio_path: str
    backend: str = "audio-separator"
    owns_work_dir: bool = False

    def cleanup(self) -> None:
        if self.owns_work_dir:
            _safe_rmtree(self.work_dir)


def separate_vocals_and_background(input_media_path: str, work_dir: str | None = None) -> SeparationResult:
    """Split input media into vocals and background tracks.

    Raises RuntimeError on any failure. The caller must not fall back to the
    original audio because that would reintroduce the original dialogue.
    """
    if not os.path.exists(input_media_path):
        raise RuntimeError(f"输入文件不存在: {input_media_path}")

    separator_bin = _resolve_audio_separator_bin()
    created_temp_dir = work_dir is None
    if work_dir is None:
        work_dir = tempfile.mkdtemp(prefix="trans_sep_")
    else:
        _safe_rmtree(work_dir)
        os.makedirs(work_dir, exist_ok=True)
    source_audio = os.path.join(work_dir, "source.wav")
    raw_output_dir = os.path.join(work_dir, "separator_output")
    vocals_path = os.path.join(work_dir, "vocals.wav")
    dialogue_path = os.path.join(work_dir, "dialogue.wav")
    background_path = os.path.join(work_dir, "background.wav")
    os.makedirs(raw_output_dir, exist_ok=True)

    try:
        print("[0/6] 人声分离...")
        print(f"  - 分离工具: {separator_bin}")
        _extract_audio_to_wav(input_media_path, source_audio)
        _run_audio_separator(separator_bin, source_audio, raw_output_dir)
        raw_vocals, raw_background = _find_separator_outputs(raw_output_dir)
        _normalize_audio(raw_vocals, vocals_path, sample_rate=16000, channels=1)
        _normalize_audio(raw_background, background_path, sample_rate=44100, channels=2)
        _build_dialogue_enhanced_audio(vocals_path, dialogue_path)
        _assert_nonempty(vocals_path, "vocals.wav")
        _assert_nonempty(dialogue_path, "dialogue.wav")
        _assert_nonempty(background_path, "background.wav")
        print(f"  - 人声轨: {vocals_path}")
        print(f"  - 对白识别轨: {dialogue_path}")
        print(f"  - 背景轨: {background_path}")
        return SeparationResult(
            vocals_path=vocals_path,
            dialogue_path=dialogue_path,
            background_path=background_path,
            work_dir=work_dir,
            source_audio_path=source_audio,
            owns_work_dir=created_temp_dir,
        )
    except Exception:
        if created_temp_dir:
            _safe_rmtree(work_dir)
        raise


def build_background_with_non_speech_vocals(
    separation_result: SeparationResult,
    speech_segments: list[dict],
    video_duration: float,
    pad_seconds: float = 0.18,
    fade_seconds: float = 0.05,
) -> str:
    """Return background plus vocal content outside ASR speech segments.

    This keeps background song vocals when they do not overlap with detected
    dialogue, while removing original dialogue from the final dubbing bed.
    """
    output_path = os.path.join(separation_result.work_dir, "background_with_non_speech_vocals.wav")
    _mix_background_with_non_speech_vocals(
        vocals_path=separation_result.vocals_path,
        background_path=separation_result.background_path,
        output_path=output_path,
        speech_segments=speech_segments,
        video_duration=video_duration,
        pad_seconds=pad_seconds,
        fade_seconds=fade_seconds,
    )
    _assert_nonempty(output_path, "background_with_non_speech_vocals.wav")
    return output_path


def _mix_background_with_non_speech_vocals(
    vocals_path: str,
    background_path: str,
    output_path: str,
    speech_segments: list[dict],
    video_duration: float,
    pad_seconds: float,
    fade_seconds: float,
) -> None:
    import librosa
    import numpy as np
    import soundfile as sf

    sample_rate = 44100
    vocals, _ = librosa.load(vocals_path, sr=sample_rate, mono=False)
    background, _ = librosa.load(background_path, sr=sample_rate, mono=False)
    if vocals.ndim == 1:
        vocals = np.stack([vocals, vocals], axis=0)
    if background.ndim == 1:
        background = np.stack([background, background], axis=0)

    target_len = min(int(video_duration * sample_rate), vocals.shape[1], background.shape[1])
    vocals = vocals[:, :target_len]
    background = background[:, :target_len]

    speech_mask = np.zeros(target_len, dtype=np.float32)
    for segment in speech_segments:
        start = float(segment.get("start", 0.0))
        end = float(segment.get("end", start))
        start_idx = max(0, int((start - pad_seconds) * sample_rate))
        end_idx = min(target_len, int((end + pad_seconds) * sample_rate))
        if end_idx > start_idx:
            speech_mask[start_idx:end_idx] = 1.0

    fade_len = int(fade_seconds * sample_rate)
    if fade_len > 1:
        kernel = np.hanning(fade_len * 2 + 1)
        kernel = kernel / kernel.sum()
        speech_mask = np.convolve(speech_mask, kernel, mode="same")
        speech_mask = np.clip(speech_mask, 0.0, 1.0)

    non_speech_vocals = vocals * (1.0 - speech_mask[None, :])
    mixed = background + non_speech_vocals
    peak = float(np.max(np.abs(mixed))) if mixed.size else 1.0
    if peak > 1.0:
        mixed = mixed / peak * 0.98
    sf.write(output_path, mixed.T, sample_rate)


def _resolve_audio_separator_bin() -> str:
    env_bin = os.environ.get("AUDIO_SEPARATOR_BIN", "").strip()
    if env_bin and os.path.exists(env_bin):
        return env_bin

    project_root = Path(__file__).resolve().parent
    candidates = [
        project_root / "separation_env" / "Scripts" / "audio-separator.exe",
        project_root / "separation_env" / "Scripts" / "audio-separator.bat",
        project_root / "separation_env" / "bin" / "audio-separator",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    found = shutil.which("audio-separator")
    if found:
        return found

    raise RuntimeError(
        "找不到 audio-separator。请先创建 separation_env 并安装: "
        "python -m pip install \"audio-separator[cpu]\""
    )


def _resolve_ffmpeg_bin() -> str:
    env_bin = os.environ.get("FFMPEG_BIN", "").strip()
    if env_bin:
        return env_bin
    if shutil.which("ffmpeg"):
        return "ffmpeg"
    try:
        import imageio_ffmpeg

        bundled = imageio_ffmpeg.get_ffmpeg_exe()
        if bundled and os.path.isfile(bundled):
            return bundled
    except Exception:
        pass
    return "ffmpeg"


def _extract_audio_to_wav(input_media_path: str, output_wav: str) -> None:
    cmd = [
        _resolve_ffmpeg_bin(),
        "-y",
        "-i",
        input_media_path,
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "44100",
        "-ac",
        "2",
        output_wav,
    ]
    _run_command(cmd, "提取音频失败")


def _run_audio_separator(separator_bin: str, source_audio: str, output_dir: str) -> None:
    cmd = [
        separator_bin,
        source_audio,
        "--model_filename",
        DEFAULT_SEPARATOR_MODEL,
        "--output_dir",
        output_dir,
        "--output_format",
        "WAV",
    ]
    _run_command(cmd, "audio-separator 分离失败")


def _find_separator_outputs(output_dir: str) -> tuple[str, str]:
    wav_files = [p for p in Path(output_dir).rglob("*.wav") if p.is_file()]
    if not wav_files:
        raise RuntimeError(f"audio-separator 未生成 WAV 输出: {output_dir}")

    vocals = _pick_output(
        wav_files,
        required=("vocals",),
        forbidden=("instrumental", "no_vocals", "no-vocals", "background", "accompaniment"),
    )
    background = _pick_output(
        wav_files,
        required=("instrumental", "no_vocals", "no-vocals", "background", "accompaniment"),
        forbidden=(),
    )
    if not vocals or not background:
        names = ", ".join(str(p.name) for p in wav_files)
        raise RuntimeError(f"无法识别分离输出的人声/背景轨。生成文件: {names}")
    return str(vocals), str(background)


def _pick_output(paths: list[Path], required: tuple[str, ...], forbidden: tuple[str, ...]) -> Optional[Path]:
    for path in paths:
        name = path.name.lower()
        if any(token in name for token in required) and not any(token in name for token in forbidden):
            return path
    return None


def _normalize_audio(input_wav: str, output_wav: str, sample_rate: int, channels: int) -> None:
    cmd = [
        _resolve_ffmpeg_bin(),
        "-y",
        "-i",
        input_wav,
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(sample_rate),
        "-ac",
        str(channels),
        output_wav,
    ]
    _run_command(cmd, f"标准化音频失败: {input_wav}")


def _build_dialogue_enhanced_audio(input_wav: str, output_wav: str) -> None:
    """Create a mono ASR track optimized for dialogue recognition.

    This is intentionally generic: it does not use per-video keywords or text
    hints. The filter chain reduces rumble, very high-frequency music content,
    and stabilizes loudness before ASR.
    """
    cmd = [
        _resolve_ffmpeg_bin(),
        "-y",
        "-i",
        input_wav,
        "-af",
        "highpass=f=120,lowpass=f=6200,afftdn=nf=-28,compand=attacks=0.04:decays=0.25:points=-80/-80|-45/-45|-35/-28|-20/-16|0/-4,dynaudnorm=f=120:g=5:p=0.8",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        output_wav,
    ]
    _run_command(cmd, f"生成对白识别轨失败: {input_wav}")


def _assert_nonempty(path: str, label: str) -> None:
    if not os.path.exists(path) or os.path.getsize(path) <= 1024:
        raise RuntimeError(f"{label} 输出无效或过小: {path}")


def _run_command(cmd: list[str], error_prefix: str) -> None:
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=_subprocess_env_with_ffmpeg(),
    )
    if result.returncode != 0:
        stderr = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(f"{error_prefix}: {stderr[-1200:]}")


def _subprocess_env_with_ffmpeg() -> dict[str, str]:
    env = os.environ.copy()
    ffmpeg_bin = _resolve_ffmpeg_bin()
    ffmpeg_dir = os.path.dirname(ffmpeg_bin) if os.path.isfile(ffmpeg_bin) else ""
    if ffmpeg_dir and os.path.basename(ffmpeg_bin).lower() not in ("ffmpeg", "ffmpeg.exe"):
        ffmpeg_dir = _ensure_ffmpeg_alias(ffmpeg_bin)
    if ffmpeg_dir:
        env["PATH"] = ffmpeg_dir + os.pathsep + env.get("PATH", "")
    return env


def _ensure_ffmpeg_alias(ffmpeg_bin: str) -> str:
    alias_dir = os.path.join(tempfile.gettempdir(), "trans_ffmpeg_alias")
    os.makedirs(alias_dir, exist_ok=True)
    alias_name = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
    alias_path = os.path.join(alias_dir, alias_name)
    if os.path.exists(alias_path):
        return alias_dir
    try:
        os.symlink(ffmpeg_bin, alias_path)
    except OSError:
        shutil.copy2(ffmpeg_bin, alias_path)
    return alias_dir


def _safe_rmtree(path: str, retries: int = 5, delay: float = 0.2) -> None:
    for attempt in range(retries):
        try:
            if os.path.exists(path):
                shutil.rmtree(path)
            return
        except PermissionError:
            if attempt < retries - 1:
                time.sleep(delay)
        except OSError:
            return
