"""Validation helpers for resumable pipeline stage outputs."""

from __future__ import annotations

from pathlib import Path
import json
import subprocess
import wave
from typing import Any


def validate_audio_stage(
    background_path: str | Path,
    dialogue_path: str | Path,
    expected_duration: float | None = None,
) -> tuple[bool, str]:
    for label, path in (("background", background_path), ("dialogue", dialogue_path)):
        ok, reason = _validate_wav(path, expected_duration=expected_duration)
        if not ok:
            return False, f"{label} 音频无效: {reason}"
    return True, "ok"


def validate_recognition_stage(
    segments_path: str | Path,
    text_path: str | Path,
    video_duration: float | None = None,
) -> tuple[bool, str]:
    segments_path = Path(segments_path)
    text_path = Path(text_path)
    if not _has_content(segments_path) or not _has_content(text_path):
        return False, "识别结果文件为空或缺失"
    try:
        segments = json.loads(segments_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, f"识别 JSON 不可读取: {exc}"
    return validate_segments_list(segments, video_duration=video_duration)


def validate_segments_list(
    segments: Any,
    video_duration: float | None = None,
) -> tuple[bool, str]:
    if not isinstance(segments, list) or not segments:
        return False, "识别片段为空"

    previous_start = -0.001
    for idx, segment in enumerate(segments):
        if not isinstance(segment, dict):
            return False, f"第 {idx} 段不是对象"
        text = str(segment.get("text") or "").strip()
        if not text:
            return False, f"第 {idx} 段文本为空"
        try:
            start = float(segment["start"])
            end = float(segment["end"])
        except Exception:
            return False, f"第 {idx} 段时间无效"
        if start < -0.05 or end <= start:
            return False, f"第 {idx} 段时间无效"
        if start + 0.05 < previous_start:
            return False, f"第 {idx} 段时间顺序无效"
        if video_duration is not None and end > float(video_duration) + 2.0:
            return False, f"第 {idx} 段超过视频时长"
        previous_start = start
    return True, "ok"


def validate_translation_stage(
    translated_path: str | Path,
    text_path: str | Path,
    pending_path: str | Path,
    source_segments: list[dict],
) -> tuple[bool, str]:
    translated_path = Path(translated_path)
    text_path = Path(text_path)
    pending_path = Path(pending_path)
    if not _has_content(translated_path) or not _has_content(text_path):
        return False, "翻译结果文件为空或缺失"

    try:
        translated = json.loads(translated_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, f"翻译 JSON 不可读取: {exc}"

    if not isinstance(translated, list) or len(translated) != len(source_segments):
        return False, "翻译条数与识别条数不一致"

    for idx, (item, source) in enumerate(zip(translated, source_segments)):
        if not isinstance(item, dict):
            return False, f"第 {idx} 条翻译不是对象"
        if str(item.get("text") or "").strip() != str(source.get("text") or "").strip():
            return False, f"第 {idx} 条源文本已变化"
        if not str(item.get("translated") or "").strip():
            return False, f"第 {idx} 条译文为空"

    pending = _load_json(pending_path, default=[])
    if pending:
        return False, "存在待回补翻译"
    return True, "ok"


def validate_speaker_gender_stage(output_path: str | Path) -> tuple[bool, str]:
    output_path = Path(output_path)
    if not _has_content(output_path):
        return False, "说话人结果为空或缺失"
    try:
        data = json.loads(output_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, f"说话人 JSON 不可读取: {exc}"
    if not isinstance(data, dict):
        return False, "说话人结果不是对象"
    return True, "ok"


def validate_tts_stage(
    segments_path: str | Path,
    timeline_path: str | Path,
    expected_count: int | None = None,
) -> tuple[bool, str]:
    segments_path = Path(segments_path)
    timeline_path = Path(timeline_path)
    if not _has_content(segments_path) or not _has_content(timeline_path):
        return False, "TTS 结果文件为空或缺失"

    try:
        segments = json.loads(segments_path.read_text(encoding="utf-8"))
        timeline = json.loads(timeline_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, f"TTS JSON 不可读取: {exc}"

    if not isinstance(segments, list) or not isinstance(timeline, list):
        return False, "TTS 结果格式无效"
    if expected_count is not None and len(segments) != expected_count:
        return False, "TTS 条数与翻译条数不一致"
    if len(timeline) != len(segments):
        return False, "TTS 时间线条数不一致"

    previous_end = -0.001
    for idx, item in enumerate(segments):
        if not isinstance(item, dict):
            return False, f"第 {idx} 条 TTS 不是对象"
        if not item.get("success"):
            return False, f"第 {idx} 条 TTS 失败"
        path = Path(str(item.get("path") or ""))
        if not path.exists() or path.stat().st_size <= 512:
            return False, f"第 {idx} 条 TTS 音频缺失"
        digest = str(item.get("cache_digest") or "")
        if not digest or digest not in path.name:
            return False, f"第 {idx} 条 TTS digest 不匹配"
        ok, reason = _validate_wav(path)
        if not ok:
            return False, f"第 {idx} 条 TTS 音频无效: {reason}"

    for idx, item in enumerate(timeline):
        if not isinstance(item, dict):
            return False, f"第 {idx} 条时间线不是对象"
        try:
            start = float(item["planned_start"])
            end = float(item["planned_end"])
        except Exception:
            return False, f"第 {idx} 条时间线无效"
        if start < -0.05 or end <= start:
            return False, f"第 {idx} 条时间线无效"
        if start + 0.05 < previous_end:
            return False, f"第 {idx} 条时间线重叠"
        previous_end = end
    return True, "ok"


def validate_composition_stage(
    output_path: str | Path,
    expected_min_duration: float | None = None,
) -> tuple[bool, str]:
    output_path = Path(output_path)
    if not output_path.exists():
        return False, "最终视频不存在"
    if output_path.stat().st_size <= 10 * 1024:
        return False, "最终视频文件过小"
    if expected_min_duration is not None:
        duration = _probe_stream_duration(output_path)
        if duration is None:
            return False, "最终视频时长不可读取"
        if duration + 0.3 < float(expected_min_duration):
            return False, f"最终视频时长不足 {duration:.2f}s，预期至少 {expected_min_duration:.2f}s"

        audio_duration = _probe_stream_duration(output_path, stream_selector="a:0")
        if audio_duration is None:
            if not _has_audio_stream(output_path):
                return False, "最终视频音频流不可读取"
            audio_duration = duration
        if audio_duration + 0.3 < float(expected_min_duration):
            return False, f"最终音频时长不足 {audio_duration:.2f}s，预期至少 {expected_min_duration:.2f}s"
    return True, "ok"


def _validate_wav(path: str | Path, expected_duration: float | None = None) -> tuple[bool, str]:
    path = Path(path)
    if not _has_content(path):
        return False, "文件为空或缺失"
    try:
        with wave.open(str(path), "rb") as wav:
            frames = wav.getnframes()
            rate = wav.getframerate()
            if rate <= 0 or frames <= 0:
                return False, "时长无效"
            duration = frames / float(rate)
    except Exception as exc:
        return False, f"不可读取: {exc}"

    if expected_duration is not None:
        tolerance = max(2.5, float(expected_duration) * 0.08)
        if abs(duration - float(expected_duration)) > tolerance:
            return False, f"时长异常 {duration:.2f}s，预期约 {expected_duration:.2f}s"
    return True, "ok"


def _has_content(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def _load_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _probe_stream_duration(path: Path, stream_selector: str | None = None) -> float | None:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
    ]
    if stream_selector:
        cmd.extend(["-select_streams", stream_selector])
        cmd[4] = "stream=duration"
    cmd.append(str(path))
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
        )
        value = (result.stdout or "").strip().splitlines()[0]
        return float(value)
    except Exception:
        return None


def _has_audio_stream(path: Path) -> bool:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=codec_type",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
        )
        return "audio" in (result.stdout or "").strip().lower()
    except Exception:
        return False
