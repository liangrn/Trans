"""Validation helpers for resumable pipeline stage outputs."""

from __future__ import annotations

from pathlib import Path
import json
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


def validate_composition_stage(output_path: str | Path) -> tuple[bool, str]:
    output_path = Path(output_path)
    if not output_path.exists():
        return False, "最终视频不存在"
    if output_path.stat().st_size <= 10 * 1024:
        return False, "最终视频文件过小"
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
