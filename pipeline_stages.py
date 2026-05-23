"""Reusable pipeline stages with resumable cache files."""

from __future__ import annotations

from pathlib import Path
import json
import shutil
import time

from audio_separation import SeparationResult, separate_vocals_and_background
from asr_recognition import transcribe_chinese_audio
from ocr_recognition import get_ocr_subtitle_segments
from pipeline_cache import (
    PipelineRun,
    atomic_write_json,
    cascade_delete_dependents,
    is_stage_complete,
    mark_stage_complete,
)
from stage_validators import (
    validate_audio_stage,
    validate_segments_list,
    validate_recognition_stage,
    validate_composition_stage,
    validate_speaker_gender_stage,
    validate_tts_stage,
    validate_translation_stage,
)
from translation_cache import translate_segments_with_cache, translate_text_with_google


def get_or_create_audio_stage(
    run: PipelineRun,
    input_video_path: str,
    video_duration: float | None = None,
    cascade: bool = True,
) -> SeparationResult:
    stage_dir = run.stage_dir("audio")
    background_path = stage_dir / "background.wav"
    dialogue_path = stage_dir / "dialogue.wav"

    if is_stage_complete(run, "audio", [background_path, dialogue_path]):
        valid, reason = validate_audio_stage(background_path, dialogue_path, expected_duration=video_duration)
        if valid:
            print(f"  - 复用音频分离阶段: {stage_dir}")
            return SeparationResult(
                vocals_path="",
                dialogue_path=str(dialogue_path),
                background_path=str(background_path),
                work_dir=str(stage_dir),
                source_audio_path="",
                owns_work_dir=False,
            )
        print(f"  - 音频分离阶段无效，重新生成: {reason}")

    if cascade:
        cascade_delete_dependents(run, "audio")
    _reset_stage_dir(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)
    work_dir = stage_dir / "_work"
    result = separate_vocals_and_background(input_video_path, work_dir=str(work_dir))
    shutil.copy2(result.background_path, background_path)
    shutil.copy2(result.dialogue_path, dialogue_path)

    result.cleanup()
    valid, reason = validate_audio_stage(background_path, dialogue_path, expected_duration=video_duration)
    if not valid:
        raise RuntimeError(f"音频分离产物无效: {reason}")
    mark_stage_complete(
        run,
        "audio",
        {
            "outputs": {
                "background": str(background_path),
                "dialogue": str(dialogue_path),
            }
        },
    )
    return SeparationResult(
        vocals_path="",
        dialogue_path=str(dialogue_path),
        background_path=str(background_path),
        work_dir=str(stage_dir),
        source_audio_path="",
        owns_work_dir=False,
    )


def get_or_create_recognition_stage(
    run: PipelineRun,
    input_video_path: str,
    video_duration: float,
    asr_audio_path: str | None = None,
    try_ocr: bool = True,
    cascade: bool = True,
) -> list[dict]:
    stage_dir = run.stage_dir("recognition")
    segments_path = stage_dir / "recognized_segments.json"
    text_path = stage_dir / "recognized_text.txt"

    if is_stage_complete(run, "recognition", [segments_path, text_path]):
        valid, reason = validate_recognition_stage(segments_path, text_path, video_duration=video_duration)
        if valid:
            print(f"  - 复用文本识别阶段: {stage_dir}")
            return json.loads(segments_path.read_text(encoding="utf-8"))
        print(f"  - 文本识别阶段无效，重新生成: {reason}")

    if cascade:
        cascade_delete_dependents(run, "recognition")
    _reset_stage_dir(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)
    ocr_segments = []
    if try_ocr:
        ocr_segments = get_ocr_subtitle_segments(
            input_video_path,
            video_duration,
            cache_dir=stage_dir / "blocks",
        )
    if ocr_segments:
        segments = ocr_segments
        source = "ocr"
        print(f"  - 使用 OCR 硬字幕: {len(segments)} 段，跳过 ASR")
    else:
        if not asr_audio_path:
            raise RuntimeError("OCR 不可用，且没有 dialogue.wav 供 ASR 兜底")
        print("  - 未检测到可用 OCR 字幕，使用 faster-whisper 兜底识别 (CPU)")
        segments = transcribe_chinese_audio(asr_audio_path)
        source = "asr"

    valid, reason = validate_segments_list(segments, video_duration=video_duration)
    if not valid:
        raise RuntimeError(f"文本识别产物无效: {reason}")
    atomic_write_json(segments_path, segments)
    _atomic_write_text(text_path, _format_segments_text(segments))
    mark_stage_complete(run, "recognition", {"source": source, "outputs": [str(segments_path), str(text_path)]})
    return segments


def get_or_create_translation_stage(
    run: PipelineRun,
    segments: list[dict],
    target_language: str,
    translator=translate_text_with_google,
    max_workers: int = 4,
    max_retries: int = 5,
    retry_base_delay: float = 1.0,
    recovery_rounds: int = 3,
    recovery_delays: tuple[float, ...] = (30.0, 90.0, 180.0),
) -> list[dict]:
    stage_dir = run.stage_dir("translation")
    translated_path = stage_dir / "translated_segments.json"
    text_path = stage_dir / "translated_text.txt"
    pending_path = stage_dir / "translation_pending.json"

    if is_stage_complete(run, "translation", [translated_path, text_path]):
        valid, reason = validate_translation_stage(translated_path, text_path, pending_path, segments)
        if valid:
            print(f"  - 复用翻译阶段: {stage_dir}")
            return json.loads(translated_path.read_text(encoding="utf-8"))
        print(f"  - 翻译阶段需要回补/重建: {reason}")

    cascade_delete_dependents(run, "translation")
    started = time.time()
    results = translate_segments_with_cache(
        segments,
        target_language,
        stage_dir=stage_dir,
        translator=translator,
        max_workers=max_workers,
        max_retries=max_retries,
        retry_base_delay=retry_base_delay,
        recovery_rounds=recovery_rounds,
        recovery_delays=recovery_delays,
    )
    valid, reason = validate_translation_stage(translated_path, text_path, pending_path, segments)
    if not valid:
        print(f"  - 翻译阶段仍有问题: {reason}")
        print("  - 停止：不会进入 TTS，避免生成错误配音")
        raise RuntimeError(f"翻译阶段未完成: {reason}")
    mark_stage_complete(
        run,
        "translation",
        {
            "elapsed": time.time() - started,
            "outputs": [str(translated_path), str(text_path)],
        },
    )
    return results


def get_or_create_speaker_gender_stage(run: PipelineRun, wait_func) -> dict:
    stage_dir = run.stage_dir("speaker_gender")
    output_path = stage_dir / "speaker_gender.json"
    if is_stage_complete(run, "speaker_gender", [output_path]):
        valid, reason = validate_speaker_gender_stage(output_path)
        if valid:
            print(f"  - 复用说话人/男女声阶段: {stage_dir}")
            return json.loads(output_path.read_text(encoding="utf-8"))
        print(f"  - 说话人/男女声阶段无效，重新生成: {reason}")

    cascade_delete_dependents(run, "speaker_gender")
    _reset_stage_dir(stage_dir)
    speaker_map = wait_func()
    atomic_write_json(output_path, speaker_map)
    valid, reason = validate_speaker_gender_stage(output_path)
    if not valid:
        raise RuntimeError(f"说话人/男女声产物无效: {reason}")
    mark_stage_complete(run, "speaker_gender", {"outputs": [str(output_path)]})
    return speaker_map


def mark_tts_stage_complete(
    run: PipelineRun,
    tts_segments: list[dict],
    timeline: list[dict] | None = None,
) -> None:
    stage_dir = run.stage_dir("tts")
    outputs = []

    segments_path = stage_dir / "tts_segments.json"
    atomic_write_json(segments_path, tts_segments)
    outputs.append(str(segments_path))

    if timeline is not None:
        timeline_path = stage_dir / "tts_timeline.json"
        atomic_write_json(timeline_path, timeline)
        outputs.append(str(timeline_path))

    if timeline is not None:
        valid, reason = validate_tts_stage(segments_path, timeline_path)
        if not valid:
            raise RuntimeError(f"TTS 阶段产物无效: {reason}")

    mark_stage_complete(run, "tts", {"outputs": outputs})


def invalidate_composition_for_tts(run: PipelineRun) -> None:
    cascade_delete_dependents(run, "tts")


def mark_composition_stage_complete(
    run: PipelineRun,
    output_video_path: str,
    expected_min_duration: float | None = None,
) -> None:
    valid, reason = validate_composition_stage(output_video_path, expected_min_duration=expected_min_duration)
    if not valid:
        raise RuntimeError(f"最终合成产物无效: {reason}")
    mark_stage_complete(run, "composition", {"outputs": [output_video_path]})


def _format_segments_text(segments: list[dict]) -> str:
    lines = []
    for segment in segments:
        confidence = segment.get("confidence")
        suffix = f"  (conf={confidence:.2f})" if isinstance(confidence, (int, float)) else ""
        lines.append(f"[{segment['start']:.2f}-{segment['end']:.2f}] {segment['text']}{suffix}")
    return "\n".join(lines)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(text, encoding="utf-8")
    tmp_path.replace(path)


def _reset_stage_dir(stage_dir: Path) -> None:
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)
