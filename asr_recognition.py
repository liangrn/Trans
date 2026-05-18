"""Chinese ASR helpers backed by faster-whisper in the main environment."""

import os

from faster_whisper import WhisperModel

#small,medium,large-v3
DEFAULT_WHISPER_MODEL_SIZE = os.environ.get("WHISPER_MODEL_SIZE", "medium").strip() or "medium"


def transcribe_chinese_audio(audio_path: str) -> list[dict]:
    """Transcribe Chinese speech and return text segments with timestamps."""
    if not os.path.exists(audio_path):
        raise RuntimeError(f"ASR 输入音频不存在: {audio_path}")

    print(f"  - 使用 faster-whisper {DEFAULT_WHISPER_MODEL_SIZE} (CPU, int8)")
    model = WhisperModel(
        DEFAULT_WHISPER_MODEL_SIZE,
        device="cpu",
        compute_type="int8",
    )
    segments, _info = model.transcribe(
        audio_path,
        language="zh",
        task="transcribe",
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
        beam_size=5,
        best_of=5,
        temperature=0.0,
        condition_on_previous_text=False,
    )
    return _normalize_segments(segments)


def _normalize_segments(raw_segments) -> list[dict]:
    normalized = []
    for segment in raw_segments:
        text = _to_simplified(str(getattr(segment, "text", "")).strip())
        if not text:
            continue
        start = max(0.0, float(getattr(segment, "start", 0.0)))
        end = max(start, float(getattr(segment, "end", start)))
        if end - start < 0.3:
            end = start + 0.3
        normalized.append(
            {
                "text": text,
                "start": start,
                "end": end,
                "duration": end - start,
            }
        )
    return normalized


def _to_simplified(text: str) -> str:
    try:
        from zhconv import convert

        return convert(text, "zh-cn")
    except Exception:
        return text
