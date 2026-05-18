"""Hard subtitle OCR helpers backed by PaddleOCR in a separate `ocr_env`."""

from pathlib import Path
import json
import os
import shutil
import subprocess
import tempfile
import time


def get_ocr_subtitle_segments(video_path: str, video_duration: float, cache_dir: str | Path | None = None) -> list[dict]:
    """Return OCR subtitle segments when the video has usable hard subtitles."""
    if not os.path.exists(video_path):
        raise RuntimeError(f"OCR 输入视频不存在: {video_path}")

    block_cache_path = None
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        block_cache_path = cache_dir / "block_000.json"
        if block_cache_path.exists():
            raw_segments = json.loads(block_cache_path.read_text(encoding="utf-8"))
            segments = _normalize_ocr_segments(raw_segments, video_duration)
            return segments if _is_usable_ocr_result(segments) else []

    ocr_python = _resolve_ocr_python()
    output_json = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
    output_txt = tempfile.NamedTemporaryFile(delete=False, suffix=".txt")
    output_json_path = output_json.name
    output_txt_path = output_txt.name
    output_json.close()
    output_txt.close()

    try:
        cmd = [
            ocr_python,
            str(Path(__file__).resolve().parent / "ocr_subtitle_probe.py"),
            "--input_video",
            video_path,
            "--output_txt",
            output_txt_path,
            "--output_json",
            output_json_path,
            "--interval",
            "0.33",
        "--fast_interval",
        "0.75",
        "--crop_top",
        "0.58",
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if result.returncode != 0:
            err = (result.stderr or result.stdout or "").strip()
            print(f"  - OCR 字幕识别失败，将使用 ASR: {err[-600:]}")
            return []

        raw_segments = json.loads(Path(output_json_path).read_text(encoding="utf-8"))
        if block_cache_path is not None:
            block_cache_path.write_text(
                json.dumps(raw_segments, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        segments = _normalize_ocr_segments(raw_segments, video_duration)
        return segments if _is_usable_ocr_result(segments) else []
    finally:
        _safe_remove(output_json_path)
        _safe_remove(output_txt_path)


def _resolve_ocr_python() -> str:
    env_python = os.environ.get("OCR_PYTHON", "").strip()
    if env_python and os.path.exists(env_python):
        return env_python

    project_root = Path(__file__).resolve().parent
    candidates = [
        project_root / "ocr_env" / "Scripts" / "python.exe",
        project_root / "ocr_env" / "bin" / "python",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    found = shutil.which("python")
    if found:
        return found

    raise RuntimeError(
        "找不到 OCR 运行环境。请先创建 ocr_env 并安装: "
        "python -m pip install paddleocr paddlepaddle opencv-python-headless"
    )


def _normalize_ocr_segments(raw_segments: list[dict], video_duration: float) -> list[dict]:
    normalized = []
    for segment in raw_segments:
        text = _to_simplified(str(segment.get("text", "")).strip())
        if not text or not _has_chinese(text):
            continue
        start = max(0.0, float(segment.get("start", 0.0)))
        end = min(video_duration, max(start + 0.3, float(segment.get("end", start + 0.3))))
        confidence = float(segment.get("confidence", 0.0))
        if len(text) == 1 and confidence < 0.9:
            continue
        normalized.append(
            {
                "text": text,
                "start": start,
                "end": end,
                "duration": end - start,
                "confidence": confidence,
                "source": "ocr",
            }
        )
    normalized.sort(key=lambda item: item["start"])
    return _avoid_overlaps(normalized)


def _is_usable_ocr_result(segments: list[dict]) -> bool:
    if len(segments) < 3:
        return False
    total_chars = sum(len(segment["text"]) for segment in segments)
    avg_conf = sum(segment["confidence"] for segment in segments) / len(segments)
    return total_chars >= 20 and avg_conf >= 0.65


def _avoid_overlaps(segments: list[dict]) -> list[dict]:
    for index, segment in enumerate(segments[:-1]):
        next_start = segments[index + 1]["start"]
        if segment["end"] > next_start:
            segment["end"] = max(segment["start"] + 0.3, next_start - 0.05)
            segment["duration"] = segment["end"] - segment["start"]
    return segments


def _has_chinese(text: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in text)


def _to_simplified(text: str) -> str:
    try:
        from zhconv import convert

        return convert(text, "zh-cn")
    except Exception:
        return text


def _safe_remove(path: str, retries: int = 5, delay: float = 0.15) -> None:
    for attempt in range(retries):
        try:
            if os.path.exists(path):
                os.remove(path)
            return
        except PermissionError:
            if attempt < retries - 1:
                time.sleep(delay)
        except OSError:
            return
