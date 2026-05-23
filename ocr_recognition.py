"""Hard subtitle OCR helpers backed by PaddleOCR in a separate `ocr_env`."""

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher
import json
import os
import shutil
import subprocess
import tempfile
import time


OCR_BLOCK_SECONDS = 300.0
OCR_BLOCK_WORKERS = 2
OCR_BLOCK_OVERLAP_SECONDS = 1.0
OCR_BLOCK_FORMAT = "ocr_block_v2"


def get_ocr_subtitle_segments(video_path: str, video_duration: float, cache_dir: str | Path | None = None) -> list[dict]:
    """Return OCR subtitle segments when the video has usable hard subtitles."""
    if not os.path.exists(video_path):
        raise RuntimeError(f"OCR 输入视频不存在: {video_path}")

    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        raw_segments = _get_or_create_ocr_blocks(video_path, video_duration, cache_dir)
        segments = _normalize_ocr_segments(raw_segments, video_duration)
        segments, cleanup_report = _merge_adjacent_ocr_duplicates(segments)
        _write_ocr_cleanup_report(cache_dir.parent / "ocr_cleanup_report.json", cleanup_report)
        return segments if _is_usable_ocr_result(segments) else []

    raw_segments = _run_ocr_probe(video_path)
    segments = _normalize_ocr_segments(raw_segments, video_duration)
    segments, _cleanup_report = _merge_adjacent_ocr_duplicates(segments)
    return segments if _is_usable_ocr_result(segments) else []


def _get_or_create_ocr_blocks(video_path: str, video_duration: float, cache_dir: Path) -> list[dict]:
    block_ranges = _build_ocr_block_ranges(video_duration)
    blocks: dict[int, list[dict]] = {}
    missing_blocks: list[tuple[int, float, float]] = []
    for block_index, (start_time, end_time) in enumerate(block_ranges):
        block_path = cache_dir / f"block_{block_index:03d}.json"
        cached = _load_cached_ocr_block(block_path, start_time, end_time)
        if cached is None:
            missing_blocks.append((block_index, start_time, end_time))
        else:
            blocks[block_index] = cached

    if missing_blocks:
        if len(missing_blocks) == 1:
            block_index, start_time, end_time = missing_blocks[0]
            blocks[block_index] = _run_and_cache_ocr_block(
                video_path, video_duration, cache_dir, block_index, start_time, end_time
            )
        else:
            workers = max(1, min(OCR_BLOCK_WORKERS, len(missing_blocks)))
            print(f"  - OCR 分块并行: {len(missing_blocks)} 个缺失块, workers={workers}")
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(
                        _run_and_cache_ocr_block,
                        video_path,
                        video_duration,
                        cache_dir,
                        block_index,
                        start_time,
                        end_time,
                    ): block_index
                    for block_index, start_time, end_time in missing_blocks
                }
                for future in as_completed(futures):
                    blocks[futures[future]] = future.result()

    merged_segments = [
        segment
        for block_index in range(len(block_ranges))
        for segment in blocks.get(block_index, [])
    ]
    merged_segments.sort(key=lambda item: float(item.get("start", 0.0)))
    return merged_segments


def _run_and_cache_ocr_block(
    video_path: str,
    video_duration: float,
    cache_dir: Path,
    block_index: int,
    start_time: float,
    end_time: float,
) -> list[dict]:
    probe_start = max(0.0, start_time - OCR_BLOCK_OVERLAP_SECONDS)
    probe_end = min(float(video_duration or end_time), end_time + OCR_BLOCK_OVERLAP_SECONDS)
    raw_segments = _run_ocr_probe(video_path, start_time=probe_start, end_time=probe_end)
    block_segments = [
        segment for segment in raw_segments
        if start_time <= float(segment.get("start", 0.0)) < end_time
    ]
    _write_ocr_block(cache_dir / f"block_{block_index:03d}.json", start_time, end_time, block_segments)
    return block_segments


def _build_ocr_block_ranges(video_duration: float, block_seconds: float = OCR_BLOCK_SECONDS) -> list[tuple[float, float]]:
    duration = max(0.0, float(video_duration or 0.0))
    if duration <= 0:
        return [(0.0, 0.0)]
    ranges = []
    start = 0.0
    while start < duration:
        end = min(duration, start + block_seconds)
        ranges.append((start, end))
        start = end
    return ranges


def _load_cached_ocr_block(block_path: Path, start_time: float, end_time: float) -> list[dict] | None:
    if not block_path.exists():
        return None
    try:
        data = json.loads(block_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(data, dict) or data.get("format") != OCR_BLOCK_FORMAT:
        return None
    if abs(float(data.get("start", -1.0)) - float(start_time)) > 0.01:
        return None
    if abs(float(data.get("end", -1.0)) - float(end_time)) > 0.01:
        return None
    segments = data.get("segments")
    if not isinstance(segments, list):
        return None
    return segments


def _write_ocr_block(block_path: Path, start_time: float, end_time: float, segments: list[dict]) -> None:
    payload = {
        "format": OCR_BLOCK_FORMAT,
        "start": start_time,
        "end": end_time,
        "segments": segments,
    }
    tmp_path = block_path.with_suffix(block_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(block_path)


def _write_ocr_blocks(cache_dir: Path, block_ranges: list[tuple[float, float]], segments: list[dict]) -> None:
    for block_index, (start_time, end_time) in enumerate(block_ranges):
        block_segments = [
            segment for segment in segments
            if start_time <= float(segment.get("start", 0.0)) < end_time
        ]
        _write_ocr_block(cache_dir / f"block_{block_index:03d}.json", start_time, end_time, block_segments)


def _run_ocr_probe(video_path: str, start_time: float | None = None, end_time: float | None = None) -> list[dict]:
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
            "0.5",
            "--fast_interval",
            "0.75",
            "--crop_top",
            "0.58",
        ]
        if start_time is not None:
            cmd.extend(["--start_time", f"{start_time:.3f}"])
        if end_time is not None:
            cmd.extend(["--end_time", f"{end_time:.3f}"])
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if result.returncode != 0:
            err = (result.stderr or result.stdout or "").strip()
            if "ConvertPirAttribute2RuntimeAttribute" in err or "onednn_instruction" in err:
                print("  - OCR 字幕识别失败：PaddleOCR/oneDNN 兼容问题，已退回 ASR")
            print(f"  - OCR 字幕识别失败，将使用 ASR: {err[-600:]}")
            return []

        return json.loads(Path(output_json_path).read_text(encoding="utf-8"))
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
        "python -m pip install \"paddlepaddle==3.3.0\" \"paddleocr>=3.3.0,<3.4.0\" opencv-python-headless"
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


def _merge_adjacent_ocr_duplicates(
    segments: list[dict],
    max_gap: float = 0.35,
    max_span: float = 2.0,
) -> tuple[list[dict], list[dict]]:
    """Merge adjacent OCR-only segments caused by repeated frame sampling."""
    cleaned = []
    report = []
    index = 0
    while index < len(segments):
        segment = segments[index]
        if segment.get("source") != "ocr":
            cleaned.append(segment)
            index += 1
            continue

        candidates = [segment]
        cursor = index + 1
        while cursor < len(segments):
            next_segment = segments[cursor]
            if next_segment.get("source") != "ocr":
                break
            gap = float(next_segment.get("start", 0.0)) - float(candidates[-1].get("end", 0.0))
            span = float(next_segment.get("end", 0.0)) - float(candidates[0].get("start", 0.0))
            if gap > max_gap or span > max_span:
                break
            candidates.append(next_segment)
            cursor += 1

        merged = None
        merged_count = 1
        merged_group = None
        for count in range(len(candidates), 1, -1):
            merged = _try_merge_ocr_duplicate_group(candidates[:count])
            if merged is not None:
                merged_count = count
                merged_group = candidates[:count]
                break

        # Allow only the dominant-variant pair rule to inspect a longer 2-segment span.
        if merged is None and index + 1 < len(segments):
            next_segment = segments[index + 1]
            if next_segment.get("source") == "ocr":
                gap = float(next_segment.get("start", 0.0)) - float(segment.get("end", 0.0))
                pair_span = float(next_segment.get("end", 0.0)) - float(segment.get("start", 0.0))
                if gap <= max_gap and pair_span <= 5.0:
                    pair_text = _choose_dominant_variant_pair([segment, next_segment])
                    if pair_text:
                        merged = (_build_merged_ocr_segment([segment, next_segment], pair_text), "dominant_variant_pair")
                        merged_count = 2
                        merged_group = [segment, next_segment]

        if merged is None:
            cleaned.append(segment)
            index += 1
            continue

        merged_segment, reason = merged
        cleaned.append(merged_segment)
        report.append(
            {
                "reason": reason,
                "original": [dict(item) for item in (merged_group or candidates[:merged_count])],
                "merged": dict(merged_segment),
            }
        )
        index += merged_count

    cleaned, noise_report = _filter_low_confidence_ocr_noise(cleaned)
    report.extend(noise_report)
    return cleaned, report


def _try_merge_ocr_duplicate_group(group: list[dict]) -> tuple[dict, str] | None:
    texts = [str(item.get("text", "")).strip() for item in group]
    if len(texts) < 2 or any(_chinese_count(text) < 2 for text in texts):
        return None

    confident_longer_variant = _choose_confident_longer_variant_pair(group)
    if confident_longer_variant:
        return _build_merged_ocr_segment(group, confident_longer_variant), "confident_longer_variant_pair"

    short_completion = _choose_short_completion(group)
    if short_completion:
        return _build_merged_ocr_segment(group, short_completion), "short_completion"

    noisy_long_to_clean_short = _choose_noisy_long_to_clean_short(group)
    if noisy_long_to_clean_short:
        return _build_merged_ocr_segment(group, noisy_long_to_clean_short), "noisy_long_to_clean_short"

    prefix_to_clean_long = _choose_prefix_to_clean_long(group)
    if prefix_to_clean_long:
        return _build_merged_ocr_segment(group, prefix_to_clean_long), "prefix_to_clean_long"

    stable_prefix = _choose_stable_prefix_noise(group)
    if stable_prefix:
        return _build_merged_ocr_segment(group, stable_prefix), "stable_prefix_noise"

    embedded_core = _recover_embedded_clean_core(group)
    if embedded_core:
        return _build_merged_ocr_segment(group, embedded_core), "embedded_core_recovery"

    if _is_near_duplicate_group(texts):
        return _build_merged_ocr_segment(group, _choose_representative_ocr_text(group)), "near_duplicate"

    return None


def _choose_short_completion(group: list[dict]) -> str | None:
    if len(group) != 2:
        return None
    first = str(group[0].get("text", "")).strip()
    second = str(group[1].get("text", "")).strip()
    if first == second:
        return None
    shorter, longer = (first, second) if len(first) < len(second) else (second, first)
    if len(shorter) > 4 or not longer.startswith(shorter):
        return None
    if len(longer) - len(shorter) not in (1, 2):
        return None
    short_conf = float((group[0] if first == shorter else group[1]).get("confidence", 0.0))
    long_conf = float((group[0] if first == longer else group[1]).get("confidence", 0.0))
    if long_conf < short_conf - 0.08:
        return None
    return longer


def _choose_noisy_long_to_clean_short(group: list[dict]) -> str | None:
    if len(group) != 2:
        return None
    first = str(group[0].get("text", "")).strip()
    second = str(group[1].get("text", "")).strip()
    if first == second:
        return None
    shorter, longer = (first, second) if len(first) < len(second) else (second, first)
    short_segment = group[0] if first == shorter else group[1]
    long_segment = group[0] if first == longer else group[1]
    short_conf = float(short_segment.get("confidence", 0.0))
    long_conf = float(long_segment.get("confidence", 0.0))
    min_clean_length = 3 if long_conf < 0.9 and short_conf >= 0.95 else 4
    if _chinese_count(shorter) < min_clean_length or not longer.startswith(shorter):
        return None
    suffix = longer[len(shorter):]
    if _chinese_count(suffix) < 3:
        return None
    if long_conf > short_conf - 0.12:
        return None
    return shorter


def _choose_confident_longer_variant_pair(group: list[dict]) -> str | None:
    if len(group) != 2:
        return None
    first = str(group[0].get("text", "")).strip()
    second = str(group[1].get("text", "")).strip()
    if abs(len(first) - len(second)) > 1:
        return None
    if _chinese_count(first) < 4 or _chinese_count(second) < 4:
        return None
    if SequenceMatcher(None, first, second).ratio() < 0.78:
        return None

    first_conf = float(group[0].get("confidence", 0.0))
    second_conf = float(group[1].get("confidence", 0.0))
    if max(first_conf, second_conf) < 0.90 or min(first_conf, second_conf) >= 0.90:
        return None

    first_duration = float(group[0].get("end", 0.0)) - float(group[0].get("start", 0.0))
    second_duration = float(group[1].get("end", 0.0)) - float(group[1].get("start", 0.0))
    if max(first_duration, second_duration) < 0.8:
        return None
    if abs(first_duration - second_duration) < 0.45:
        return None

    first_score = (first_conf, first_duration, len(first))
    second_score = (second_conf, second_duration, len(second))
    return first if first_score >= second_score else second


def _choose_prefix_to_clean_long(group: list[dict]) -> str | None:
    if len(group) < 3:
        return None
    texts = [str(item.get("text", "")).strip() for item in group]
    longest_text = max(texts, key=len)
    longest_segment = max(group, key=lambda item: len(str(item.get("text", "")).strip()))
    if float(longest_segment.get("confidence", 0.0)) < 0.95:
        return None

    shorter_texts = [text for text in texts if text != longest_text and len(text) <= len(longest_text) - 3]
    if len(shorter_texts) < 2:
        return None
    shorter_prefix = _common_prefix(shorter_texts)
    if _chinese_count(shorter_prefix) < 4 or not longest_text.startswith(shorter_prefix):
        return None
    return longest_text


def _choose_stable_prefix_noise(group: list[dict]) -> str | None:
    if len(group) < 3:
        return None
    texts = [str(item.get("text", "")).strip() for item in group]
    prefix = _common_prefix(texts)
    if _chinese_count(prefix) < 4:
        return None
    shortest = min(texts, key=len)
    if len(prefix) < max(4, int(len(shortest) * 0.75)):
        return None

    suffixes = [text[len(prefix):] for text in texts]
    non_empty_suffixes = [suffix for suffix in suffixes if suffix]
    if len(non_empty_suffixes) < 2 or len(set(non_empty_suffixes)) < 2:
        return None

    if prefix not in texts:
        return None

    prefix_confidences = [
        float(item.get("confidence", 0.0))
        for item, text in zip(group, texts)
        if text == prefix
    ]
    suffixed_confidences = [
        float(item.get("confidence", 0.0))
        for item, suffix in zip(group, suffixes)
        if suffix
    ]
    if prefix_confidences and suffixed_confidences:
        if max(suffixed_confidences) > max(prefix_confidences) - 0.05:
            return None

    return prefix


def _recover_embedded_clean_core(group: list[dict]) -> str | None:
    if len(group) < 3:
        return None
    texts = [str(item.get("text", "")).strip() for item in group]
    short_texts = [text for text in texts if len(text) <= 4]
    long_texts = [text for text in texts if len(text) >= 5]
    if len(short_texts) < 2 or not long_texts:
        return None

    short_prefix = _common_prefix(short_texts)
    if len(short_prefix) < 3:
        return None

    short_stems = {text[:-1] for text in short_texts if len(text) >= 4}
    if len(short_stems) != 1:
        return None

    for long_text in long_texts:
        for start_index in range(max(0, len(long_text) - 3)):
            candidate = long_text[start_index:]
            if not candidate.startswith(short_prefix):
                continue
            if len(candidate) < len(short_texts[0]):
                continue
            core = candidate[:len(short_texts[0])]
            if core.startswith(short_prefix) and _chinese_count(core) >= 4 and core not in short_texts:
                return core
    return None


def _choose_dominant_variant_pair(group: list[dict]) -> str | None:
    if len(group) != 2:
        return None
    first = str(group[0].get("text", "")).strip()
    second = str(group[1].get("text", "")).strip()
    if abs(len(first) - len(second)) > 1:
        return None
    if _chinese_count(first) < 4 or _chinese_count(second) < 4:
        return None
    if SequenceMatcher(None, first, second).ratio() < 0.75:
        return None
    prefix = _common_prefix([first, second])
    if len(prefix) < min(len(first), len(second)) - 1:
        return None

    first_duration = float(group[0].get("end", 0.0)) - float(group[0].get("start", 0.0))
    second_duration = float(group[1].get("end", 0.0)) - float(group[1].get("start", 0.0))
    dominant_duration = max(first_duration, second_duration)
    pair_span = max(float(item.get("end", 0.0)) for item in group) - min(float(item.get("start", 0.0)) for item in group)
    if dominant_duration < 1.5 or pair_span < 1.5 or pair_span > 5.0:
        return None

    first_score = (first_duration, float(group[0].get("confidence", 0.0)), len(first))
    second_score = (second_duration, float(group[1].get("confidence", 0.0)), len(second))
    return first if first_score >= second_score else second


def _is_near_duplicate_group(texts: list[str]) -> bool:
    if len(texts) < 2:
        return False
    if max(texts.count(text) for text in set(texts)) < 2:
        return False
    if max(len(text) for text in texts) - min(len(text) for text in texts) > 2:
        return False
    similarities = [
        SequenceMatcher(None, texts[i], texts[j]).ratio()
        for i in range(len(texts))
        for j in range(i + 1, len(texts))
    ]
    return bool(similarities) and min(similarities) >= 0.78


def _choose_representative_ocr_text(group: list[dict]) -> str:
    texts = [str(item.get("text", "")).strip() for item in group]
    counts = {text: texts.count(text) for text in set(texts)}
    best_text = texts[0]
    best_score = (-1, -1.0, -1)
    for text in counts:
        confidence = max(
            float(item.get("confidence", 0.0))
            for item in group
            if str(item.get("text", "")).strip() == text
        )
        score = (counts[text], confidence, len(text))
        if score > best_score:
            best_text = text
            best_score = score
    return best_text


def _build_merged_ocr_segment(group: list[dict], text: str) -> dict:
    start = min(float(item.get("start", 0.0)) for item in group)
    end = max(float(item.get("end", start + 0.3)) for item in group)
    confidence = max(float(item.get("confidence", 0.0)) for item in group)
    return {
        "text": text,
        "start": start,
        "end": end,
        "duration": end - start,
        "confidence": confidence,
        "source": "ocr",
    }


def _filter_low_confidence_ocr_noise(segments: list[dict]) -> tuple[list[dict], list[dict]]:
    cleaned = []
    report = []
    for index, segment in enumerate(segments):
        if _is_low_confidence_ocr_noise(segments, index):
            report.append(
                {
                    "reason": "low_confidence_noise",
                    "original": [dict(segment)],
                    "removed": dict(segment),
                }
            )
            continue
        cleaned.append(segment)
    return cleaned, report


def _is_low_confidence_ocr_noise(segments: list[dict], index: int) -> bool:
    segment = segments[index]
    if segment.get("source") != "ocr":
        return False

    text = str(segment.get("text", "")).strip()
    confidence = float(segment.get("confidence", 0.0))
    if confidence >= 0.90 or _chinese_count(text) < 2:
        return False
    if _has_adjacent_ocr_text_support(segments, index):
        return False
    if _has_low_confidence_duplicate_neighbor(segments, index):
        return True

    duration = float(segment.get("end", 0.0)) - float(segment.get("start", 0.0))
    chinese_chars = _chinese_count(text)

    if confidence < 0.65:
        return duration <= 0.55 or chinese_chars <= 6
    return duration <= 0.55 and chinese_chars >= 4


def _has_adjacent_ocr_text_support(segments: list[dict], index: int) -> bool:
    text = str(segments[index].get("text", "")).strip()
    start = float(segments[index].get("start", 0.0))
    end = float(segments[index].get("end", start))
    for neighbor_index in (index - 1, index + 1):
        if neighbor_index < 0 or neighbor_index >= len(segments):
            continue
        neighbor = segments[neighbor_index]
        if neighbor.get("source") != "ocr":
            continue
        if float(neighbor.get("confidence", 0.0)) < 0.90:
            continue
        neighbor_text = str(neighbor.get("text", "")).strip()
        if not neighbor_text:
            continue
        gap = (
            start - float(neighbor.get("end", 0.0))
            if neighbor_index < index
            else float(neighbor.get("start", 0.0)) - end
        )
        if gap > 0.5:
            continue
        shorter, longer = (text, neighbor_text) if len(text) <= len(neighbor_text) else (neighbor_text, text)
        if _chinese_count(shorter) >= 2 and shorter in longer:
            return True
        similarity = SequenceMatcher(None, text, neighbor_text).ratio()
        if similarity >= 0.82:
            return True
        if abs(len(text) - len(neighbor_text)) <= 1 and similarity >= 0.72:
            return True
    return False


def _has_low_confidence_duplicate_neighbor(segments: list[dict], index: int) -> bool:
    text = str(segments[index].get("text", "")).strip()
    start = float(segments[index].get("start", 0.0))
    end = float(segments[index].get("end", start))
    for neighbor_index in (index - 1, index + 1):
        if neighbor_index < 0 or neighbor_index >= len(segments):
            continue
        neighbor = segments[neighbor_index]
        if neighbor.get("source") != "ocr" or float(neighbor.get("confidence", 0.0)) >= 0.90:
            continue
        neighbor_text = str(neighbor.get("text", "")).strip()
        if not neighbor_text or abs(len(text) - len(neighbor_text)) > 2:
            continue
        gap = (
            start - float(neighbor.get("end", 0.0))
            if neighbor_index < index
            else float(neighbor.get("start", 0.0)) - end
        )
        if gap <= 0.5 and SequenceMatcher(None, text, neighbor_text).ratio() >= 0.72:
            return True
    return False


def _common_prefix(texts: list[str]) -> str:
    if not texts:
        return ""
    prefix = texts[0]
    for text in texts[1:]:
        while prefix and not text.startswith(prefix):
            prefix = prefix[:-1]
    return prefix


def _chinese_count(text: str) -> int:
    return sum(1 for char in text if "\u4e00" <= char <= "\u9fff")


def _write_ocr_cleanup_report(path: Path, report: list[dict]) -> None:
    if not report:
        if path.exists():
            _safe_remove(str(path))
        return
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)


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
