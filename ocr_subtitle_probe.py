"""Probe hard subtitles in a video with PaddleOCR.

This is a standalone validation tool. It does not participate in the main
pipeline yet.
"""

import argparse
import difflib
import json
from pathlib import Path

import cv2

WATERMARK_KEYWORDS = (
    "请勿模仿",
    "勿模仿",
    "热门短剧",
    "热门短刷",
    "本故事纯属虚构",
    "本故事纯属",
    "纯属虚构",
)
SUBTITLE_Y_MIN_RATIO = 0.58
SUBTITLE_Y_MAX_RATIO = 0.92
SUBTITLE_CENTER_TOLERANCE_RATIO = 0.36


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_video", required=True)
    parser.add_argument("--output_txt", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--interval", type=float, default=0.33)
    parser.add_argument("--fast_interval", type=float, default=0.75)
    parser.add_argument("--crop_top", type=float, default=0.58)
    parser.add_argument("--min_conf", type=float, default=0.55)
    args = parser.parse_args()

    segments = extract_hard_subtitles(
        args.input_video,
        interval=args.interval,
        fast_interval=args.fast_interval,
        crop_top=args.crop_top,
        min_conf=args.min_conf,
    )
    Path(args.output_json).write_text(
        json.dumps(segments, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = [f"[{s['start']:.2f}-{s['end']:.2f}] {s['text']}  (conf={s['confidence']:.2f})" for s in segments]
    Path(args.output_txt).write_text("\n".join(lines), encoding="utf-8")
    print(args.output_txt)
    print(args.output_json)
    print(f"segments={len(segments)}")
    return 0


def extract_hard_subtitles(video_path: str, interval: float, fast_interval: float, crop_top: float, min_conf: float) -> list[dict]:
    from paddleocr import PaddleOCR

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    duration = frame_count / fps if fps > 0 else 0
    fast_step = max(1, int(round(fps * fast_interval)))
    fine_step = max(1, int(round(fps * interval)))

    ocr = PaddleOCR(
        lang="ch",
        ocr_version="PP-OCRv4",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        text_det_limit_side_len=960,
    )
    active_windows = _scan_active_windows(cap, ocr, fps, frame_count, fast_step, crop_top, min_conf)
    samples = []
    for start_frame, end_frame in active_windows:
        frame_index = start_frame
        while frame_index <= end_frame:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()
            if not ok:
                break
            timestamp = frame_index / fps
            text, confidence = _read_subtitle_region(ocr, frame, crop_top)
            if text and confidence >= min_conf:
                samples.append({"time": timestamp, "text": text, "confidence": confidence})
            frame_index += fine_step

    cap.release()
    samples = _filter_persistent_text_samples(samples)
    segments = _merge_samples(samples, max_gap=interval * 2.5, sample_interval=interval)
    segments = _filter_noisy_segments(segments)
    return _dedupe_adjacent_segments(segments)


def _scan_active_windows(cap, ocr, fps: float, frame_count: float, step: int, crop_top: float, min_conf: float) -> list[tuple[int, int]]:
    windows = []
    current_start = None
    frame_index = 0
    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok:
            break
        text, confidence = _read_subtitle_region(ocr, frame, crop_top)
        if text and confidence >= min_conf:
            if current_start is None:
                current_start = max(0, frame_index - step)
        elif current_start is not None:
            windows.append((current_start, min(int(frame_count), frame_index + step)))
            current_start = None
        frame_index += step
        if frame_count and frame_index > frame_count:
            break
    if current_start is not None:
        windows.append((current_start, int(frame_count)))
    return _merge_windows(windows, max_gap=step * 2)


def _merge_windows(windows: list[tuple[int, int]], max_gap: int) -> list[tuple[int, int]]:
    if not windows:
        return []
    merged = [windows[0]]
    for start, end in windows[1:]:
        prev_start, prev_end = merged[-1]
        if start - prev_end <= max_gap:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def _read_subtitle_region(ocr, frame, crop_top: float) -> tuple[str, float]:
    height = frame.shape[0]
    y0 = int(height * crop_top)
    crop = frame[y0:, :]
    result = ocr.predict(crop)
    candidates = []
    for item in result:
        data = item if isinstance(item, dict) else {}
        rec_texts = data.get("rec_texts") or []
        rec_scores = data.get("rec_scores") or []
        rec_boxes = _first_nonempty_boxes(
            data.get("rec_boxes"),
            data.get("dt_polys"),
            data.get("rec_polys"),
        )
        for index, (text, score) in enumerate(zip(rec_texts, rec_scores)):
            text = _clean_text(text)
            bbox = _normalize_bbox(rec_boxes[index] if index < len(rec_boxes) else None, y_offset=y0)
            if text and bbox:
                candidates.append({"text": text, "confidence": float(score), "bbox": bbox})
    return _choose_dialogue_text(candidates, frame.shape)


def _choose_dialogue_text(candidates: list[dict], frame_shape) -> tuple[str, float]:
    frame_height, frame_width = frame_shape[:2]
    filtered = [
        candidate
        for candidate in candidates
        if _is_dialogue_candidate(candidate, frame_width, frame_height)
    ]
    if not filtered:
        return "", 0.0

    filtered.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))
    lines = _group_lines(filtered)
    if not lines:
        return "", 0.0

    scored_lines = [(_score_line(line, frame_width, frame_height), line) for line in lines]
    scored_lines.sort(key=lambda item: item[0], reverse=True)
    best_score, best_line = scored_lines[0]

    # Merge an adjacent second line when it is close and also subtitle-like.
    best_y1 = min(item["bbox"][1] for item in best_line)
    best_y2 = max(item["bbox"][3] for item in best_line)
    chosen = best_line[:]
    for score, line in scored_lines[1:]:
        y1 = min(item["bbox"][1] for item in line)
        y2 = max(item["bbox"][3] for item in line)
        close = min(abs(y1 - best_y2), abs(best_y1 - y2)) <= frame_height * 0.08
        if close and score >= best_score * 0.75:
            chosen.extend(line)
            break

    chosen.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))
    text = "".join(item["text"] for item in chosen)
    confidence = sum(item["confidence"] for item in chosen) / len(chosen)
    return text, confidence


def _is_dialogue_candidate(candidate: dict, frame_width: int, frame_height: int) -> bool:
    text = candidate["text"]
    if _is_watermark_text(text):
        return False
    x1, y1, x2, y2 = candidate["bbox"]
    box_width = max(1, x2 - x1)
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    if not (frame_height * SUBTITLE_Y_MIN_RATIO <= center_y <= frame_height * SUBTITLE_Y_MAX_RATIO):
        return False
    if abs(center_x - frame_width / 2) > frame_width * SUBTITLE_CENTER_TOLERANCE_RATIO:
        return False
    if x1 < frame_width * 0.08 or x2 > frame_width * 0.92:
        return False
    if box_width < frame_width * 0.035 and len(text) < 2:
        return False
    return True


def _is_watermark_text(text: str) -> bool:
    return any(keyword in text for keyword in WATERMARK_KEYWORDS)


def _group_lines(candidates: list[dict]) -> list[list[dict]]:
    lines: list[list[dict]] = []
    for candidate in candidates:
        y1, y2 = candidate["bbox"][1], candidate["bbox"][3]
        center_y = (y1 + y2) / 2
        matched = None
        for line in lines:
            line_center = sum((item["bbox"][1] + item["bbox"][3]) / 2 for item in line) / len(line)
            line_height = max(item["bbox"][3] - item["bbox"][1] for item in line)
            if abs(center_y - line_center) <= max(12, line_height * 0.75):
                matched = line
                break
        if matched is None:
            lines.append([candidate])
        else:
            matched.append(candidate)
    return lines


def _score_line(line: list[dict], frame_width: int, frame_height: int) -> float:
    x1 = min(item["bbox"][0] for item in line)
    x2 = max(item["bbox"][2] for item in line)
    y1 = min(item["bbox"][1] for item in line)
    y2 = max(item["bbox"][3] for item in line)
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    confidence = sum(item["confidence"] for item in line) / len(line)
    center_score = 1.0 - min(1.0, abs(center_x - frame_width / 2) / (frame_width * 0.35))
    bottom_score = 1.0 - min(1.0, abs(center_y - frame_height * 0.72) / (frame_height * 0.22))
    text_len = sum(len(item["text"]) for item in line)
    length_score = min(1.0, text_len / 6)
    return confidence * 2.0 + center_score + bottom_score + length_score


def _normalize_bbox(raw_box, y_offset: int) -> tuple[int, int, int, int] | None:
    if raw_box is None:
        return None
    try:
        if hasattr(raw_box, "tolist"):
            raw_box = raw_box.tolist()
        if len(raw_box) == 4 and all(isinstance(value, (int, float)) for value in raw_box):
            x1, y1, x2, y2 = raw_box
        else:
            xs = [point[0] for point in raw_box]
            ys = [point[1] for point in raw_box]
            x1, x2 = min(xs), max(xs)
            y1, y2 = min(ys), max(ys)
        return int(x1), int(y1 + y_offset), int(x2), int(y2 + y_offset)
    except Exception:
        return None


def _first_nonempty_boxes(*values):
    for value in values:
        if value is None:
            continue
        try:
            if hasattr(value, "size") and value.size == 0:
                continue
            if len(value) == 0:
                continue
        except TypeError:
            continue
        return value
    return []


def _merge_samples(samples: list[dict], max_gap: float, sample_interval: float) -> list[dict]:
    merged = []
    current = None
    for sample in samples:
        if current and _similar(current["text"], sample["text"]) >= 0.82 and sample["time"] - current["end"] <= max_gap:
            current["end"] = sample["time"]
            current["text"] = _pick_better_text(current["text"], sample["text"])
            current["scores"].append(sample["confidence"])
            continue
        if current:
            merged.append(_finalize(current, sample_interval))
        current = {
            "start": sample["time"],
            "end": sample["time"],
            "text": sample["text"],
            "scores": [sample["confidence"]],
        }
    if current:
        merged.append(_finalize(current, sample_interval))
    return merged


def _filter_persistent_text_samples(
    samples: list[dict],
    min_count: int = 4,
    min_span: float = 8.0,
    max_cluster_gap: float = 1.2,
) -> list[dict]:
    """Drop text that stays continuously on screen like a watermark."""
    grouped: dict[str, list[dict]] = {}
    for sample in samples:
        grouped.setdefault(sample["text"], []).append(sample)

    persistent_texts = set()
    for text, items in grouped.items():
        if _has_persistent_cluster(sorted(items, key=lambda item: item["time"]), min_count, min_span, max_cluster_gap):
            persistent_texts.add(text)

    if not persistent_texts:
        return samples
    return [sample for sample in samples if sample["text"] not in persistent_texts]


def _has_persistent_cluster(items: list[dict], min_count: int, min_span: float, max_cluster_gap: float) -> bool:
    cluster = []
    for item in items:
        if not cluster or item["time"] - cluster[-1]["time"] <= max_cluster_gap:
            cluster.append(item)
        else:
            if _cluster_is_persistent(cluster, min_count, min_span):
                return True
            cluster = [item]
    return _cluster_is_persistent(cluster, min_count, min_span)


def _cluster_is_persistent(cluster: list[dict], min_count: int, min_span: float) -> bool:
    if len(cluster) < min_count:
        return False
    return cluster[-1]["time"] - cluster[0]["time"] >= min_span


def _filter_noisy_segments(segments: list[dict]) -> list[dict]:
    noisy_indexes = set()
    for index, segment in enumerate(segments):
        if _is_noisy_segment(segment):
            noisy_indexes.add(index)
    noisy_indexes.update(_find_noisy_similarity_cluster_indexes(segments, noisy_indexes))

    filtered = []
    for index, segment in enumerate(segments):
        if index in noisy_indexes:
            continue
        filtered.append(segment)
    return filtered


def _dedupe_adjacent_segments(
    segments: list[dict],
    max_gap: float = 0.12,
    similarity_threshold: float = 0.82,
) -> list[dict]:
    deduped = []
    current = None
    for segment in segments:
        if current and _segments_are_duplicate(current, segment, max_gap, similarity_threshold):
            current["end"] = max(float(current["end"]), float(segment["end"]))
            current["confidence"] = max(float(current.get("confidence", 0.0)), float(segment.get("confidence", 0.0)))
            if len(str(segment.get("text", ""))) > len(str(current.get("text", ""))):
                current["text"] = segment["text"]
            continue
        if current:
            current["duration"] = float(current["end"]) - float(current["start"])
            deduped.append(current)
        current = dict(segment)
    if current:
        current["duration"] = float(current["end"]) - float(current["start"])
        deduped.append(current)
    return deduped


def _segments_are_duplicate(
    first: dict,
    second: dict,
    max_gap: float,
    similarity_threshold: float,
) -> bool:
    gap = float(second.get("start", 0.0)) - float(first.get("end", 0.0))
    if gap > max_gap:
        return False
    first_text = str(first.get("text", ""))
    second_text = str(second.get("text", ""))
    if _similar(first_text, second_text) >= similarity_threshold:
        return True
    return _cyclic_text_similarity(first_text, second_text) >= similarity_threshold


def _cyclic_text_similarity(first: str, second: str) -> float:
    if not first or not second or len(first) != len(second):
        return 0.0
    doubled = first + first
    best = 0.0
    for start in range(len(first)):
        candidate = doubled[start:start + len(first)]
        best = max(best, _similar(candidate, second))
    return best


def _is_noisy_segment(segment: dict) -> bool:
    text = str(segment.get("text", ""))
    if not text:
        return True
    chinese_count = sum("\u4e00" <= char <= "\u9fff" for char in text)
    ascii_noise_count = sum(char.isascii() and (char.isalnum() or not char.isspace()) for char in text)
    latin_digit_count = sum(char.isascii() and char.isalnum() for char in text)
    if chinese_count >= 2 and latin_digit_count >= 2:
        return True
    if ascii_noise_count and ascii_noise_count >= max(2, chinese_count):
        return True
    if any(char in text for char in "&@#$%^*_+=<>\\|~"):
        return True
    duration = float(segment.get("end", 0.0)) - float(segment.get("start", 0.0))
    confidence = float(segment.get("confidence", 0.0))
    if duration < 0.35 and len(text) <= 3 and confidence < 0.99:
        return True
    return False


def _find_noisy_similarity_cluster_indexes(
    segments: list[dict],
    seed_indexes: set[int],
    window_seconds: float = 5.0,
    min_cluster_size: int = 3,
) -> set[int]:
    cluster_indexes: set[int] = set()
    for seed_index in sorted(seed_indexes):
        seed = segments[seed_index]
        seed_start = float(seed.get("start", 0.0))
        seed_text = _normalize_noise_text(str(seed.get("text", "")))
        if not seed_text:
            continue
        candidates = []
        for index, segment in enumerate(segments):
            start = float(segment.get("start", 0.0))
            if start < seed_start - 0.05 or start - seed_start > window_seconds:
                continue
            text = _normalize_noise_text(str(segment.get("text", "")))
            if not text:
                continue
            if _texts_are_noise_similar(seed_text, text):
                candidates.append(index)
        if len(candidates) >= min_cluster_size:
            cluster_indexes.update(candidates)
    return cluster_indexes


def _texts_are_noise_similar(a: str, b: str) -> bool:
    if a in b or b in a:
        return True
    return _similar(a, b) >= 0.55


def _normalize_noise_text(text: str) -> str:
    return "".join(char for char in text if "\u4e00" <= char <= "\u9fff")


def _finalize(segment: dict, sample_interval: float) -> dict:
    end = segment["end"] + sample_interval
    return {
        "start": segment["start"],
        "end": max(end, segment["start"] + 0.3),
        "text": segment["text"],
        "confidence": sum(segment["scores"]) / len(segment["scores"]),
        "source": "ocr",
    }


def _similar(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


def _pick_better_text(a: str, b: str) -> str:
    return b if len(b) > len(a) else a


def _clean_text(text: str) -> str:
    return "".join(ch for ch in str(text).strip() if not ch.isspace())


if __name__ == "__main__":
    raise SystemExit(main())
