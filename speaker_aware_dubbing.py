#!/usr/bin/env python3
"""
speaker_aware_dubbing.py — 说话人感知配音模块（优化版）

性能优化点：
1. 说话人分离 与 video_dubbing 的 ASR+翻译 并行执行（最大收益）
   - 通过 run_diarization_async() 提前在后台线程启动分离
   - ASR 结束后调用 wait_diarization() 获取结果，几乎零等待
2. 性别识别多线程并行（各说话人互不依赖）
3. Embedding 计算多线程并行
4. 逻辑与 test_diarization.py 完全一致，准确率不变
"""

import os
import warnings
import tempfile
import subprocess
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple, Optional

import numpy as np

warnings.filterwarnings("ignore")


# =====================================================================
# 配置 —— 与 test_diarization.py 完全一致
# =====================================================================
class Config:
    SAMPLE_RATE = 16000
    MIN_SEGMENT_DURATION = 0.5

    DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"

    CLUSTERING_THRESHOLD = 0.45
    MIN_DURATION_OFF = 0.1

    MIN_SPEAKERS = 2
    MAX_SPEAKERS = 8

    MERGE_COSINE_THRESHOLD = 0.82
    GENDER_CONF_THRESHOLD = 0.55

    # 性别识别并行线程数（说话人数量通常 2~8，全部并行即可）
    GENDER_WORKERS = 4
    # Embedding 并行线程数
    EMBEDDING_WORKERS = 4

    # 稳定优先：性别以 speaker 级长语音聚合为准，短字幕片段只做未匹配兜底。
    PER_SEGMENT_GENDER = False
    MIN_SEGMENT_SPEAKER_OVERLAP = 0.25
    MIN_SEGMENT_SPEAKER_RATIO = 0.35

    HF_TOKEN = os.environ.get("HF_TOKEN", "")


# =====================================================================
# 优化：异步启动说话人分离（与 ASR 并行）
# =====================================================================

_diarization_future = None      # Future 对象
_diarization_executor = None    # ThreadPoolExecutor，保持引用防止GC
_diarization_waveform = None    # 供后续步骤复用的 waveform
_diarization_sr = None
_diarization_audio_path = None  # 临时文件路径，用完后清理


def run_diarization_async(video_path: str, hf_token: str = None):
    """
    在后台线程启动说话人分离，立即返回。
    在 video_dubbing.py 的 ASR 开始前调用，让分离和 ASR 并行跑。

    用法：
        run_diarization_async(video_path, hf_token)   # ASR 开始前调用
        ... ASR + 翻译 ...
        speaker_map = wait_diarization()              # ASR 结束后取结果
    """
    global _diarization_future, _diarization_waveform, _diarization_sr
    global _diarization_audio_path

    token = hf_token or Config.HF_TOKEN
    if not token:
        return

    global _diarization_executor
    _diarization_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="diarize")
    _diarization_future = _diarization_executor.submit(_run_full_pipeline, video_path, token)
    print("[说话人识别] 已在后台启动，与 ASR 并行运行...")


def wait_diarization() -> Dict[str, Dict]:
    """
    等待后台分离完成并返回结果。
    若 run_diarization_async 未调用，直接返回空字典。
    """
    global _diarization_future
    if _diarization_future is None:
        return {}
    try:
        result = _diarization_future.result()  # 阻塞直到完成
        _diarization_future = None
        return result
    except Exception as e:
        print(f"[说话人识别] 后台任务异常: {e}")
        _diarization_future = None
        return {}
    finally:
        global _diarization_executor
        if _diarization_executor is not None:
            _diarization_executor.shutdown(wait=False)
            _diarization_executor = None


# =====================================================================
# 对外接口 1：同步版（不用异步时的简单调用方式）
# =====================================================================
def analyze_speakers_for_video(
    video_path: str,
    hf_token: str = None
) -> Dict[str, Dict]:
    """同步版：直接阻塞等待结果"""
    token = hf_token or Config.HF_TOKEN
    if not token:
        print("[说话人识别] 未设置 HF_TOKEN，跳过")
        return {}
    return _run_full_pipeline(video_path, token)


# =====================================================================
# 对外接口 2：分配声音
# =====================================================================
def build_speaker_voice_map(
    speaker_map: Dict[str, Dict],
    target_lang: str,
    available_voices: Dict[str, Dict],
    fallback_voice_key: str
) -> Dict[str, str]:
    """按说话人性别顺序分配不同声音"""
    if not speaker_map:
        return {}

    lang_prefix = _get_lang_prefix(target_lang)
    male_voices = sorted([
        k for k in available_voices
        if k.startswith(lang_prefix) and ("_male_" in k or "_m0" in k)
    ])
    female_voices = sorted([
        k for k in available_voices
        if k.startswith(lang_prefix) and ("_female_" in k or "_f0" in k)
    ])

    if not male_voices and not female_voices:
        male_voices = sorted([k for k in available_voices if "_vctk_vits_m" in k])
        female_voices = sorted([k for k in available_voices if "_vctk_vits_f" in k])

    sorted_speakers = sorted(
        speaker_map.items(),
        key=lambda x: x[1].get("total_duration", 0),
        reverse=True
    )

    male_idx = 0
    female_idx = 0
    result = {}

    print("[说话人识别] Speaker 默认声音分配:")
    for speaker_id, info in sorted_speakers:
        gender = info.get("subtitle_gender") or info.get("gender", "unknown")
        if gender == "unknown":
            gender = "female" if ("female" in fallback_voice_key or "_f0" in fallback_voice_key) else "male"

        if gender == "male":
            voice_key = male_voices[male_idx % len(male_voices)] if male_voices else fallback_voice_key
            male_idx += 1
        else:
            voice_key = female_voices[female_idx % len(female_voices)] if female_voices else fallback_voice_key
            female_idx += 1

        result[speaker_id] = voice_key
        gz = "男" if gender == "male" else "女"
        print(f"  {speaker_id} ({gz}) → {voice_key}")

    return result


# =====================================================================
# 对外接口 3：按时间点查声音
# =====================================================================
def get_voice_for_segment(
    seg_start: float,
    seg_end: float,
    speaker_map: Dict[str, Dict],
    speaker_voice_map: Dict[str, str],
    fallback_voice_key: str,
    available_voices: Dict = None,
    target_lang: str = "en"
) -> str:
    """
    找与该 ASR 片段时间重叠最多的说话人片段，返回其 voice_key

    如果启用了 per_segment 模式，会根据片段的性别选择对应声音
    """
    if not speaker_map:
        return fallback_voice_key

    best_speaker = None
    best_overlap = 0.0

    for speaker_id, info in speaker_map.items():
        for diar_start, diar_end in info.get("segments", []):
            overlap = max(0.0, min(seg_end, diar_end) - max(seg_start, diar_start))
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = speaker_id

    seg_duration = max(0.0, seg_end - seg_start)
    overlap_ratio = best_overlap / seg_duration if seg_duration > 0 else 0.0
    if (
        best_speaker is None
        or (
            best_overlap < Config.MIN_SEGMENT_SPEAKER_OVERLAP
            and overlap_ratio < Config.MIN_SEGMENT_SPEAKER_RATIO
        )
    ):
        return fallback_voice_key

    alignment = _find_subtitle_alignment(seg_start, seg_end, speaker_map.get(best_speaker, {}))
    aligned_gender = None
    aligned_confidence = 0.0
    if alignment:
        aligned_gender = (
            alignment.get("final_gender")
            or alignment.get("smoothed_gender")
            or alignment.get("segment_gender")
        )
        aligned_confidence = alignment.get(
            "final_confidence",
            alignment.get("smoothed_confidence", alignment.get("segment_confidence", 0.0))
        )
    speaker_default_voice = speaker_voice_map.get(best_speaker)
    speaker_default_gender = _voice_key_gender(speaker_default_voice) if speaker_default_voice else "unknown"
    if (
        aligned_gender in ("male", "female")
        and aligned_confidence >= 0.85
        and available_voices
        and speaker_default_gender != aligned_gender
    ):
        return _get_voice_by_gender(aligned_gender, target_lang, available_voices, fallback_voice_key)

    if speaker_default_voice:
        return speaker_default_voice

    speaker_gender = (
        speaker_map.get(best_speaker, {}).get("subtitle_gender")
        or speaker_map.get(best_speaker, {}).get("gender")
    )
    if speaker_gender in ("male", "female") and available_voices:
        return _get_voice_by_gender(speaker_gender, target_lang, available_voices, fallback_voice_key)

    return fallback_voice_key


def explain_segment_voice_alignment(
    seg_start: float,
    seg_end: float,
    text: str,
    speaker_map: Dict[str, Dict],
    speaker_voice_map: Dict[str, str],
    fallback_voice_key: str,
    available_voices: Dict = None,
    target_lang: str = "en",
) -> Dict:
    """Return diagnostic info for subtitle-to-speaker voice assignment."""
    best_speaker = None
    best_overlap = 0.0
    for speaker_id, info in speaker_map.items():
        for diar_start, diar_end in info.get("segments", []):
            overlap = max(0.0, min(seg_end, diar_end) - max(seg_start, diar_start))
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = speaker_id

    seg_duration = max(0.0, seg_end - seg_start)
    overlap_ratio = best_overlap / seg_duration if seg_duration > 0 else 0.0
    matched = (
        best_speaker is not None
        and (
            best_overlap >= Config.MIN_SEGMENT_SPEAKER_OVERLAP
            or overlap_ratio >= Config.MIN_SEGMENT_SPEAKER_RATIO
        )
    )
    speaker_info = speaker_map.get(best_speaker, {}) if matched else {}
    alignment = _find_subtitle_alignment(seg_start, seg_end, speaker_info) if matched else None
    speaker_gender = speaker_info.get("gender", "unknown")
    subtitle_gender = speaker_info.get("subtitle_gender", "unknown")
    segment_gender = alignment.get("segment_gender", "unknown") if alignment else "unknown"
    smoothed_gender = alignment.get("smoothed_gender", "unknown") if alignment else "unknown"
    final_gender = alignment.get("final_gender", "unknown") if alignment else "unknown"
    segment_confidence = alignment.get("segment_confidence", 0.0) if alignment else 0.0
    final_confidence = alignment.get(
        "final_confidence",
        alignment.get("smoothed_confidence", segment_confidence) if alignment else 0.0,
    ) if alignment else 0.0
    f0_gender = alignment.get("f0_gender", "unknown") if alignment else "unknown"
    f0_confidence = alignment.get("f0_confidence", 0.0) if alignment else 0.0
    selected_voice = get_voice_for_segment(
        seg_start,
        seg_end,
        speaker_map,
        speaker_voice_map,
        fallback_voice_key,
        available_voices=available_voices,
        target_lang=target_lang,
    )
    speaker_default_voice = speaker_voice_map.get(best_speaker) if matched else None
    if not matched:
        voice_source = "fallback"
    elif speaker_default_voice and selected_voice == speaker_default_voice:
        voice_source = "speaker_default"
    elif selected_voice == fallback_voice_key and not speaker_default_voice:
        voice_source = "fallback"
    else:
        voice_source = "segment_override"
    review_reasons = _segment_review_reasons(
        matched=matched,
        overlap_ratio=overlap_ratio,
        duration=seg_duration,
        speaker_gender=speaker_gender,
        subtitle_gender=subtitle_gender,
        segment_gender=segment_gender,
        segment_confidence=segment_confidence,
        final_gender=final_gender,
        final_confidence=final_confidence,
        f0_gender=f0_gender,
        f0_confidence=f0_confidence,
        selected_voice=selected_voice,
        fallback_voice_key=fallback_voice_key,
    )
    return {
        "text": text,
        "start": seg_start,
        "end": seg_end,
        "duration": seg_duration,
        "speaker": best_speaker if matched else None,
        "overlap": best_overlap,
        "overlap_ratio": overlap_ratio,
        "speaker_gender": speaker_gender,
        "subtitle_gender": subtitle_gender,
        "segment_gender": segment_gender,
        "smoothed_gender": smoothed_gender,
        "final_gender": final_gender,
        "final_reason": alignment.get("final_reason", "unknown") if alignment else "unknown",
        "segment_confidence": segment_confidence,
        "final_confidence": final_confidence,
        "f0_gender": f0_gender,
        "f0_confidence": f0_confidence,
        "voice": selected_voice,
        "voice_gender": _voice_key_gender(selected_voice),
        "voice_source": voice_source,
        "speaker_default_voice": speaker_default_voice,
        "needs_review": bool(review_reasons),
        "review_reasons": review_reasons,
    }


def print_voice_alignment_summary(alignment_report: List[Dict], fallback_voice_key: str) -> None:
    """Print final per-segment voice diagnostics without changing assignment logic."""
    if not alignment_report:
        return

    voice_counts = Counter(item.get("voice", "unknown") for item in alignment_report)
    gender_counts = Counter(item.get("voice_gender", "unknown") for item in alignment_report)
    source_counts = Counter(item.get("voice_source", "unknown") for item in alignment_report)
    review_count = sum(1 for item in alignment_report if item.get("needs_review"))
    reason_counts = Counter(
        reason
        for item in alignment_report
        for reason in item.get("review_reasons", [])
    )

    print("[说话人识别] 最终片段配音声音统计:")
    print("  voice_key:")
    for voice_key, count in sorted(voice_counts.items()):
        print(f"    {voice_key} ({_voice_key_gender(voice_key)}): {count} 片段")

    print("  voice_gender:")
    for gender, count in sorted(gender_counts.items()):
        gz = {"male": "男", "female": "女", "unknown": "未知"}.get(gender, gender)
        print(f"    {gz}: {count} 片段")

    print("  voice_source:")
    source_labels = {
        "speaker_default": "speaker 默认 voice",
        "segment_override": "片段级 gender 覆盖",
        "fallback": "fallback voice",
    }
    for source, count in sorted(source_counts.items()):
        print(f"    {source}: {count} 片段 ({source_labels.get(source, source)})")

    print(f"  needs_review: {review_count}/{len(alignment_report)}")
    if reason_counts:
        print("  review_reasons:")
        for reason, count in reason_counts.most_common():
            print(f"    {reason}: {count}")

    print(f"  fallback voice: {fallback_voice_key} ({_voice_key_gender(fallback_voice_key)})")


def _segment_review_reasons(
    matched: bool,
    overlap_ratio: float,
    duration: float,
    speaker_gender: str,
    subtitle_gender: str,
    segment_gender: str,
    segment_confidence: float,
    final_gender: str,
    final_confidence: float,
    f0_gender: str,
    f0_confidence: float,
    selected_voice: str,
    fallback_voice_key: str,
) -> List[str]:
    reasons = []
    if not matched:
        reasons.append("no_speaker_match")
    elif overlap_ratio < 0.5:
        reasons.append("low_speaker_overlap")

    if duration < 0.8:
        reasons.append("short_segment")

    if speaker_gender not in ("male", "female") and subtitle_gender not in ("male", "female"):
        reasons.append("unknown_speaker_gender")

    if (
        speaker_gender in ("male", "female")
        and segment_gender in ("male", "female")
        and speaker_gender != segment_gender
        and segment_confidence >= 0.85
    ):
        reasons.append("segment_speaker_gender_conflict")

    if (
        speaker_gender in ("male", "female")
        and final_gender in ("male", "female")
        and speaker_gender != final_gender
        and final_confidence >= 0.85
    ):
        reasons.append("final_speaker_gender_conflict")

    if (
        segment_gender in ("male", "female")
        and f0_gender in ("male", "female")
        and segment_gender != f0_gender
        and segment_confidence >= 0.75
        and f0_confidence >= 0.65
    ):
        reasons.append("ecapa_f0_conflict")

    if selected_voice == fallback_voice_key and not matched:
        reasons.append("fallback_voice")

    return sorted(set(reasons))


def _voice_key_gender(voice_key: str) -> str:
    if "_f0" in voice_key or "_female_" in voice_key:
        return "female"
    if "_m0" in voice_key or "_male_" in voice_key:
        return "male"
    return "unknown"


def enrich_speaker_map_with_subtitle_genders(
    audio_path: str,
    recognized_segments: List[Dict],
    speaker_map: Dict[str, Dict],
) -> Dict[str, Dict]:
    """
    Use OCR/ASR subtitle time spans only to stabilize speaker gender assignment.

    The primary speaker timeline still comes from pyannote on dialogue.wav. This
    pass classifies each recognized subtitle span, aligns it to the best speaker
    by overlap, and stores a conservative speaker-level subtitle_gender vote.
    """
    if not speaker_map or not recognized_segments:
        return speaker_map

    try:
        import librosa
        from gender_classifier import GenderClassifier, Config as GC
    except Exception as e:
        print(f"  [说话人识别] 字幕段性别校准不可用: {e}")
        return speaker_map

    try:
        waveform, sr = librosa.load(audio_path, sr=Config.SAMPLE_RATE, mono=True)
    except Exception as e:
        print(f"  [说话人识别] 字幕段性别校准读取音频失败: {e}")
        return speaker_map

    classifier = GenderClassifier()
    classifier.load_models()

    votes = {sp: {"male": 0.0, "female": 0.0} for sp in speaker_map}
    alignments = {sp: [] for sp in speaker_map}
    all_alignments = []
    audio_duration = len(waveform) / sr

    for idx, seg in enumerate(recognized_segments):
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start))
        text = str(seg.get("text") or seg.get("original_text") or "")
        best_speaker, best_overlap, overlap_ratio = _best_speaker_for_span(start, end, speaker_map)
        if best_speaker is None:
            continue

        padded_start = max(0.0, start - 0.08)
        padded_end = min(audio_duration, end + 0.08)
        audio = waveform[int(padded_start * sr):int(padded_end * sr)]
        gender, confidence, method, f0_gender, f0_confidence = _classify_subtitle_audio_segment(
            classifier, audio, sr, GC
        )

        duration = max(0.0, end - start)
        if gender in ("male", "female"):
            votes[best_speaker][gender] += duration * max(confidence, 0.01)

        alignment = {
            "idx": seg.get("idx", idx),
            "speaker": best_speaker,
            "text": text,
            "start": start,
            "end": end,
            "overlap": best_overlap,
            "overlap_ratio": overlap_ratio,
            "segment_gender": gender,
            "segment_confidence": confidence,
            "segment_method": method,
            "f0_gender": f0_gender,
            "f0_confidence": f0_confidence,
        }
        alignments[best_speaker].append(alignment)
        all_alignments.append(alignment)

    decoded_alignments = _decode_subtitle_gender_sequence(all_alignments, votes)
    decoded_by_speaker = {sp: [] for sp in speaker_map}
    for alignment in decoded_alignments:
        decoded_by_speaker.setdefault(alignment["speaker"], []).append(alignment)

    for speaker_id, info in speaker_map.items():
        speaker_votes = votes.get(speaker_id, {"male": 0.0, "female": 0.0})
        male_score = speaker_votes["male"]
        female_score = speaker_votes["female"]
        total = male_score + female_score
        subtitle_gender = "unknown"
        subtitle_confidence = 0.0
        if total > 0:
            subtitle_gender = "male" if male_score >= female_score else "female"
            subtitle_confidence = max(male_score, female_score) / total
        info["subtitle_gender"] = subtitle_gender if subtitle_confidence >= 0.55 else "unknown"
        info["subtitle_gender_confidence"] = subtitle_confidence
        info["subtitle_gender_votes"] = speaker_votes
        info["subtitle_alignments"] = decoded_by_speaker.get(speaker_id, [])
        if info["subtitle_gender"] != "unknown":
            gz = "男" if info["subtitle_gender"] == "male" else "女"
            print(f"  [字幕校准] {speaker_id}: {gz} (置信度={subtitle_confidence:.2f}, "
                  f"男={male_score:.2f}, 女={female_score:.2f})")

    return speaker_map


def _decode_subtitle_gender_sequence(alignments: List[Dict], votes: Dict[str, Dict[str, float]]) -> List[Dict]:
    """Decode final per-subtitle gender using acoustic evidence plus local turn continuity."""
    ordered = [dict(item) for item in sorted(alignments, key=lambda item: (item.get("start", 0.0), item.get("end", 0.0)))]

    speaker_dominant = {}
    for speaker_id, speaker_votes in votes.items():
        male_score = speaker_votes.get("male", 0.0)
        female_score = speaker_votes.get("female", 0.0)
        if male_score + female_score <= 0:
            speaker_dominant[speaker_id] = "unknown"
        else:
            speaker_dominant[speaker_id] = "male" if male_score >= female_score else "female"

    for item in ordered:
        gender = item.get("segment_gender", "unknown")
        confidence = float(item.get("segment_confidence", 0.0) or 0.0)
        item["final_gender"] = gender if gender in ("male", "female") and confidence >= 0.85 else "unknown"
        item["final_confidence"] = confidence if item["final_gender"] != "unknown" else 0.0
        item["final_reason"] = "segment" if item["final_gender"] != "unknown" else "unknown"

    # If pyannote starts a long same-speaker turn slightly late/early, the first
    # one or two subtitle snippets can be classified like the previous speaker.
    # Correct only the start of a speaker-local run with enough following
    # evidence. Do not let the speaker majority override arbitrary middle spans.
    by_speaker = {}
    for item in ordered:
        by_speaker.setdefault(item.get("speaker"), []).append(item)
    for speaker_id, speaker_items in by_speaker.items():
        dominant = speaker_dominant.get(speaker_id, "unknown")
        if dominant not in ("male", "female") or len(speaker_items) < 4:
            continue
        for idx, item in enumerate(speaker_items[:2]):
            if item.get("final_gender") == dominant:
                continue
            later = speaker_items[idx + 1:idx + 5]
            close_later = [
                candidate for candidate in later
                if candidate.get("start", 0.0) - item.get("end", 0.0) <= 5.0
            ]
            dominant_later = sum(1 for candidate in close_later if candidate.get("final_gender") == dominant)
            if dominant_later >= 2:
                item["final_gender"] = dominant
                item["final_confidence"] = max(float(item.get("final_confidence", 0.0) or 0.0), 0.86)
                item["final_reason"] = "speaker_turn_start"

    # Local continuity: fix only an isolated flip between same-gender neighbors.
    # This avoids sequence-level drift across real speaker changes.
    for idx, item in enumerate(ordered):
        gender = item.get("final_gender")
        if gender not in ("male", "female"):
            continue

        if 0 < idx < len(ordered) - 1:
            prev_item = ordered[idx - 1]
            next_item = ordered[idx + 1]
            prev_gender = prev_item.get("final_gender")
            next_gender = next_item.get("final_gender")
            if (
                prev_gender == next_gender
                and prev_gender in ("male", "female")
                and prev_gender != gender
                and item.get("start", 0.0) - prev_item.get("end", 0.0) <= 1.5
                and next_item.get("start", 0.0) - item.get("end", 0.0) <= 1.5
            ):
                item["final_gender"] = prev_gender
                item["final_confidence"] = max(float(item.get("final_confidence", 0.0) or 0.0), 0.86)
                item["final_reason"] = "between_same_neighbors"

    return ordered


def _smooth_subtitle_alignments(alignments: List[Dict], dominant_gender: str = "unknown") -> List[Dict]:
    """Correct isolated high-confidence segment flips within a local speaker run."""
    ordered = sorted(alignments, key=lambda item: (item.get("start", 0.0), item.get("end", 0.0)))
    result = []
    for idx, item in enumerate(ordered):
        current = dict(item)
        current_gender = current.get("segment_gender", "unknown")
        current_conf = float(current.get("segment_confidence", 0.0) or 0.0)
        current["smoothed_gender"] = current_gender
        current["smoothed_confidence"] = current_conf
        current["smooth_reason"] = "segment"

        if idx >= 2 and current_gender in ("male", "female"):
            prev_1 = result[idx - 1]
            prev_2 = result[idx - 2]
            prev_gender_1 = prev_1.get("smoothed_gender") or prev_1.get("segment_gender")
            prev_gender_2 = prev_2.get("smoothed_gender") or prev_2.get("segment_gender")
            gap_1 = float(current.get("start", 0.0)) - float(prev_1.get("end", 0.0))
            gap_2 = float(prev_1.get("start", 0.0)) - float(prev_2.get("end", 0.0))
            if (
                prev_gender_1 == prev_gender_2
                and prev_gender_1 in ("male", "female")
                and prev_gender_1 == dominant_gender
                and prev_gender_1 != current_gender
                and gap_1 <= 1.5
                and gap_2 <= 1.5
            ):
                current["smoothed_gender"] = prev_gender_1
                current["smoothed_confidence"] = max(
                    current_conf,
                    float(prev_1.get("smoothed_confidence", prev_1.get("segment_confidence", 0.0)) or 0.0),
                    float(prev_2.get("smoothed_confidence", prev_2.get("segment_confidence", 0.0)) or 0.0),
                )
                current["smooth_reason"] = "previous_two_segments"

        result.append(current)
    return result


def _classify_subtitle_audio_segment(classifier, audio, sr, gender_config):
    if audio is None or len(audio) == 0:
        return "unknown", 0.0, "empty", "unknown", 0.0

    ecapa_gender, ecapa_confidence = (None, 0.0)
    if classifier._model_ready is None:
        classifier.load_models()
    if classifier._model_ready:
        ecapa_gender, ecapa_confidence = classifier._predict_one(audio)

    f0_gender, f0_confidence = classifier._classify_segment_by_f0(audio, sr)

    if ecapa_gender in ("male", "female") and ecapa_confidence >= 0.85:
        return ecapa_gender, ecapa_confidence, "ecapa", f0_gender, f0_confidence
    if f0_gender in ("male", "female") and f0_confidence >= 0.65:
        return f0_gender, f0_confidence, "f0", f0_gender, f0_confidence
    return "unknown", 0.0, "unknown", f0_gender, f0_confidence


def _best_speaker_for_span(seg_start: float, seg_end: float, speaker_map: Dict[str, Dict]):
    best_speaker = None
    best_overlap = 0.0
    for speaker_id, info in speaker_map.items():
        total_overlap = 0.0
        for diar_start, diar_end in info.get("segments", []):
            total_overlap += max(0.0, min(seg_end, diar_end) - max(seg_start, diar_start))
        if total_overlap > best_overlap:
            best_overlap = total_overlap
            best_speaker = speaker_id

    seg_duration = max(0.0, seg_end - seg_start)
    overlap_ratio = best_overlap / seg_duration if seg_duration > 0 else 0.0
    if (
        best_speaker is None
        or (
            best_overlap < Config.MIN_SEGMENT_SPEAKER_OVERLAP
            and overlap_ratio < Config.MIN_SEGMENT_SPEAKER_RATIO
        )
    ):
        return None, best_overlap, overlap_ratio
    return best_speaker, best_overlap, overlap_ratio


def _find_subtitle_alignment(seg_start: float, seg_end: float, speaker_info: Dict) -> Optional[Dict]:
    best = None
    best_delta = None
    for alignment in speaker_info.get("subtitle_alignments", []):
        delta = abs(float(alignment.get("start", 0.0)) - seg_start) + abs(float(alignment.get("end", 0.0)) - seg_end)
        if best_delta is None or delta < best_delta:
            best = alignment
            best_delta = delta
    if best is not None and best_delta is not None and best_delta <= 0.25:
        return best
    return None


def _get_voice_by_gender(
    gender: str,
    target_lang: str,
    available_voices: Dict,
    fallback_voice_key: str
) -> str:
    """根据性别和语言选择合适的声音"""
    lang_prefix = {
        "en": "en", "ja": "ja", "ko": "ko", "id": "id",
        "vi": "vi", "es": "es", "tr": "tr", "pt": "pt",
        "hi": "hi", "ar": "ar", "th": "th", "fr": "fr",
        "de": "de", "it": "it", "zh": "zh", "ru": "ru",
    }.get(target_lang.lower(), "en")

    if gender == "male":
        voices = sorted([
            k for k in available_voices
            if k.startswith(lang_prefix) and ("_m0" in k or "_male_" in k)
        ])
    else:  # female
        voices = sorted([
            k for k in available_voices
            if k.startswith(lang_prefix) and ("_f0" in k or "_female_" in k)
        ])

    # 如果没有对应语言的声音，使用英文
    if not voices:
        if gender == "male":
            voices = sorted([k for k in available_voices if "_vctk_vits_m" in k])
        else:
            voices = sorted([k for k in available_voices if "_vctk_vits_f" in k])

    return voices[0] if voices else fallback_voice_key


# =====================================================================
# 内部流程
# =====================================================================

def _run_full_pipeline(video_path: str, token: str) -> Dict[str, Dict]:
    """完整的分析流程（提取音频→分离→性别→合并）"""
    audio_path = None
    try:
        t0 = time.time()

        # 1. 提取音频
        print("[说话人识别] 提取音频...")
        audio_path = _extract_audio_temp(video_path)
        if not audio_path:
            return {}

        # 2. 加载音频（只加载一次）
        import librosa
        waveform, sr = librosa.load(audio_path, sr=Config.SAMPLE_RATE, mono=True)
        duration = len(waveform) / sr
        print(f"[说话人识别] 音频时长: {duration:.1f}s")

        # 3. 说话人分离
        t1 = time.time()
        diarization = _perform_diarization(audio_path, token)
        print(f"  分离耗时: {time.time()-t1:.1f}s")

        speaker_segments = _parse_diarization(diarization)
        if not speaker_segments:
            print("[说话人识别] 未检测到说话人")
            return {}

        # 4. 性别识别（多线程并行）
        t2 = time.time()
        speaker_info = _identify_genders_parallel(waveform, sr, speaker_segments)
        print(f"  性别识别耗时: {time.time()-t2:.1f}s")

        # 5. 合并过度分割（embedding 并行计算）
        t3 = time.time()
        print("\n[说话人识别] 后处理：检查是否需要合并...")
        speaker_info = _merge_oversplit_speakers_parallel(speaker_info, waveform, sr)
        print(f"  合并检查耗时: {time.time()-t3:.1f}s")

        print(f"[说话人识别] 总耗时: {time.time()-t0:.1f}s")
        _print_summary(speaker_info)
        return speaker_info

    except Exception as e:
        print(f"[说话人识别] 失败: {e}")
        import traceback
        traceback.print_exc()
        return {}
    finally:
        if audio_path:
            _safe_remove(audio_path)


def _resolve_ffmpeg_bin() -> str:
    """与 video_dubbing.py 相同的 ffmpeg 路径解析逻辑：
    环境变量 FFMPEG_BIN > PATH > imageio-ffmpeg 捆绑二进制
    """
    import shutil as _shutil
    env_bin = os.environ.get('FFMPEG_BIN', '').strip()
    if env_bin:
        return env_bin
    if _shutil.which('ffmpeg'):
        return 'ffmpeg'
    try:
        import imageio_ffmpeg
        bundled = imageio_ffmpeg.get_ffmpeg_exe()
        if bundled and os.path.isfile(bundled):
            return bundled
    except Exception:
        pass
    return 'ffmpeg'

def _safe_remove(path: str, retries: int = 5, delay: float = 0.15) -> None:
    """安全删除，解决 Windows 文件句柄延迟释放的 PermissionError。"""
    import time as _time
    for attempt in range(retries):
        try:
            if os.path.exists(path):
                os.remove(path)
            return
        except PermissionError:
            if attempt < retries - 1:
                _time.sleep(delay)
        except OSError:
            return


def _extract_audio_temp(video_path: str) -> Optional[str]:
    ffmpeg_bin = _resolve_ffmpeg_bin()
    try:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        audio_path = tmp.name
        tmp.close()

        ext = os.path.splitext(video_path)[1].lower()
        if ext in (".wav", ".mp3", ".flac", ".m4a", ".ogg"):
            import librosa, soundfile as sf
            wav, sr = librosa.load(video_path, sr=Config.SAMPLE_RATE, mono=True)
            sf.write(audio_path, wav, Config.SAMPLE_RATE)
        else:
            cmd = [
                ffmpeg_bin, "-y", "-i", video_path,
                "-vn", "-acodec", "pcm_s16le",
                "-ar", str(Config.SAMPLE_RATE), "-ac", "1",
                audio_path
            ]
            # encoding='utf-8' 防止 Windows cp936/cp1252 编码问题
            result = subprocess.run(
                cmd, capture_output=True, text=True,
                encoding='utf-8', errors='replace'
            )
            if result.returncode != 0:
                print(f"  [错误] ffmpeg: {result.stderr.strip()[-300:]}")
                return None
        return audio_path
    except Exception as e:
        print(f"  [错误] 音频提取异常: {e}")
        return None


def _perform_diarization(audio_path: str, token: str):
    """与 test_diarization.py perform_diarization() 完全相同"""
    from pyannote.audio import Pipeline
    import torch, librosa

    print(f"[说话人识别] 运行分离 (模型: {Config.DIARIZATION_MODEL})...")
    # hf_hub 0.20+ 将 use_auth_token 改为 token，优先用新参数，旧版 pyannote 回退
    try:
        pipeline = Pipeline.from_pretrained(Config.DIARIZATION_MODEL, token=token)
    except TypeError:
        # 极少数旧版 pyannote (<3.0) 仍只认 use_auth_token
        pipeline = Pipeline.from_pretrained(Config.DIARIZATION_MODEL, use_auth_token=token)
    pipeline = pipeline.to(torch.device("cpu"))
    print("  使用 CPU 模式")

    try:
        pipeline.instantiate({
            "segmentation": {"min_duration_off": Config.MIN_DURATION_OFF},
            "clustering": {"threshold": Config.CLUSTERING_THRESHOLD}
        })
    except Exception as e:
        print(f"  [提示] 参数微调跳过: {e}")

    waveform, sr = librosa.load(audio_path, sr=Config.SAMPLE_RATE, mono=True)
    wav_tensor = torch.from_numpy(waveform.copy()).float().unsqueeze(0)
    audio_input = {"waveform": wav_tensor, "sample_rate": sr}

    try:
        output = pipeline(audio_input,
                          min_speakers=Config.MIN_SPEAKERS,
                          max_speakers=Config.MAX_SPEAKERS)
    except TypeError:
        output = pipeline(audio_input)

    return output


def _parse_diarization(diarization) -> Dict[str, List[Tuple[float, float]]]:
    """与 test_diarization.py parse_diarization() 完全相同"""
    speaker_segments: Dict[str, List[Tuple[float, float]]] = {}

    def add(start, end, speaker):
        if end - start < Config.MIN_SEGMENT_DURATION:
            return
        spk = (f"SPEAKER_{int(speaker):02d}"
               if str(speaker).lstrip("-").isdigit() else str(speaker))
        speaker_segments.setdefault(spk, []).append((float(start), float(end)))

    if hasattr(diarization, "speaker_diarization"):
        for turn, speaker in diarization.speaker_diarization:
            add(turn.start, turn.end, speaker)
    elif hasattr(diarization, "itertracks"):
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            add(turn.start, turn.end, speaker)
    else:
        raise RuntimeError(f"未知 diarization 类型: {type(diarization)}")

    print(f"  检测到 {len(speaker_segments)} 个说话人:")
    for sp, segs in sorted(speaker_segments.items()):
        total = sum(e - s for s, e in segs)
        print(f"    {sp}: {len(segs)} 段, 共 {total:.1f}s")

    return speaker_segments


def _identify_genders_parallel(
    waveform: np.ndarray,
    sr: int,
    speaker_segments: Dict[str, List[Tuple[float, float]]]
) -> Dict[str, Dict]:
    """
    性别识别 —— 多线程并行版
    注意：GenderClassifier 是单例，模型只加载一次，线程安全（只读推理）。
    """
    from gender_classifier import GenderClassifier, Config as GC

    per_segment = Config.PER_SEGMENT_GENDER
    mode_str = "（按片段）" if per_segment else "（按说话人投票）"
    print(f"[说话人识别] 性别识别{mode_str}（并行）...")

    # 预先加载模型，避免多线程竞争初始化
    classifier = GenderClassifier()
    classifier.load_models()

    def process_one(speaker, segments):
        total_duration = sum(e - s for s, e in segments)
        valid_count = sum(1 for s, e in segments if e - s >= GC.MIN_DURATION)

        if per_segment:
            # 每个片段独立判断
            segment_results = classifier.classify_each_segment(
                waveform, sr, segments, speaker_id=speaker
            )
            # 统计投票
            votes = {"male": 0, "female": 0, "unknown": 0}
            for seg_info in segment_results:
                votes[seg_info["gender"]] += 1
            main_gender = max(votes, key=votes.get)
            confidence = votes[main_gender] / len(segment_results) if segment_results else 0

            return speaker, {
                "gender": main_gender,
                "confidence": confidence,
                "segments": segments,
                "segment_genders": segment_results,  # 每个片段的性别
                "total_duration": total_duration,
                "segment_count": len(segments),
                "details": {"method": "per_segment", "votes": votes},
            }
        else:
            # 按说话人投票聚合
            gender, confidence, details = classifier.classify_speaker_segments(
                waveform, sr, segments, speaker_id=speaker
            )
            if confidence < Config.GENDER_CONF_THRESHOLD:
                gender = "unknown"

            return speaker, {
                "gender": gender,
                "confidence": confidence,
                "segments": segments,
                "total_duration": total_duration,
                "segment_count": len(segments),
                "details": details,
            }

    speaker_info = {}
    n_speakers = len(speaker_segments)
    # 说话人数量通常 ≤ 8，全部并行
    workers = min(Config.GENDER_WORKERS, n_speakers)

    with ThreadPoolExecutor(max_workers=workers,
                            thread_name_prefix="gender") as executor:
        futures = {
            executor.submit(process_one, sp, segs): sp
            for sp, segs in speaker_segments.items()
        }
        for future in as_completed(futures):
            try:
                speaker, info = future.result()
                if info.get("segment_genders"):
                    # 按片段模式
                    votes = info["details"].get("votes", {})
                    print(f"  {speaker}: 男={votes.get('male',0)} 女={votes.get('female',0)} "
                          f"未知={votes.get('unknown',0)}")
                else:
                    # 按说话人投票模式
                    gz = {"male": "男", "female": "女", "unknown": "未知"}[info["gender"]]
                    print(f"  {speaker}: {gz} (置信度={info['confidence']:.2f}, "
                          f"方法={info['details'].get('method', '?')})")
                speaker_info[speaker] = info
            except Exception as e:
                sp = futures[future]
                print(f"  {sp}: 性别识别失败 ({e})")
                speaker_info[sp] = {
                    "gender": "unknown", "confidence": 0.0,
                    "segments": speaker_segments[sp],
                    "total_duration": sum(e - s for s, e in speaker_segments[sp]),
                    "segment_count": len(speaker_segments[sp]),
                    "details": {"error": str(e)},
                }

    return speaker_info


def _merge_oversplit_speakers_parallel(
    speaker_info: Dict[str, Dict],
    waveform: np.ndarray,
    sr: int
) -> Dict[str, Dict]:
    """
    合并过度分割的说话人 —— Embedding 计算并行版
    逻辑与 test_diarization.py merge_oversplit_speakers() 完全相同
    """
    if len(speaker_info) <= 1:
        return speaker_info

    print("  [合并检查] 并行计算 embedding...")

    from gender_classifier import GenderClassifier
    classifier = GenderClassifier()

    if not classifier.load_models() or classifier._ecapa_model is None:
        print("  ECAPA 不可用，跳过合并")
        return speaker_info

    def compute_embedding(sp, info):
        segs = sorted(info["segments"], key=lambda x: x[1]-x[0], reverse=True)[:3]
        embs = []
        for start, end in segs:
            if end - start < 1.0:
                continue
            seg = waveform[int(start*sr):int(end*sr)]
            emb = classifier.get_embedding(seg, sr)
            if emb is not None:
                embs.append(emb)
        if embs:
            return sp, np.mean(embs, axis=0)
        return sp, None

    # 并行计算所有说话人的 embedding
    speaker_embeddings = {}
    workers = min(Config.EMBEDDING_WORKERS, len(speaker_info))
    with ThreadPoolExecutor(max_workers=workers,
                            thread_name_prefix="embed") as executor:
        futures = {
            executor.submit(compute_embedding, sp, info): sp
            for sp, info in speaker_info.items()
        }
        for future in as_completed(futures):
            sp, emb = future.result()
            if emb is not None:
                speaker_embeddings[sp] = emb

    # 以下合并逻辑与原版完全相同
    speakers = list(speaker_info.keys())
    merged = {}

    for i, sp1 in enumerate(speakers):
        if sp1 in merged:
            continue
        for sp2 in speakers[i+1:]:
            if sp2 in merged:
                continue
            g1 = speaker_info[sp1]["gender"]
            g2 = speaker_info[sp2]["gender"]
            if g1 != g2 and "unknown" not in (g1, g2):
                continue
            if sp1 not in speaker_embeddings or sp2 not in speaker_embeddings:
                continue
            e1, e2 = speaker_embeddings[sp1], speaker_embeddings[sp2]
            cos = float(np.dot(e1, e2) / (np.linalg.norm(e1)*np.linalg.norm(e2) + 1e-8))
            if cos >= Config.MERGE_COSINE_THRESHOLD:
                print(f"  合并 {sp2} → {sp1} (余弦={cos:.3f})")
                merged[sp2] = sp1

    if not merged:
        print("  无需合并")
        return speaker_info

    new_info = {}
    for sp, info in speaker_info.items():
        canonical = merged.get(sp, sp)
        if canonical not in new_info:
            new_info[canonical] = {**info,
                                   "segments": list(info["segments"]),
                                   "merged_from": []}
        else:
            new_info[canonical]["segments"].extend(info["segments"])
            new_info[canonical]["total_duration"] += info["total_duration"]
            new_info[canonical]["segment_count"] += info["segment_count"]
            new_info[canonical]["merged_from"].append(sp)
            if info["confidence"] > new_info[canonical]["confidence"]:
                new_info[canonical]["gender"] = info["gender"]
                new_info[canonical]["confidence"] = info["confidence"]

    final = {}
    for i, (sp, info) in enumerate(sorted(new_info.items())):
        name = f"SPEAKER_{i:02d}"
        final[name] = info
        if info.get("merged_from"):
            print(f"  {name} ← {sp} + {', '.join(info['merged_from'])}")

    print(f"  合并结果: {len(speaker_info)} → {len(final)} 个说话人")
    return final


def _get_lang_prefix(target_lang: str) -> str:
    lang_map = {
        "en": "en", "ja": "ja", "ko": "ko", "id": "id",
        "vi": "vi", "es": "es", "tr": "tr", "pt": "pt",
        "hi": "hi", "ar": "ar", "th": "th", "fr": "fr",
        "de": "de", "it": "it", "zh": "zh", "ru": "ru",
    }
    return lang_map.get(target_lang.lower(), "en")


def _print_summary(speaker_map: Dict[str, Dict]):
    print("\n[说话人识别] 结果:")
    print("-" * 50)
    for sp, info in sorted(speaker_map.items()):
        gz = {"male": "男", "female": "女", "unknown": "未知"}.get(info["gender"], "?")
        print(f"  {sp}: {gz}  时长={info['total_duration']:.1f}s  "
              f"片段={len(info['segments'])}")
    print("-" * 50)
