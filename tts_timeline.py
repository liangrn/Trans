"""Timeline scheduling helpers for translated TTS dubbing."""

from __future__ import annotations

from typing import Any


def build_tts_timeline(
    segments: list[dict[str, Any]],
    raw_durations: list[float],
    video_duration: float,
    max_speed_factor: float = 1.35,
    review_duration_ratio: float = 3.0,
    review_delay_threshold: float = 5.0,
) -> list[dict[str, Any]]:
    """Create a first-pass schedule from original timings and raw TTS durations."""
    timeline: list[dict[str, Any]] = []
    previous_end = 0.0
    last_index = len(segments) - 1

    for idx, (segment, raw_duration) in enumerate(zip(segments, raw_durations)):
        original_start = float(segment["start"])
        original_end = float(segment["end"])
        original_duration = float(segment.get("original_duration") or max(0.0, original_end - original_start))
        planned_start = max(original_start, previous_end)
        next_original_start = (
            float(segments[idx + 1]["start"]) if idx < last_index else float(video_duration)
        )
        window_until_next = max(0.0, next_original_start - planned_start)
        compressed_duration = float(raw_duration)
        speed_factor = 1.0
        overflow_reason = "fits_in_window"

        if compressed_duration > window_until_next + 0.05:
            min_duration_at_cap = compressed_duration / max_speed_factor if compressed_duration > 0 else 0.0
            if min_duration_at_cap <= window_until_next + 0.05 and window_until_next > 0.05:
                compressed_duration = window_until_next
                speed_factor = min(max_speed_factor, raw_duration / window_until_next)
                overflow_reason = "speedup_to_fit"
            else:
                compressed_duration = min_duration_at_cap
                speed_factor = min(max_speed_factor, max(1.0, raw_duration / compressed_duration))
                overflow_reason = "tail_freeze" if idx == last_index else "cascade_delay"

        planned_end = planned_start + compressed_duration
        delay_from_original = max(0.0, planned_start - original_start)
        used_gap_after = max(0.0, min(planned_end, next_original_start) - original_end)
        needs_review = (
            compressed_duration > original_duration * review_duration_ratio
            or delay_from_original > review_delay_threshold
        )

        entry = {
            "idx": idx,
            "original_start": original_start,
            "original_end": original_end,
            "original_duration": original_duration,
            "translated_text": segment["translated_text"],
            "raw_tts_duration": float(raw_duration),
            "target_duration": compressed_duration,
            "planned_start": planned_start,
            "planned_end": planned_end,
            "speed_factor": speed_factor,
            "delay_from_original": delay_from_original,
            "used_gap_after": used_gap_after,
            "overflow_reason": overflow_reason,
            "freeze_tail": idx == last_index and planned_end > video_duration + 0.05,
            "needs_review": needs_review,
        }
        timeline.append(entry)
        previous_end = planned_end

    return timeline


def finalize_tts_timeline(
    planned_timeline: list[dict[str, Any]],
    adjusted_durations: list[float],
    video_duration: float,
    review_duration_ratio: float = 3.0,
    review_delay_threshold: float = 5.0,
) -> list[dict[str, Any]]:
    """Reflow the final schedule from measured post-speed-adjustment durations."""
    final_timeline: list[dict[str, Any]] = []
    previous_end = 0.0
    last_index = len(planned_timeline) - 1

    for idx, (entry, adjusted_duration) in enumerate(zip(planned_timeline, adjusted_durations)):
        original_start = float(entry["original_start"])
        original_end = float(entry["original_end"])
        actual_duration = float(adjusted_duration)
        planned_start = max(original_start, previous_end)
        planned_end = planned_start + actual_duration
        next_original_start = (
            float(planned_timeline[idx + 1]["original_start"]) if idx < last_index else float(video_duration)
        )
        delay_from_original = max(0.0, planned_start - original_start)
        used_gap_after = max(0.0, min(planned_end, next_original_start) - original_end)
        overflow_reason = "fits_in_window"
        if idx == last_index and planned_end > video_duration + 0.05:
            overflow_reason = "tail_freeze"
        elif planned_end > next_original_start + 0.05:
            overflow_reason = "cascade_delay"
        elif actual_duration < entry["raw_tts_duration"] - 0.05:
            overflow_reason = "speedup_to_fit"

        final_entry = dict(entry)
        final_entry.update(
            {
                "target_duration": actual_duration,
                "planned_start": planned_start,
                "planned_end": planned_end,
                "delay_from_original": delay_from_original,
                "used_gap_after": used_gap_after,
                "overflow_reason": overflow_reason,
                "freeze_tail": idx == last_index and planned_end > video_duration + 0.05,
                "needs_review": (
                    actual_duration > final_entry["original_duration"] * review_duration_ratio
                    or delay_from_original > review_delay_threshold
                ),
            }
        )
        final_timeline.append(final_entry)
        previous_end = planned_end

    return final_timeline
