from tts_timeline import build_tts_timeline, finalize_tts_timeline


def _segments():
    return [
        {"start": 0.0, "end": 1.0, "original_duration": 1.0, "translated_text": "a"},
        {"start": 1.6, "end": 2.6, "original_duration": 1.0, "translated_text": "b"},
        {"start": 5.0, "end": 6.0, "original_duration": 1.0, "translated_text": "c"},
    ]


def test_short_segments_keep_original_timing():
    timeline = build_tts_timeline(_segments(), [0.8, 0.9, 0.7], 8.0, max_speed_factor=1.35)

    assert timeline[0]["planned_start"] == 0.0
    assert timeline[0]["planned_end"] == 0.8
    assert timeline[0]["speed_factor"] == 1.0
    assert timeline[1]["planned_start"] == 1.6
    assert timeline[2]["planned_start"] == 5.0


def test_mid_sentence_speedup_fits_next_window_without_delay():
    timeline = build_tts_timeline(_segments(), [2.0, 0.9, 0.7], 8.0, max_speed_factor=1.35)

    assert timeline[0]["overflow_reason"] == "speedup_to_fit"
    assert abs(timeline[0]["planned_end"] - 1.6) < 1e-6
    assert timeline[1]["planned_start"] == 1.6


def test_cascade_delay_until_later_gap_then_recovers():
    timeline = build_tts_timeline(_segments(), [2.4, 2.2, 0.7], 8.0, max_speed_factor=1.35)

    assert timeline[0]["overflow_reason"] == "cascade_delay"
    assert timeline[1]["planned_start"] > 1.6
    assert timeline[2]["planned_start"] == 5.0


def test_tail_freeze_marks_last_segment_when_it_runs_past_video_end():
    segments = [
        {"start": 0.0, "end": 1.0, "original_duration": 1.0, "translated_text": "a"},
        {"start": 2.0, "end": 3.0, "original_duration": 1.0, "translated_text": "b"},
    ]
    timeline = build_tts_timeline(segments, [0.8, 3.0], 4.0, max_speed_factor=1.1)

    assert timeline[-1]["freeze_tail"] is True
    assert timeline[-1]["overflow_reason"] == "tail_freeze"


def test_finalize_uses_measured_adjusted_duration_for_real_schedule():
    planned = build_tts_timeline(_segments(), [1.5, 0.9, 0.7], 8.0, max_speed_factor=1.35)
    final = finalize_tts_timeline(planned, [1.58, 0.9, 0.7], 8.0)

    assert abs(final[0]["planned_end"] - 1.58) < 1e-6
    assert abs(final[1]["planned_start"] - 1.6) < 1e-6
