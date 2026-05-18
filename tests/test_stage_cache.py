import json
from pathlib import Path

from pipeline_cache import (
    PipelineRun,
    atomic_write_json,
    build_run_manifest,
    build_stage_manifest,
    get_pipeline_run,
    is_stage_complete,
    mark_stage_complete,
)


def test_stage_directory_uses_output_filename(tmp_path):
    output_video = tmp_path / "output" / "0_dubbed_en.mp4"
    run = get_pipeline_run(
        input_video_path=str(tmp_path / "input" / "0.mp4"),
        output_video_path=str(output_video),
        target_language="en",
        selected_voice_key="en_vctk_vits_m001",
        extra_params={"mode": "single"},
    )

    assert run.root_dir == output_video.parent / "0_dubbed_en"
    assert run.stage_dir("audio") == output_video.parent / "0_dubbed_en" / "01_audio"
    assert run.stage_dir("recognition") == output_video.parent / "0_dubbed_en" / "02_recognition"
    assert run.stage_dir("translation") == output_video.parent / "0_dubbed_en" / "04_translation"


def test_stage_complete_requires_done_file_and_outputs(tmp_path):
    run = PipelineRun(
        root_dir=tmp_path / "output" / "demo",
        manifest={"command": {"target_language": "en"}},
    )
    audio_dir = run.stage_dir("audio")
    audio_dir.mkdir(parents=True)
    background = audio_dir / "background.wav"
    dialogue = audio_dir / "dialogue.wav"

    assert not is_stage_complete(run, "audio", [background, dialogue])

    background.write_bytes(b"x" * 2048)
    dialogue.write_bytes(b"x" * 2048)
    mark_stage_complete(run, "audio", {"outputs": [str(background), str(dialogue)]})

    assert is_stage_complete(run, "audio", [background, dialogue])

    dialogue.unlink()
    assert not is_stage_complete(run, "audio", [background, dialogue])


def test_manifest_changes_when_output_name_changes(tmp_path):
    input_video = tmp_path / "0.mp4"
    input_video.write_bytes(b"video")

    first = build_run_manifest(
        input_video_path=str(input_video),
        output_video_path=str(tmp_path / "out_a.mp4"),
        target_language="en",
        selected_voice_key="voice_a",
        extra_params={"workers": 4},
    )
    second = build_run_manifest(
        input_video_path=str(input_video),
        output_video_path=str(tmp_path / "out_b.mp4"),
        target_language="en",
        selected_voice_key="voice_a",
        extra_params={"workers": 4},
    )

    assert first["command"]["output_video_name"] == "out_a.mp4"
    assert second["command"]["output_video_name"] == "out_b.mp4"
    assert first != second


def test_atomic_write_json_never_leaves_tmp_file(tmp_path):
    path = tmp_path / "data.json"
    atomic_write_json(path, {"ok": True})

    assert json.loads(path.read_text(encoding="utf-8")) == {"ok": True}
    assert not list(tmp_path.glob("*.tmp"))


def test_audio_stage_manifest_ignores_unrelated_runtime_params(tmp_path):
    input_video = tmp_path / "0.mp4"
    input_video.write_bytes(b"video")
    output_video = tmp_path / "out.mp4"

    first = build_run_manifest(
        input_video_path=str(input_video),
        output_video_path=str(output_video),
        target_language="en",
        selected_voice_key="voice_a",
        extra_params={"max_speed_factor": 1.35, "workers": 10},
    )
    second = build_run_manifest(
        input_video_path=str(input_video),
        output_video_path=str(output_video),
        target_language="en",
        selected_voice_key="voice_a",
        extra_params={"max_speed_factor": 1.5, "workers": 2},
    )

    assert first != second
    assert build_stage_manifest(first, "audio") == build_stage_manifest(second, "audio")
    assert build_stage_manifest(first, "translation") == build_stage_manifest(second, "translation")
    assert build_stage_manifest(first, "tts") != build_stage_manifest(second, "tts")


def test_audio_stage_requires_readable_wav_outputs(tmp_path):
    from stage_validators import validate_audio_stage

    background = tmp_path / "background.wav"
    dialogue = tmp_path / "dialogue.wav"
    background.write_bytes(b"not a wav")
    dialogue.write_bytes(b"not a wav")

    ok, reason = validate_audio_stage(background, dialogue, expected_duration=10.0)

    assert not ok
    assert "不可读取" in reason or "时长" in reason


def test_recognition_stage_rejects_malformed_segments(tmp_path):
    from stage_validators import validate_recognition_stage

    segments = tmp_path / "recognized_segments.json"
    text = tmp_path / "recognized_text.txt"
    atomic_write_json(segments, [{"text": "", "start": 3.0, "end": 2.0}])
    text.write_text("", encoding="utf-8")

    ok, reason = validate_recognition_stage(segments, text, video_duration=5.0)

    assert not ok
    assert "无效" in reason or "空" in reason


def test_composition_stage_rejects_tiny_output(tmp_path):
    from pipeline_stages import mark_composition_stage_complete

    run = PipelineRun(
        root_dir=tmp_path / "output" / "demo",
        manifest={"command": {"target_language": "en"}},
    )
    output = tmp_path / "tiny.mp4"
    output.write_bytes(b"x")

    try:
        mark_composition_stage_complete(run, str(output))
    except RuntimeError as exc:
        assert "最终合成产物无效" in str(exc)
    else:
        raise AssertionError("tiny composition output should be rejected")
