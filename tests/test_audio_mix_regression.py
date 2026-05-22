from pathlib import Path
import math
import struct
import wave


def _write_test_wav(path, duration=3.2, sample_rate=22050, frequency=440, amplitude=1000):
    frame_count = int(duration * sample_rate)
    with wave.open(str(path), "w") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        frames = [
            struct.pack(
                "<h",
                int(amplitude * math.sin(2 * math.pi * frequency * i / sample_rate)),
            )
            for i in range(frame_count)
        ]
        wav_file.writeframes(b"".join(frames))


def test_tts_cache_changes_when_segment_text_changes(tmp_path, monkeypatch):
    import video_dubbing

    calls = []

    def fake_synthesize(_tts_model, _speaker_idx, text, output_path, target_lang=None):
        calls.append((text, target_lang, Path(output_path).name))
        Path(output_path).write_bytes((text * 200).encode("utf-8"))

    monkeypatch.setattr(video_dubbing, "synthesize_speech_coqui_single", fake_synthesize)

    first_segments = [{"idx": 24, "translated_text": "I'll get out first"}]
    second_segments = [{"idx": 24, "translated_text": "According to legend"}]

    first_results, _ = video_dubbing.generate_tts_parallel(
        first_segments,
        tts_model=object(),
        speaker_idx="p225",
        target_lang="en",
        max_workers=1,
        cache_dir=tmp_path,
    )
    second_results, _ = video_dubbing.generate_tts_parallel(
        second_segments,
        tts_model=object(),
        speaker_idx="p225",
        target_lang="en",
        max_workers=1,
        cache_dir=tmp_path,
    )

    assert len(calls) == 2
    assert first_results[0]["temp_tts_file"] != second_results[0]["temp_tts_file"]


def test_tts_cache_regenerates_unreadable_cached_file(tmp_path, monkeypatch):
    import video_dubbing

    calls = []

    def fake_synthesize(_tts_model, _speaker_idx, text, output_path, target_lang=None):
        calls.append(text)
        Path(output_path).write_bytes(b"valid wav content")

    monkeypatch.setattr(video_dubbing, "synthesize_speech_coqui_single", fake_synthesize)
    bad_bytes = b"bad" * 300
    monkeypatch.setattr(video_dubbing, "_probe_audio_duration", lambda path: 1.0 if Path(path).read_bytes() != bad_bytes else (_ for _ in ()).throw(RuntimeError("bad wav")))

    segments = [{"idx": 1, "translated_text": "hello"}]
    cache_name = video_dubbing._build_tts_cache_filename(segments[0], 0, "p225", "en")
    (tmp_path / cache_name).write_bytes(bad_bytes)

    results, _ = video_dubbing.generate_tts_parallel(
        segments,
        tts_model=object(),
        speaker_idx="p225",
        target_lang="en",
        max_workers=1,
        cache_dir=tmp_path,
    )

    assert calls == ["hello"]
    assert results[0]["cached"] is False


def test_xtts_clone_uses_match_fallback_speed(tmp_path):
    import video_dubbing

    class FakeXTTS:
        def __init__(self):
            self._xtts_language = "en"
            self._speaker_wav = str(tmp_path / "ref.wav")
            self._requires_speaker_wav = True
            self.calls = []

        def tts_to_file(self, **kwargs):
            self.calls.append(kwargs)
            Path(kwargs["file_path"]).write_bytes(b"valid wav content" * 64)

    ref_path = tmp_path / "ref.wav"
    ref_path.write_bytes(b"ref")
    output_path = tmp_path / "out.wav"
    tts = FakeXTTS()

    video_dubbing.synthesize_speech_coqui_single(
        tts,
        speaker_idx=None,
        text="Hello there.",
        output_file=str(output_path),
        target_lang="en",
    )

    assert len(tts.calls) == 1
    assert tts.calls[0]["speed"] == video_dubbing.XTTS_CLONE_SPEED


def test_clone_reference_rejects_mixed_gender_subtitle_evidence():
    import video_dubbing

    safe, reason = video_dubbing._speaker_clone_reference_is_safe({
        "gender": "female",
        "subtitle_gender": "female",
        "subtitle_alignments": [
            {"start": 0.0, "end": 4.0, "final_gender": "female", "final_confidence": 0.90},
            {"start": 5.0, "end": 7.0, "final_gender": "male", "final_confidence": 0.90},
        ],
    })

    assert safe is False
    assert "mixed-gender" in reason


def test_clone_reference_allows_consistent_gender_subtitle_evidence():
    import video_dubbing

    speaker_info = {
        "gender": "male",
        "subtitle_gender": "male",
        "subtitle_alignments": [
            {"start": 0.0, "end": 2.0, "final_gender": "male", "final_confidence": 0.90},
            {"start": 3.0, "end": 5.0, "final_gender": "male", "final_confidence": 0.90},
        ],
    }
    segments, gender = video_dubbing._speaker_reference_segments(speaker_info)
    safe, reason = video_dubbing._speaker_clone_reference_is_safe(speaker_info, segments, gender)

    assert safe is True
    assert reason == "ok"


def test_clone_reference_segments_keep_only_matching_high_confidence_gender():
    import video_dubbing

    segments, gender = video_dubbing._speaker_reference_segments({
        "gender": "unknown",
        "subtitle_gender": "male",
        "subtitle_alignments": [
            {"start": 1.0, "end": 2.0, "final_gender": "male", "final_confidence": 0.90},
            {"start": 3.0, "end": 4.0, "final_gender": "female", "final_confidence": 0.95},
            {"start": 5.0, "end": 6.0, "final_gender": "male", "final_confidence": 0.60},
            {"start": 7.0, "end": 8.0, "segment_gender": "male", "segment_confidence": 0.90},
        ],
    })

    assert gender == "male"
    assert segments == [(1.0, 2.0, 1.0), (7.0, 8.0, 1.0)]


def test_clone_reference_writer_accepts_triplet_reference_segments(tmp_path):
    import video_dubbing

    source = tmp_path / "source_(Vocals)_model_bs_roformer_ep_317_sdr_12.wav"
    output = tmp_path / "speaker_ref.wav"
    _write_test_wav(source, duration=4.0)

    success, status = video_dubbing._write_speaker_reference_from_source(
        source,
        {"segments": [(0.0, 1.0)]},
        output,
        reference_segments=[(0.0, 3.0, 3.0)],
    )

    assert success is True
    assert status == "created"
    assert video_dubbing._probe_audio_duration(str(output)) >= 2.5


def test_clone_reference_requires_source_vocals_and_skips_dialogue_fallback(tmp_path):
    import video_dubbing

    dialogue = tmp_path / "dialogue.wav"
    _write_test_wav(dialogue, duration=4.0)
    refs_dir = tmp_path / "refs"

    ref_path, ref_info = video_dubbing._build_speaker_reference_audio(
        dialogue,
        "SPEAKER_00",
        {"segments": [(0.0, 4.0)]},
        refs_dir,
        reference_segments=[(0.0, 3.0, 3.0)],
        expected_manifest={
            "clone_profile": video_dubbing.XTTS_CLONE_PROFILE,
            "reference_gender": "male",
            "selected_segments": [[0.0, 3.0, 3.0]],
            "source_path": None,
        },
        previous_manifest=None,
    )

    assert ref_path is None
    assert "无可用参考音频源" in ref_info["reason"]
    assert not (refs_dir / "SPEAKER_00.wav").exists()


def test_clone_reference_rebuilds_when_manifest_mismatches(tmp_path):
    import video_dubbing

    audio_dir = tmp_path / "01_audio"
    source_dir = audio_dir / "_work" / "separator_output"
    source_dir.mkdir(parents=True)
    source = source_dir / "source_(Vocals)_model_bs_roformer_ep_317_sdr_12.wav"
    dialogue = audio_dir / "dialogue.wav"
    refs_dir = tmp_path / "refs"
    refs_dir.mkdir()
    output = refs_dir / "SPEAKER_00.wav"
    _write_test_wav(source, duration=5.0, frequency=440)
    _write_test_wav(dialogue, duration=5.0, frequency=880)
    _write_test_wav(output, duration=3.0, frequency=220)
    old_mtime = output.stat().st_mtime_ns

    ref_path, ref_info = video_dubbing._build_speaker_reference_audio(
        dialogue,
        "SPEAKER_00",
        {"segments": [(0.0, 5.0)]},
        refs_dir,
        reference_segments=[(0.0, 3.0, 3.0)],
        expected_manifest={
            "clone_profile": video_dubbing.XTTS_CLONE_PROFILE,
            "reference_gender": "male",
            "selected_segments": [[0.0, 3.0, 3.0]],
            "source_path": str(source),
        },
        previous_manifest={
            "clone_profile": "old_profile",
            "reference_gender": "female",
            "selected_segments": [[1.0, 4.0, 3.0]],
            "source_path": str(source),
        },
    )

    assert ref_path == str(output)
    assert ref_info["status"] == "created"
    assert ref_info["clone_profile"] == video_dubbing.XTTS_CLONE_PROFILE
    assert ref_info["reference_gender"] == "male"
    assert ref_info["selected_segments"] == [[0.0, 3.0, 3.0]]
    assert ref_info["source_path"] == str(source)
    assert output.stat().st_mtime_ns != old_mtime


def test_clone_builder_returns_empty_when_clone_is_disabled(tmp_path, monkeypatch):
    import video_dubbing

    class FakeRun:
        def stage_dir(self, stage_name):
            return tmp_path / stage_name

    def fail_if_called(*args, **kwargs):
        raise AssertionError("clone builder should not be called when clone is disabled")

    monkeypatch.setattr(video_dubbing, "_is_xtts_v2_downloaded", lambda: True)
    monkeypatch.setattr(video_dubbing, "_speaker_reference_sources", fail_if_called)
    monkeypatch.setattr(video_dubbing, "_build_speaker_reference_audio", fail_if_called)
    monkeypatch.setattr(video_dubbing, "_speaker_clone_reference_is_safe", fail_if_called)

    clone_voices, clone_voice_map = video_dubbing._build_speaker_clone_voices(
        FakeRun(),
        str(tmp_path / "dialogue.wav"),
        {
            "SPEAKER_00": {
                "gender": "female",
                "subtitle_gender": "female",
                "subtitle_alignments": [
                    {"start": 0.0, "end": 2.0, "final_gender": "female", "final_confidence": 0.95},
                ],
            }
        },
        "en",
        enable_clone_voice=False,
    )

    assert clone_voices == {}
    assert clone_voice_map == {}


def test_dubbing_uses_separated_background_and_never_original_audio():
    source = Path(__file__).resolve().parents[1] / "video_dubbing.py"
    text = source.read_text(encoding="utf-8")

    assert "get_or_create_audio_stage" in text
    assert "asr_audio_path = separation_result.dialogue_path" in text
    assert "background_audio_path = separation_result.background_path" in text
    assert "background_audio = AudioFileClip(background_audio_path)" in text
    assert "all_audio_clips = [background_audio] + final_audio_clips_for_composition" in text
    assert "background_audio_clips" not in text
    assert "original_video.audio.set_duration(video_duration)" not in text
    assert "background_volume" not in text


def test_default_voice_exists_in_voice_table():
    source = Path(__file__).resolve().parents[1] / "video_dubbing.py"
    text = source.read_text(encoding="utf-8")

    assert 'default="en_vctk_vits_m001"' in text
    assert '"en_vctk_vits_m001":' in text


def test_speaker_voice_mapping_uses_subtitle_gender_before_raw_speaker_gender():
    from speaker_aware_dubbing import build_speaker_voice_map

    voices = {
        "en_vctk_vits_m001": {},
        "en_vctk_vits_f001": {},
    }
    speaker_map = {
        "SPEAKER_00": {
            "gender": "female",
            "subtitle_gender": "male",
            "total_duration": 3.0,
        }
    }

    result = build_speaker_voice_map(speaker_map, "en", voices, "en_vctk_vits_m001")

    assert result["SPEAKER_00"] == "en_vctk_vits_m001"


def test_segment_voice_uses_speaker_gender_not_short_segment_female():
    from speaker_aware_dubbing import get_voice_for_segment

    voices = {
        "en_vctk_vits_m001": {},
        "en_vctk_vits_f001": {},
    }
    speaker_map = {
        "SPEAKER_00": {
            "gender": "female",
            "subtitle_gender": "male",
            "segments": [(8.5, 15.0)],
            "subtitle_alignments": [
                {
                    "start": 8.89,
                    "end": 9.55,
                    "segment_gender": "male",
                    "smoothed_gender": "male",
                    "segment_confidence": 0.90,
                    "smoothed_confidence": 0.90,
                },
                {
                    "start": 12.55,
                    "end": 13.55,
                    "segment_gender": "male",
                    "smoothed_gender": "male",
                    "segment_confidence": 0.90,
                    "smoothed_confidence": 0.90,
                },
                {
                    "start": 14.2,
                    "end": 15.5,
                    "segment_gender": "female",
                    "smoothed_gender": "male",
                    "segment_confidence": 0.90,
                    "smoothed_confidence": 0.90,
                }
            ],
        }
    }
    speaker_voice_map = {"SPEAKER_00": "en_vctk_vits_m001"}

    voice = get_voice_for_segment(
        14.2,
        15.5,
        speaker_map,
        speaker_voice_map,
        "en_vctk_vits_m001",
        available_voices=voices,
        target_lang="en",
    )

    assert voice == "en_vctk_vits_m001"


def test_strong_male_subtitle_segment_can_override_female_speaker_vote():
    from speaker_aware_dubbing import get_voice_for_segment

    voices = {
        "en_vctk_vits_m001": {},
        "en_vctk_vits_f001": {},
    }
    speaker_map = {
        "SPEAKER_04": {
            "gender": "female",
            "subtitle_gender": "female",
            "segments": [(7.15, 7.86)],
            "subtitle_alignments": [
                {
                    "start": 7.22,
                    "end": 8.22,
                    "segment_gender": "male",
                    "segment_confidence": 0.90,
                }
            ],
        }
    }
    speaker_voice_map = {"SPEAKER_04": "en_vctk_vits_f001"}

    voice = get_voice_for_segment(
        7.22,
        8.22,
        speaker_map,
        speaker_voice_map,
        "en_vctk_vits_m001",
        available_voices=voices,
        target_lang="en",
    )

    assert voice == "en_vctk_vits_m001"


def test_segment_voice_keeps_speaker_specific_voice_for_same_gender():
    from speaker_aware_dubbing import get_voice_for_segment

    voices = {
        "en_vctk_vits_m001": {},
        "en_vctk_vits_f001": {},
        "en_vctk_vits_f002": {},
    }
    speaker_map = {
        "SPEAKER_01": {
            "gender": "female",
            "subtitle_gender": "female",
            "segments": [(20.0, 24.0)],
            "subtitle_alignments": [
                {
                    "start": 21.0,
                    "end": 22.0,
                    "final_gender": "female",
                    "final_confidence": 0.95,
                }
            ],
        }
    }
    speaker_voice_map = {"SPEAKER_01": "en_vctk_vits_f002"}

    voice = get_voice_for_segment(
        21.0,
        22.0,
        speaker_map,
        speaker_voice_map,
        "en_vctk_vits_m001",
        available_voices=voices,
        target_lang="en",
    )

    assert voice == "en_vctk_vits_f002"


def test_segment_voice_downgrades_polluted_clone_on_confident_gender_conflict():
    from speaker_aware_dubbing import get_voice_for_segment

    voices = {
        "en_vctk_vits_m001": {},
        "en_vctk_vits_f001": {},
    }
    speaker_map = {
        "SPEAKER_01": {
            "gender": "female",
            "subtitle_gender": "unknown",
            "segments": [(7.0, 12.0)],
            "subtitle_alignments": [
                {
                    "start": 8.0,
                    "end": 9.0,
                    "final_gender": "male",
                    "final_confidence": 0.90,
                }
            ],
        }
    }
    speaker_voice_map = {"SPEAKER_01": "clone_SPEAKER_01"}

    voice = get_voice_for_segment(
        8.0,
        9.0,
        speaker_map,
        speaker_voice_map,
        "en_vctk_vits_m001",
        available_voices=voices,
        target_lang="en",
    )

    assert voice == "en_vctk_vits_m001"


def test_segment_voice_keeps_clone_when_speaker_context_is_stable():
    from speaker_aware_dubbing import get_voice_for_segment

    voices = {
        "en_vctk_vits_m001": {},
        "en_vctk_vits_f001": {},
    }
    speaker_map = {
        "SPEAKER_01": {
            "gender": "male",
            "subtitle_gender": "male",
            "segments": [(17.0, 20.5)],
            "subtitle_alignments": [
                {
                    "start": 18.4,
                    "end": 19.3,
                    "final_gender": "female",
                    "final_confidence": 0.90,
                }
            ],
        }
    }
    speaker_voice_map = {"SPEAKER_01": "clone_SPEAKER_01"}

    voice = get_voice_for_segment(
        18.4,
        19.3,
        speaker_map,
        speaker_voice_map,
        "en_vctk_vits_m001",
        available_voices=voices,
        target_lang="en",
    )

    assert voice == "clone_SPEAKER_01"


def test_segment_voice_downgrades_clone_when_only_subtitle_gender_conflicts():
    from speaker_aware_dubbing import get_voice_for_segment

    voices = {
        "en_vctk_vits_m001": {},
        "en_vctk_vits_f001": {},
    }
    speaker_map = {
        "SPEAKER_01": {
            "gender": "unknown",
            "subtitle_gender": "male",
            "segments": [(74.0, 98.0)],
            "subtitle_alignments": [
                {
                    "start": 75.56,
                    "end": 77.11,
                    "final_gender": "female",
                    "final_confidence": 0.90,
                }
            ],
        }
    }
    speaker_voice_map = {"SPEAKER_01": "clone_SPEAKER_01"}

    voice = get_voice_for_segment(
        75.56,
        77.11,
        speaker_map,
        speaker_voice_map,
        "en_vctk_vits_m001",
        available_voices=voices,
        target_lang="en",
    )

    assert voice == "en_vctk_vits_f001"


def test_voice_alignment_diagnostics_marks_clone_gender_downgrade_as_segment_override():
    from speaker_aware_dubbing import explain_segment_voice_alignment

    voices = {
        "en_vctk_vits_m001": {},
        "en_vctk_vits_f001": {},
    }
    speaker_map = {
        "SPEAKER_01": {
            "gender": "female",
            "subtitle_gender": "unknown",
            "segments": [(7.0, 12.0)],
            "subtitle_alignments": [
                {
                    "start": 8.0,
                    "end": 9.0,
                    "final_gender": "male",
                    "final_confidence": 0.90,
                }
            ],
        }
    }
    speaker_voice_map = {"SPEAKER_01": "clone_SPEAKER_01"}

    report = explain_segment_voice_alignment(
        8.0,
        9.0,
        "高置信男声",
        speaker_map,
        speaker_voice_map,
        "en_vctk_vits_m001",
        available_voices=voices,
        target_lang="en",
    )

    assert report["voice"] == "en_vctk_vits_m001"
    assert report["voice_gender"] == "male"
    assert report["voice_source"] == "segment_override"
    assert report["speaker_default_voice"] == "clone_SPEAKER_01"


def test_voice_alignment_diagnostics_marks_review_reasons():
    from speaker_aware_dubbing import explain_segment_voice_alignment

    voices = {
        "en_vctk_vits_m001": {},
        "en_vctk_vits_f001": {},
    }
    speaker_map = {
        "SPEAKER_00": {
            "gender": "female",
            "subtitle_gender": "female",
            "segments": [(10.0, 10.35)],
            "subtitle_alignments": [
                {
                    "start": 10.0,
                    "end": 10.7,
                    "segment_gender": "male",
                    "segment_confidence": 0.90,
                    "final_gender": "male",
                    "final_confidence": 0.90,
                    "final_reason": "segment",
                    "f0_gender": "female",
                    "f0_confidence": 0.70,
                }
            ],
        }
    }
    speaker_voice_map = {"SPEAKER_00": "en_vctk_vits_f001"}

    report = explain_segment_voice_alignment(
        10.0,
        10.79,
        "短句",
        speaker_map,
        speaker_voice_map,
        "en_vctk_vits_m001",
        available_voices=voices,
        target_lang="en",
    )

    assert report["voice"] == "en_vctk_vits_m001"
    assert report["voice_gender"] == "male"
    assert report["voice_source"] == "segment_override"
    assert report["speaker_default_voice"] == "en_vctk_vits_f001"
    assert report["needs_review"] is True
    assert "short_segment" in report["review_reasons"]
    assert "low_speaker_overlap" in report["review_reasons"]
    assert "segment_speaker_gender_conflict" in report["review_reasons"]
    assert "ecapa_f0_conflict" in report["review_reasons"]


def test_voice_alignment_diagnostics_marks_speaker_default_source():
    from speaker_aware_dubbing import explain_segment_voice_alignment

    voices = {
        "en_vctk_vits_m001": {},
        "en_vctk_vits_f001": {},
    }
    speaker_map = {
        "SPEAKER_00": {
            "gender": "female",
            "subtitle_gender": "female",
            "segments": [(1.0, 3.0)],
            "subtitle_alignments": [],
        }
    }
    speaker_voice_map = {"SPEAKER_00": "en_vctk_vits_f001"}

    report = explain_segment_voice_alignment(
        1.2,
        2.4,
        "默认女声",
        speaker_map,
        speaker_voice_map,
        "en_vctk_vits_m001",
        available_voices=voices,
        target_lang="en",
    )

    assert report["voice"] == "en_vctk_vits_f001"
    assert report["voice_source"] == "speaker_default"
    assert report["speaker_default_voice"] == "en_vctk_vits_f001"


def test_voice_alignment_diagnostics_marks_fallback_source():
    from speaker_aware_dubbing import explain_segment_voice_alignment

    voices = {
        "en_vctk_vits_m001": {},
        "en_vctk_vits_f001": {},
    }
    speaker_map = {
        "SPEAKER_00": {
            "gender": "female",
            "subtitle_gender": "female",
            "segments": [(20.0, 21.0)],
            "subtitle_alignments": [],
        }
    }
    speaker_voice_map = {"SPEAKER_00": "en_vctk_vits_f001"}

    report = explain_segment_voice_alignment(
        1.0,
        2.0,
        "无匹配",
        speaker_map,
        speaker_voice_map,
        "en_vctk_vits_m001",
        available_voices=voices,
        target_lang="en",
    )

    assert report["voice"] == "en_vctk_vits_m001"
    assert report["voice_source"] == "fallback"
    assert report["speaker_default_voice"] is None
    assert "fallback_voice" in report["review_reasons"]


def test_voice_alignment_summary_prints_final_voice_counts(capsys):
    from speaker_aware_dubbing import print_voice_alignment_summary

    print_voice_alignment_summary(
        [
            {
                "voice": "en_vctk_vits_f001",
                "voice_gender": "female",
                "voice_source": "speaker_default",
                "needs_review": False,
                "review_reasons": [],
            },
            {
                "voice": "en_vctk_vits_m001",
                "voice_gender": "male",
                "voice_source": "segment_override",
                "needs_review": True,
                "review_reasons": ["final_speaker_gender_conflict"],
            },
            {
                "voice": "en_vctk_vits_m001",
                "voice_gender": "male",
                "voice_source": "fallback",
                "needs_review": True,
                "review_reasons": ["fallback_voice"],
            },
        ],
        "en_vctk_vits_m001",
    )

    output = capsys.readouterr().out
    assert "最终片段配音声音统计" in output
    assert "en_vctk_vits_f001 (female): 1 片段" in output
    assert "en_vctk_vits_m001 (male): 2 片段" in output
    assert "segment_override: 1" in output
    assert "fallback voice: en_vctk_vits_m001 (male)" in output


def test_runtime_and_windows_install_are_cpu_only():
    root = Path(__file__).resolve().parents[1]
    checked_files = [
        "video_dubbing.py",
        "video_subtitles_only.py",
        "speaker_aware_dubbing.py",
        "gender_classifier.py",
        "test_diarization.py",
        "requirements.txt",
        "install_windows.bat",
    ]

    combined = "\n".join(
        (root / path).read_text(encoding="utf-8")
        for path in checked_files
        if (root / path).exists()
    )

    forbidden = [
        "torch.cuda.is_available()",
        'torch.device("cuda")',
        '"cuda"',
        "onnxruntime-gpu",
        "cu121",
        "cu128",
    ]
    for token in forbidden:
        assert token not in combined


def test_gender_model_loading_is_cwd_independent_and_repo_consistent():
    source = Path(__file__).resolve().parents[1] / "gender_classifier.py"
    text = source.read_text(encoding="utf-8")

    assert 'JaesungHuh/voice-gender-classifier' in text
    assert 'JaesungHuh/ecapa-gender' not in text
    assert 'filename="model.pt"' not in text
    assert 'sys.path.insert(0, "voice-gender-classifier")' not in text


def test_windows_install_script_has_preflight_logging_and_post_checks():
    source = Path(__file__).resolve().parents[1] / "install_windows.bat"
    text = source.read_text(encoding="utf-8")

    assert "set LOG_FILE=%~dp0install_windows.log" in text
    assert "call :check_required_file \"voice-gender-classifier\\model.py\"" in text
    assert "call :run_pip_install" in text
    assert "HF_TOKEN is not set" in text
    assert "python video_dubbing.py --help" in text


def test_vocal_separation_is_required_without_cli_switches():
    root = Path(__file__).resolve().parents[1]
    dubbing = (root / "video_dubbing.py").read_text(encoding="utf-8")
    subtitles = (root / "video_subtitles_only.py").read_text(encoding="utf-8")
    helper = (root / "audio_separation.py").read_text(encoding="utf-8")
    installer = (root / "install_windows.bat").read_text(encoding="utf-8")

    forbidden_cli = [
        "--vocal_separation",
        "--keep_background_audio",
        "--background_volume",
    ]
    combined = dubbing + subtitles
    for token in forbidden_cli:
        assert token not in combined

    assert "get_or_create_audio_stage," in dubbing
    assert "get_or_create_audio_stage" not in subtitles
    assert "dialogue_path" in helper
    assert "compand=" in helper
    assert "afftdn=" in helper
    assert "asr_audio_path = separation_result.dialogue_path" in dubbing
    assert "separation_result" not in subtitles
    assert "get_or_create_recognition_stage(" in dubbing
    assert "get_or_create_recognition_stage(" in subtitles
    assert "final_clip = final_clip.set_audio(original_video.audio)" in subtitles
    assert "audio-separator" in helper
    assert "separation_env" in installer
    assert "audio-separator[cpu]" in installer
    assert "source.wav" not in " ".join(
        line.strip() for line in helper.splitlines() if "print(" in line
    )


def test_dubbing_uses_faster_whisper_fallback_and_builds_background_from_asr_mask():
    root = Path(__file__).resolve().parents[1]
    dubbing = (root / "video_dubbing.py").read_text(encoding="utf-8")
    subtitles = (root / "video_subtitles_only.py").read_text(encoding="utf-8")
    helper = (root / "audio_separation.py").read_text(encoding="utf-8")
    asr = (root / "asr_recognition.py").read_text(encoding="utf-8")
    installer = (root / "install_windows.bat").read_text(encoding="utf-8")
    requirements = (root / "requirements.txt").read_text(encoding="utf-8")

    assert "get_or_create_recognition_stage" in dubbing
    assert "get_or_create_recognition_stage" in subtitles
    assert "background_audio_path = separation_result.background_path" in dubbing
    assert "def build_background_with_non_speech_vocals(" in helper
    assert "speech_segments" in helper
    assert "WhisperModel" in asr
    assert 'DEFAULT_WHISPER_MODEL_SIZE = os.environ.get("WHISPER_MODEL_SIZE", "medium")' in asr
    assert 'device="cpu"' in asr
    assert 'compute_type="int8"' in asr
    assert "faster-whisper" in requirements
    assert "Install faster-whisper" in installer
    assert "funasr" not in installer.lower()
    assert "asr_env" not in installer


def test_ocr_is_primary_text_source_and_asr_is_fallback_only():
    root = Path(__file__).resolve().parents[1]
    dubbing = (root / "video_dubbing.py").read_text(encoding="utf-8")
    subtitles = (root / "video_subtitles_only.py").read_text(encoding="utf-8")
    stages = (root / "pipeline_stages.py").read_text(encoding="utf-8")
    ocr = (root / "ocr_recognition.py").read_text(encoding="utf-8")

    assert "get_or_create_recognition_stage(" in dubbing
    assert "get_or_create_recognition_stage(" in subtitles
    assert "from ocr_recognition import get_ocr_subtitle_segments" in stages
    assert "get_ocr_subtitle_segments(" in stages
    assert "transcribe_chinese_audio(asr_audio_path)" in stages
    assert "ocr_env" in ocr
    assert "ocr_subtitle_probe.py" in ocr


def test_pipeline_uses_output_filename_stage_directory_and_translation_cache():
    root = Path(__file__).resolve().parents[1]
    dubbing = (root / "video_dubbing.py").read_text(encoding="utf-8")
    subtitles = (root / "video_subtitles_only.py").read_text(encoding="utf-8")
    cache = (root / "pipeline_cache.py").read_text(encoding="utf-8")
    stages = (root / "pipeline_stages.py").read_text(encoding="utf-8")
    translation = (root / "translation_cache.py").read_text(encoding="utf-8")

    assert "get_pipeline_run(" in dubbing
    assert "get_pipeline_run(" in subtitles
    assert "ThreadPoolExecutor(max_workers=2)" in dubbing
    assert "ThreadPoolExecutor(max_workers=2)" not in subtitles
    assert "output_path.stem" in cache
    assert "run_manifest.json" in cache
    assert "stage.done.json" in cache
    assert "01_audio" in cache
    assert "02_recognition" in cache
    assert "04_translation" in cache
    assert "translation_pending.json" in translation
    assert "translation_report.json" in translation
    assert "translate_segments_with_cache(" in stages
    assert "recognized_segments.json" in stages
    assert "recognized_text.txt" in stages


def test_incremental_upgrade_doc_allows_arbitrary_input_and_output_paths():
    root = Path(__file__).resolve().parents[1]
    doc = (root / "WINDOWS_INCREMENTAL_UPGRADE.md").read_text(encoding="utf-8")

    assert "用户可用任意输入文件和任意输出路径" in doc
    assert "<输出视频所在目录>\\<输出文件名不含扩展名>\\" in doc
    assert "--output_video D:\\result\\movie_en.mp4" in doc
    assert "D:\\result\\movie_en\\" in doc
    assert "--output_dir D:\\result" in doc
    assert "input\\0.mp4" not in doc
    assert "output\\0_dubbed_en.mp4" not in doc


def test_dubbing_uses_real_tts_timeline_and_freezes_tail_for_overflow():
    root = Path(__file__).resolve().parents[1]
    dubbing = (root / "video_dubbing.py").read_text(encoding="utf-8")
    stages = (root / "pipeline_stages.py").read_text(encoding="utf-8")
    timeline = (root / "tts_timeline.py").read_text(encoding="utf-8")

    assert "build_tts_timeline(" in dubbing
    assert "finalize_tts_timeline(" in dubbing
    assert "tts_timeline.json" in stages
    assert "freeze_tail" in timeline
    assert '"-frames:v", "1"' in dubbing
    assert 'pipeline_run.stage_dir("composition")' in dubbing
    assert 'frozen_frame_path = str(composition_stage_dir / "frozen_tail.png")' in dubbing
    assert '"-sseof", "-1"' in dubbing
    assert '"-update", "1"' in dubbing
    assert 'Image.open(frozen_frame_path).convert("RGB")' in dubbing
    assert '冻结帧文件未生成' in dubbing
    assert 'ImageClip(frozen_frame)' in dubbing
    assert "frozen_tail.temp_path = frozen_frame_path" in dubbing
    assert "final_output_duration = max(video_duration, final_timeline[-1][\"planned_end\"])" in dubbing
    assert "set_duration(final_output_duration)" in dubbing
    assert "_est_tts" not in dubbing
    assert "available_duration" not in dubbing
    assert "adjusted_start" not in dubbing
    assert "original_duration'] * 1.5" not in dubbing


def test_cached_speaker_stage_does_not_restart_background_diarization():
    root = Path(__file__).resolve().parents[1]
    dubbing = (root / "video_dubbing.py").read_text(encoding="utf-8")

    assert 'speaker_stage_path = pipeline_run.stage_dir("speaker_gender") / "speaker_gender.json"' in dubbing
    assert 'speaker_stage_ready = is_stage_complete(' in dubbing
    assert 'if _hf_token and not speaker_stage_ready:' in dubbing


def test_old_inline_translation_path_removed_from_dubbing():
    root = Path(__file__).resolve().parents[1]
    dubbing = (root / "video_dubbing.py").read_text(encoding="utf-8")

    assert "def translate_text(" not in dubbing
    assert "def translate_segments_parallel(" not in dubbing
    assert "GoogleTranslator(" not in dubbing
