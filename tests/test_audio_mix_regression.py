from pathlib import Path


def test_dubbing_mix_replaces_original_audio_track_by_default():
    source = Path(__file__).resolve().parents[1] / "video_dubbing.py"
    text = source.read_text(encoding="utf-8")

    assert "silent_audio = create_silent_audio(video_duration)" in text
    assert "all_audio_clips = [silent_audio] + final_audio_clips_for_composition" in text
    assert "background_audio_clips" not in text
    assert "original_video.audio.set_duration(video_duration)" not in text
    assert "background_volume" not in text


def test_default_voice_exists_in_voice_table():
    source = Path(__file__).resolve().parents[1] / "video_dubbing.py"
    text = source.read_text(encoding="utf-8")

    assert 'default="en_vctk_vits_m001"' in text
    assert '"en_vctk_vits_m001":' in text


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
        (root / path).read_text(encoding="utf-8") for path in checked_files
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
