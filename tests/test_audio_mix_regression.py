from pathlib import Path


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
