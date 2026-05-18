import json

from translation_cache import translate_segments_with_cache


def test_translation_cache_reuses_successful_results(tmp_path):
    calls = []

    def translator(text, target_lang):
        calls.append((text, target_lang))
        return f"{text}-{target_lang}"

    segments = [{"text": "你好", "start": 0.0, "end": 1.0, "duration": 1.0}]
    first = translate_segments_with_cache(
        segments,
        "en",
        stage_dir=tmp_path,
        translator=translator,
        max_workers=1,
        max_retries=1,
    )
    second = translate_segments_with_cache(
        segments,
        "en",
        stage_dir=tmp_path,
        translator=translator,
        max_workers=1,
        max_retries=1,
    )

    assert first[0]["translated"] == "你好-en"
    assert second[0]["translated"] == "你好-en"
    assert calls == [("你好", "en")]


def test_translation_failure_writes_pending_and_uses_original(tmp_path):
    def translator(text, target_lang):
        raise RuntimeError("network down")

    segments = [{"text": "网络失败", "start": 0.0, "end": 1.0, "duration": 1.0}]
    results = translate_segments_with_cache(
        segments,
        "en",
        stage_dir=tmp_path,
        translator=translator,
        max_workers=1,
        max_retries=2,
        retry_base_delay=0.0,
    )

    pending = json.loads((tmp_path / "translation_pending.json").read_text(encoding="utf-8"))
    report = json.loads((tmp_path / "translation_report.json").read_text(encoding="utf-8"))

    assert results[0]["translated"] == "网络失败"
    assert results[0]["fallback_original"] is True
    assert pending[0]["text"] == "网络失败"
    assert report["failed"] == 1


def test_translation_pending_is_recovered_on_next_run(tmp_path):
    attempts = {"count": 0}

    def translator(text, target_lang):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise RuntimeError("temporary")
        return "Recovered"

    segments = [{"text": "先失败", "start": 0.0, "end": 1.0, "duration": 1.0}]
    translate_segments_with_cache(
        segments,
        "en",
        stage_dir=tmp_path,
        translator=translator,
        max_workers=1,
        max_retries=1,
        retry_base_delay=0.0,
    )
    recovered = translate_segments_with_cache(
        segments,
        "en",
        stage_dir=tmp_path,
        translator=translator,
        max_workers=1,
        max_retries=1,
        retry_base_delay=0.0,
    )

    pending = json.loads((tmp_path / "translation_pending.json").read_text(encoding="utf-8"))
    assert recovered[0]["translated"] == "Recovered"
    assert pending == []
