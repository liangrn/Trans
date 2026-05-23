from pathlib import Path

import ocr_recognition


def test_ocr_blocks_only_run_missing_ranges(tmp_path, monkeypatch):
    cache_dir = tmp_path / "blocks"
    cache_dir.mkdir()
    ocr_recognition._write_ocr_block(
        cache_dir / "block_000.json",
        0.0,
        300.0,
        [{"start": 10.0, "end": 11.0, "text": "已缓存", "confidence": 0.99}],
    )

    calls = []

    def fake_run(video_path, start_time=None, end_time=None):
        calls.append((round(start_time, 1), round(end_time, 1)))
        return [
            {"start": start_time + 0.2, "end": start_time + 0.8, "text": "重叠外", "confidence": 0.9},
            {"start": max(300.0, start_time + 1.2), "end": max(300.6, start_time + 1.8), "text": "补块", "confidence": 0.9},
        ]

    monkeypatch.setattr(ocr_recognition, "_run_ocr_probe", fake_run)

    segments = ocr_recognition._get_or_create_ocr_blocks("video.mp4", 650.0, cache_dir)

    assert sorted(calls) == [(299.0, 601.0), (599.0, 650.0)]
    assert [segment["text"] for segment in segments] == ["已缓存", "补块", "补块"]
    assert (cache_dir / "block_001.json").exists()
    assert (cache_dir / "block_002.json").exists()


def test_ocr_blocks_reuse_all_cached_ranges(tmp_path, monkeypatch):
    cache_dir = tmp_path / "blocks"
    cache_dir.mkdir()
    for index, start, end in [(0, 0.0, 300.0), (1, 300.0, 450.0)]:
        ocr_recognition._write_ocr_block(
            cache_dir / f"block_{index:03d}.json",
            start,
            end,
            [{"start": start + 1.0, "end": start + 2.0, "text": f"缓存{index}", "confidence": 0.99}],
        )

    def fail_run(*args, **kwargs):
        raise AssertionError("cached OCR blocks should not run probe")

    monkeypatch.setattr(ocr_recognition, "_run_ocr_probe", fail_run)

    segments = ocr_recognition._get_or_create_ocr_blocks("video.mp4", 450.0, cache_dir)

    assert [segment["text"] for segment in segments] == ["缓存0", "缓存1"]
