"""Retrying translation cache for resumable processing."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import hashlib
import json
import random
import re
import time
from typing import Callable

from deep_translator import GoogleTranslator
from pipeline_cache import atomic_write_json


Translator = Callable[[str, str], str]


def translate_text_with_google(text: str, target_lang: str) -> str:
    """Shared Google Translate entrypoint for pipeline stages."""
    text = str(text or "").strip()
    if not text:
        return text

    if re.search(r'[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]', text):
        try:
            from zhconv import convert
            text = convert(text, 'zh-cn')
        except Exception:
            pass

    normalized_target = 'zh-CN' if target_lang.startswith('zh') else target_lang
    translator = GoogleTranslator(source='auto', target=normalized_target)
    result = translator.translate(text)

    if result.strip() == text.strip() and not re.match(r'^[\s\W]+$', text):
        try:
            translator_zh = GoogleTranslator(source='zh-CN', target=normalized_target)
            retry_result = translator_zh.translate(text)
            if retry_result.strip() != text.strip():
                return retry_result
        except Exception:
            pass

    return result


def translate_segments_with_cache(
    segments: list[dict],
    target_lang: str,
    stage_dir: str | Path,
    translator: Translator,
    max_workers: int = 4,
    max_retries: int = 5,
    retry_base_delay: float = 0.8,
) -> list[dict]:
    stage_dir = Path(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)
    cache_path = stage_dir / "translation_cache.json"
    pending_path = stage_dir / "translation_pending.json"
    report_path = stage_dir / "translation_report.json"
    text_path = stage_dir / "translated_text.txt"
    segments_path = stage_dir / "translated_segments.json"

    cache = _load_json(cache_path, default={})
    results: list[dict | None] = [None] * len(segments)

    def work(item: tuple[int, dict]) -> tuple[int, dict]:
        idx, segment = item
        text = str(segment.get("text", "")).strip()
        key = _cache_key(text, target_lang)
        base = {
            "idx": idx,
            "text": text,
            "start": segment["start"],
            "end": segment["end"],
            "duration": segment.get("duration") or segment.get("original_duration"),
        }
        cached = cache.get(key)
        if cached and cached.get("status") == "ok":
            return idx, {**base, "translated": cached["translated"], "fallback_original": False}

        try:
            translated = _translate_with_retries(
                text,
                target_lang,
                translator,
                max_retries=max_retries,
                retry_base_delay=retry_base_delay,
            )
            cache[key] = {
                "source_text": text,
                "target_lang": target_lang,
                "status": "ok",
                "translated": translated,
                "updated_at": time.time(),
            }
            return idx, {**base, "translated": translated, "fallback_original": False}
        except Exception as exc:
            cache[key] = {
                "source_text": text,
                "target_lang": target_lang,
                "status": "failed",
                "error": str(exc),
                "updated_at": time.time(),
            }
            return idx, {
                **base,
                "translated": text,
                "fallback_original": True,
                "error": str(exc),
            }

    with ThreadPoolExecutor(max_workers=max(1, min(max_workers, 4))) as executor:
        futures = [executor.submit(work, item) for item in enumerate(segments)]
        for future in as_completed(futures):
            idx, result = future.result()
            results[idx] = result

    final_results = [result for result in results if result is not None]
    pending = [
        {
            "idx": result["idx"],
            "index": result["idx"],
            "text": result["text"],
            "source_text": result["text"],
            "target_lang": target_lang,
            "error": result.get("error", ""),
            "retry_count": max_retries,
        }
        for result in final_results
        if result.get("fallback_original")
    ]
    report = {
        "total": len(final_results),
        "success": len(final_results) - len(pending),
        "failed": len(pending),
        "has_pending": bool(pending),
    }
    lines = [
        f"[{item['start']:.2f}-{item['end']:.2f}] {item['text']} -> {item['translated']}"
        for item in final_results
    ]
    atomic_write_json(cache_path, cache)
    atomic_write_json(pending_path, pending)
    atomic_write_json(report_path, report)
    atomic_write_json(segments_path, final_results)
    _atomic_write_text(text_path, "\n".join(lines))
    return final_results


def _translate_with_retries(
    text: str,
    target_lang: str,
    translator: Translator,
    max_retries: int,
    retry_base_delay: float,
) -> str:
    try:
        return _translate_once_with_retries(
            text,
            target_lang,
            translator,
            max_retries=max_retries,
            retry_base_delay=retry_base_delay,
        )
    except Exception as exc:
        if not _should_split_text(text):
            raise
        translated_parts = []
        try:
            for part in _split_text(text):
                translated_parts.append(
                    _translate_once_with_retries(
                        part,
                        target_lang,
                        translator,
                        max_retries=max_retries,
                        retry_base_delay=retry_base_delay,
                    )
                )
            return " ".join(part for part in translated_parts if part)
        except Exception:
            raise exc


def _translate_once_with_retries(
    text: str,
    target_lang: str,
    translator: Translator,
    max_retries: int,
    retry_base_delay: float,
) -> str:
    last_error: Exception | None = None
    for attempt in range(max(1, max_retries)):
        try:
            return translator(text, target_lang)
        except Exception as exc:
            last_error = exc
            if attempt < max_retries - 1 and retry_base_delay > 0:
                delay = retry_base_delay * (2**attempt) + random.uniform(0, retry_base_delay)
                time.sleep(delay)

    raise last_error or RuntimeError("翻译失败")


def _cache_key(text: str, target_lang: str) -> str:
    raw = f"{target_lang}\n{text}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()


def _should_split_text(text: str) -> bool:
    return len(text) >= 12 and any(mark in text for mark in "，。！？；,.!?;")


def _split_text(text: str) -> list[str]:
    parts = [part.strip() for part in re.split(r"(?<=[，。！？；,.!?;])", text) if part.strip()]
    return parts or [text]


def _load_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(text, encoding="utf-8")
    tmp_path.replace(path)
