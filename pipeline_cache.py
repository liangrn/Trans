"""Stage cache helpers for resumable video processing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import os
import time
from typing import Any


STAGE_DIRS = {
    "audio": "01_audio",
    "recognition": "02_recognition",
    "speaker_gender": "03_speaker_gender",
    "translation": "04_translation",
    "tts": "05_tts",
    "composition": "06_composition",
}

STAGE_MANIFEST_FIELDS = {
    "audio": ("input", "command.output_video_name", "command.output_video_path"),
    "recognition": ("input", "command.output_video_name", "command.output_video_path"),
    "speaker_gender": ("input", "command.output_video_name", "command.output_video_path"),
    "translation": ("input", "command.output_video_name", "command.output_video_path", "command.target_language"),
    "tts": ("input", "command"),
    "composition": ("input", "command"),
}


@dataclass
class PipelineRun:
    root_dir: Path
    manifest: dict[str, Any]

    def stage_dir(self, stage: str) -> Path:
        dirname = STAGE_DIRS.get(stage)
        if not dirname:
            raise KeyError(f"未知阶段: {stage}")
        return self.root_dir / dirname

    @property
    def manifest_path(self) -> Path:
        return self.root_dir / "run_manifest.json"


def get_pipeline_run(
    input_video_path: str,
    output_video_path: str,
    target_language: str,
    selected_voice_key: str | None = None,
    extra_params: dict[str, Any] | None = None,
) -> PipelineRun:
    output_path = Path(output_video_path)
    root_dir = output_path.parent / output_path.stem
    manifest = build_run_manifest(
        input_video_path=input_video_path,
        output_video_path=output_video_path,
        target_language=target_language,
        selected_voice_key=selected_voice_key,
        extra_params=extra_params or {},
    )
    root_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = root_dir / "run_manifest.json"
    if not manifest_path.exists() or _read_json(manifest_path) != manifest:
        atomic_write_json(manifest_path, manifest)
    return PipelineRun(root_dir=root_dir, manifest=manifest)


def build_run_manifest(
    input_video_path: str,
    output_video_path: str,
    target_language: str,
    selected_voice_key: str | None = None,
    extra_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    input_path = Path(input_video_path).expanduser().resolve(strict=False)
    output_path = Path(output_video_path).expanduser().resolve(strict=False)
    stat = input_path.stat() if input_path.exists() else None
    return {
        "input": {
            "path": str(input_path),
            "size": stat.st_size if stat else None,
            "mtime": stat.st_mtime if stat else None,
        },
        "command": {
            "output_video_name": output_path.name,
            "output_video_path": str(output_path),
            "target_language": target_language,
            "voice": selected_voice_key,
            "params": extra_params or {},
        },
    }


def is_stage_complete(run: PipelineRun, stage: str, outputs: list[str | Path]) -> bool:
    stage_dir = run.stage_dir(stage)
    done_path = stage_dir / "stage.done.json"
    if not done_path.exists():
        return False
    try:
        done = _read_json(done_path)
    except Exception:
        return False
    expected_manifest = build_stage_manifest(run.manifest, stage)
    actual_manifest = done.get("stage_manifest") or build_stage_manifest(done.get("manifest", {}), stage)
    if actual_manifest != expected_manifest:
        return False
    for output in outputs:
        path = Path(output)
        if not path.exists() or path.stat().st_size <= 0:
            return False
    return True


def mark_stage_complete(run: PipelineRun, stage: str, data: dict[str, Any] | None = None) -> None:
    stage_dir = run.stage_dir(stage)
    stage_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "stage": stage,
        "completed_at": time.time(),
        "manifest": run.manifest,
        "stage_manifest": build_stage_manifest(run.manifest, stage),
    }
    if data:
        payload.update(data)
    atomic_write_json(stage_dir / "stage.done.json", payload)


def atomic_write_json(path: str | Path, data: dict[str, Any] | list[Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp_path, path)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def build_stage_manifest(manifest: dict[str, Any], stage: str) -> dict[str, Any]:
    fields = STAGE_MANIFEST_FIELDS.get(stage, ("input", "command"))
    stage_manifest: dict[str, Any] = {}
    for field in fields:
        _set_nested(stage_manifest, field, _get_nested(manifest, field))
    return stage_manifest


def _get_nested(data: dict[str, Any], field: str) -> Any:
    current: Any = data
    for part in field.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def _set_nested(target: dict[str, Any], field: str, value: Any) -> None:
    parts = field.split(".")
    current = target
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    current[parts[-1]] = value
