# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

**IMPORTANT**: All commands must be run in the `iai` conda environment:

```bash
conda activate iai
# Then run any python commands
```

## Project Overview

Video translation and dubbing tool with multi-speaker support. Features speaker diarization, gender recognition, and voice-appropriate TTS synthesis. Supports both online (Google Translate) and offline (NLLB-200) translation modes.

## Running Commands

```bash
# Single video with dubbing (multi-speaker voice synthesis)
python video_dubbing.py --mode single --input_video input/video.mp4 --target_lang en --output_video output/result.mp4

# Single video with subtitles only
python video_subtitles_only.py --mode single --input_video input/video.mp4 --target_lang en --output_video output/result.mp4

# Batch process videos (subtitles only)
python video_subtitles_only.py --mode batch --input_dir ./input --output_dir ./output --target_lang en

# Batch process and merge into single video
python video_subtitles_only.py --mode batch_merge --input_dir ./input --output_dir ./output --target_lang en --merged_filename final.mp4

# Merge existing videos only (no processing)
python video_subtitles_only.py --mode merge_only --output_dir ./output --merged_filename final.mp4

# Standalone speaker diarization + gender recognition
python test_diarization.py input/video.mp4
python test_diarization.py input/video.mp4 --threshold 0.5 --min-speakers 2 --max-speakers 4
python test_diarization.py input/video.mp4 --num-speakers 4  # Force exact speaker count
```

## Supported Target Languages

`en`, `ja`, `ko`, `zh`, `fr`, `de`, `es`, `pt`, `ru`, `it`, `tr`, `ar`, `hi`, `th`, `vi`, `id`

## Architecture

### Processing Pipeline (video_dubbing.py)

```
[1/6] ASR (faster-whisper)          → Extract text segments with timestamps
[2/6] Speaker Diarization (async)   → pyannote speaker separation (runs in parallel with ASR)
[3/6] Gender Recognition            → ECAPA-TDNN model + F0 fallback
[4/6] Translation                   → Google Translate / NLLB-200
[5/6] TTS Generation                → Multi-voice synthesis (male/female voices)
[6/6] Video Composition             → MoviePy + FFmpeg final output
```

### Key Modules

| File | Lines | Purpose |
|------|-------|---------|
| `video_dubbing.py` | ~2000 | Main pipeline with multi-speaker TTS |
| `speaker_aware_dubbing.py` | ~550 | Async diarization, speaker merging, voice mapping |
| `gender_classifier.py` | ~280 | Singleton GenderClassifier with ECAPA model |
| `test_diarization.py` | ~440 | Standalone diarization + gender test script |
| `video_subtitles_only.py` | ~1040 | Subtitle-only version (no TTS) |
| `本地翻译版本/video_dubbing.py` | - | Dubbing with offline NLLB translation |
| `本地翻译版本/local_translator.py` | - | NLLB-200 3.3B translation wrapper |

### Core Classes

**GenderClassifier** (singleton pattern):
- Loads `JaesungHuh/ecapa-gender` model for gender classification
- Loads `speechbrain/spkrec-ecapa-voxceleb` for speaker embeddings
- Primary: ECAPA model prediction with voting across segments
- Fallback: F0-based classification (P25 < 155Hz = male, P25 > 195Hz = female)

**SpeakerAwareDubbing**:
- Async diarization execution with `run_diarization_async()` / `wait_diarization()`
- Parallel gender recognition via `_identify_genders_parallel()`
- Speaker merging via embedding cosine similarity (threshold: 0.82)
- Voice mapping: male speakers → male TTS voice, female speakers → female TTS voice

### Configuration Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `DIARIZATION_MODEL` | `pyannote/speaker-diarization-community-1` | Speaker separation model |
| `CLUSTERING_THRESHOLD` | 0.45 | Lower = more speakers detected |
| `MIN_SPEAKERS` / `MAX_SPEAKERS` | 2 / 8 | Speaker count bounds |
| `MERGE_COSINE_THRESHOLD` | 0.82 | Speaker merging similarity threshold |
| `GENDER_CONF_THRESHOLD` | 0.55 | Gender prediction confidence threshold |

### Time Alignment Strategy

1. Pre-estimate TTS duration based on text length
2. Generate TTS audio
3. Post-shift segments if TTS runs longer than original
4. Maintain natural speech timing

### Environment Variables

- `FFMPEG_BIN`: Override ffmpeg executable path (default: `ffmpeg`)
- `SUBTITLE_FONT_PATH`: Custom subtitle font file path

### Video Merging

Uses FFmpeg concat demuxer for lossless, fast merging. Requires videos to have identical codec, resolution, and pixel format. Run `check_videos_compatible()` to verify before merging.

## Dependencies

### Core
- `faster-whisper`, `torch` (ASR)
- `deep_translator` (Google Translate)
- `transformers`, `torch` (NLLB - offline mode)
- `TTS` (Coqui TTS for voice synthesis)
- `moviepy`, `PIL/Pillow` (video/image processing)
- `ffmpeg`, `ffprobe` (external binaries)

### Speaker Analysis
- `pyannote.audio` (speaker diarization)
- `speechbrain` (ECAPA embeddings + gender model)
- `huggingface_hub` (model downloads)
- `librosa`, `scipy` (audio processing)

## Directory Structure

```
Trans/
├── input/                    # Source videos
├── output/                   # Processed videos
├── tmp/                      # Historical versions and experiments
├── pretrained_models/        # Downloaded model weights
├── 本地翻译版本/              # Offline translation variant
├── video_dubbing.py          # Main dubbing pipeline
├── speaker_aware_dubbing.py  # Speaker analysis module
├── gender_classifier.py      # Gender classification module
├── test_diarization.py       # Diarization test script
└── video_subtitles_only.py   # Subtitle-only script
```

## Model Downloads

First run will automatically download:
- `pyannote/speaker-diarization-community-1` (diarization)
- `JaesungHuh/ecapa-gender` (gender classification)
- `speechbrain/spkrec-ecapa-voxceleb` (speaker embeddings)

Note: pyannote models require accepting user conditions on HuggingFace.