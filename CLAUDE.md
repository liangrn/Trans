# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

**IMPORTANT**: All commands must be run in the `iai` conda environment:

```bash
conda activate iai
# Then run any python commands
```

## Project Overview

Video translation and dubbing tool with multi-speaker support. Features speaker diarization, gender recognition, and voice-appropriate TTS synthesis. Translation uses the online Google Translate flow via `deep_translator`; the old `本地翻译版本/` NLLB offline variant has been removed. Runtime and installation are CPU-only to reduce Windows dependency conflicts.

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

`en`, `ja`, `ko`, `zh`, `fr`, `de`, `es`, `pt`, `ru`, `it`, `tr`, `ar`, `hi`

## Architecture

### Processing Pipeline (video_dubbing.py)

```
[0/6] Vocal Separation              → audio-separator in separation_env creates background/dialogue tracks
[1/6] Text Recognition              → OCR first via ocr_env; faster-whisper in trans_env only as fallback
[2/6] Speaker + Gender Analysis     → pyannote + ECAPA/F0 on dialogue audio
[3/6] Translation                   → Google Translate via deep_translator
[4/6] TTS Generation                → Multi-voice synthesis (male/female voices)
[5/6] Video Composition             → MoviePy + FFmpeg final output
```

### Key Modules

| File | Lines | Purpose |
|------|-------|---------|
| `video_dubbing.py` | ~2000 | Main pipeline with multi-speaker TTS |
| `speaker_aware_dubbing.py` | ~550 | Async diarization, speaker merging, voice mapping |
| `gender_classifier.py` | ~280 | Singleton GenderClassifier with ECAPA model |
| `test_diarization.py` | ~440 | Standalone diarization + gender test script |
| `video_subtitles_only.py` | ~1040 | Subtitle-only version (no TTS) |

### Core Classes

**GenderClassifier** (singleton pattern):
- Loads bundled `voice-gender-classifier/model.py` by absolute path, then uses `JaesungHuh/voice-gender-classifier` as the remote model source
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
| `DIARIZATION_MODEL` | `pyannote/speaker-diarization-3.1` | Speaker separation model |
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
- `faster-whisper` in the main environment (ASR fallback when OCR is unusable)
- `deep_translator` (Google Translate)
- `coqui-tts`, `transformers`, CPU `torch` (Coqui TTS for voice synthesis)
- `moviepy`, `PIL/Pillow` (video/image processing)
- `ffmpeg`, `ffprobe` (external binaries)
- `audio-separator[cpu]` in `separation_env` (vocal/background separation)
- `paddleocr`, `paddlepaddle`, `opencv-python-headless` in `ocr_env` (hard subtitle OCR)

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
├── voice-gender-classifier/  # Bundled local gender model source; must be deployed together
├── video_dubbing.py          # Main dubbing pipeline
├── speaker_aware_dubbing.py  # Speaker analysis module
├── gender_classifier.py      # Gender classification module
├── test_diarization.py       # Diarization test script
└── video_subtitles_only.py   # Subtitle-only script
```

## Model Downloads

First run will automatically download:
- `pyannote/speaker-diarization-3.1` (diarization)
- `JaesungHuh/voice-gender-classifier` (gender classification remote source)
- `speechbrain/spkrec-ecapa-voxceleb` (speaker embeddings)

Note: pyannote models require accepting user conditions on HuggingFace.
