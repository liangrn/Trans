# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Video translation and dubbing tool that processes videos through speech recognition, translation, and optional voice synthesis. Supports both online (Google Translate) and offline (NLLB-200) translation modes.

## Running Commands

```bash
# Single video with dubbing (voice synthesis)
python video_dubbing.py --mode single --input_video input/video.mp4 --target_lang en --output_video output/result.mp4

# Single video with subtitles only
python video_subtitles_only.py --mode single --input_video input/video.mp4 --target_lang en --output_video output/result.mp4

# Batch process videos (subtitles only)
python video_subtitles_only.py --mode batch --input_dir ./input --output_dir ./output --target_lang en

# Batch process and merge into single video
python video_subtitles_only.py --mode batch_merge --input_dir ./input --output_dir ./output --target_lang en --merged_filename final.mp4

# Merge existing videos only (no processing)
python video_subtitles_only.py --mode merge_only --output_dir ./output --merged_filename final.mp4
```

## Supported Target Languages

`en`, `ja`, `ko`, `zh`, `fr`, `de`, `es`, `pt`, `ru`, `it`, `tr`, `ar`, `hi`, `th`, `vi`, `id`

## Architecture

### Processing Pipeline

1. **ASR (Speech Recognition)**: faster-whisper extracts text segments with timestamps
2. **Translation**: Google Translator (online) or NLLB-200 3.3B (offline)
3. **Voice Synthesis** (dubbing only): Coqui TTS generates target language audio
4. **Subtitle Rendering**: PIL generates transparent subtitle images with font fallback
5. **Video Composition**: MoviePy + FFmpeg combines video, audio, and subtitles

### Key Modules

| File | Purpose |
|------|---------|
| `video_dubbing.py` | Full pipeline with TTS voice synthesis |
| `video_subtitles_only.py` | Subtitle-only version, no voice synthesis |
| `本地翻译版本/video_dubbing.py` | Dubbing with offline NLLB translation |
| `本地翻译版本/local_translator.py` | NLLB-200 3.3B translation wrapper |

### Environment Variables

- `FFMPEG_BIN`: Override ffmpeg executable path (default: `ffmpeg`)
- `SUBTITLE_FONT_PATH`: Custom subtitle font file path

### Video Merging

Uses FFmpeg concat demuxer for lossless, fast merging. Requires videos to have identical codec, resolution, and pixel format. Run `check_videos_compatible()` to verify before merging.

## Dependencies

- faster-whisper, torch (ASR)
- deep_translator (Google Translate)
- transformers, torch (NLLB - offline mode)
- TTS (Coqui TTS for voice synthesis)
- moviepy, PIL/Pillow (video/image processing)
- ffmpeg, ffprobe (external binaries for video processing)

## Directory Structure

```
Trans/
├── input/              # Source videos
├── output/             # Processed videos
├── tmp/                # Historical versions and experiments
├── 本地翻译版本/        # Offline translation variant
├── video_dubbing.py    # Main dubbing script
└── video_subtitles_only.py  # Subtitle-only script
```