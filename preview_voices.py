import argparse
import os
import platform
import shutil
import subprocess
import tempfile

from video_dubbing import (
    get_available_coqui_voices,
    load_coqui_tts_model,
    synthesize_speech_coqui_single,
)


PREVIEW_TEXT = "Hello, this is a voice preview for video dubbing."


def play_audio_blocking(audio_path):
    system = platform.system().lower()

    if system == "windows":
        escaped = audio_path.replace("'", "''")
        command = (
            "$player = New-Object System.Media.SoundPlayer "
            f"'{escaped}'; "
            "$player.Load(); "
            "$player.PlaySync()"
        )
        subprocess.run(["powershell", "-NoProfile", "-Command", command], check=True)
        return

    if system == "darwin":
        if not shutil.which("afplay"):
            raise RuntimeError("未找到 afplay，无法播放试听音频")
        subprocess.run(["afplay", audio_path], check=True)
        return

    ffplay = shutil.which("ffplay")
    if ffplay:
        subprocess.run(
            [ffplay, "-nodisp", "-autoexit", "-loglevel", "error", audio_path],
            check=True,
        )
        return

    aplay = shutil.which("aplay")
    if aplay:
        subprocess.run([aplay, audio_path], check=True)
        return

    raise RuntimeError("未找到可用播放器：需要 ffplay 或 aplay")


def _target_lang_for_voice(key, voice_config):
    language = voice_config.get("language")
    if language:
        return language
    return key.split("_", 1)[0] if "_" in key else "en"


def _model_cache_key(voice_config):
    return (
        voice_config.get("model_name", ""),
        voice_config.get("language", ""),
        voice_config.get("speaker_wav", ""),
        voice_config.get("reference_voice_key", ""),
    )


def preview_voice(key, voice_config, model_cache):
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_path = temp_file.name

        cache_key = _model_cache_key(voice_config)
        if cache_key not in model_cache:
            model_cache[cache_key] = load_coqui_tts_model(
                voice_config,
                gpu_is_available=False,
            )
        tts_model, default_speaker_idx = model_cache[cache_key]
        speaker_idx = voice_config.get("speaker_idx", default_speaker_idx)
        synthesize_speech_coqui_single(
            tts_model,
            speaker_idx,
            PREVIEW_TEXT,
            temp_path,
            target_lang=_target_lang_for_voice(key, voice_config),
        )
        play_audio_blocking(temp_path)
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


def main():
    parser = argparse.ArgumentParser(
        description="逐个试听 video_dubbing.py 中配置的 Coqui 声音"
    )
    parser.add_argument(
        "--prefix",
        default="en_vctk",
        help="只试听 key 以该前缀开头的声音，默认: en_vctk",
    )
    args = parser.parse_args()

    voices = get_available_coqui_voices()
    selected = [(key, cfg) for key, cfg in voices.items() if key.startswith(args.prefix)]

    if not selected:
        print(f"没有找到 prefix='{args.prefix}' 的声音。")
        return 0

    print(f"固定试听文本: {PREVIEW_TEXT}")
    print(f"匹配声音数量: {len(selected)}")
    print("说明: 按回车试听下一个，输入 q 后回车退出。\n")

    model_cache = {}
    try:
        for index, (key, voice_config) in enumerate(selected, start=1):
            print("=" * 72)
            print(f"[{index}/{len(selected)}]")
            print(f"Key: {key}")
            print(f"description: {voice_config.get('description', '')}")
            print(f"model_name: {voice_config.get('model_name', '')}")
            print(f"language: {voice_config.get('language', '')}")
            print(f"speaker_idx: {voice_config.get('speaker_idx', '')}")
            print(f"reference_voice_key: {voice_config.get('reference_voice_key', '')}")
            print(f"speaker_wav: {voice_config.get('speaker_wav', '')}")
            print("=" * 72)

            try:
                preview_voice(key, voice_config, model_cache)
            except Exception as exc:
                print(f"试听失败: {exc}")

            if index < len(selected):
                user_input = input("按回车试听下一个，输入 q 退出: ").strip().lower()
                if user_input == "q":
                    break
    finally:
        for tts_model, _speaker_idx in model_cache.values():
            del tts_model

    print("试听结束。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
