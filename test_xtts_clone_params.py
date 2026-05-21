import argparse
import json
from pathlib import Path

from TTS.api import TTS


XTTS_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"


PRESETS = {
    "current": {},
    "clear_low_temp": {
        "temperature": 0.55,
        "top_p": 0.75,
        "top_k": 35,
        "repetition_penalty": 8.0,
    },
    "clear_mid": {
        "temperature": 0.65,
        "top_p": 0.8,
        "top_k": 45,
        "repetition_penalty": 8.5,
    },
    "stable_greedy": {
        "temperature": 0.5,
        "top_p": 0.7,
        "top_k": 30,
        "repetition_penalty": 7.0,
        "do_sample": False,
    },
    "slightly_fast": {
        "temperature": 0.6,
        "top_p": 0.8,
        "top_k": 40,
        "repetition_penalty": 8.0,
        "speed": 1.08,
    },
}


def main():
    parser = argparse.ArgumentParser(description="Generate XTTS clone parameter previews from speaker_refs.")
    parser.add_argument(
        "--refs-dir",
        default="output/8_dubbed_en/05_tts/speaker_refs",
        help="Directory containing SPEAKER_*.wav reference files.",
    )
    parser.add_argument(
        "--out-dir",
        default="output/8_dubbed_en/xtts_param_previews",
        help="Directory for generated preview wav files.",
    )
    parser.add_argument(
        "--text",
        default="I have shown great sincerity. Take the money and leave my daughter.",
        help="Preview text to synthesize.",
    )
    parser.add_argument("--language", default="en", help="XTTS language code.")
    parser.add_argument(
        "--speakers",
        nargs="*",
        default=None,
        help="Optional speaker ids, for example SPEAKER_02. Defaults to all SPEAKER_*.wav.",
    )
    args = parser.parse_args()

    refs_dir = Path(args.refs_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    refs = sorted(refs_dir.glob("SPEAKER_*.wav"))
    if args.speakers:
        wanted = {speaker if speaker.endswith(".wav") else f"{speaker}.wav" for speaker in args.speakers}
        refs = [path for path in refs if path.name in wanted]
    if not refs:
        raise SystemExit(f"No speaker refs found in {refs_dir}")

    print(f"Loading {XTTS_MODEL}...")
    tts = TTS(model_name=XTTS_MODEL, progress_bar=False, gpu=False)
    manifest = {
        "text": args.text,
        "language": args.language,
        "refs_dir": str(refs_dir),
        "presets": PRESETS,
        "outputs": [],
    }

    for ref in refs:
        speaker_out = out_dir / ref.stem
        speaker_out.mkdir(parents=True, exist_ok=True)
        for preset_name, kwargs in PRESETS.items():
            output_path = speaker_out / f"{preset_name}.wav"
            print(f"{ref.stem} / {preset_name} -> {output_path}")
            tts.tts_to_file(
                text=args.text,
                file_path=str(output_path),
                language=args.language,
                speaker_wav=str(ref),
                split_sentences=False,
                **kwargs,
            )
            manifest["outputs"].append(
                {
                    "speaker": ref.stem,
                    "preset": preset_name,
                    "path": str(output_path),
                    "kwargs": kwargs,
                }
            )

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
