import argparse
from collections import defaultdict

from TTS.api import TTS

from video_dubbing import get_available_coqui_voices


VCTK_MODEL = "tts_models/en/vctk/vits"
XTTS_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"
KNOWN_VCTK_SPEAKERS = {
    # Existing curated voice keys in video_dubbing.py. This validation is
    # intentionally static so the script does not instantiate or download models.
    "p225", "p227", "p231", "p232", "p233", "p236", "p237", "p239",
    "p240", "p243", "p244", "p245", "p246", "p247", "p248", "p249",
    "p250", "p251", "p253", "p256", "p259", "p261", "p263", "p264",
    "p265", "p266",
}
EXPECTED_XTTS_LANGS = {
    "ja", "ko", "zh", "es", "fr", "de", "it",
    "pt", "pl", "tr", "ru", "nl", "ar", "hi",
}
EXPECTED_REFERENCE_KEYS = {
    "male": {
        "en_vctk_vits_m001",
        "en_vctk_vits_m002",
        "en_vctk_vits_m003",
        "en_vctk_vits_m004",
        "en_vctk_vits_m005",
    },
    "female": {
        "en_vctk_vits_f001",
        "en_vctk_vits_f002",
        "en_vctk_vits_f003",
        "en_vctk_vits_f004",
        "en_vctk_vits_f005",
    },
}


def infer_lang_and_gender(voice_key):
    parts = voice_key.split("_")
    lang = parts[0] if parts else "unknown"
    if "_male_" in voice_key or "_m0" in voice_key:
        gender = "male"
    elif "_female_" in voice_key or "_f0" in voice_key:
        gender = "female"
    else:
        gender = "unknown"
    return lang, gender


def validate_voices(voices, official_models):
    issues = []
    for key, config in voices.items():
        model_name = config.get("model_name")
        speaker_idx = config.get("speaker_idx")
        reference_voice_key = config.get("reference_voice_key")

        if model_name not in official_models:
            issues.append(f"{key}: model not in Coqui index: {model_name}")
            continue

        if speaker_idx and model_name == VCTK_MODEL and speaker_idx not in KNOWN_VCTK_SPEAKERS:
            issues.append(f"{key}: VCTK speaker not in static allowlist: {speaker_idx}")
        elif speaker_idx and model_name != VCTK_MODEL:
            issues.append(f"{key}: speaker_idx requires model-specific validation: {speaker_idx}")

        if reference_voice_key:
            if model_name != XTTS_MODEL:
                issues.append(f"{key}: reference_voice_key is only valid for XTTS voices")
            if reference_voice_key not in voices:
                issues.append(f"{key}: reference voice key does not exist: {reference_voice_key}")

        if "_native_" in key and ("_male_" in key or "_female_" in key):
            issues.append(f"{key}: native key must not enter automatic male/female rotation")

    for lang in sorted(EXPECTED_XTTS_LANGS):
        for gender in ("male", "female"):
            expected_refs = EXPECTED_REFERENCE_KEYS[gender]
            for index in range(1, 6):
                key = f"{lang}_{gender}_{index:03d}"
                config = voices.get(key)
                if not config:
                    issues.append(f"{key}: missing XTTS clone voice")
                    continue
                if config.get("model_name") != XTTS_MODEL:
                    issues.append(f"{key}: expected XTTS model, got {config.get('model_name')}")
                if config.get("reference_voice_key") not in expected_refs:
                    issues.append(f"{key}: invalid reference voice {config.get('reference_voice_key')}")

    return issues


def print_coverage(voices):
    coverage = defaultdict(lambda: defaultdict(int))
    for key in voices:
        lang, gender = infer_lang_and_gender(key)
        coverage[lang][gender] += 1

    print("\nCoverage by voice key naming:")
    for lang in sorted(coverage):
        counts = coverage[lang]
        print(
            f"  {lang}: male={counts['male']} "
            f"female={counts['female']} unknown={counts['unknown']}"
        )

    below_target = []
    for lang, counts in sorted(coverage.items()):
        if counts["male"] < 5 or counts["female"] < 5:
            below_target.append((lang, counts["male"], counts["female"]))

    if below_target:
        print("\nLanguages below 5 male + 5 female voice keys:")
        for lang, male_count, female_count in below_target:
            print(f"  {lang}: male={male_count}, female={female_count}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Validate get_available_coqui_voices() against Coqui's model index "
            "without downloading model weights."
        )
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print each configured voice key and model.",
    )
    args = parser.parse_args()

    official_models = set(TTS().list_models())
    voices = get_available_coqui_voices()
    issues = validate_voices(voices, official_models)

    print(f"Official Coqui entries: {len(official_models)}")
    print(f"Configured voices: {len(voices)}")

    if args.list:
        print("\nConfigured voices:")
        for key, config in sorted(voices.items()):
            speaker = config.get("speaker_idx", "")
            print(f"  {key}: {config.get('model_name')} {speaker}".rstrip())

    print_coverage(voices)

    if issues:
        print("\nValidation issues:")
        for issue in issues:
            print(f"  - {issue}")
        return 1

    print("\nValidation passed: configured models are present in Coqui's index.")
    print("XTTS clone rotation keys and reference voice keys are complete.")
    print("No model weights were loaded or downloaded by this script.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
