import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import preview_voices


def test_preview_text_uses_voice_language_default():
    text = preview_voices._preview_text_for_voice(
        "ja_male_001",
        {"language": "ja"},
    )

    assert text == preview_voices.PREVIEW_TEXTS["ja"]


def test_preview_text_normalizes_zh_alias():
    text = preview_voices._preview_text_for_voice(
        "zh_female_001",
        {"language": "zh"},
    )

    assert text == preview_voices.PREVIEW_TEXTS["zh-cn"]


def test_preview_text_can_be_overridden():
    text = preview_voices._preview_text_for_voice(
        "ko_male_001",
        {"language": "ko"},
        custom_text="自定义试听文本",
    )

    assert text == "自定义试听文本"
