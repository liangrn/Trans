from ocr_subtitle_probe import _choose_dialogue_text, _filter_persistent_text_samples
from ocr_subtitle_probe import _first_nonempty_boxes


def test_ocr_filters_fixed_warning_words_and_keeps_bottom_center_dialogue():
    frame_shape = (1080, 1920, 3)
    candidates = [
        {
            "text": "请勿模仿热门短剧本故事纯属虚构",
            "confidence": 0.98,
            "bbox": (450, 980, 1470, 1030),
        },
        {
            "text": "瑾然我来接你回家",
            "confidence": 0.95,
            "bbox": (650, 830, 1270, 895),
        },
    ]

    text, confidence = _choose_dialogue_text(candidates, frame_shape)

    assert text == "瑾然我来接你回家"
    assert confidence == 0.95


def test_ocr_keeps_short_center_dialogue_in_lower_third():
    frame_shape = (1024, 576, 3)
    candidates = [
        {"text": "陆时钦", "confidence": 0.99, "bbox": (234, 661, 342, 701)},
    ]

    text, confidence = _choose_dialogue_text(candidates, frame_shape)

    assert text == "陆时钦"
    assert confidence == 0.99


def test_ocr_ignores_center_text_above_dialogue_band():
    frame_shape = (1024, 576, 3)
    candidates = [
        {"text": "中部说明文字", "confidence": 0.99, "bbox": (160, 500, 416, 550)},
    ]

    text, confidence = _choose_dialogue_text(candidates, frame_shape)

    assert text == ""
    assert confidence == 0.0


def test_ocr_keeps_dialogue_in_red_box_like_lower_band():
    frame_shape = (1024, 576, 3)
    candidates = [
        {"text": "你们每娶一个老婆", "confidence": 0.98, "bbox": (120, 705, 470, 765)},
        {"text": "热门短剧", "confidence": 0.97, "bbox": (220, 900, 360, 945)},
        {"text": "本故事纯属虚构", "confidence": 0.97, "bbox": (160, 960, 430, 1005)},
    ]

    text, confidence = _choose_dialogue_text(candidates, frame_shape)

    assert text == "你们每娶一个老婆"
    assert confidence == 0.98


def test_ocr_ignores_non_center_or_edge_text():
    frame_shape = (1080, 1920, 3)
    candidates = [
        {"text": "右上角水印", "confidence": 0.99, "bbox": (1560, 80, 1900, 140)},
        {"text": "底部左侧贴片", "confidence": 0.99, "bbox": (0, 780, 360, 850)},
    ]

    text, confidence = _choose_dialogue_text(candidates, frame_shape)

    assert text == ""
    assert confidence == 0.0


def test_ocr_selects_best_dialogue_line_instead_of_concatenating_all_boxes():
    frame_shape = (1080, 1920, 3)
    candidates = [
        {"text": "本故事纯属虚构", "confidence": 0.95, "bbox": (710, 980, 1210, 1030)},
        {"text": "陆瑾然的大哥", "confidence": 0.96, "bbox": (720, 790, 1200, 850)},
        {"text": "陆时钦", "confidence": 0.97, "bbox": (845, 860, 1080, 920)},
    ]

    text, _confidence = _choose_dialogue_text(candidates, frame_shape)

    assert text == "陆瑾然的大哥陆时钦"


def test_ocr_box_selection_handles_numpy_arrays():
    import numpy as np

    empty = np.array([])
    boxes = np.array([[1, 2, 3, 4]])

    selected = _first_nonempty_boxes(empty, boxes)

    assert selected is boxes


def test_ocr_filters_persistent_repeated_text_samples():
    samples = [
        {"time": 0.0, "text": "固定水印", "confidence": 0.99},
        {"time": 3.0, "text": "固定水印", "confidence": 0.99},
        {"time": 6.0, "text": "固定水印", "confidence": 0.99},
        {"time": 9.0, "text": "固定水印", "confidence": 0.99},
        {"time": 4.0, "text": "正常对白", "confidence": 0.96},
    ]

    filtered = _filter_persistent_text_samples(samples)

    assert [item["text"] for item in filtered] == ["正常对白"]
