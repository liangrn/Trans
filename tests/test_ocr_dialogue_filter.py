from ocr_subtitle_probe import _choose_dialogue_text, _create_paddleocr, _dedupe_adjacent_segments, _filter_noisy_segments, _filter_persistent_text_samples
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
        {"time": 0.8, "text": "固定水印", "confidence": 0.99},
        {"time": 1.6, "text": "固定水印", "confidence": 0.99},
        {"time": 2.4, "text": "固定水印", "confidence": 0.99},
        {"time": 3.2, "text": "固定水印", "confidence": 0.99},
        {"time": 4.0, "text": "固定水印", "confidence": 0.99},
        {"time": 4.8, "text": "固定水印", "confidence": 0.99},
        {"time": 5.6, "text": "固定水印", "confidence": 0.99},
        {"time": 6.4, "text": "固定水印", "confidence": 0.99},
        {"time": 7.2, "text": "固定水印", "confidence": 0.99},
        {"time": 8.0, "text": "固定水印", "confidence": 0.99},
        {"time": 4.0, "text": "正常对白", "confidence": 0.96},
    ]

    filtered = _filter_persistent_text_samples(samples)

    assert [item["text"] for item in filtered] == ["正常对白"]


def test_ocr_keeps_repeated_dialogue_when_it_appears_in_separate_clusters():
    samples = [
        {"time": 1.08, "text": "沈遇", "confidence": 0.99},
        {"time": 1.40, "text": "沈遇", "confidence": 0.99},
        {"time": 16.76, "text": "沈遇", "confidence": 0.99},
        {"time": 17.08, "text": "沈遇", "confidence": 0.99},
        {"time": 28.28, "text": "你装什么贞节烈女呢", "confidence": 0.99},
        {"time": 28.60, "text": "你装什么贞节烈女呢", "confidence": 0.99},
    ]

    filtered = _filter_persistent_text_samples(samples)

    assert filtered == samples


def test_ocr_filters_noisy_tail_segments_with_latin_artifacts():
    segments = [
        {"start": 98.68, "end": 99.97, "text": "说我攀高枝", "confidence": 0.99},
        {"start": 100.92, "end": 102.47, "text": "他才是高枝", "confidence": 0.96},
        {"start": 102.52, "end": 102.82, "text": "地提高枝he", "confidence": 0.83},
        {"start": 102.84, "end": 103.14, "text": "高枝hes", "confidence": 0.83},
        {"start": 103.16, "end": 104.39, "text": "高枝he&", "confidence": 0.91},
        {"start": 104.44, "end": 104.74, "text": "he&高枝", "confidence": 0.95},
        {"start": 104.76, "end": 105.06, "text": "高枝he&", "confidence": 0.85},
        {"start": 105.08, "end": 105.38, "text": "地提高枝he8", "confidence": 0.86},
        {"start": 105.40, "end": 105.99, "text": "地才是高枝he&", "confidence": 0.86},
        {"start": 106.04, "end": 106.34, "text": "高枝", "confidence": 0.97},
        {"start": 106.36, "end": 107.27, "text": "地才是高枝", "confidence": 0.87},
        {"start": 107.32, "end": 107.44, "text": "高枝", "confidence": 0.98},
    ]

    filtered = _filter_noisy_segments(segments)

    assert [item["text"] for item in filtered] == ["说我攀高枝", "他才是高枝"]


def test_ocr_dedupes_adjacent_rotated_subtitle_segments():
    segments = [
        {"start": 27.33, "end": 27.66, "text": "哇！我直接变身熔岩巨鲨", "confidence": 0.99},
        {"start": 27.67, "end": 28.00, "text": "我直接变身熔岩巨鲨哇！", "confidence": 0.99},
        {"start": 28.00, "end": 30.33, "text": "哇！我直接变身熔岩巨鲨", "confidence": 0.99},
    ]

    deduped = _dedupe_adjacent_segments(segments)

    assert len(deduped) == 1
    assert deduped[0]["text"] == "哇！我直接变身熔岩巨鲨"
    assert deduped[0]["start"] == 27.33
    assert deduped[0]["end"] == 30.33


def test_paddleocr_init_disables_mkldnn():
    calls = []

    class FakePaddleOCR:
        def __init__(self, **kwargs):
            calls.append(kwargs)

    _create_paddleocr(FakePaddleOCR)

    assert calls[0]["enable_mkldnn"] is False


def test_paddleocr_init_retries_when_enable_mkldnn_is_unsupported():
    calls = []

    class FakePaddleOCR:
        def __init__(self, **kwargs):
            calls.append(kwargs)
            if "enable_mkldnn" in kwargs:
                raise TypeError("unexpected keyword argument 'enable_mkldnn'")

    _create_paddleocr(FakePaddleOCR)

    assert "enable_mkldnn" in calls[0]
    assert "enable_mkldnn" not in calls[1]
