from ocr_recognition import _merge_adjacent_ocr_duplicates


def _seg(text, start, end, confidence=0.95):
    return {
        "text": text,
        "start": start,
        "end": end,
        "duration": end - start,
        "confidence": confidence,
        "source": "ocr",
    }


def test_merges_adjacent_near_duplicate_ocr_text():
    segments = [
        _seg("好好过日子", 323.13, 323.46, 0.94),
        _seg("好好过目子", 323.47, 324.13, 0.93),
        _seg("好好过日子", 324.13, 324.46, 0.96),
    ]

    cleaned, report = _merge_adjacent_ocr_duplicates(segments)

    assert len(cleaned) == 1
    assert cleaned[0]["text"] == "好好过日子"
    assert cleaned[0]["start"] == 323.13
    assert cleaned[0]["end"] == 324.46
    assert report[0]["reason"] == "near_duplicate"


def test_merges_short_completion_to_full_text():
    segments = [
        _seg("饭来", 464.73, 465.06, 1.00),
        _seg("饭来了", 465.07, 466.40, 0.97),
    ]

    cleaned, report = _merge_adjacent_ocr_duplicates(segments)

    assert [item["text"] for item in cleaned] == ["饭来了"]
    assert cleaned[0]["start"] == 464.73
    assert cleaned[0]["end"] == 466.40
    assert report[0]["reason"] == "short_completion"


def test_trims_unstable_noise_tail_to_stable_prefix():
    segments = [
        _seg("好好跟着为夫过日子", 529.40, 529.73, 1.00),
        _seg("好好跟着为夫过日子福力建牛肉面", 529.73, 530.40, 0.82),
        _seg("好好跟着为夫过日子番茄鸡牛肉面", 530.40, 531.06, 0.86),
    ]

    cleaned, report = _merge_adjacent_ocr_duplicates(segments)

    assert [item["text"] for item in cleaned] == ["好好跟着为夫过日子"]
    assert cleaned[0]["start"] == 529.40
    assert cleaned[0]["end"] == 531.06
    assert report[0]["reason"] == "stable_prefix_noise"


def test_does_not_merge_adjacent_different_dialogue():
    segments = [
        _seg("只要你们", 528.73, 529.39, 0.99),
        _seg("好好跟着为夫过日子", 529.40, 529.73, 1.00),
    ]

    cleaned, report = _merge_adjacent_ocr_duplicates(segments)

    assert [item["text"] for item in cleaned] == ["只要你们", "好好跟着为夫过日子"]
    assert report == []


def test_does_not_merge_two_near_duplicates_without_majority():
    segments = [
        _seg("遭歹人陷害", 376.20, 376.86, 0.92),
        _seg("遭列人陷害", 376.87, 378.20, 0.95),
    ]

    cleaned, report = _merge_adjacent_ocr_duplicates(segments)

    assert [item["text"] for item in cleaned] == ["遭歹人陷害", "遭列人陷害"]
    assert report == []


def test_keeps_longer_higher_confidence_variant_when_short_low_confidence_duplicate_precedes_it():
    segments = [
        _seg("遭歹人陷害", 376.20, 376.53, 0.88),
        _seg("遭列人陷害", 376.53, 378.20, 0.92),
    ]

    cleaned, report = _merge_adjacent_ocr_duplicates(segments)

    assert [item["text"] for item in cleaned] == ["遭列人陷害"]
    assert cleaned[0]["start"] == 376.20
    assert cleaned[0]["end"] == 378.20
    assert report[0]["reason"] == "confident_longer_variant_pair"


def test_does_not_merge_repeated_text_far_apart():
    segments = [
        _seg("沈遇", 1.08, 1.73, 1.00),
        _seg("沈遇", 16.76, 17.73, 1.00),
    ]

    cleaned, report = _merge_adjacent_ocr_duplicates(segments)

    assert [item["text"] for item in cleaned] == ["沈遇", "沈遇"]
    assert report == []


def test_merges_noisy_long_text_to_clean_short_text():
    segments = [
        _seg("可一旦提升起来生牛自临", 505.73, 506.06, 0.72),
        _seg("可一旦提升起来", 506.07, 507.06, 1.00),
    ]

    cleaned, report = _merge_adjacent_ocr_duplicates(segments)

    assert [item["text"] for item in cleaned] == ["可一旦提升起来"]
    assert cleaned[0]["start"] == 505.73
    assert cleaned[0]["end"] == 507.06
    assert report[0]["reason"] == "noisy_long_to_clean_short"


def test_recovers_embedded_clean_core_from_noisy_cluster():
    segments = [
        _seg("小小公鞋", 509.07, 509.40, 0.82),
        _seg("小小公至", 509.40, 510.06, 0.84),
        _seg("苗茄小小公主", 510.07, 510.40, 0.78),
    ]

    cleaned, report = _merge_adjacent_ocr_duplicates(segments)

    assert [item["text"] for item in cleaned] == ["小小公主"]
    assert cleaned[0]["start"] == 509.07
    assert cleaned[0]["end"] == 510.40
    assert report[0]["reason"] == "embedded_core_recovery"


def test_keeps_dominant_variant_for_long_adjacent_pair():
    segments = [
        _seg("你来打我塞", 700.60, 700.93, 0.95),
        _seg("你来打我噻", 700.93, 704.60, 0.98),
    ]

    cleaned, report = _merge_adjacent_ocr_duplicates(segments)

    assert [item["text"] for item in cleaned] == ["你来打我噻"]
    assert cleaned[0]["start"] == 700.60
    assert cleaned[0]["end"] == 704.60
    assert report[0]["reason"] == "dominant_variant_pair"


def test_prefers_clean_long_sentence_over_noisy_prefix_variants():
    segments = [
        _seg("保险服务?通", 149.56, 149.89, 0.68),
        _seg("保险服务D建财通", 149.89, 150.22, 0.76),
        _seg("保险服务但我的余额却没变", 150.56, 150.89, 0.99),
    ]

    cleaned, report = _merge_adjacent_ocr_duplicates(segments)

    assert [item["text"] for item in cleaned] == ["保险服务但我的余额却没变"]
    assert cleaned[0]["start"] == 149.56
    assert cleaned[0]["end"] == 150.89
    assert report[0]["reason"] == "prefix_to_clean_long"


def test_filters_isolated_very_low_confidence_short_noise():
    segments = [
        _seg("从今天开始", 512.00, 512.70, 0.99),
        _seg("智茄星牛面", 512.73, 513.06, 0.58),
        _seg("我们好好过", 513.40, 514.20, 0.98),
    ]

    cleaned, report = _merge_adjacent_ocr_duplicates(segments)

    assert [item["text"] for item in cleaned] == ["从今天开始", "我们好好过"]
    assert report[0]["reason"] == "low_confidence_noise"
    assert report[0]["removed"]["text"] == "智茄星牛面"


def test_filters_mid_confidence_short_noise_with_no_context_support():
    segments = [
        _seg("从今天开始", 440.60, 441.30, 0.99),
        _seg("热谢短剧", 441.40, 441.73, 0.76),
        _seg("我们好好过", 442.20, 443.00, 0.98),
    ]

    cleaned, report = _merge_adjacent_ocr_duplicates(segments)

    assert [item["text"] for item in cleaned] == ["从今天开始", "我们好好过"]
    assert report[0]["reason"] == "low_confidence_noise"
    assert report[0]["removed"]["text"] == "热谢短剧"


def test_filters_low_confidence_neighbor_pair_without_high_confidence_support():
    segments = [
        _seg("那种东西", 439.07, 440.06, 1.00),
        _seg("热谢短剧", 441.40, 441.73, 0.76),
        _seg("热习短剧", 441.73, 442.40, 0.88),
        _seg("我才不吃", 442.40, 443.40, 1.00),
    ]

    cleaned, report = _merge_adjacent_ocr_duplicates(segments)

    assert [item["text"] for item in cleaned] == ["那种东西", "我才不吃"]
    assert [item["removed"]["text"] for item in report] == ["热谢短剧", "热习短剧"]


def test_keeps_short_low_confidence_dialogue():
    segments = [
        _seg("你先走吧", 44.20, 45.00, 0.99),
        _seg("毕竟", 45.22, 45.89, 0.73),
        _seg("这件事不简单", 46.30, 47.20, 0.98),
    ]

    cleaned, report = _merge_adjacent_ocr_duplicates(segments)

    assert [item["text"] for item in cleaned] == ["你先走吧", "毕竟", "这件事不简单"]
    assert report == []


def test_keeps_low_confidence_text_with_high_confidence_neighbor_support():
    segments = [
        _seg("竞然有肉", 516.40, 516.73, 0.96),
        _seg("竟然有肉", 516.73, 517.06, 0.89),
        _seg("番茄鸡蛋牛肉面", 517.07, 518.06, 0.96),
    ]

    cleaned, report = _merge_adjacent_ocr_duplicates(segments)

    assert [item["text"] for item in cleaned] == ["竞然有肉", "竟然有肉", "番茄鸡蛋牛肉面"]
    assert report == []


def test_keeps_long_low_confidence_text_without_duplicate_neighbor():
    segments = [
        _seg("相公", 562.07, 563.06, 1.00),
        _seg("已经亥时了", 563.07, 564.40, 0.88),
        _seg("该就寝了", 564.40, 565.40, 1.00),
    ]

    cleaned, report = _merge_adjacent_ocr_duplicates(segments)

    assert [item["text"] for item in cleaned] == ["相公", "已经亥时了", "该就寝了"]
    assert report == []


def test_merges_short_clean_text_over_low_confidence_noisy_extension():
    segments = [
        _seg("好吃吧鸡牛电酒", 481.73, 482.06, 0.81),
        _seg("好吃吧", 482.07, 482.73, 1.00),
    ]

    cleaned, report = _merge_adjacent_ocr_duplicates(segments)

    assert [item["text"] for item in cleaned] == ["好吃吧"]
    assert report[0]["reason"] == "noisy_long_to_clean_short"
