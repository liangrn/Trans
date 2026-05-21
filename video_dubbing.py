import argparse
import subprocess
import os
import tempfile
import textwrap
import sys
import glob
import math
import json
import warnings
import hashlib

# ===== TTS 导入：优先 coqui-tts（社区 fork），回退到原版 TTS =====
try:
    from TTS.api import TTS
except ImportError:
    raise ImportError(
        "TTS 包未安装。请运行: pip install coqui-tts>=0.24.0\n"
        "coqui-tts 是 Coqui TTS 的社区维护 fork，支持 torch 2.x 且 Windows 有预编译 wheel。"
    )

import torch
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

# ===== MoviePy 1.x 导入（requirements.txt 已锁定 moviepy==1.0.3）=====
# MoviePy 2.0 删除了 moviepy.editor，本项目使用 1.x API，不支持 2.x。
try:
    from moviepy.editor import VideoFileClip, AudioFileClip, CompositeVideoClip, concatenate_videoclips
    from moviepy.audio.AudioClip import CompositeAudioClip
except ImportError as _mpy_err:
    raise ImportError(
        f"MoviePy 1.x 导入失败: {_mpy_err}\n"
        "请确认已安装 moviepy==1.0.3（MoviePy 2.x 已移除 moviepy.editor 模块）。\n"
        "修复命令: pip install moviepy==1.0.3"
    )

from PIL import Image, ImageDraw, ImageFont
from pipeline_cache import get_pipeline_run, get_stage_source
from pipeline_cache import is_stage_complete
from pipeline_stages import (
    get_or_create_audio_stage,
    get_or_create_recognition_stage,
    get_or_create_speaker_gender_stage,
    get_or_create_translation_stage,
    invalidate_composition_for_tts,
    mark_composition_stage_complete,
    mark_tts_stage_complete,
)
from tts_timeline import build_tts_timeline, finalize_tts_timeline

warnings.filterwarnings("ignore", message="You are sending unauthenticated requests to the HF Hub")

# ===== 说话人感知配音 =====
try:
    from speaker_aware_dubbing import (
        run_diarization_async,
        wait_diarization,
        build_speaker_voice_map,
        get_voice_for_segment,
        explain_segment_voice_alignment,
        print_voice_alignment_summary,
        enrich_speaker_map_with_subtitle_genders,
    )
    SPEAKER_AWARE_AVAILABLE = True
except ImportError as _e:
    print(f'[警告] 说话人模块未找到: {_e}，使用单一声音')
    SPEAKER_AWARE_AVAILABLE = False
# ===========================

import platform
from pathlib import Path
import shutil
import time

def _safe_remove(path: str, retries: int = 5, delay: float = 0.15) -> None:
    """安全删除临时文件，解决 Windows 上文件句柄未释放导致的 PermissionError。
    Windows 下 MoviePy/GC 可能延迟释放文件句柄，重试+延迟可规避此问题。
    """
    import time as _time
    for attempt in range(retries):
        try:
            if os.path.exists(path):
                os.remove(path)
            return
        except PermissionError:
            if attempt < retries - 1:
                _time.sleep(delay)
        except OSError:
            return  # 文件不存在或其他无害错误，直接忽略

# 配置：可通过环境变量或命令行覆盖 ffmpeg 可执行文件与字幕字体路径
# - 使用环境变量 `FFMPEG_BIN` 覆盖 ffmpeg 可执行文件
# - 使用环境变量 `SUBTITLE_FONT_PATH` 指定优先使用的字体文件路径
FONT_PATH_OVERRIDE = os.environ.get('SUBTITLE_FONT_PATH', None)
_SUBTITLE_FONT_CACHE = {}
_SUBTITLE_FONT_LOGGED = set()


def _subtitle_font_log_once(key, message):
    if key not in _SUBTITLE_FONT_LOGGED:
        print(message)
        _SUBTITLE_FONT_LOGGED.add(key)

def _resolve_ffmpeg_bin() -> str:
    """解析 ffmpeg 可执行文件路径，优先级：
    1. 环境变量 FFMPEG_BIN（用户显式指定）
    2. 系统 PATH 中的 ffmpeg
    3. imageio-ffmpeg 捆绑的 ffmpeg 二进制（requirements.txt 已引入）
       在 Windows 上无需手动安装 ffmpeg 或配置 PATH。
    返回可直接传给 subprocess 的路径字符串。
    """
    # 1. 环境变量最高优先
    env_bin = os.environ.get('FFMPEG_BIN', '').strip()
    if env_bin:
        return env_bin
    # 2. PATH 中存在 ffmpeg
    if shutil.which('ffmpeg'):
        return 'ffmpeg'
    # 3. imageio-ffmpeg 捆绑二进制（Windows 友好）
    try:
        import imageio_ffmpeg
        bundled = imageio_ffmpeg.get_ffmpeg_exe()
        if bundled and os.path.isfile(bundled):
            return bundled
    except Exception:
        pass
    # 4. 找不到时返回字符串 'ffmpeg'，后续调用会抛出清晰的 FileNotFoundError
    return 'ffmpeg'

FFMPEG_BIN = _resolve_ffmpeg_bin()

XTTS_V2_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"
XTTS_REQUIRED_FILES = ("model.pth", "config.json", "vocab.json", "speakers_xtts.pth")
XTTS_LANGUAGE_MAP = {
    "en": "en", "ja": "ja", "ko": "ko", "zh": "zh-cn",
    "es": "es", "fr": "fr", "de": "de", "it": "it",
    "pt": "pt", "pl": "pl", "tr": "tr", "ru": "ru",
    "nl": "nl", "ar": "ar", "hi": "hi",
}

# ===== 注入 FFMPEG_BIN 到 moviepy 1.x 配置 =====
# moviepy 1.x 通过 moviepy.config.FFMPEG_BINARY 决定调用哪个 ffmpeg，
# write_videofile 不接受 ffmpeg_exe 参数，必须在此处覆盖配置。
try:
    import moviepy.config as _mpy_cfg
    _mpy_cfg.FFMPEG_BINARY = FFMPEG_BIN
except Exception:
    pass

# 中文常见语气词列表（用于过滤ASR识别的无效片段）
FILLER_WORDS = {
    '嗯', '啊', '呃', '哦', '唔', '诶', '哎', '哎哟', '哎呀',
    '哼', '哈', '嘿', '哇', '咦', '噢', '噢噢',
    '嗯嗯', '啊啊', '呃呃', '哦哦'
}

def is_filler_word(text):
    """检查文本是否只包含语气词"""
    import re
    # 移除所有标点和空格
    clean_text = re.sub(r'[^\w]', '', text.strip())
    return clean_text in FILLER_WORDS or clean_text == ''

def run_ffmpeg_cmd(cmd_list):
    try:
        # 如果命令以 'ffmpeg' 开头，使用可配置的 FFMPEG_BIN 替换
        cmd = list(cmd_list)
        if cmd and str(cmd[0]).lower() == 'ffmpeg':
            cmd[0] = FFMPEG_BIN
        cmd_str = [str(arg) for arg in cmd]
        result = subprocess.run(cmd_str, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return result.stdout.decode()
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode() if e.stderr else str(e)
        print(f"FFmpeg 错误: {stderr}")
        raise e


def adjust_audio_speed_ffmpeg(input_file, output_file, target_duration, max_speed_factor=2.0, min_speed_factor=0.5):
    """
    调整音频速度：
    - TTS 比目标时长短 → 原速播放，不拉伸（剩余时间自动静音）
    - TTS 比目标时长长 → 加速压缩，最大 max_speed_factor
    """
    original_clip = AudioFileClip(input_file)
    original_duration = original_clip.duration
    original_clip.close()
    if original_duration <= 0:
        import shutil
        shutil.copy(input_file, output_file)
        return original_duration, 1.0
    raw_speed_factor = original_duration / target_duration
    # TTS 比目标短：原速播放，不拉伸
    if raw_speed_factor <= 1.0:
        import shutil
        shutil.copy(input_file, output_file)
        return original_duration, 1.0
    # TTS 比目标长：加速压缩
    if raw_speed_factor > max_speed_factor:
        print(f"    - 警告: 速度因子 {raw_speed_factor:.2f}x 超过上限 {max_speed_factor}x，将使用 {max_speed_factor}x")
        speed_factor = max_speed_factor
        final_duration = original_duration / speed_factor
    else:
        speed_factor = raw_speed_factor
        final_duration = target_duration

    # 如果速度变化在可接受范围内，直接使用ffmpeg调整
    if abs(speed_factor - 1.0) < 0.05:  # 5%以内的变化视为不需要调整
        import shutil
        shutil.copy(input_file, output_file)
        return final_duration, speed_factor

    # 构建atempo滤镜链
    atempo_values = []
    current_factor = speed_factor
    # 处理超出范围的速度因子
    while current_factor > 2.0:
        atempo_values.append(2.0)
        current_factor /= 2.0
    while current_factor < 0.5:
        atempo_values.append(0.5)
        current_factor /= 0.5
    # 添加剩余因子
    if 0.5 <= current_factor <= 2.0 and abs(current_factor - 1.0) > 0.05:
        atempo_values.append(current_factor)

    if not atempo_values:
        import shutil
        shutil.copy(input_file, output_file)
        return final_duration, 1.0

    filter_chain = ",".join([f"atempo={val}" for val in atempo_values])
    cmd = [
        "ffmpeg", "-y",
        "-i", input_file,
        "-af", filter_chain,
        "-c:a", "pcm_s16le",
        "-ar", "44100",
        "-ac", "2",
        output_file
    ]
    try:
        run_ffmpeg_cmd(cmd)
        return final_duration, speed_factor
    except Exception as e:
        print(f"    - FFmpeg调整失败: {e}")
        import shutil
        shutil.copy(input_file, output_file)
        return original_duration, 1.0


def _probe_audio_duration(audio_path: str) -> float:
    clip = AudioFileClip(audio_path)
    try:
        return float(clip.duration or 0.0)
    finally:
        clip.close()


def _extend_video_with_frozen_tail(
    video_clip,
    source_video_path: str,
    target_duration: float,
    frozen_frame_path: str,
):
    if target_duration <= video_clip.duration + 0.05:
        return video_clip

    fps = video_clip.fps if video_clip.fps else 30
    freeze_time = max(0.0, video_clip.duration - (1.0 / fps))
    freeze_duration = target_duration - video_clip.duration
    os.makedirs(os.path.dirname(frozen_frame_path), exist_ok=True)
    from moviepy.video.VideoClip import ImageClip

    attempts = [
        [
            "ffmpeg", "-y",
            "-sseof", "-1",
            "-i", source_video_path,
            "-update", "1",
            "-frames:v", "1",
            frozen_frame_path,
        ],
        [
            "ffmpeg", "-y",
            "-ss", f"{freeze_time:.3f}",
            "-i", source_video_path,
            "-update", "1",
            "-frames:v", "1",
            frozen_frame_path,
        ],
    ]

    image = None
    last_error = None
    for cmd in attempts:
        try:
            run_ffmpeg_cmd(cmd)
            if not os.path.exists(frozen_frame_path) or os.path.getsize(frozen_frame_path) <= 0:
                raise RuntimeError(f"冻结帧文件未生成: {frozen_frame_path}")
            image = Image.open(frozen_frame_path).convert("RGB")
            break
        except Exception as exc:
            last_error = exc
            if os.path.exists(frozen_frame_path):
                _safe_remove(frozen_frame_path)

    if image is None:
        raise RuntimeError(f"无法提取视频尾帧: {last_error}")

    frozen_frame = np.array(image)
    frozen_tail = ImageClip(frozen_frame).set_duration(freeze_duration)
    frozen_tail.temp_path = frozen_frame_path
    return concatenate_videoclips([video_clip, frozen_tail], method="chain")

def get_available_coqui_voices():
    """获取可用的 Coqui TTS 声音/模型列表。

    英语使用 VCTK 原生多说话人模型。非英语的规则 key
    (<lang>_male_001..005 / <lang>_female_001..005) 使用 XTTS-v2
    跨语言克隆英文 VCTK 前 5 男声/女声，保证多说话人配音有稳定
    的轮换音色。真实单语种 Coqui 模型保留为 native key，供手动选择。
    """
    XTTS_V2 = XTTS_V2_MODEL

    voices = {
        # ==================== 英语 (English) — tts_models/en/vctk/vits ====================
        # VCTK 数据集，109 个英国英语说话人，VITS 架构，无需参考音频
        "en_vctk_vits_m001": {"model_name": "tts_models/en/vctk/vits", "speaker_idx": "p231", "description": "VITS (VCTK, 男声1, 深沉)"},
        "en_vctk_vits_m002": {"model_name": "tts_models/en/vctk/vits", "speaker_idx": "p232", "description": "VITS (VCTK, 男声2, 温和)"},
        "en_vctk_vits_m003": {"model_name": "tts_models/en/vctk/vits", "speaker_idx": "p233", "description": "VITS (VCTK, 男声3, 年轻)"},
        "en_vctk_vits_m004": {"model_name": "tts_models/en/vctk/vits", "speaker_idx": "p236", "description": "VITS (VCTK, 男声4, 沉稳)"},
        "en_vctk_vits_m005": {"model_name": "tts_models/en/vctk/vits", "speaker_idx": "p239", "description": "VITS (VCTK, 男声5, 磁性)"},
        "en_vctk_vits_m006": {"model_name": "tts_models/en/vctk/vits", "speaker_idx": "p245", "description": "VITS (VCTK, 男声6, 浑厚)"},
        "en_vctk_vits_m007": {"model_name": "tts_models/en/vctk/vits", "speaker_idx": "p248", "description": "VITS (VCTK, 男声7, 年轻)"},
        "en_vctk_vits_m008": {"model_name": "tts_models/en/vctk/vits", "speaker_idx": "p251", "description": "VITS (VCTK, 男声8, 中性)"},
        "en_vctk_vits_m009": {"model_name": "tts_models/en/vctk/vits", "speaker_idx": "p253", "description": "VITS (VCTK, 男声9, 温和)"},
        "en_vctk_vits_m010": {"model_name": "tts_models/en/vctk/vits", "speaker_idx": "p256", "description": "VITS (VCTK, 男声10, 沉稳)"},
        "en_vctk_vits_m011": {"model_name": "tts_models/en/vctk/vits", "speaker_idx": "p259", "description": "VITS (VCTK, 男声11, 浑厚)"},
        "en_vctk_vits_m012": {"model_name": "tts_models/en/vctk/vits", "speaker_idx": "p261", "description": "VITS (VCTK, 男声12, 磁性)"},
        "en_vctk_vits_m013": {"model_name": "tts_models/en/vctk/vits", "speaker_idx": "p263", "description": "VITS (VCTK, 男声13, 年轻)"},
        "en_vctk_vits_m014": {"model_name": "tts_models/en/vctk/vits", "speaker_idx": "p264", "description": "VITS (VCTK, 男声14, 沉稳)"},
        "en_vctk_vits_m015": {"model_name": "tts_models/en/vctk/vits", "speaker_idx": "p265", "description": "VITS (VCTK, 男声15, 温和)"},
        "en_vctk_vits_m016": {"model_name": "tts_models/en/vctk/vits", "speaker_idx": "p266", "description": "VITS (VCTK, 男声16, 磁性)"},
        "en_vctk_vits_f001": {"model_name": "tts_models/en/vctk/vits", "speaker_idx": "p225", "description": "VITS (VCTK, 女声1, 甜美清晰)"},
        "en_vctk_vits_f002": {"model_name": "tts_models/en/vctk/vits", "speaker_idx": "p227", "description": "VITS (VCTK, 女声2, 明亮活泼)"},
        "en_vctk_vits_f003": {"model_name": "tts_models/en/vctk/vits", "speaker_idx": "p237", "description": "VITS (VCTK, 女声3, 成熟稳重)"},
        "en_vctk_vits_f004": {"model_name": "tts_models/en/vctk/vits", "speaker_idx": "p240", "description": "VITS (VCTK, 女声4, 清脆悦耳)"},
        "en_vctk_vits_f005": {"model_name": "tts_models/en/vctk/vits", "speaker_idx": "p243", "description": "VITS (VCTK, 女声5, 明亮自信)"},
        "en_vctk_vits_f006": {"model_name": "tts_models/en/vctk/vits", "speaker_idx": "p244", "description": "VITS (VCTK, 女声6, 清新活泼)"},
        "en_vctk_vits_f007": {"model_name": "tts_models/en/vctk/vits", "speaker_idx": "p246", "description": "VITS (VCTK, 女声7, 温柔细腻)"},
        "en_vctk_vits_f008": {"model_name": "tts_models/en/vctk/vits", "speaker_idx": "p247", "description": "VITS (VCTK, 女声8, 优雅知性)"},
        "en_vctk_vits_f009": {"model_name": "tts_models/en/vctk/vits", "speaker_idx": "p249", "description": "VITS (VCTK, 女声9, 开朗热情)"},
        "en_vctk_vits_f010": {"model_name": "tts_models/en/vctk/vits", "speaker_idx": "p250", "description": "VITS (VCTK, 女声10, 柔和亲切)"},
        # 英语单说话人高质量备选
        "en_ljspeech_vits":   {"model_name": "tts_models/en/ljspeech/vits",    "description": "VITS (LJSpeech, 女声, 标准美式)"},

        # ==================== 真实存在的 Coqui 单语种模型（手动选择，不参与自动轮换） ====================
        #"de_native_thorsten_vits": {"model_name": "tts_models/de/thorsten/vits", "description": "Native (德语, Thorsten, VITS)"},
        #"de_native_thorsten_tacotron_ddc": {"model_name": "tts_models/de/thorsten/tacotron2-DDC", "description": "Native (德语, Thorsten, Tacotron2-DDC)"},
        #"de_native_thorsten_tacotron_dca": {"model_name": "tts_models/de/thorsten/tacotron2-DCA", "description": "Native (德语, Thorsten, Tacotron2-DCA)"},
        #"ja_native_kokoro": {"model_name": "tts_models/ja/kokoro/tacotron2-DDC", "description": "Native (日语, Kokoro, Tacotron2-DDC)"},
        #"zh_native_baker": {"model_name": "tts_models/zh-CN/baker/tacotron2-DDC-GST", "description": "Native (中文, Baker, Tacotron2-DDC-GST)"},
        #"es_native_mai": {"model_name": "tts_models/es/mai/tacotron2-DDC", "description": "Native (西班牙语, Mai, Tacotron2-DDC)"},
        #"es_native_css10_vits": {"model_name": "tts_models/es/css10/vits", "description": "Native (西班牙语, CSS10, VITS)"},
        #"fr_native_mai": {"model_name": "tts_models/fr/mai/tacotron2-DDC", "description": "Native (法语, Mai, Tacotron2-DDC)"},
        #"fr_native_css10_vits": {"model_name": "tts_models/fr/css10/vits", "description": "Native (法语, CSS10, VITS)"},
        #"it_native_mai_m_vits": {"model_name": "tts_models/it/mai_male/vits", "description": "Native (意大利语, Mai, 男声, VITS)"},
        #"it_native_mai_m_glow_tts": {"model_name": "tts_models/it/mai_male/glow-tts", "description": "Native (意大利语, Mai, 男声, Glow-TTS)"},
        #"it_native_mai_f_vits": {"model_name": "tts_models/it/mai_female/vits", "description": "Native (意大利语, Mai, 女声, VITS)"},
        #"it_native_mai_f_glow_tts": {"model_name": "tts_models/it/mai_female/glow-tts", "description": "Native (意大利语, Mai, 女声, Glow-TTS)"},
        #"nl_native_mai": {"model_name": "tts_models/nl/mai/tacotron2-DDC", "description": "Native (荷兰语, Mai, Tacotron2-DDC)"},
        #"nl_native_css10_vits": {"model_name": "tts_models/nl/css10/vits", "description": "Native (荷兰语, CSS10, VITS)"},
        #"pl_native_mai_f_vits": {"model_name": "tts_models/pl/mai_female/vits", "description": "Native (波兰语, Mai, 女声, VITS)"},
        #"pt_native_cv_vits": {"model_name": "tts_models/pt/cv/vits", "description": "Native (葡萄牙语, Common Voice, VITS)"},
        #"tr_native_common_voice_glow_tts": {"model_name": "tts_models/tr/common-voice/glow-tts", "description": "Native (土耳其语, Common Voice, Glow-TTS)"},
    }

    _add_xtts_clone_voices(voices, XTTS_V2)
    return voices


def _add_xtts_clone_voices(voices, xtts_model_name):
    xtts_langs = {
        "ja": {"language": "ja", "name": "日语"},
        "ko": {"language": "ko", "name": "韩语"},
        "zh": {"language": "zh-cn", "name": "中文"},
        "es": {"language": "es", "name": "西班牙语"},
        "fr": {"language": "fr", "name": "法语"},
        "de": {"language": "de", "name": "德语"},
        "it": {"language": "it", "name": "意大利语"},
        "pt": {"language": "pt", "name": "葡萄牙语"},
        "pl": {"language": "pl", "name": "波兰语"},
        "tr": {"language": "tr", "name": "土耳其语"},
        "ru": {"language": "ru", "name": "俄语"},
        "nl": {"language": "nl", "name": "荷兰语"},
        "ar": {"language": "ar", "name": "阿拉伯语"},
        "hi": {"language": "hi", "name": "印地语"},
    }
    reference_voices = {
        "male": [
            "en_vctk_vits_m001",
            "en_vctk_vits_m002",
            "en_vctk_vits_m003",
            "en_vctk_vits_m004",
            "en_vctk_vits_m005",
        ],
        "female": [
            "en_vctk_vits_f001",
            "en_vctk_vits_f002",
            "en_vctk_vits_f003",
            "en_vctk_vits_f004",
            "en_vctk_vits_f005",
        ],
    }

    for lang_key, lang_info in xtts_langs.items():
        for gender, ref_keys in reference_voices.items():
            gender_label = "男声" if gender == "male" else "女声"
            for index, reference_voice_key in enumerate(ref_keys, start=1):
                voice_key = f"{lang_key}_{gender}_{index:03d}"
                voices[voice_key] = {
                    "model_name": xtts_model_name,
                    "language": lang_info["language"],
                    "reference_voice_key": reference_voice_key,
                    "description": (
                        f"XTTS-v2 ({lang_info['name']}, {gender_label}{index}, "
                        f"克隆自 {reference_voice_key})"
                    ),
                }


def generate_tts_parallel(segments_data, tts_model, speaker_idx, target_lang,
                          max_workers=3, max_speed_factor=2.0, min_speed_factor=0.5,
                          cache_dir=None):
    """并行生成 TTS 音频

    TTS 合成在 CPU-only 部署中较耗时，建议使用较小的线程数 (2-3)

    Args:
        segments_data: 翻译后的片段数据列表
        tts_model: TTS 模型实例
        speaker_idx: 说话人索引
        target_lang: 目标语言代码
        max_workers: 最大并行线程数 (默认: 3)
        max_speed_factor: 最大语速加速倍数
        min_speed_factor: 最小语速减慢倍数

    Returns:
        成功生成的音频片段列表，包含音频文件路径和相关信息
    """
    if not segments_data:
        return []

    total = len(segments_data)
    temp_files_to_cleanup = []
    results = [None] * total
    cache_path = Path(cache_dir) if cache_dir else None
    if cache_path:
        cache_path.mkdir(parents=True, exist_ok=True)

    print(f"  - 开始并行生成 {total} 个 TTS 音频 (线程数: {max_workers})...")
    start_time = time.time()
    completed = [0]

    def generate_one(seg_data, idx):
        """生成单个 TTS 音频"""
        try:
            seg_data.setdefault("_tts_model_name", getattr(tts_model, "_model_name", None))
            seg_data.setdefault("_speaker_wav", getattr(tts_model, "_speaker_wav", None))
            seg_data.setdefault("_tts_language", getattr(tts_model, "_xtts_language", None))
            seg_data.setdefault("_tts_generation_profile", getattr(tts_model, "_tts_generation_profile", None))
            if cache_path:
                cache_digest = _build_tts_cache_digest(seg_data, idx, speaker_idx, target_lang)
                temp_tts_file = str(cache_path / _build_tts_cache_filename(seg_data, idx, speaker_idx, target_lang))
                if os.path.exists(temp_tts_file) and os.path.getsize(temp_tts_file) > 512:
                    if _is_valid_cached_tts_file(temp_tts_file):
                        return {
                            "idx": idx,
                            "temp_tts_file": temp_tts_file,
                            "seg_data": seg_data,
                            "success": True,
                            "cached": True,
                            "cache_digest": cache_digest,
                        }
                    _safe_remove(temp_tts_file)
            else:
                cache_digest = _build_tts_cache_digest(seg_data, idx, speaker_idx, target_lang)
                tmp_orig = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
                temp_tts_file = tmp_orig.name
                tmp_orig.close()

            # 生成 TTS
            synthesize_speech_coqui_single(tts_model, speaker_idx,
                                          seg_data["translated_text"], temp_tts_file,
                                          target_lang=target_lang)

            completed[0] += 1
            if completed[0] % 5 == 0 or completed[0] == total:
                print(f"    - TTS 生成进度: {completed[0]}/{total}")

            return {
                "idx": idx,
                "temp_tts_file": temp_tts_file,
                "seg_data": seg_data,
                "success": True,
                "cached": False,
                "cache_digest": cache_digest,
            }
        except Exception as e:
            print(f"    - 片段 {idx} TTS 生成失败: {e}")
            return {
                "idx": idx,
                "temp_tts_file": None,
                "seg_data": seg_data,
                "success": False,
                "error": str(e)
            }

    # 尝试并行处理，如果资源不足则降级为顺序处理
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(generate_one, seg, i): i
                       for i, seg in enumerate(segments_data)}

            for future in as_completed(futures):
                try:
                    result = future.result()
                    results[result["idx"]] = result
                    if result["temp_tts_file"] and not result.get("cached"):
                        temp_files_to_cleanup.append(result["temp_tts_file"])
                except Exception as e:
                    idx = futures[future]
                    print(f"    - 片段 {idx} 处理异常: {e}")
                    results[idx] = {
                        "idx": idx,
                        "temp_tts_file": None,
                        "seg_data": segments_data[idx],
                        "success": False,
                        "error": str(e)
                    }

        elapsed = time.time() - start_time
        success_count = sum(1 for r in results if r and r.get("success"))
        print(f"  - TTS 生成完成: {success_count}/{total} 个片段, 耗时 {elapsed:.1f}s")

    except Exception as e:
        print(f"  - 并行 TTS 失败，降级为顺序处理: {e}")
        # 降级为顺序处理
        for i, seg_data in enumerate(segments_data):
            results[i] = generate_one(seg_data, i)
            if results[i]["temp_tts_file"] and not results[i].get("cached"):
                temp_files_to_cleanup.append(results[i]["temp_tts_file"])

    return results, temp_files_to_cleanup


def _build_tts_cache_filename(seg_data, idx, speaker_idx, target_lang):
    stable_idx = _stable_tts_segment_idx(seg_data, idx)
    digest = _build_tts_cache_digest(seg_data, idx, speaker_idx, target_lang)
    return f"segment_{stable_idx:04d}_{digest}.wav"


def _stable_tts_segment_idx(seg_data, idx):
    stable_idx = seg_data.get("idx", idx)
    try:
        return int(stable_idx)
    except (TypeError, ValueError):
        return idx


def _build_tts_cache_digest(seg_data, idx, speaker_idx, target_lang):
    text = seg_data.get("translated_text") or seg_data.get("text") or ""
    payload = json.dumps(
        {
            "target_lang": target_lang,
            "speaker_idx": speaker_idx,
            "model_name": seg_data.get("_tts_model_name"),
            "speaker_wav": seg_data.get("_speaker_wav"),
            "speaker_wav_signature": _speaker_wav_signature(seg_data.get("_speaker_wav")),
            "tts_language": seg_data.get("_tts_language"),
            "tts_generation_profile": seg_data.get("_tts_generation_profile"),
            "voice_key": seg_data.get("_voice_key"),
            "text": text,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def _is_valid_cached_tts_file(path):
    try:
        if not path or not os.path.exists(path) or os.path.getsize(path) <= 512:
            return False
        return _probe_audio_duration(path) > 0.1
    except Exception:
        return False


def _get_xtts_language(target_lang):
    base_lang = (target_lang or "en").lower().split("-")[0].split("_")[0]
    return XTTS_LANGUAGE_MAP.get(base_lang, base_lang)


def _get_xtts_model_dir():
    try:
        from trainer.io import get_user_data_dir
        base_dir = Path(get_user_data_dir("tts"))
    except Exception:
        if sys.platform == "darwin":
            base_dir = Path.home() / "Library" / "Application Support" / "tts"
        elif os.name == "nt":
            base_dir = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "tts"
        else:
            base_dir = Path.home() / ".local" / "share" / "tts"
    return base_dir / "tts_models--multilingual--multi-dataset--xtts_v2"


def _is_xtts_v2_downloaded():
    model_dir = _get_xtts_model_dir()
    return model_dir.exists() and all((model_dir / name).is_file() for name in XTTS_REQUIRED_FILES)


def _speaker_wav_signature(path):
    if not path or not os.path.exists(path):
        return None
    try:
        stat = os.stat(path)
        return {
            "path": os.path.abspath(path),
            "size": stat.st_size,
            "mtime_ns": getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1_000_000_000)),
        }
    except OSError:
        return {"path": os.path.abspath(path)}


def _wav_rms(path):
    import wave

    try:
        with wave.open(str(path), "rb") as wav_file:
            frames = wav_file.readframes(wav_file.getnframes())
            sample_width = wav_file.getsampwidth()
        if not frames:
            return 0.0
        if sample_width == 2:
            data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        elif sample_width == 4:
            data = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
        else:
            data = np.frombuffer(frames, dtype=np.uint8).astype(np.float32)
            data = (data - 128.0) / 128.0
        if data.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(np.square(data))))
    except Exception:
        return 0.0


def _is_valid_speaker_reference(path, min_duration=2.5, min_rms=0.001):
    try:
        if not path or not os.path.exists(path) or os.path.getsize(path) <= 512:
            return False
        return _probe_audio_duration(str(path)) >= min_duration and _wav_rms(path) >= min_rms
    except Exception:
        return False


def _speaker_reference_sources(dialogue_path):
    dialogue = Path(dialogue_path) if dialogue_path else None
    audio_dir = dialogue.parent if dialogue else None
    sources = []
    if audio_dir:
        separator_dir = audio_dir / "_work" / "separator_output"
        if separator_dir.exists():
            for path in sorted(separator_dir.glob("*Vocals*.wav")):
                sources.append(("source_vocals", path))
        sources.append(("work_vocals", audio_dir / "_work" / "vocals.wav"))
    if dialogue:
        sources.append(("dialogue", dialogue))

    seen = set()
    unique_sources = []
    for label, path in sources:
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen:
            continue
        seen.add(key)
        unique_sources.append((label, path))
    return unique_sources


def _write_speaker_reference_from_source(source_path, speaker_info, output_path):
    if not source_path or not os.path.exists(source_path):
        return False, "源文件不存在"

    source_duration = _probe_audio_duration(str(source_path))
    if source_duration <= 0.1:
        return False, "源文件时长异常"

    segments = speaker_info.get("segments", [])
    if not segments:
        return False, "无 speaker 片段"

    audio = None
    try:
        audio = AudioFileClip(str(source_path))
        audio_duration = max(0.0, float(audio.duration or source_duration))
        candidates = []
        for raw_start, raw_end in segments:
            start = max(0.0, min(float(raw_start), max(0.0, audio_duration - 0.05)))
            end = max(start + 0.05, min(float(raw_end), max(0.05, audio_duration - 0.02)))
            duration = end - start
            if duration >= 0.45:
                candidates.append((start, end, duration))
        if not candidates:
            return False, "无有效裁剪片段"

        candidates.sort(key=lambda item: item[2], reverse=True)
        clips = []
        total_duration = 0.0
        try:
            for start, end, duration in candidates:
                clips.append(audio.subclip(start, end))
                total_duration += duration
                if total_duration >= 10.0:
                    break
            if total_duration < 2.5:
                return False, f"参考音频过短: {total_duration:.2f}s"
            if len(clips) == 1:
                clips[0].write_audiofile(str(output_path), fps=22050, nbytes=2, codec="pcm_s16le", verbose=False, logger=None)
            else:
                from moviepy.editor import concatenate_audioclips
                combined = concatenate_audioclips(clips)
                combined.write_audiofile(str(output_path), fps=22050, nbytes=2, codec="pcm_s16le", verbose=False, logger=None)
                combined.close()
        finally:
            for clip in clips:
                clip.close()

        if not _is_valid_speaker_reference(output_path):
            _safe_remove(str(output_path))
            return False, "参考音频无效或接近静音"
        return True, "created"
    except Exception as e:
        _safe_remove(str(output_path))
        return False, str(e)
    finally:
        if audio is not None:
            audio.close()


def _build_speaker_reference_audio(dialogue_path, speaker_id, speaker_info, refs_dir):
    refs_dir.mkdir(parents=True, exist_ok=True)
    output_path = refs_dir / f"{speaker_id}.wav"
    if _is_valid_speaker_reference(output_path):
        return str(output_path), {
            "status": "cached",
            "source": "cached",
            "duration": _probe_audio_duration(str(output_path)),
            "rms": _wav_rms(output_path),
        }

    failures = []
    for source_label, source_path in _speaker_reference_sources(dialogue_path):
        success, status = _write_speaker_reference_from_source(source_path, speaker_info, output_path)
        if success:
            return str(output_path), {
                "status": status,
                "source": source_label,
                "source_path": str(source_path),
                "duration": _probe_audio_duration(str(output_path)),
                "rms": _wav_rms(output_path),
            }
        failures.append(f"{source_label}: {status}")
    return None, {
        "status": "failed",
        "source": None,
        "reason": "; ".join(failures) if failures else "无可用参考音频源",
    }


def _build_speaker_clone_voices(pipeline_run, dialogue_path, speaker_map, target_lang):
    if not speaker_map:
        return {}, {}
    if not _is_xtts_v2_downloaded():
        print("  [Speaker Clone] XTTS-v2 未完整下载，跳过原视频音色克隆，使用原有声音分配")
        return {}, {}

    refs_dir = pipeline_run.stage_dir("tts") / "speaker_refs"
    xtts_language = _get_xtts_language(target_lang)
    clone_voices = {}
    clone_voice_map = {}
    manifest = {}
    print("  [Speaker Clone] 构建原视频说话人参考音频...")
    for speaker_id, info in sorted(speaker_map.items()):
        ref_path, ref_info = _build_speaker_reference_audio(dialogue_path, speaker_id, info, refs_dir)
        manifest[speaker_id] = ref_info
        if not ref_path:
            print(f"    {speaker_id}: 跳过 ({ref_info.get('reason') or ref_info.get('status')})")
            continue
        voice_key = f"clone_{speaker_id}"
        clone_voices[voice_key] = {
            "model_name": XTTS_V2_MODEL,
            "language": xtts_language,
            "speaker_wav": ref_path,
            "requires_speaker_wav": True,
            "description": f"XTTS-v2 clone from original video speaker {speaker_id}",
        }
        clone_voice_map[speaker_id] = voice_key
        print(
            f"    {speaker_id}: {ref_info.get('status')} "
            f"from {ref_info.get('source')} -> {ref_path}"
        )
    try:
        manifest_path = refs_dir / "speaker_refs_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"  [Speaker Clone] 写入参考音频 manifest 失败: {e}")
    return clone_voices, clone_voice_map


def _load_reference_tts_model(voice_config):
    model_name = voice_config.get("model_name")
    speaker_idx = voice_config.get("speaker_idx")
    print(f"  - 加载参考声音模型: {model_name}")
    tts = TTS(model_name=model_name, progress_bar=True, gpu=False)
    tts._xtts_language = None
    tts._model_name = model_name
    tts._speaker_wav = None
    return tts, speaker_idx


def _get_or_create_xtts_reference_wav(reference_voice_key):
    """为 XTTS 跨语言克隆生成英文 VCTK 参考音频缓存。"""
    voices = get_available_coqui_voices()
    reference_config = voices.get(reference_voice_key)
    if not reference_config:
        raise ValueError(f"reference_voice_key 不存在: {reference_voice_key}")
    if reference_config.get("model_name") == XTTS_V2_MODEL:
        raise ValueError(f"reference_voice_key 不能指向 XTTS 克隆声音: {reference_voice_key}")

    cache_dir = Path(__file__).resolve().parent / "pretrained_models" / "xtts_voice_refs"
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_path = cache_dir / f"{reference_voice_key}.wav"
    if _is_valid_cached_tts_file(str(output_path)):
        return str(output_path)

    print(f"  - 生成 XTTS 参考音频: {reference_voice_key}")
    ref_tts = None
    try:
        ref_tts, ref_speaker_idx = _load_reference_tts_model(reference_config)
        synthesize_speech_coqui_single(
            ref_tts,
            ref_speaker_idx,
            "This is a clear reference voice for multilingual dubbing.",
            str(output_path),
            target_lang="en",
        )
        if not _is_valid_cached_tts_file(str(output_path)):
            raise RuntimeError(f"参考音频生成失败或无效: {output_path}")
        return str(output_path)
    finally:
        if ref_tts is not None:
            del ref_tts


def load_coqui_tts_model(voice_config, gpu_is_available=False):
    """加载 Coqui TTS 模型。

    问题7修复：原代码在模型加载失败时静默降级为英语 ljspeech 模型，
    导致输出实际上是英语配音但用户毫无察觉。
    修复：失败时明确打印警告并说明降级原因；仅允许降级到英语基准模型，
    且此时返回 fallback=True 标记供调用方决定是否继续。
    同时支持 XTTS-v2 的 language 参数（通过 voice_config["language"] 传入）。
    """
    FALLBACK_MODEL = "tts_models/en/ljspeech/vits"
    model_name = voice_config.get("model_name", FALLBACK_MODEL)
    speaker_idx = voice_config.get("speaker_idx", None)
    # XTTS-v2 需要额外的 language 参数，从配置中读取
    xtts_language = voice_config.get("language", None)
    speaker_wav = voice_config.get("speaker_wav")
    reference_voice_key = voice_config.get("reference_voice_key")
    requires_speaker_wav = bool(voice_config.get("requires_speaker_wav"))

    print(f"  - 加载TTS模型: {model_name}")
    if xtts_language:
        print(f"    语言参数: {xtts_language}")
    if reference_voice_key:
        speaker_wav = _get_or_create_xtts_reference_wav(reference_voice_key)
        requires_speaker_wav = True
        print(f"    参考音色: {reference_voice_key}")
    if model_name == XTTS_V2_MODEL and requires_speaker_wav and not (speaker_wav and os.path.isfile(speaker_wav)):
        raise RuntimeError("XTTS 克隆声音缺少有效 speaker_wav")

    try:
        tts = TTS(model_name=model_name, progress_bar=True, gpu=False)
        # 将 xtts_language 存入模型实例，供 synthesize_speech_coqui_single 使用
        tts._xtts_language = xtts_language
        tts._model_name = model_name
        tts._speaker_wav = speaker_wav
        tts._requires_speaker_wav = requires_speaker_wav
        tts._tts_generation_profile = "xtts_clone_slightly_fast_v1" if speaker_wav else "default"
        return tts, speaker_idx
    except Exception as e:
        # 明确警告：不再静默，避免用户不知情地收到英语配音
        print(f"  ⚠ [警告] TTS 模型加载失败: {model_name}")
        print(f"    错误详情: {e}")
        if model_name == XTTS_V2_MODEL and requires_speaker_wav:
            raise RuntimeError(
                f"XTTS 克隆模型加载失败，不能降级为非克隆声音。\n原始错误: {e}"
            ) from e
        if model_name == FALLBACK_MODEL:
            # 基准模型本身也失败，无法继续
            raise RuntimeError(
                f"TTS 基准模型 {FALLBACK_MODEL} 加载失败，请检查 coqui-tts 安装。\n"
                f"原始错误: {e}"
            ) from e
        print(f"  ⚠ 降级到英语基准模型: {FALLBACK_MODEL}")
        print(f"    注意：当前目标语言配音将使用英语模型合成，音质/语言可能不匹配！")
        try:
            tts = TTS(model_name=FALLBACK_MODEL, progress_bar=True, gpu=False)
            tts._xtts_language = None
            tts._model_name = FALLBACK_MODEL
            tts._speaker_wav = None
            tts._requires_speaker_wav = False
            tts._tts_generation_profile = "default"
            tts._is_fallback = True  # 标记为降级，供上层判断
            return tts, None
        except Exception as e2:
            raise RuntimeError(
                f"TTS 基准模型 {FALLBACK_MODEL} 加载也失败了。\n"
                f"请运行: pip install coqui-tts>=0.24.0\n"
                f"原始错误: {e2}"
            ) from e2

def synthesize_speech_coqui_single(tts_instance, speaker_idx, text, output_file, target_lang='en'):
    """生成单个语音片段 - 增强版（处理短文本/语言不匹配/错误回退）"""
    import re
    
    # ===== 1. 文本预处理 =====
    original_text = text.strip()
    
    # 过滤纯标点/空文本
    if not original_text or re.match(r'^[\s\W]+$', original_text):
        print(f"    - 跳过无效文本: '{original_text}'")
        _create_silent_audio(output_file, duration=0.3)
        return
    
    # 处理超短文本（<2字符）- VITS 模型要求
    if len(original_text) < 2:
        # 根据语言添加安全填充
        if target_lang.startswith(('zh', 'ja', 'ko')):
            text = original_text + "。"  # 东亚语言用句号
        else:
            text = original_text + "."   # 拉丁语系用句号
        print(f"    - 文本过短增强: '{original_text}' -> '{text}'")
    
    # 繁体转简体（TTS 模型通常只支持简体）
    if target_lang.startswith('zh'):
        try:
            from zhconv import convert
            text = convert(text, 'zh-cn')
        except:
            pass
    
    #print(f"    - TTS输入: '{text}' (原: '{original_text}')")
    
    # ===== 2. 生成音频（带重试机制）=====
    # 读取 XTTS-v2 的 language 参数（由 load_coqui_tts_model 存入实例）
    xtts_language = getattr(tts_instance, '_xtts_language', None)
    is_xtts = xtts_language is not None

    max_retries = 2
    for attempt in range(max_retries + 1):
        try:
            # 选择正确的调用方式
            if is_xtts:
                # XTTS-v2：必须传入 language 参数；speaker_wav 可选（提供时克隆音色）
                speaker_wav = getattr(tts_instance, '_speaker_wav', None)
                if speaker_wav and os.path.isfile(speaker_wav):
                    tts_instance.tts_to_file(
                        text=text, file_path=output_file,
                        language=xtts_language, speaker_wav=speaker_wav,
                        split_sentences=False,
                        temperature=0.6,
                        top_p=0.8,
                        top_k=40,
                        repetition_penalty=8.0,
                        speed=1.08,
                    )
                elif getattr(tts_instance, '_requires_speaker_wav', False):
                    raise RuntimeError("XTTS 克隆声音缺少有效 speaker_wav，拒绝使用默认 speaker")
                else:
                    tts_instance.tts_to_file(
                        text=text, file_path=output_file,
                        language=xtts_language,
                        speaker="Claribel Dervla"  # XTTS-v2 内置默认说话人
                    )
            elif speaker_idx and hasattr(tts_instance.synthesizer.tts_model, 'speaker_manager'):
                tts_instance.tts_to_file(text=text, file_path=output_file, speaker=speaker_idx)
            else:
                tts_instance.tts_to_file(text=text, file_path=output_file)
            
            # 验证音频有效性
            if os.path.exists(output_file) and os.path.getsize(output_file) > 512:
                return  # 成功
                
            raise Exception("音频文件过小（可能生成失败）")
            
        except Exception as e:
            error_msg = str(e)
            print(f"    - TTS尝试 #{attempt+1} 失败: {error_msg[:80]}")
            
            # VITS 特定错误处理：短文本导致的维度错误
            if any(k in error_msg.lower() for k in ['dimension', 'squeeze', 'index', 'shape']):
                if attempt == 0:
                    # 添加语言特定填充词
                    if target_lang.startswith('zh'):
                        text = text.rstrip('。') + " 嗯。"
                    elif target_lang.startswith('ja'):
                        text = text.rstrip('。') + " あの。"
                    elif target_lang.startswith('ko'):
                        text = text.rstrip('。') + " 어。"
                    else:
                        text = text.rstrip('.') + " uh."
                    print(f"    - 添加填充词重试: '{text}'")
                    continue
            
            # 最后一次失败：生成静音替代
            if attempt == max_retries:
                print(f"    - 所有尝试失败，生成静音替代 (0.3s)")
                _create_silent_audio(output_file, duration=min(0.5, max(0.2, len(text)*0.1)))
                return

def _create_silent_audio(output_path, duration=0.3, sample_rate=22050):
    """创建指定时长的静音WAV文件"""
    import numpy as np
    import wave
    
    n_frames = int(duration * sample_rate)
    # 16-bit stereo 静音数据
    silent_data = b'\x00\x00' * n_frames * 2
    
    with wave.open(output_path, 'wb') as wav_file:
        wav_file.setnchannels(2)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(silent_data)

def create_subtitle_clip(text, start_time, duration, video_width, video_height, language_code='en'):
    """创建字幕片段 - 修复字体加载问题"""
    # ========== 1. 字体大小配置 ==========
    font_sizes = {
        'zh': 34, 'ja': 32, 'ko': 34, 'ar': 32, 'th': 32, 'hi': 32,
        'vi': 30, 'id': 30, 'tr': 30, 'pt': 30, 'es': 30, 'fr': 30, 'ru': 30,
        'default': 32
    }
    base_lang = language_code.split('-')[0].split('_')[0]
    base_font_size = font_sizes.get(base_lang, font_sizes['default'])
    # 根据视频分辨率调整
    if video_height >= 1080:
        font_size = int(base_font_size * 1.4)
        estimated_height = 180
    elif video_height >= 720:
        font_size = int(base_font_size * 1.2)
        estimated_height = 150
    else:
        font_size = int(base_font_size * 1.0)
        estimated_height = 120
    img_width = video_width
    # ========== 2. 创建透明背景 ==========
    subtitle_img = Image.new('RGBA', (img_width, estimated_height), color=(0, 0, 0, 0))
    draw = ImageDraw.Draw(subtitle_img)

    import sys
    font_cache_key = (sys.platform, base_lang, font_size, FONT_PATH_OVERRIDE or "")
    cached_font = _SUBTITLE_FONT_CACHE.get(font_cache_key)

    # ========== 3. 修复字体加载函数 ==========
    def get_font_for_mac(size):
        """Mac专用字体加载"""
        # Mac字体路径
        mac_font_paths = [
            # 1. Arial Unicode (最全)
            "/Library/Fonts/Arial Unicode.ttf",
            # 2. Apple SD Gothic Neo (韩文)
            "/System/Library/Fonts/Apple SD Gothic Neo.ttc",
            # 3. AppleGothic
            "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
            # 4. PingFang (中文)
            "/System/Library/Fonts/PingFang.ttc",
            # 5. Hiragino Sans
            "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
            # 6. Helvetica
            "/System/Library/Fonts/Helvetica.ttc",
            # 7. Arial
            "/System/Library/Fonts/Arial.ttf",
        ]
        # 尝试加载字体
        # 优先尝试外部指定的字体路径（环境变量或命令行传入）
        if FONT_PATH_OVERRIDE and os.path.exists(FONT_PATH_OVERRIDE):
            try:
                _subtitle_font_log_once(("try", FONT_PATH_OVERRIDE, size), f"    - 尝试加载覆盖字体: {os.path.basename(FONT_PATH_OVERRIDE)}")
                # 对于 .ttc/.ttf 都尝试直接加载
                font = ImageFont.truetype(FONT_PATH_OVERRIDE, size=size)
                _subtitle_font_log_once(("ok", FONT_PATH_OVERRIDE, size), f"    - 成功加载覆盖字体: {FONT_PATH_OVERRIDE}")
                return font
            except Exception as e:
                _subtitle_font_log_once(("fail", FONT_PATH_OVERRIDE, size), f"    - 覆盖字体加载失败: {e}")
        
        for font_path in mac_font_paths:
            if os.path.exists(font_path):
                try:
                    #print(f"    - 尝试加载: {os.path.basename(font_path)}")
                    if font_path.endswith('.ttc'):
                        # 对于字体集合，尝试不同索引
                        for index in [0, 1, 2]:
                            try:
                                font = ImageFont.truetype(font_path, size=size, index=index)
                                # 测试字体
                                test_text = "Test"
                                bbox = font.getbbox(test_text)
                                if bbox:
                                    #print(f"    - 成功加载: {os.path.basename(font_path)} (索引:{index})")
                                    return font
                            except:
                                continue
                    else:
                        font = ImageFont.truetype(font_path, size=size)
                        # 测试字体
                        test_text = "Test"
                        bbox = font.getbbox(test_text)
                        if bbox:
                            #print(f"    - 成功加载: {os.path.basename(font_path)}")
                            return font
                except Exception as e:
                    _subtitle_font_log_once(("fail", font_path, size), f"    - 加载失败 {os.path.basename(font_path)}: {e}")
                    continue
        # 如果所有字体都失败，返回默认字体
        _subtitle_font_log_once(("default", "mac", size), "    - 使用PIL默认字体")
        return ImageFont.load_default()

    def get_font_for_windows(size):
        """Windows专用字体加载
        修复：FONT_PATH_OVERRIDE 优先级最高，放在最前检查（原代码优先级逻辑反了）。
        """
        # ① 最高优先级：用户通过环境变量或命令行指定的自定义字体
        if FONT_PATH_OVERRIDE and os.path.exists(FONT_PATH_OVERRIDE):
            try:
                _subtitle_font_log_once(("try", FONT_PATH_OVERRIDE, size), f"    - 尝试加载覆盖字体: {os.path.basename(FONT_PATH_OVERRIDE)}")
                font = ImageFont.truetype(FONT_PATH_OVERRIDE, size=size)
                _subtitle_font_log_once(("ok", FONT_PATH_OVERRIDE, size), f"    - 成功加载覆盖字体: {FONT_PATH_OVERRIDE}")
                return font
            except Exception as e:
                _subtitle_font_log_once(("fail", FONT_PATH_OVERRIDE, size), f"    - 覆盖字体加载失败（将继续尝试系统字体）: {e}")

        # ② 系统字体目录扫描
        windows_dirs = [
            os.path.join(os.environ.get('WINDIR', 'C:\\Windows'), 'Fonts'),
            'C:\\Windows\\Fonts',
            'D:\\Windows\\Fonts',
        ]
        windows_fonts = [
            'malgun.ttf',      # 韩文
            'gulim.ttc',       # 韩文
            'arialuni.ttf',    # Arial Unicode
            'msyh.ttc',        # 中文
            'msgothic.ttc',    # 日文
            'segoeui.ttf',
            'tahoma.ttf',
            'arial.ttf',
        ]

        for font_dir in windows_dirs:
            if os.path.exists(font_dir):
                for font_name in windows_fonts:
                    font_path = os.path.join(font_dir, font_name)
                    if os.path.exists(font_path):
                        try:
                            _subtitle_font_log_once(("try", font_path, size), f"    - 尝试加载: {font_name}")
                            if font_name.endswith('.ttc'):
                                font = ImageFont.truetype(font_path, size=size, index=0)
                            else:
                                font = ImageFont.truetype(font_path, size=size)
                            # 测试字体可正常渲染
                            bbox = font.getbbox("Test")
                            if bbox:
                                _subtitle_font_log_once(("ok", font_path, size), f"    - 成功加载: {font_name}")
                                return font
                        except Exception as e:
                            _subtitle_font_log_once(("fail", font_path, size), f"    - 加载失败 {font_name}: {e}")
                            continue

        _subtitle_font_log_once(("default", "windows", size), "    - 使用PIL默认字体")
        return ImageFont.load_default()

    # 根据系统选择字体加载函数
    if cached_font is not None:
        font = cached_font
    else:
        if sys.platform == 'darwin':
            #print("    - 系统: macOS")
            font = get_font_for_mac(font_size)
        elif sys.platform.startswith('win'):
            #print("    - 系统: Windows")
            font = get_font_for_windows(font_size)
        else:
            #print("    - 系统: Linux/其他")
            font = ImageFont.load_default()
        _SUBTITLE_FONT_CACHE[font_cache_key] = font

    # ========== 4. 确保font是有效的字体对象 ==========
    if font is None or not hasattr(font, 'getbbox'):
        print("    - 警告: 字体对象无效，使用默认字体")
        font = ImageFont.load_default()

    # ========== 5. 字符宽度配置 ==========
    char_widths = {
        'zh': 22, 'ja': 22, 'ko': 24, 'ar': 20, 'th': 22,
        'hi': 22, 'vi': 18, 'id': 18, 'tr': 18, 'pt': 18,
        'es': 18, 'fr': 18, 'ru': 18, 'default': 18
    }
    avg_char_width = char_widths.get(base_lang, char_widths['default'])
    max_chars = max(10, int(img_width * 0.9 / avg_char_width))  # 使用90%宽度，减少行数

    # ========== 6. 文本换行（按单词换行，避免截断） ==========
    def smart_wrap(text, max_chars, lang):
        """智能文本换行 - 按单词边界换行，避免单词截断"""
        if not text:
            return []

        # 中文、日文、韩文等不使用空格分词的语言，按字符换行
        if lang in ['zh', 'ja', 'ko']:
            if len(text) <= max_chars:
                return [text]
            lines = []
            current_line = ""
            for char in text:
                if len(current_line) >= max_chars:
                    lines.append(current_line)
                    current_line = char
                else:
                    current_line += char
            if current_line:
                lines.append(current_line)
            return lines

        # 其他语言（英文等）按单词换行
        words = text.split(' ')
        if not words:
            return [text]

        lines = []
        current_line = ""

        for word in words:
            # 如果当前行为空，直接添加单词
            if not current_line:
                current_line = word
            # 如果添加这个单词不超过最大长度，添加到当前行
            elif len(current_line) + 1 + len(word) <= max_chars:
                current_line += ' ' + word
            else:
                # 当前行已满，开始新行
                if current_line:
                    lines.append(current_line)
                current_line = word

        # 添加最后一行
        if current_line:
            lines.append(current_line)

        return lines if lines else [text]

    lines = smart_wrap(text, max_chars, base_lang)

    # ========== 7. 计算位置 ==========
    # 计算每行高度
    line_heights = []
    for line in lines:
        try:
            bbox = font.getbbox(line)
            if bbox:
                height = bbox[3] - bbox[1]
            else:
                height = font_size
        except:
            height = font_size
        line_heights.append(height)

    # 计算总高度
    total_height = 0
    for h in line_heights:
        total_height += h + 8  # 8像素行间距
    if line_heights:
        total_height -= 8  # 减去最后一行的额外间距

    # 垂直位置（从底部开始）
    y_start = estimated_height - total_height - 20
    if y_start < 10:
        y_start = 10

    # ========== 8. 绘制文字（简化版确保可靠） ==========
    current_y = y_start
    for i, line in enumerate(lines):
        # 计算文本宽度
        try:
            bbox = font.getbbox(line)
            if bbox:
                line_width = bbox[2] - bbox[0]
            else:
                line_width = len(line) * avg_char_width
        except:
            line_width = len(line) * avg_char_width

        # 水平居中
        text_x = (img_width - line_width) // 2

        # 设置文字颜色
        if base_lang == 'ko':
            text_color = (255, 255, 200, 255)  # 韩文用淡黄色
        else:
            text_color = (255, 255, 255, 255)  # 其他用白色

        # 首先绘制文字描边（使用stroke_width参数，更平滑清晰）
        outline_color = (0, 0, 0, 255)
        stroke_width = 2  # 描边宽度（适中）

        # 尝试使用PIL的stroke参数（Pillow 8.0+支持）
        try:
            draw.text((text_x, current_y), line,
                      fill=text_color, font=font,
                      stroke_width=stroke_width, stroke_fill=outline_color)
            #print(f"    - 成功绘制（stroke模式）: {line[:20]}...")
        except TypeError:
            # 旧版Pillow不支持stroke，使用传统描边方式
            for dx, dy in [(3, 0), (-3, 0), (0, 3), (0, -3),
                           (2, 2), (2, -2), (-2, 2), (-2, -2)]:
                try:
                    draw.text((text_x + dx, current_y + dy), line,
                              fill=outline_color, font=font)
                except Exception as e:
                    print(f"    - 描边绘制失败: {e}")
            # 绘制主文字
            try:
                draw.text((text_x, current_y), line,
                          fill=text_color, font=font)
                print(f"    - 成功绘制（传统模式）: {line[:20]}...")
            except Exception as e:
                print(f"    - 主文字绘制失败: {e}")
        except Exception as e:
            print(f"    - 绘制失败: {e}")

        # 更新Y位置
        current_y += line_heights[i] + 8

    # ========== 9. 保存和返回 ==========
    # 修复：mkstemp 返回的 fd 必须先 os.close()，否则 Windows 保持句柄导致后续 PermissionError
    temp_img_fd, temp_img_path = tempfile.mkstemp(suffix='.png')
    os.close(temp_img_fd)
    try:
        subtitle_img.save(temp_img_path, format='PNG', optimize=True)
        #print(f"    - 字幕图片保存: {temp_img_path}")
    except Exception as e:
        print(f"    - 图片保存失败: {e}")
        # 创建简单的错误图片
        error_img = Image.new('RGBA', (100, 50), color=(255, 0, 0, 128))
        error_img.save(temp_img_path, format='PNG')

    from moviepy.video.VideoClip import ImageClip
    try:
        clip = ImageClip(temp_img_path, duration=duration).set_start(start_time).set_position(('center', 'bottom'))
        clip.temp_path = temp_img_path
        #print(f"    - 字幕片段创建成功: {duration:.2f}s")
        return clip
    except Exception as e:
        print(f"    - 创建ImageClip失败: {e}")
        # 返回一个空片段
        from moviepy.video.VideoClip import ColorClip
        empty_clip = ColorClip(size=(10, 10), color=(0, 0, 0), duration=0.1).set_start(start_time)
        empty_clip.temp_path = None
        return empty_clip

def process_single_video(input_video_path, target_language, selected_voice_key, output_video_path, max_speed_factor, min_speed_factor, available_voices, parallel=True, workers=10, tts_workers=3):
    """处理单个视频的函数 - 修复资源泄漏版

    Args:
        input_video_path: 输入视频路径
        target_language: 目标语言代码
        selected_voice_key: 选择的语音配置键
        output_video_path: 输出视频路径
        max_speed_factor: 最大速度因子
        min_speed_factor: 最小速度因子
        available_voices: 可用语音配置字典
        parallel: 是否启用并行翻译 (默认: True)
        workers: 并行翻译线程数 (默认: 10)
        tts_workers: 并行 TTS 生成线程数 (默认: 3)
    """
    import gc
    import numpy as np
    from moviepy.audio.AudioClip import AudioArrayClip

    if not os.path.exists(input_video_path):
        print(f"错误: 文件不存在 - {input_video_path}")
        return False
    
    print(f"--- 开始处理: {input_video_path} ---")
    # 强制垃圾回收
    gc.collect()
    
    original_video = None
    base_video = None
    final_audio_track = None
    video_with_new_audio = None
    final_video_with_subtitles = None
    separation_result = None
    final_audio_clips_for_composition = []
    subtitle_clips = []
    temp_files_to_cleanup = []
    adjusted_tts_files = []
    speaker_diarization_started = False

    def _cleanup_background_speaker_task():
        nonlocal speaker_diarization_started
        if speaker_diarization_started and SPEAKER_AWARE_AVAILABLE:
            try:
                wait_diarization()
            except Exception:
                pass
            speaker_diarization_started = False
    
    try:
        # 加载原视频
        original_video = VideoFileClip(input_video_path)
        video_duration = original_video.duration
        video_width = original_video.w
        video_height = original_video.h
        
        print(f"视频时长: {video_duration:.2f}s")

        pipeline_run = get_pipeline_run(
            input_video_path=input_video_path,
            output_video_path=output_video_path,
            target_language=target_language,
            selected_voice_key=selected_voice_key,
            extra_params={
                "max_speed_factor": max_speed_factor,
                "min_speed_factor": min_speed_factor,
                "parallel": parallel,
                "workers": workers,
                "tts_workers": tts_workers,
            },
        )

        # Step 1: 文本识别（优先 OCR 硬字幕；无可用字幕才跑 ASR）
        print("\n[1/6] 文本识别...")
        
        try:
            audio_stage_dir = pipeline_run.stage_dir("audio")
            audio_stage_ready = is_stage_complete(
                pipeline_run,
                "audio",
                [audio_stage_dir / "background.wav", audio_stage_dir / "dialogue.wav"],
            )
            recognition_source = get_stage_source(pipeline_run, "recognition")

            if not audio_stage_ready and recognition_source != "ocr":
                separation_result = get_or_create_audio_stage(pipeline_run, input_video_path, video_duration)
                segments = get_or_create_recognition_stage(
                    pipeline_run,
                    input_video_path,
                    video_duration,
                    separation_result.dialogue_path,
                )
            else:
                # OCR 和音频分离互不依赖，先并行启动；只有 OCR 不可用时才等待 dialogue.wav 做 ASR。
                with ThreadPoolExecutor(max_workers=2) as stage_executor:
                    audio_future = stage_executor.submit(
                        get_or_create_audio_stage, pipeline_run, input_video_path, video_duration
                    )
                    recognition_future = stage_executor.submit(
                        get_or_create_recognition_stage,
                        pipeline_run,
                        input_video_path,
                        video_duration,
                        None,
                    )
                    try:
                        segments = recognition_future.result()
                        separation_result = audio_future.result()
                    except Exception:
                        separation_result = audio_future.result()
                        segments = get_or_create_recognition_stage(
                            pipeline_run,
                            input_video_path,
                            video_duration,
                            separation_result.dialogue_path,
                            try_ocr=False,
                        )

            asr_audio_path = separation_result.dialogue_path
            background_audio_path = separation_result.background_path

            # ===== 在文本识别后启动说话人分离；输入依赖 dialogue.wav =====
            if SPEAKER_AWARE_AVAILABLE:
                _hf_token = os.environ.get('HF_TOKEN', '')
                speaker_stage_path = pipeline_run.stage_dir("speaker_gender") / "speaker_gender.json"
                speaker_stage_ready = is_stage_complete(
                    pipeline_run,
                    "speaker_gender",
                    [speaker_stage_path],
                )
                if _hf_token and not speaker_stage_ready:
                    run_diarization_async(asr_audio_path, _hf_token)
                    speaker_diarization_started = True
            # =========================================================
            
            original_segments_data = []
            for segment in segments:
                if segment["start"] >= video_duration:
                    continue
                actual_end = min(segment["end"], video_duration)
                duration = actual_end - segment["start"]
                # 过滤短片段
                if duration < 0.3:
                    continue

                # 过滤语气词片段
                text = segment["text"].strip()
                if is_filler_word(text):
                    print(f"  - 过滤语气词: [{segment['start']:.1f}s] '{text}'")
                    continue

                original_segments_data.append({
                    "text": text,
                    "start": segment["start"],
                    "end": actual_end,
                    "original_duration": duration
                })
            print(f"  - 识别到 {len(original_segments_data)} 个有效语音片段")
            
            if not original_segments_data:
                print("错误: 未识别到任何有效语音。请检查人声分离结果、OCR 环境或 faster-whisper 模型。")
                return False
                
        except Exception as e:
            print(f"  - 语音识别失败: {e}")
            return False
        
        gc.collect()

        # ===== Step 2: 说话人分离在后台运行，先执行翻译以重叠耗时 =====
        speaker_map = {}
        speaker_voice_map = {}
        if SPEAKER_AWARE_AVAILABLE:
            print('\n[2/6] 说话人识别...')
            _hf_token = os.environ.get('HF_TOKEN', '')
            if _hf_token and speaker_diarization_started:
                print('  [说话人识别] 已在后台运行，将与翻译并行')
            elif _hf_token:
                print('  [说话人识别] 将复用缓存或在翻译后获取结果')
            else:
                print('  [跳过] 未设置 HF_TOKEN')
        # =============================================================
        
        # Step 3: 翻译
        print(f"\n[3/6] 翻译为 {target_language}...")
        translated_segments_data = []

        try:
            translated_results = get_or_create_translation_stage(
                pipeline_run,
                original_segments_data,
                target_language,
                max_workers=workers if parallel else 1,
            )
        except Exception as e:
            print(f"  - 翻译失败，已停止后续 TTS/合成: {e}")
            _cleanup_background_speaker_task()
            return False
        for result in translated_results:
            translated_segments_data.append({
                "idx": result.get("idx"),
                "original_text": result["text"],
                "translated_text": result["translated"],
                "original_duration": result["duration"],
                "start": result["start"],
                "end": result["end"]
            })

        # ===== 翻译完成后再获取说话人分离结果 =====
        if SPEAKER_AWARE_AVAILABLE:
            _hf_token = os.environ.get('HF_TOKEN', '')
            if _hf_token:
                print('\n[2/6] 获取说话人分离结果...')
                speaker_map = get_or_create_speaker_gender_stage(pipeline_run, wait_diarization)
                speaker_diarization_started = False
                if not speaker_map:
                    print('  [说话人识别] 结果为空，使用单一声音')

        # ===== 分配说话人声音 =====
        runtime_available_voices = available_voices
        if SPEAKER_AWARE_AVAILABLE and speaker_map:
            print('\n[3/6] 分配说话人声音...')
            speaker_map = enrich_speaker_map_with_subtitle_genders(
                asr_audio_path,
                original_segments_data,
                speaker_map,
            )
            speaker_voice_map = build_speaker_voice_map(
                speaker_map, target_language, available_voices, selected_voice_key
            )
            clone_voices, clone_voice_map = _build_speaker_clone_voices(
                pipeline_run,
                asr_audio_path,
                speaker_map,
                target_language,
            )
            if clone_voices:
                runtime_available_voices = dict(available_voices)
                runtime_available_voices.update(clone_voices)
                speaker_voice_map.update(clone_voice_map)
                print(f"  [Speaker Clone] 启用 {len(clone_voices)} 个原视频克隆声音")
        # ==========================

        # Step 4: 加载TTS模型
        print("\n[4/6] 加载TTS模型...")
        voice_config = runtime_available_voices.get(selected_voice_key,
                                            {"model_name": "tts_models/en/ljspeech/vits"})
        
        tts_model = None
        tts_speaker_idx = None
        if not (SPEAKER_AWARE_AVAILABLE and speaker_map and speaker_voice_map):
            try:
                tts_model, tts_speaker_idx = load_coqui_tts_model(voice_config, gpu_is_available=False)
            except Exception as e:
                print(f"  - TTS模型加载失败: {e}")
                return False
        else:
            print("  - 多说话人模式：按片段声音分组延迟加载 TTS 模型")
        
        # Step 5: 生成并调整TTS音频
        print("\n[5/6] 生成并调整配音音频...")
        invalidate_composition_for_tts(pipeline_run)

        # 5.1 并行生成所有 TTS 音频
        # 说话人感知：为每个片段标记对应的 voice_key
        tts_cache_dir = pipeline_run.stage_dir("tts") / "clips"
        if SPEAKER_AWARE_AVAILABLE and speaker_map and speaker_voice_map:
            voice_alignment_report = []
            for seg in translated_segments_data:
                seg['_voice_key'] = get_voice_for_segment(
                    seg['start'], seg['end'],
                    speaker_map, speaker_voice_map, selected_voice_key,
                    available_voices=runtime_available_voices,
                    target_lang=target_language
                )
                voice_alignment_report.append(explain_segment_voice_alignment(
                    seg['start'], seg['end'],
                    seg.get('original_text') or seg.get('translated_text', ''),
                    speaker_map, speaker_voice_map, selected_voice_key,
                    available_voices=runtime_available_voices,
                    target_lang=target_language,
                ))
            print_voice_alignment_summary(voice_alignment_report, selected_voice_key)
            try:
                diagnostics_path = pipeline_run.stage_dir("speaker_gender") / "aligned_segments.json"
                from pipeline_cache import atomic_write_json
                atomic_write_json(diagnostics_path, voice_alignment_report)
            except Exception as e:
                print(f"  [说话人识别] 对齐诊断写入失败: {e}")
            # 按 voice_key 分组，每组用自己的 TTS 模型生成
            tts_results, tts_temp_files = _generate_tts_multi_voice(
                translated_segments_data, target_language,
                runtime_available_voices, selected_voice_key,
                False, max_workers=tts_workers,
                cache_root=tts_cache_dir,
            )
        else:
            tts_results, tts_temp_files = generate_tts_parallel(
                translated_segments_data, tts_model, tts_speaker_idx, target_language,
                max_workers=tts_workers,
                cache_dir=tts_cache_dir,
            )

        successful_tts_results = [
            result for result in tts_results
            if result and result.get("success") and result.get("temp_tts_file")
        ]
        if len(successful_tts_results) != len(translated_segments_data):
            failed_indexes = [
                str(result.get("idx", idx)) if result else str(idx)
                for idx, result in enumerate(tts_results)
                if not (result and result.get("success") and result.get("temp_tts_file"))
            ]
            print(f"  - TTS 生成不完整，失败片段: {', '.join(failed_indexes)}")
            return False
        if not successful_tts_results:
            print("  - TTS 未生成任何有效音频片段")
            return False

        raw_tts_durations = []
        ordered_segments = []
        for tts_result in successful_tts_results:
            raw_duration = _probe_audio_duration(tts_result["temp_tts_file"])
            raw_tts_durations.append(raw_duration)
            ordered_segments.append(tts_result["seg_data"])

        planned_timeline = build_tts_timeline(
            ordered_segments,
            raw_tts_durations,
            video_duration,
            max_speed_factor=max_speed_factor,
        )
        print(f"  - 时间调度完成: {len(planned_timeline)} 段")

        adjusted_durations = []

        # 5.2 顺序处理音频速度调整和片段创建
        for order_idx, (tts_result, timeline_entry) in enumerate(zip(successful_tts_results, planned_timeline)):
            seg_data = tts_result["seg_data"]
            temp_tts_file = tts_result["temp_tts_file"]

            print(f"\n--- 处理片段 {order_idx+1}/{len(successful_tts_results)} ---")
            print(f"    原时间: [{seg_data['start']:.2f}s -> {seg_data['end']:.2f}s], 时长: {seg_data['original_duration']:.2f}s")

            # 调整音频速度
            tmp_adj = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            temp_adjusted_file = tmp_adj.name
            tmp_adj.close()
            temp_files_to_cleanup.append(temp_adjusted_file)
            adjusted_tts_files.append(temp_adjusted_file)

            try:
                final_duration, speed_factor = adjust_audio_speed_ffmpeg(
                    temp_tts_file,
                    temp_adjusted_file,
                    timeline_entry["target_duration"],
                    max_speed_factor=max_speed_factor,
                    min_speed_factor=min_speed_factor
                )
                adjusted_durations.append(_probe_audio_duration(temp_adjusted_file))
                print(
                    f"    调整后时长: {final_duration:.2f}s, 速度: {speed_factor:.2f}x, "
                    f"目标窗口: {timeline_entry['target_duration']:.2f}s"
                )
            except Exception as e:
                print(f"    - 音频速度调整失败: {e}")
                adjusted_durations.append(max(0.0, timeline_entry["target_duration"]))
                continue

            # 清理当前片段的临时 TTS 文件
            if temp_tts_file and os.path.exists(temp_tts_file) and not tts_result.get("cached"):
                try:
                    if not str(temp_tts_file).startswith(str(tts_cache_dir)):
                        _safe_remove(temp_tts_file)
                except:
                    pass

        # 清理剩余的临时 TTS 文件
        for temp_file in tts_temp_files:
            if temp_file and os.path.exists(temp_file):
                try:
                    if not str(temp_file).startswith(str(tts_cache_dir)):
                        os.remove(temp_file)
                except:
                    pass

        final_timeline = finalize_tts_timeline(
            planned_timeline,
            adjusted_durations,
            video_duration,
        )
        final_output_duration = max(video_duration, final_timeline[-1]["planned_end"])

        for tts_result, timeline_entry, adjusted_file in zip(
            successful_tts_results,
            final_timeline,
            adjusted_tts_files,
        ):
            try:
                raw_clip = AudioFileClip(adjusted_file)
                if raw_clip.duration > timeline_entry["target_duration"] + 0.05:
                    raw_clip = raw_clip.subclip(0, timeline_entry["target_duration"])
                adjusted_clip = raw_clip.set_start(timeline_entry["planned_start"])
                final_audio_clips_for_composition.append(adjusted_clip)
            except Exception as e:
                print(f"    - 创建音频片段失败: {e}")
                continue

            if timeline_entry["target_duration"] > 0.1:
                try:
                    subtitle_clip = create_subtitle_clip(
                        tts_result["seg_data"]["translated_text"],
                        timeline_entry["planned_start"],
                        timeline_entry["target_duration"],
                        video_width,
                        video_height,
                        target_language
                    )
                    subtitle_clips.append(subtitle_clip)
                except Exception as e:
                    print(f"    - 创建字幕片段失败: {e}")
        
        # 释放 TTS 模型内存
        mark_tts_stage_complete(
            pipeline_run,
            [
                {
                    "idx": result.get("idx"),
                    "path": result.get("temp_tts_file"),
                    "success": result.get("success"),
                    "cache_digest": result.get("cache_digest"),
                }
                for result in tts_results
                if result
            ],
            timeline=final_timeline,
        )
        del tts_model
        gc.collect()
        
        # Step 6: 合成最终视频
        print("\n[6/6] 合成最终视频...")
        print("  - 合成音频轨道 (使用绝对时间对齐)...")
        
        try:
            composition_stage_dir = pipeline_run.stage_dir("composition")
            frozen_frame_path = str(composition_stage_dir / "frozen_tail.png")
            # 6.1 使用分离后的背景轨，保留 BGM/环境声但不保留原对白
            background_audio = AudioFileClip(background_audio_path).set_start(0)
            background_audio = CompositeAudioClip([background_audio]).set_duration(final_output_duration)
            print("  - 使用分离后的背景轨并替换原视频对白")
            
            # 6.2 组合音频（总时长跟随最终排程；尾部不足部分保持静音）
            all_audio_clips = [background_audio] + final_audio_clips_for_composition
            final_audio_track = CompositeAudioClip(all_audio_clips)
            final_audio_track = final_audio_track.set_duration(final_output_duration)

            # 6.3 应用音频到视频
            base_video = _extend_video_with_frozen_tail(
                original_video,
                input_video_path,
                final_output_duration,
                frozen_frame_path,
            )
            video_with_new_audio = base_video.set_audio(final_audio_track)
            
            # 6.4 添加字幕
            print(f"  - 添加 {len(subtitle_clips)} 个字幕片段...")
            if subtitle_clips:
                final_video_with_subtitles = CompositeVideoClip(
                    [video_with_new_audio] + subtitle_clips,
                    size=original_video.size,
                ).set_duration(final_output_duration)
            else:
                final_video_with_subtitles = video_with_new_audio.set_duration(final_output_duration)
            
            # 6.5 输出视频 - 修复参数避免卡死
            print(f"  - 正在输出视频到: {output_video_path}")
            print("  - 此过程可能需要较长时间，请耐心等待...")
            
            final_video_with_subtitles.write_videofile(
                output_video_path,
                codec='libx264',
                audio_codec='aac',
                fps=original_video.fps if original_video.fps else 30,
                preset='medium',
                ffmpeg_params=['-crf', '23', '-pix_fmt', 'yuv420p'],
                threads=2,  # 减少线程数避免卡死
                logger='bar',  # 显示进度条
                temp_audiofile='temp-audio.m4a',
                remove_temp=True
            )
            
            # 验证输出文件
            if os.path.exists(output_video_path):
                file_size = os.path.getsize(output_video_path)
                if file_size > 10240:  # 大于10KB
                    print(f"  - ✓ 文件写入成功: {file_size/1024/1024:.1f} MB")
                    mark_composition_stage_complete(
                        pipeline_run,
                        output_video_path,
                        expected_min_duration=final_output_duration,
                    )
                    return True
                else:
                    print(f"  - ✗ 输出文件过小: {file_size} bytes")
                    return False
            else:
                print(f"  - ✗ 输出文件不存在")
                return False
                
        except Exception as e:
            print(f"  - 视频合成失败: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    finally:
        # ========== 关键：确保所有资源都被释放 ==========
        print("\n- 清理资源...")
        _cleanup_background_speaker_task()
        
        # 关闭所有音频片段
        for clip in final_audio_clips_for_composition:
            try:
                clip.close()
            except:
                pass
        
        # 关闭所有字幕片段并删除临时图片
        for clip in subtitle_clips:
            try:
                clip.close()
                if hasattr(clip, 'temp_path') and clip.temp_path and os.path.exists(clip.temp_path):
                    _safe_remove(clip.temp_path)
            except:
                pass
        
        # 删除所有临时音频文件
        for temp_file in temp_files_to_cleanup:
            try:
                if os.path.exists(temp_file):
                    _safe_remove(temp_file)
            except:
                pass
        
        # 关闭 MoviePy 对象
        for obj in [final_video_with_subtitles, video_with_new_audio, base_video, final_audio_track, original_video]:
            if obj:
                try:
                    temp_path = getattr(obj, "temp_path", None)
                    obj.close()
                    if temp_path and os.path.exists(temp_path):
                        _safe_remove(temp_path)
                except:
                    pass

        if separation_result:
            try:
                separation_result.cleanup()
            except:
                pass
        
        # 强制垃圾回收
        gc.collect()
        
        print("- 资源清理完成")

def get_video_stream_info(video_path):
    """使用ffprobe获取视频流关键参数（增强错误处理）"""
    try:
        cmd = [
            'ffprobe',
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_streams',
            '-select_streams', 'v:0',
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=10)
        
        if result.returncode != 0:
            # 尝试备用命令（某些FFmpeg版本需要）
            cmd_alt = cmd[:-1] + ['-show_format'] + [video_path]
            result = subprocess.run(cmd_alt, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=10)
            if result.returncode != 0:
                return None
        
        info = json.loads(result.stdout)
        
        # 优先使用streams，回退到format
        streams = info.get('streams', [])
        if not streams and 'format' in info:
            # 从format中提取基本信息（有限）
            return {
                'width': 0,
                'height': 0,
                'codec': 'unknown',
                'profile': '',
                'pix_fmt': 'unknown',
                'r_frame_rate': '30/1',
                'bit_rate': '0'
            }
        
        if not streams:
            return None
        
        stream = streams[0]
        return {
            'width': int(stream.get('width', 0)),
            'height': int(stream.get('height', 0)),
            'codec': stream.get('codec_name', 'unknown'),
            'profile': stream.get('profile', ''),
            'pix_fmt': stream.get('pix_fmt', 'unknown'),
            'r_frame_rate': stream.get('r_frame_rate', '30/1'),
            'bit_rate': stream.get('bit_rate', '0')
        }
    except Exception as e:
        print(f"  ⚠️  ffprobe失败 {os.path.basename(video_path)}: {str(e)[:60]}")
        return None

def check_videos_compatible(video_files, strict_mode=False):
    """检查视频是否可无损合并（修复空流检测）"""
    if not video_files:
        return False, "无视频文件"
    
    # 获取基准视频信息
    base_info = get_video_stream_info(video_files[0])
    if not base_info:
        return False, f"无法读取基准视频: {os.path.basename(video_files[0])}"
    
    print(f"  ✓ 基准: {base_info['width']}x{base_info['height']}, {base_info['codec']}, {base_info['pix_fmt']}")
    
    # 检查所有视频
    for i, path in enumerate(video_files[1:], 1):
        info = get_video_stream_info(path)
        if not info:
            return False, f"无法读取: {os.path.basename(path)}"
        
        # 关键参数必须一致（宽松模式允许bit_rate/profile差异）
        mismatches = []
        if info['width'] != base_info['width']:
            mismatches.append(f"宽度 {info['width']}≠{base_info['width']}")
        if info['height'] != base_info['height']:
            mismatches.append(f"高度 {info['height']}≠{base_info['height']}")
        if info['codec'] != base_info['codec']:
            mismatches.append(f"编码 {info['codec']}≠{base_info['codec']}")
        if info['pix_fmt'] != base_info['pix_fmt']:
            mismatches.append(f"像素格式 {info['pix_fmt']}≠{base_info['pix_fmt']}")
        
        if mismatches:
            return False, (
                f"参数不兼容 [{i+1}]: {os.path.basename(path)}\n"
                f"  基准: {base_info['width']}x{base_info['height']}, {base_info['codec']}, {base_info['pix_fmt']}\n"
                f"  当前: {info['width']}x{info['height']}, {info['codec']}, {info['pix_fmt']}\n"
                f"  差异: {', '.join(mismatches)}"
            )
    
    return True, "✓ 所有视频参数兼容，可无损合并"

def merge_videos_ffmpeg_safe(video_files, output_path):
    """使用FFmpeg concat demuxer安全合并（增强错误诊断）"""
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    
    # 创建临时文件列表（UTF-8 + 路径转义）
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8', newline='') as f:
        filelist_path = f.name
        for video_path in video_files:
            # 路径标准化（Mac/Windows通用）
            abs_path = os.path.abspath(video_path).replace('\\', '/')
            # 转义单引号（FFmpeg要求）
            escaped_path = abs_path.replace("'", "'\\''")
            f.write(f"file '{escaped_path}'\n")
    
    try:
        # FFmpeg合并命令
        cmd = [
            'ffmpeg',
            '-f', 'concat',
            '-safe', '0',
            '-i', filelist_path,
            '-c', 'copy',
            '-fflags', '+genpts',
            '-movflags', '+faststart',  # 优化网络播放
            '-y',
            output_path
        ]
        
        print(f"\n  → 执行无损合并 ({len(video_files)}个视频)...")
        print(f"     输出: {os.path.abspath(output_path)}")
        
        # 执行合并（捕获完整输出用于诊断）
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding='utf-8',
            timeout=3600  # 1小时超时
        )
        
        # 诊断输出
        if result.returncode != 0:
            print(f"\n  ✗ FFmpeg合并失败 (退出码: {result.returncode})")
            print(f"     错误输出:\n{result.stderr[:500]}")  # 仅显示前500字符
            return False
        
        # 验证输出文件（宽松条件：>10KB）
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            if file_size > 10240:  # >10KB
                print(f"     ✓ 合并成功: {os.path.basename(output_path)} ({file_size/1024/1024:.1f} MB)")
                return True
            else:
                print(f"     ✗ 输出文件过小 ({file_size} bytes)，可能为空文件")
                return False
        else:
            print(f"     ✗ 输出文件不存在: {output_path}")
            # 尝试查找可能生成的文件
            output_dir = os.path.dirname(output_path) or '.'
            candidates = [f for f in os.listdir(output_dir) if f.endswith('.mp4') or f.endswith('.mkv')]
            if candidates:
                print(f"     提示: 目录中找到其他视频文件: {candidates[:3]}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"  ✗ 合并超时 (1小时)")
        return False
    except Exception as e:
        print(f"  ✗ 合并异常: {type(e).__name__}: {str(e)}")
        return False
    finally:
        # 延迟清理临时文件（避免竞态条件）
        time.sleep(0.1)
        if os.path.exists(filelist_path):
            try:
                _safe_remove(filelist_path)
            except Exception as e:
                print(f"  ⚠️  临时文件清理失败: {e}")

def merge_videos_from_directory(output_dir, merged_filename, video_extension=".mp4"):
    """优化版视频合并（修复验证逻辑 + 增强诊断）"""
    print(f"\n{'='*60}")
    print(f"🚀 优化视频合并 (FFmpeg Concat Demuxer)")
    print(f"{'='*60}")
    
    # 标准化路径
    output_dir = os.path.abspath(output_dir)
    output_path = os.path.join(output_dir, merged_filename)
    
    # 查找视频文件
    pattern = os.path.join(output_dir, f"*{video_extension}")
    video_files = sorted(glob.glob(pattern))
    
    if not video_files:
        print(f"❌ 错误: 目录 '{output_dir}' 中无 {video_extension} 文件")
        # 调试：列出目录内容
        print(f"   目录内容: {os.listdir(output_dir)[:10]}")
        return False
    
    print(f"📁 找到 {len(video_files)} 个视频 (目录: {output_dir}):")
    for i, vf in enumerate(video_files[:min(5, len(video_files))], 1):
        size_mb = os.path.getsize(vf) / (1024*1024)
        print(f"   [{i}] {os.path.basename(vf)} ({size_mb:.1f} MB)")
    if len(video_files) > 5:
        print(f"   ... 共 {len(video_files)} 个文件")
    
    # 检查兼容性（宽松模式）
    print(f"\n🔍 检查视频参数兼容性...")
    compatible, msg = check_videos_compatible(video_files)
    
    if not compatible:
        print(f"⚠️  {msg}")
        # 自动转码合并（可选，此处仅提示）
        print("\n💡 建议: 重新生成视频时统一输出参数")
        print("   在 process_single_video 中固定:")
        print("   codec='libx264', audio_codec='aac', fps=30, preset='medium'")
        return False
    
    # 执行合并
    success = merge_videos_ffmpeg_safe(video_files, output_path)
    
    if success:
        # 最终验证
        if os.path.exists(output_path) and os.path.getsize(output_path) > 10240:
            final_size = os.path.getsize(output_path) / (1024*1024)
            total_input = sum(os.path.getsize(v) for v in video_files) / (1024*1024)
            print(f"\n✅ 合并成功!")
            print(f"   输出路径: {output_path}")
            print(f"   大小: {final_size:.1f} MB (输入总大小: {total_input:.1f} MB)")
            print(f"   耗时: 约 {len(video_files) * 2 // 60}m{len(video_files) * 2 % 60}s")
            return True
        else:
            print(f"\n❌ 输出文件验证失败: {output_path}")
            if os.path.exists(output_path):
                print(f"   文件大小: {os.path.getsize(output_path)} bytes")
            return False
    else:
        print(f"\n❌ 合并失败，请检查:")
        print(f"   1. 输出目录是否存在: {output_dir}")
        print(f"   2. 磁盘空间是否充足 (需要 {sum(os.path.getsize(v) for v in video_files)/(1024*1024*1024):.1f} GB)")
        print(f"   3. 视频文件是否被其他程序占用")
        return False

def batch_process_videos(input_dir, output_dir, target_language, selected_voice_key, max_speed_factor, min_speed_factor, available_voices, video_extension=".mp4", parallel=True, workers=10, tts_workers=3):
    """批量处理指定目录下的所有视频文件

    Args:
        input_dir: 输入目录
        output_dir: 输出目录
        target_language: 目标语言代码
        selected_voice_key: 选择的语音配置键
        max_speed_factor: 最大速度因子
        min_speed_factor: 最小速度因子
        available_voices: 可用语音配置字典
        video_extension: 视频文件扩展名
        parallel: 是否启用并行翻译
        workers: 并行翻译线程数
        tts_workers: 并行 TTS 生成线程数
    """
    print(f"\n--- 开始批量处理视频 ---")
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"目标语言: {target_language}")
    print(f"选择声音: {selected_voice_key}")
    print(f"并行翻译: {'启用' if parallel else '禁用'} (线程数: {workers})")
    print(f"并行 TTS: 线程数: {tts_workers}")

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 规范化输入目录路径，转换为绝对路径，以确保 glob 正确工作
    normalized_input_dir = os.path.abspath(input_dir)
    #print(f"  - 规范化输入目录路径: {normalized_input_dir}")

    # 查找输入目录下的所有视频文件
    video_extensions = (video_extension,)
    video_files = []
    for ext in video_extensions:
        # 使用规范化后的目录路径
        pattern = os.path.join(normalized_input_dir, f"*{ext}")
        print(f"  - 搜索模式: {pattern}")
        found_files = glob.glob(pattern)
        video_files.extend(found_files)
        # Also check uppercase extension
        #pattern_upper = os.path.join(normalized_input_dir, f"*{ext.upper()}")
        #print(f"  - 搜索模式 (大写): {pattern_upper}")
        #found_files_upper = glob.glob(pattern_upper)
        #video_files.extend(found_files_upper)

    # 再次规范化找到的文件路径（可选，但有助于调试）
    video_files = [os.path.abspath(f) for f in video_files]

    if not video_files:
        print(f"错误: 在目录 '{normalized_input_dir}' 中没有找到扩展名为 '{video_extension}' 的视频文件。")
        return

    print(f"找到 {len(video_files)} 个视频文件需要处理:")
    for vf in video_files:
        print(f"  - {vf}") # 打印完整路径以便确认

    # 遍历处理每个视频
    for input_video_path in video_files:
        # 生成输出文件名，保持原名但改变路径和扩展名
        # 使用原始输入路径的 basename 来构造输出名
        original_filename = os.path.basename(input_video_path)
        name_without_ext, _ = os.path.splitext(original_filename)
        output_video_path = os.path.join(output_dir, f"{name_without_ext}.mp4")

        print(f"\n--- 处理文件: {original_filename} (来自 {input_video_path}) ---")
        success = process_single_video(
            input_video_path, # 传递绝对路径给处理函数
            target_language,
            selected_voice_key,
            output_video_path,
            max_speed_factor,
            min_speed_factor,
            available_voices,
            parallel=parallel,
            workers=workers,
            tts_workers=tts_workers
        )
        if not success:
            print(f"警告: 处理 '{input_video_path}' 时失败。跳过此文件。")

    print("\n--- 所有视频处理完成 ---")

#单文件处理:python video_dubbing.py --mode single --input_video input.mp4 --target_lang pt --voice pt_male_001 --output_video output_dubbed.mp4
#多文件处理:python video_dubbing.py --mode batch --input_dir ./input --output_dir ./output --target_lang pt --voice pt_male_001
#多文件处理+合并:python video_dubbing.py --mode batch_merge --input_dir ./input --output_dir ./output --target_lang pt --voice pt_male_001 --merged_filename final_movie.mp4
#仅合并:python video_dubbing.py --mode merge_only --output_dir ./output --merged_filename final_movie.mp4
def _generate_tts_multi_voice(
    segments_data, target_lang, available_voices,
    fallback_voice_key, gpu_available, max_workers=3, cache_root=None
):
    """
    按说话人分组，为每组加载对应 TTS 模型并生成音频。
    保持原始片段顺序，返回与 generate_tts_parallel() 相同格式。
    """
    import gc

    # 收集所有不同的 voice_key
    voice_keys = list(dict.fromkeys(
        seg.get("_voice_key", fallback_voice_key) for seg in segments_data
    ))
    print(f"  - 多说话人配音: {len(voice_keys)} 种声音")
    for vk in voice_keys:
        count = sum(1 for s in segments_data if s.get("_voice_key", fallback_voice_key) == vk)
        print(f"    {vk}: {count} 片段")

    results = [None] * len(segments_data)
    all_temp_files = []
    model_cache = {}

    def prepare_voice_config(voice_config):
        prepared = dict(voice_config or {})
        if prepared.get("reference_voice_key") and not prepared.get("speaker_wav"):
            prepared["speaker_wav"] = _get_or_create_xtts_reference_wav(prepared["reference_voice_key"])
            prepared["requires_speaker_wav"] = True
        return prepared

    def model_cache_key(voice_config):
        return (
            voice_config.get("model_name", "tts_models/en/ljspeech/vits"),
            voice_config.get("language"),
        )

    for voice_key in voice_keys:
        # 找到属于该声音的片段（保留原始 index）
        group = [(i, seg) for i, seg in enumerate(segments_data)
                 if seg.get("_voice_key", fallback_voice_key) == voice_key]
        if not group:
            continue

        voice_config = prepare_voice_config(
            available_voices.get(voice_key, available_voices.get(fallback_voice_key, {}))
        )
        print(f"\n  - 加载声音模型: {voice_key}")
        try:
            cache_key = model_cache_key(voice_config)
            if cache_key in model_cache:
                tts_model = model_cache[cache_key]
                tts_speaker_idx = voice_config.get("speaker_idx")
                tts_model._xtts_language = voice_config.get("language")
                tts_model._model_name = voice_config.get("model_name")
                tts_model._speaker_wav = voice_config.get("speaker_wav")
                tts_model._requires_speaker_wav = bool(voice_config.get("requires_speaker_wav"))
                tts_model._tts_generation_profile = (
                    "xtts_clone_slightly_fast_v1" if voice_config.get("speaker_wav") else "default"
                )
                print("    - 复用已加载 TTS 模型")
            else:
                tts_model, tts_speaker_idx = load_coqui_tts_model(voice_config, gpu_available)
                model_cache[cache_key] = tts_model
        except Exception as e:
            print(f"    - 模型加载失败: {e}，使用 fallback")
            fallback_cfg = prepare_voice_config(available_voices.get(fallback_voice_key, {}))
            fallback_key = model_cache_key(fallback_cfg)
            if fallback_key in model_cache:
                tts_model = model_cache[fallback_key]
                tts_speaker_idx = fallback_cfg.get("speaker_idx")
                tts_model._xtts_language = fallback_cfg.get("language")
                tts_model._model_name = fallback_cfg.get("model_name")
                tts_model._speaker_wav = fallback_cfg.get("speaker_wav")
                tts_model._requires_speaker_wav = bool(fallback_cfg.get("requires_speaker_wav"))
                tts_model._tts_generation_profile = (
                    "xtts_clone_slightly_fast_v1" if fallback_cfg.get("speaker_wav") else "default"
                )
            else:
                tts_model, tts_speaker_idx = load_coqui_tts_model(fallback_cfg, gpu_available)
                model_cache[fallback_key] = tts_model

        # 只取该组的 seg_data 列表
        group_segs = [seg for _, seg in group]
        group_cache_dir = Path(cache_root) / voice_key if cache_root else None
        group_workers = 1 if getattr(tts_model, "_xtts_language", None) else max_workers
        group_results, group_temps = generate_tts_parallel(
            group_segs, tts_model, tts_speaker_idx, target_lang,
            max_workers=group_workers,
            cache_dir=group_cache_dir,
        )
        all_temp_files.extend(group_temps)

        # 把结果放回原始顺序
        for (orig_idx, _), res in zip(group, group_results):
            results[orig_idx] = res

        gc.collect()

    # 填补 None（不应发生，保险起见）
    for i, r in enumerate(results):
        if r is None:
            results[i] = {"idx": i, "temp_tts_file": None,
                          "seg_data": segments_data[i], "success": False}

    for tts_model in model_cache.values():
        del tts_model
    gc.collect()

    return results, all_temp_files


def main():
    parser = argparse.ArgumentParser(description="视频配音替换与批量处理工具")
    available_voices = get_available_coqui_voices()

    parser.add_argument("--mode", required=True, choices=["single", "batch", "batch_merge", "merge_only"],
                        help="运行模式: single(单文件), batch(多文件处理), batch_merge(多文件处理并合并), merge_only(仅合并)")
    parser.add_argument("--input_video", help="输入视频文件路径 (单文件模式)")
    parser.add_argument("--input_dir", help="输入视频目录路径 (多文件模式)")
    parser.add_argument("--output_dir", default="./output_videos/", help="输出视频目录路径 (多文件模式和合并模式)")
    parser.add_argument("--output_video", default="dubbed_output.mp4", help="输出视频文件路径 (单文件模式)")
    parser.add_argument("--target_lang", help="目标语言代码 (单文件和多文件处理模式),例如：en, ja, ko, fr, pt, es, id, vi, tr, hi, ar, th, de, it")
    parser.add_argument("--voice", default="en_vctk_vits_m001", choices=available_voices.keys(),
                        help="选择配音声音")
    parser.add_argument("--max_speed", type=float, default=1.5,
                        help="最大语速加速倍数 (默认: 1.5，过高会影响清晰度)")
    parser.add_argument("--min_speed", type=float, default=1.0,
                        help="最小语速倍数 (默认: 1.0，TTS短于原始时长时原速播放不拉伸)")
    parser.add_argument("--merged_filename", default="merged_output.mp4",
                        help="合并后视频的文件名 (默认: merged_output.mp4)")
    parser.add_argument("--ffmpeg_bin", dest='ffmpeg_bin', default=None,
                        help="可选: 指定 ffmpeg 可执行路径，优先级高于环境变量 FFMPEG_BIN")
    parser.add_argument("--font_path", dest='font_path', default=None,
                        help="可选: 指定字幕字体文件路径（优先于系统字体）")
    parser.add_argument("--parallel", type=lambda x: x.lower() != 'false', default=True,
                        help="启用并行翻译 (默认: True, 设置为 false 禁用)")
    parser.add_argument("--workers", type=int, default=10,
                        help="并行翻译线程数 (默认: 10, 推荐 5-20)")
    parser.add_argument("--tts_workers", type=int, default=3,
                        help="并行 TTS 生成线程数 (默认: 3, CPU 模式推荐 2-3)")

    args = parser.parse_args()

    # 如果通过命令行传入覆盖项，赋值给模块级全局变量
    global FFMPEG_BIN, FONT_PATH_OVERRIDE
    if getattr(args, 'ffmpeg_bin', None):
        FFMPEG_BIN = args.ffmpeg_bin
        print(f"- 使用自定义 ffmpeg: {FFMPEG_BIN}")
    if getattr(args, 'font_path', None):
        FONT_PATH_OVERRIDE = args.font_path
        print(f"- 使用自定义字幕字体: {FONT_PATH_OVERRIDE}")

    # --- 模式验证 ---
    if args.mode == "single":
        if not args.input_video or not args.target_lang:
            parser.error("--mode single requires --input_video and --target_lang.")
        print(f"--- 单文件处理模式 ---")
        print(f"输入视频: {args.input_video}")
        print(f"目标语言: {args.target_lang}")
        print(f"选择声音: {args.voice}")
        print(f"输出视频: {args.output_video}")
        print(f"并行翻译: {'启用' if args.parallel else '禁用'} (线程数: {args.workers})")
        print(f"并行 TTS: 线程数: {args.tts_workers}")
        success = process_single_video(
            args.input_video, args.target_lang, args.voice,
            args.output_video, args.max_speed, args.min_speed, available_voices,
            parallel=args.parallel, workers=args.workers, tts_workers=args.tts_workers
        )
        if success:
             print("\n--- 单文件处理完成 ---")
        else:
            print("\n--- 单文件处理失败 ---")

    elif args.mode == "batch":
        if not args.input_dir or not args.target_lang:
            parser.error("--mode batch requires --input_dir and --target_lang.")
        print(f"--- 多文件处理模式 ---")
        print(f"输入目录: {args.input_dir}")
        print(f"输出目录: {args.output_dir}")
        print(f"目标语言: {args.target_lang}")
        print(f"选择声音: {args.voice}")
        print(f"并行翻译: {'启用' if args.parallel else '禁用'} (线程数: {args.workers})")
        batch_process_videos(
            args.input_dir, args.output_dir, args.target_lang, args.voice,
            args.max_speed, args.min_speed, available_voices,
            parallel=args.parallel, workers=args.workers, tts_workers=args.tts_workers
        )
        print("\n--- 批量处理完成 ---")

    elif args.mode == "batch_merge":
        if not args.input_dir or not args.target_lang:
            parser.error("--mode batch_merge requires --input_dir and --target_lang.")
        print(f"--- 多文件处理并合并模式 ---")
        print(f"输入目录: {args.input_dir}")
        print(f"输出目录: {args.output_dir}")
        print(f"目标语言: {args.target_lang}")
        print(f"选择声音: {args.voice}")
        print(f"合并后文件名: {args.merged_filename}")
        print(f"并行翻译: {'启用' if args.parallel else '禁用'} (线程数: {args.workers})")
        batch_process_videos(
            args.input_dir, args.output_dir, args.target_lang, args.voice,
            args.max_speed, args.min_speed, available_voices,
            parallel=args.parallel, workers=args.workers, tts_workers=args.tts_workers
        )
        print("\n--- 批量处理完成，开始合并 ---")
        merge_videos_from_directory(args.output_dir, args.merged_filename)
        print("\n--- 处理并合并完成 ---")

    elif args.mode == "merge_only":
        print(f"--- 仅合并模式 ---")
        print(f"输出目录: {args.output_dir}")
        print(f"合并后文件名: {args.merged_filename}")
        merge_videos_from_directory(args.output_dir, args.merged_filename)
        print("\n--- 合并完成 ---")

if __name__ == "__main__":
    main()
