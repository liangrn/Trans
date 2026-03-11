import argparse
import subprocess
import os
import tempfile
import textwrap
from faster_whisper import WhisperModel
from deep_translator import GoogleTranslator
from TTS.api import TTS
import torch
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
#from moviepy.editor import VideoFileClip, AudioFileClip, CompositeVideoClip, concatenate_videoclips
#from moviepy.audio.AudioClip import CompositeAudioClip
try:
    from moviepy.editor import VideoFileClip, AudioFileClip, CompositeVideoClip, concatenate_videoclips
    from moviepy.audio.AudioClip import CompositeAudioClip
except ImportError:
    # 尝试直接导入
    import moviepy.video.io.VideoFileClip as VideoFileClip
    import moviepy.audio.io.AudioFileClip as AudioFileClip
    import moviepy.video.compositing.CompositeVideoClip as CompositeVideoClip
    import moviepy.video.compositing.concatenate as concatenate_videoclips
    import moviepy.audio.compositing.CompositeAudioClip as CompositeAudioClip
from PIL import Image, ImageDraw, ImageFont
import math
import sys
import glob
import warnings
warnings.filterwarnings("ignore", message="You are sending unauthenticated requests to the HF Hub")

import glob
import tempfile
import json
import platform
from pathlib import Path
import shutil
import time

# 配置：可通过环境变量或命令行覆盖 ffmpeg 可执行文件与字幕字体路径
# - 使用环境变量 `FFMPEG_BIN` 覆盖 ffmpeg 可执行文件（默认 'ffmpeg'）
# - 使用环境变量 `SUBTITLE_FONT_PATH` 指定优先使用的字体文件路径
FFMPEG_BIN = os.environ.get('FFMPEG_BIN', 'ffmpeg')
FONT_PATH_OVERRIDE = os.environ.get('SUBTITLE_FONT_PATH', None)

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
    调整音频速度，但限制最大/最小速度变化
    """
    original_clip = AudioFileClip(input_file)
    original_duration = original_clip.duration
    original_clip.close()
    if original_duration <= 0:
        import shutil
        shutil.copy(input_file, output_file)
        return original_duration, 1.0
    # 计算需要的速度因子，但限制在合理范围内
    raw_speed_factor = original_duration / target_duration
    # 限制速度变化范围
    if raw_speed_factor > max_speed_factor:
        print(f"    - 警告: 速度因子 {raw_speed_factor:.2f}x 超过上限 {max_speed_factor}x，将使用 {max_speed_factor}x")
        speed_factor = max_speed_factor
        final_duration = original_duration / speed_factor
    elif raw_speed_factor < min_speed_factor:
        print(f"    - 警告: 速度因子 {raw_speed_factor:.2f}x 低于下限 {min_speed_factor}x，将使用 {min_speed_factor}x")
        speed_factor = min_speed_factor
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

def get_available_coqui_voices():
    """获取可用的 Coqui TTS 声音/模型列表."""
    # 英语 印尼语 韩语 日语 越南语 西班牙语 土耳其语 葡萄牙语 印地语 阿拉伯语 泰语 法语支持
    voices = {
        # ==================== 男声模型 ====================
        # SAM Tacotron (明确的男声模型)
        "en_sam_tacotron": {"model_name": "tts_models/en/sam/tacotron-DDC", "description": "SAM Tacotron (标准男声, 清晰)"},
        # VCTK 男声 (VITS架构)
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
        # VCTK FastPitch 男声
        "en_vctk_fast_pitch_m001": {"model_name": "tts_models/en/vctk/fast_pitch", "speaker_idx": "p225", "description": "FastPitch (VCTK, 男声, 快速)"},
        "en_vctk_fast_pitch_m002": {"model_name": "tts_models/en/vctk/fast_pitch", "speaker_idx": "p231", "description": "FastPitch (VCTK, 男声, 温和)"},
        "en_vctk_fast_pitch_m003": {"model_name": "tts_models/en/vctk/fast_pitch", "speaker_idx": "p236", "description": "FastPitch (VCTK, 男声, 沉稳)"},
        "en_vctk_fast_pitch_m004": {"model_name": "tts_models/en/vctk/fast_pitch", "speaker_idx": "p239", "description": "FastPitch (VCTK, 男声, 磁性)"},
        "en_vctk_fast_pitch_m005": {"model_name": "tts_models/en/vctk/fast_pitch", "speaker_idx": "p245", "description": "FastPitch (VCTK, 男声, 浑厚)"},
        # VCTK 女声 (VITS架构) - 精选50个不同风格
        "en_vctk_vits_f001": {"model_name": "tts_models/en/vctk/vits", "speaker_idx": "p225", "description": "VITS (VCTK, 女声1, 甜美清晰)"},
        "en_vctk_vits_f002": {"model_name": "tts_models/en/vctk/vits", "speaker_idx": "p226", "description": "VITS (VCTK, 女声2, 温柔优雅)"},
        "en_vctk_vits_f003": {"model_name": "tts_models/en/vctk/vits", "speaker_idx": "p227", "description": "VITS (VCTK, 女声3, 明亮活泼)"},
        "en_vctk_vits_f004": {"model_name": "tts_models/en/vctk/vits", "speaker_idx": "p228", "description": "VITS (VCTK, 女声4, 柔和自然)"},
        "en_vctk_vits_f005": {"model_name": "tts_models/en/vctk/vits", "speaker_idx": "p229", "description": "VITS (VCTK, 女声5, 端庄大方)"},
        "en_vctk_vits_f006": {"model_name": "tts_models/en/vctk/vits", "speaker_idx": "p230", "description": "VITS (VCTK, 女声6, 清新自然)"},
        "en_vctk_vits_f007": {"model_name": "tts_models/en/vctk/vits", "speaker_idx": "p234", "description": "VITS (VCTK, 女声7, 温暖亲切)"},
        "en_vctk_vits_f008": {"model_name": "tts_models/en/vctk/vits", "speaker_idx": "p237", "description": "VITS (VCTK, 女声8, 成熟稳重)"},
        "en_vctk_vits_f009": {"model_name": "tts_models/en/vctk/vits", "speaker_idx": "p240", "description": "VITS (VCTK, 女声9, 清脆悦耳)"},
        "en_vctk_vits_f010": {"model_name": "tts_models/en/vctk/vits", "speaker_idx": "p241", "description": "VITS (VCTK, 女声10, 柔和甜美)"},
        "en_vctk_vits_f011": {"model_name": "tts_models/en/vctk/vits", "speaker_idx": "p243", "description": "VITS (VCTK, 女声11, 明亮自信)"},
        "en_vctk_vits_f012": {"model_name": "tts_models/en/vctk/vits", "speaker_idx": "p244", "description": "VITS (VCTK, 女声12, 清新活泼)"},
        "en_vctk_vits_f013": {"model_name": "tts_models/en/vctk/vits", "speaker_idx": "p246", "description": "VITS (VCTK, 女声13, 温柔细腻)"},
        "en_vctk_vits_f014": {"model_name": "tts_models/en/vctk/vits", "speaker_idx": "p247", "description": "VITS (VCTK, 女声14, 优雅知性)"},
        "en_vctk_vits_f015": {"model_name": "tts_models/en/vctk/vits", "speaker_idx": "p249", "description": "VITS (VCTK, 女声15, 开朗热情)"},
        "en_vctk_vits_f016": {"model_name": "tts_models/en/vctk/vits", "speaker_idx": "p250", "description": "VITS (VCTK, 女声16, 柔和亲切)"},
        # VCTK FastPitch 女声
        "en_vctk_fast_pitch_f001": {"model_name": "tts_models/en/vctk/fast_pitch", "speaker_idx": "p225", "description": "FastPitch (VCTK, 女声1, 快速清晰)"},
        "en_vctk_fast_pitch_f002": {"model_name": "tts_models/en/vctk/fast_pitch", "speaker_idx": "p234", "description": "FastPitch (VCTK, 女声2, 快速温暖)"},
        "en_vctk_fast_pitch_f003": {"model_name": "tts_models/en/vctk/fast_pitch", "speaker_idx": "p254", "description": "FastPitch (VCTK, 女声3, 快速活力)"},
        "en_vctk_fast_pitch_f004": {"model_name": "tts_models/en/vctk/fast_pitch", "speaker_idx": "p257", "description": "FastPitch (VCTK, 女声4, 快速甜美)"},
        "en_vctk_fast_pitch_f005": {"model_name": "tts_models/en/vctk/fast_pitch", "speaker_idx": "p280", "description": "FastPitch (VCTK, 女声5, 快速成熟)"},
        # ==================== 印尼语 (Indonesian) ====================
        # 印尼语男声
        "id_male_001": {"model_name": "tts_models/id/common-voice/vits", "speaker_idx": "id_male_1", "description": "VITS (印尼语, 男声1, 标准)"},
        "id_male_002": {"model_name": "tts_models/id/common-voice/vits", "speaker_idx": "id_male_2", "description": "VITS (印尼语, 男声2, 深沉)"},
        "id_male_003": {"model_name": "tts_models/id/common-voice/vits", "speaker_idx": "id_male_3", "description": "VITS (印尼语, 男声3, 年轻)"},
        "id_male_004": {"model_name": "tts_models/id/css10/vits", "description": "VITS (印尼语, 男声4, 清晰)"},
        "id_male_005": {"model_name": "tts_models/id/mai/vits", "description": "VITS (印尼语, 男声5, 新闻)"},
        # 印尼语女声
        "id_female_001": {"model_name": "tts_models/id/common-voice/vits", "speaker_idx": "id_female_1", "description": "VITS (印尼语, 女声1, 甜美)"},
        "id_female_002": {"model_name": "tts_models/id/common-voice/vits", "speaker_idx": "id_female_2", "description": "VITS (印尼语, 女声2, 柔和)"},
        "id_female_003": {"model_name": "tts_models/id/common-voice/vits", "speaker_idx": "id_female_3", "description": "VITS (印尼语, 女声3, 清晰)"},
        "id_female_004": {"model_name": "tts_models/id/css10/vits", "description": "VITS (印尼语, 女声4, 标准)"},
        "id_female_005": {"model_name": "tts_models/id/mai/vits", "description": "VITS (印尼语, 女声5, 优雅)"},
        # ==================== 韩语 (Korean) ====================
        # 韩语男声
        "ko_male_001": {"model_name": "tts_models/ko/css10/vits", "speaker_idx": "ko_male_1", "description": "VITS (韩语, 男声1, 标准)"},
        "ko_male_002": {"model_name": "tts_models/ko/css10/vits", "speaker_idx": "ko_male_2", "description": "VITS (韩语, 男声2, 深沉)"},
        "ko_male_003": {"model_name": "tts_models/ko/korean_multi_speaker/vits", "speaker_idx": "ko_male_3", "description": "VITS (韩语, 男声3, 新闻)"},
        "ko_male_004": {"model_name": "tts_models/ko/korean_multi_speaker/vits", "speaker_idx": "ko_male_4", "description": "VITS (韩语, 男声4, 年轻)"},
        "ko_male_005": {"model_name": "tts_models/ko/korean_multi_speaker/vits", "speaker_idx": "ko_male_5", "description": "VITS (韩语, 男声5, 温和)"},
        # 韩语女声
        "ko_female_001": {"model_name": "tts_models/ko/css10/vits", "speaker_idx": "ko_female_1", "description": "VITS (韩语, 女声1, 甜美)"},
        "ko_female_002": {"model_name": "tts_models/ko/css10/vits", "speaker_idx": "ko_female_2", "description": "VITS (韩语, 女声2, 清晰)"},
        "ko_female_003": {"model_name": "tts_models/ko/korean_multi_speaker/vits", "speaker_idx": "ko_female_3", "description": "VITS (韩语, 女声3, 标准)"},
        "ko_female_004": {"model_name": "tts_models/ko/korean_multi_speaker/vits", "speaker_idx": "ko_female_4", "description": "VITS (韩语, 女声4, 柔和)"},
        "ko_female_005": {"model_name": "tts_models/ko/korean_multi_speaker/vits", "speaker_idx": "ko_female_5", "description": "VITS (韩语, 女声5, 优雅)"},
        "ko_female_006": {"model_name": "tts_models/ko/kss/vits", "description": "VITS (韩语, 女声6, 专业)"},
        # ==================== 日语 (Japanese) ====================
        # 日语男声
        "ja_male_001": {"model_name": "tts_models/ja/kokoro/vits", "speaker_idx": "ja_male_1", "description": "VITS (日语, 男声1, 标准)"},
        "ja_male_002": {"model_name": "tts_models/ja/kokoro/vits", "speaker_idx": "ja_male_2", "description": "VITS (日语, 男声2, 深沉)"},
        "ja_male_003": {"model_name": "tts_models/ja/css10/vits", "speaker_idx": "ja_male_3", "description": "VITS (日语, 男声3, 清晰)"},
        "ja_male_004": {"model_name": "tts_models/ja/css10/vits", "speaker_idx": "ja_male_4", "description": "VITS (日语, 男声4, 年轻)"},
        "ja_male_005": {"model_name": "tts_models/ja/kokoro/tacotron2-DDC", "description": "Tacotron2 (日语, 男声5, 传统)"},
        # 日语女声
        "ja_female_001": {"model_name": "tts_models/ja/kokoro/vits", "speaker_idx": "ja_female_1", "description": "VITS (日语, 女声1, 甜美)"},
        "ja_female_002": {"model_name": "tts_models/ja/kokoro/vits", "speaker_idx": "ja_female_2", "description": "VITS (日语, 女声2, 温柔)"},
        "ja_female_003": {"model_name": "tts_models/ja/css10/vits", "speaker_idx": "ja_female_3", "description": "VITS (日语, 女声3, 标准)"},
        "ja_female_004": {"model_name": "tts_models/ja/css10/vits", "speaker_idx": "ja_female_4", "description": "VITS (日语, 女声4, 清晰)"},
        "ja_female_005": {"model_name": "tts_models/ja/kokoro/tacotron2-DDC", "description": "Tacotron2 (日语, 女声5, 传统)"},
        "ja_female_006": {"model_name": "tts_models/ja/kss/vits", "description": "VITS (日语, 女声6, 专业)"},
        # ==================== 越南语 (Vietnamese) ====================
        # 越南语男声
        "vi_male_001": {"model_name": "tts_models/vi/common-voice/vits", "speaker_idx": "vi_male_1", "description": "VITS (越南语, 男声1, 标准)"},
        "vi_male_002": {"model_name": "tts_models/vi/common-voice/vits", "speaker_idx": "vi_male_2", "description": "VITS (越南语, 男声2, 深沉)"},
        "vi_male_003": {"model_name": "tts_models/vi/css10/vits", "description": "VITS (越南语, 男声3, 清晰)"},
        "vi_male_004": {"model_name": "tts_models/vi/vivos/vits", "description": "VITS (越南语, 男声4, 新闻)"},
        "vi_male_005": {"model_name": "tts_models/vi/mai/vits", "description": "VITS (越南语, 男声5, 正式)"},
        # 越南语女声
        "vi_female_001": {"model_name": "tts_models/vi/common-voice/vits", "speaker_idx": "vi_female_1", "description": "VITS (越南语, 女声1, 甜美)"},
        "vi_female_002": {"model_name": "tts_models/vi/common-voice/vits", "speaker_idx": "vi_female_2", "description": "VITS (越南语, 女声2, 柔和)"},
        "vi_female_003": {"model_name": "tts_models/vi/css10/vits", "description": "VITS (越南语, 女声3, 清晰)"},
        "vi_female_004": {"model_name": "tts_models/vi/vivos/vits", "description": "VITS (越南语, 女声4, 标准)"},
        "vi_female_005": {"model_name": "tts_models/vi/mai/vits", "description": "VITS (越南语, 女声5, 优雅)"},
        # ==================== 西班牙语 (Spanish) ====================
        # 西班牙语男声
        "es_male_001": {"model_name": "tts_models/es/css10/vits", "speaker_idx": "es_male_1", "description": "VITS (西班牙语, 男声1, 标准)"},
        "es_male_002": {"model_name": "tts_models/es/css10/vits", "speaker_idx": "es_male_2", "description": "VITS (西班牙语, 男声2, 深沉)"},
        "es_male_003": {"model_name": "tts_models/es/common-voice/vits", "speaker_idx": "es_male_3", "description": "VITS (西班牙语, 男声3, 清晰)"},
        "es_male_004": {"model_name": "tts_models/es/mai/vits", "description": "VITS (西班牙语, 男声4, 正式)"},
        "es_male_005": {"model_name": "tts_models/es/m-ailabs/vits", "description": "VITS (西班牙语, 男声5, 新闻)"},
        # 西班牙语女声
        "es_female_001": {"model_name": "tts_models/es/css10/vits", "speaker_idx": "es_female_1", "description": "VITS (西班牙语, 女声1, 热情)"},
        "es_female_002": {"model_name": "tts_models/es/css10/vits", "speaker_idx": "es_female_2", "description": "VITS (西班牙语, 女声2, 甜美)"},
        "es_female_003": {"model_name": "tts_models/es/common-voice/vits", "speaker_idx": "es_female_3", "description": "VITS (西班牙语, 女声3, 标准)"},
        "es_female_004": {"model_name": "tts_models/es/mai/vits", "description": "VITS (西班牙语, 女声4, 优雅)"},
        "es_female_005": {"model_name": "tts_models/es/m-ailabs/vits", "description": "VITS (西班牙语, 女声5, 专业)"},
        # ==================== 土耳其语 (Turkish) ====================
        # 土耳其语男声
        "tr_male_001": {"model_name": "tts_models/tr/common-voice/vits", "speaker_idx": "tr_male_1", "description": "VITS (土耳其语, 男声1, 标准)"},
        "tr_male_002": {"model_name": "tts_models/tr/common-voice/vits", "speaker_idx": "tr_male_2", "description": "VITS (土耳其语, 男声2, 深沉)"},
        "tr_male_003": {"model_name": "tts_models/tr/css10/vits", "description": "VITS (土耳其语, 男声3, 清晰)"},
        "tr_male_004": {"model_name": "tts_models/tr/mai/vits", "description": "VITS (土耳其语, 男声4, 正式)"},
        "tr_male_005": {"model_name": "tts_models/tr/m-ailabs/vits", "description": "VITS (土耳其语, 男声5, 新闻)"},
        # 土耳其语女声
        "tr_female_001": {"model_name": "tts_models/tr/common-voice/vits", "speaker_idx": "tr_female_1", "description": "VITS (土耳其语, 女声1, 甜美)"},
        "tr_female_002": {"model_name": "tts_models/tr/common-voice/vits", "speaker_idx": "tr_female_2", "description": "VITS (土耳其语, 女声2, 柔和)"},
        "tr_female_003": {"model_name": "tts_models/tr/css10/vits", "description": "VITS (土耳其语, 女声3, 标准)"},
        "tr_female_004": {"model_name": "tts_models/tr/mai/vits", "description": "VITS (土耳其语, 女声4, 优雅)"},
        "tr_female_005": {"model_name": "tts_models/tr/m-ailabs/vits", "description": "VITS (土耳其语, 女声5, 专业)"},
        # ==================== 葡萄牙语 (Portuguese) ====================
        # 葡萄牙语男声
        "pt_male_001": {"model_name": "tts_models/pt/css10/vits", "speaker_idx": "pt_male_1", "description": "VITS (葡萄牙语, 男声1, 标准)"},
        "pt_male_002": {"model_name": "tts_models/pt/css10/vits", "speaker_idx": "pt_male_2", "description": "VITS (葡萄牙语, 男声2, 深沉)"},
        "pt_male_003": {"model_name": "tts_models/pt/common-voice/vits", "speaker_idx": "pt_male_3", "description": "VITS (葡萄牙语, 男声3, 清晰)"},
        "pt_male_004": {"model_name": "tts_models/pt/mai/vits", "description": "VITS (葡萄牙语, 男声4, 正式)"},
        "pt_male_005": {"model_name": "tts_models/pt/m-ailabs/vits", "description": "VITS (葡萄牙语, 男声5, 新闻)"},
        # 葡萄牙语女声
        "pt_female_001": {"model_name": "tts_models/pt/css10/vits", "speaker_idx": "pt_female_1", "description": "VITS (葡萄牙语, 女声1, 甜美)"},
        "pt_female_002": {"model_name": "tts_models/pt/css10/vits", "speaker_idx": "pt_female_2", "description": "VITS (葡萄牙语, 女声2, 柔和)"},
        "pt_female_003": {"model_name": "tts_models/pt/common-voice/vits", "speaker_idx": "pt_female_3", "description": "VITS (葡萄牙语, 女声3, 标准)"},
        "pt_female_004": {"model_name": "tts_models/pt/mai/vits", "description": "VITS (葡萄牙语, 女声4, 优雅)"},
        "pt_female_005": {"model_name": "tts_models/pt/m-ailabs/vits", "description": "VITS (葡萄牙语, 女声5, 专业)"},
        # ==================== 印地语 (Hindi) ====================
        # 印地语男声
        "hi_male_001": {"model_name": "tts_models/hi/common-voice/vits", "speaker_idx": "hi_male_1", "description": "VITS (印地语, 男声1, 标准)"},
        "hi_male_002": {"model_name": "tts_models/hi/common-voice/vits", "speaker_idx": "hi_male_2", "description": "VITS (印地语, 男声2, 深沉)"},
        "hi_male_003": {"model_name": "tts_models/hi/css10/vits", "description": "VITS (印地语, 男声3, 清晰)"},
        "hi_male_004": {"model_name": "tts_models/hi/mai/vits", "description": "VITS (印地语, 男声4, 正式)"},
        "hi_male_005": {"model_name": "tts_models/hi/indic-tts/vits", "description": "VITS (印地语, 男声5, 新闻)"},
        # 印地语女声
        "hi_female_001": {"model_name": "tts_models/hi/common-voice/vits", "speaker_idx": "hi_female_1", "description": "VITS (印地语, 女声1, 甜美)"},
        "hi_female_002": {"model_name": "tts_models/hi/common-voice/vits", "speaker_idx": "hi_female_2", "description": "VITS (印地语, 女声2, 柔和)"},
        "hi_female_003": {"model_name": "tts_models/hi/css10/vits", "description": "VITS (印地语, 女声3, 标准)"},
        "hi_female_004": {"model_name": "tts_models/hi/mai/vits", "description": "VITS (印地语, 女声4, 优雅)"},
        "hi_female_005": {"model_name": "tts_models/hi/indic-tts/vits", "description": "VITS (印地语, 女声5, 专业)"},
        # ==================== 阿拉伯语 (Arabic) ====================
        # 阿拉伯语男声
        "ar_male_001": {"model_name": "tts_models/ar/common-voice/vits", "speaker_idx": "ar_male_1", "description": "VITS (阿拉伯语, 男声1, 标准)"},
        "ar_male_002": {"model_name": "tts_models/ar/common-voice/vits", "speaker_idx": "ar_male_2", "description": "VITS (阿拉伯语, 男声2, 深沉)"},
        "ar_male_003": {"model_name": "tts_models/ar/css10/vits", "description": "VITS (阿拉伯语, 男声3, 清晰)"},
        "ar_male_004": {"model_name": "tts_models/ar/mai/vits", "description": "VITS (阿拉伯语, 男声4, 正式)"},
        "ar_male_005": {"model_name": "tts_models/ar/m-ailabs/vits", "description": "VITS (阿拉伯语, 男声5, 新闻)"},
        # 阿拉伯语女声
        "ar_female_001": {"model_name": "tts_models/ar/common-voice/vits", "speaker_idx": "ar_female_1", "description": "VITS (阿拉伯语, 女声1, 甜美)"},
        "ar_female_002": {"model_name": "tts_models/ar/common-voice/vits", "speaker_idx": "ar_female_2", "description": "VITS (阿拉伯语, 女声2, 柔和)"},
        "ar_female_003": {"model_name": "tts_models/ar/css10/vits", "description": "VITS (阿拉伯语, 女声3, 标准)"},
        "ar_female_004": {"model_name": "tts_models/ar/mai/vits", "description": "VITS (阿拉伯语, 女声4, 优雅)"},
        "ar_female_005": {"model_name": "tts_models/ar/m-ailabs/vits", "description": "VITS (阿拉伯语, 女声5, 专业)"},
        # ==================== 泰语 (Thai) ====================
        # 泰语男声
        "th_male_001": {"model_name": "tts_models/th/common-voice/vits", "speaker_idx": "th_male_1", "description": "VITS (泰语, 男声1, 标准)"},
        "th_male_002": {"model_name": "tts_models/th/common-voice/vits", "speaker_idx": "th_male_2", "description": "VITS (泰语, 男声2, 深沉)"},
        "th_male_003": {"model_name": "tts_models/th/css10/vits", "description": "VITS (泰语, 男声3, 清晰)"},
        "th_male_004": {"model_name": "tts_models/th/mai/vits", "description": "VITS (泰语, 男声4, 正式)"},
        "th_male_005": {"model_name": "tts_models/th/th-tts/vits", "description": "VITS (泰语, 男声5, 新闻)"},
        # 泰语女声
        "th_female_001": {"model_name": "tts_models/th/common-voice/vits", "speaker_idx": "th_female_1", "description": "VITS (泰语, 女声1, 甜美)"},
        "th_female_002": {"model_name": "tts_models/th/common-voice/vits", "speaker_idx": "th_female_2", "description": "VITS (泰语, 女声2, 柔和)"},
        "th_female_003": {"model_name": "tts_models/th/css10/vits", "description": "VITS (泰语, 女声3, 标准)"},
        "th_female_004": {"model_name": "tts_models/th/mai/vits", "description": "VITS (泰语, 女声4, 优雅)"},
        "th_female_005": {"model_name": "tts_models/th/th-tts/vits", "description": "VITS (泰语, 女声5, 专业)"},
        # ==================== 法语 (French) ====================
        # 法语男声
        "fr_male_001": {"model_name": "tts_models/fr/css10/vits", "speaker_idx": "fr_male_1", "description": "VITS (法语, 男声1, 标准)"},
        "fr_male_002": {"model_name": "tts_models/fr/css10/vits", "speaker_idx": "fr_male_2", "description": "VITS (法语, 男声2, 深沉)"},
        "fr_male_003": {"model_name": "tts_models/fr/common-voice/vits", "speaker_idx": "fr_male_3", "description": "VITS (法语, 男声3, 清晰)"},
        "fr_male_004": {"model_name": "tts_models/fr/mai/vits", "description": "VITS (法语, 男声4, 正式)"},
        "fr_male_005": {"model_name": "tts_models/fr/m-ailabs/vits", "description": "VITS (法语, 男声5, 新闻)"},
        # 法语女声
        "fr_female_001": {"model_name": "tts_models/fr/css10/vits", "speaker_idx": "fr_female_1", "description": "VITS (法语, 女声1, 优雅)"},
        "fr_female_002": {"model_name": "tts_models/fr/css10/vits", "speaker_idx": "fr_female_2", "description": "VITS (法语, 女声2, 甜美)"},
        "fr_female_003": {"model_name": "tts_models/fr/common-voice/vits", "speaker_idx": "fr_female_3", "description": "VITS (法语, 女声3, 标准)"},
        "fr_female_004": {"model_name": "tts_models/fr/mai/vits", "description": "VITS (法语, 女声4, 清晰)"},
        "fr_female_005": {"model_name": "tts_models/fr/m-ailabs/vits", "description": "VITS (法语, 女声5, 专业)"},
        # ==================== 德语 (German) ====================
        # 德语男声
        "de_male_001": {"model_name": "tts_models/de/thorsten/vits", "description": "VITS (德语, Thorsten, 男声1)"},
        "de_male_002": {"model_name": "tts_models/de/css10/vits", "speaker_idx": "de_male_1", "description": "VITS (德语, 男声2, 标准)"},
        "de_male_003": {"model_name": "tts_models/de/css10/vits", "speaker_idx": "de_male_2", "description": "VITS (德语, 男声3, 深沉)"},
        "de_male_004": {"model_name": "tts_models/de/common-voice/vits", "speaker_idx": "de_male_3", "description": "VITS (德语, 男声4, 清晰)"},
        "de_male_005": {"model_name": "tts_models/de/mai/vits", "description": "VITS (德语, 男声5, 正式)"},
        # 德语女声
        "de_female_001": {"model_name": "tts_models/de/thorsten/vits", "description": "VITS (德语, Thorsten, 女声1)"},
        "de_female_002": {"model_name": "tts_models/de/css10/vits", "speaker_idx": "de_female_1", "description": "VITS (德语, 女声2, 标准)"},
        "de_female_003": {"model_name": "tts_models/de/css10/vits", "speaker_idx": "de_female_2", "description": "VITS (德语, 女声3, 清晰)"},
        "de_female_004": {"model_name": "tts_models/de/common-voice/vits", "speaker_idx": "de_female_3", "description": "VITS (德语, 女声4, 柔和)"},
        "de_female_005": {"model_name": "tts_models/de/mai/vits", "description": "VITS (德语, 女声5, 优雅)"},
        # ==================== 意大利语 (Italian) ====================
        # 意大利语男声
        "it_male_001": {"model_name": "tts_models/it/css10/vits", "speaker_idx": "it_male_1", "description": "VITS (意大利语, 男声1, 标准)"},
        "it_male_002": {"model_name": "tts_models/it/css10/vits", "speaker_idx": "it_male_2", "description": "VITS (意大利语, 男声2, 深沉)"},
        "it_male_003": {"model_name": "tts_models/it/common-voice/vits", "speaker_idx": "it_male_3", "description": "VITS (意大利语, 男声3, 清晰)"},
        "it_male_004": {"model_name": "tts_models/it/mai/vits", "description": "VITS (意大利语, 男声4, 正式)"},
        "it_male_005": {"model_name": "tts_models/it/m-ailabs/vits", "description": "VITS (意大利语, 男声5, 新闻)"},
        # 意大利语女声
        "it_female_001": {"model_name": "tts_models/it/css10/vits", "speaker_idx": "it_female_1", "description": "VITS (意大利语, 女声1, 甜美)"},
        "it_female_002": {"model_name": "tts_models/it/css10/vits", "speaker_idx": "it_female_2", "description": "VITS (意大利语, 女声2, 优雅)"},
        "it_female_003": {"model_name": "tts_models/it/common-voice/vits", "speaker_idx": "it_female_3", "description": "VITS (意大利语, 女声3, 标准)"},
    }
    return voices

def translate_text(text, target_lang):
    """翻译文本 - 增强版（自动处理繁体中文）"""
    try:
        import re
        # 检测是否包含中文字符（繁体/简体）
        if re.search(r'[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]', text):
            try:
                from zhconv import convert
                # 繁体转简体（双重保险：先转简体再翻译）
                simplified = convert(text, 'zh-cn')
                if simplified != text:
                    print(f"    - 繁体转简体: '{text}' -> '{simplified}'")
                    text = simplified
            except ImportError:
                print("    - 未安装 zhconv，繁体中文可能翻译不准确（建议: pip install zhconv）")
        
        # Google Translator 兼容性处理
        if target_lang.startswith('zh'):
            target_lang = 'zh-CN'
        
        translator = GoogleTranslator(source='auto', target=target_lang)
        result = translator.translate(text)
        
        # 验证翻译结果（避免返回原文）
        if result.strip() == text.strip() and not re.match(r'^[\s\W]+$', text):
            print(f"    - 警告: 翻译结果与原文相同，可能识别失败")
            # 尝试强制指定源语言为中文
            try:
                translator_zh = GoogleTranslator(source='zh-CN', target=target_lang)
                result2 = translator_zh.translate(text)
                if result2.strip() != text.strip():
                    print(f"    - 回退方案成功: '{text}' -> '{result2}'")
                    return result2
            except:
                pass
        
        return result
    except Exception as e:
        print(f"  - 翻译失败: {e}")
        return text  # 回退到原文


def translate_segments_parallel(segments, target_lang, max_workers=10, show_progress=True):
    """并行翻译所有片段

    Args:
        segments: 片段列表，每个片段包含 text, start, end, duration/original_duration
        target_lang: 目标语言代码
        max_workers: 最大并行线程数
        show_progress: 是否显示进度

    Returns:
        翻译后的片段列表，保持原始顺序
    """
    if not segments:
        return []

    total = len(segments)
    completed = [0]  # 使用列表以便在闭包中修改

    def translate_one(seg, idx):
        """翻译单个片段"""
        translated = translate_text(seg["text"], target_lang)
        completed[0] += 1
        if show_progress and completed[0] % 10 == 0:
            print(f"  - 翻译进度: {completed[0]}/{total}")
        return {
            "idx": idx,
            "text": seg["text"],
            "translated": translated,
            "start": seg["start"],
            "end": seg["end"],
            "duration": seg.get("duration") or seg.get("original_duration")
        }

    print(f"  - 开始并行翻译 {total} 个片段 (线程数: {max_workers})...")
    start_time = time.time()

    results = [None] * total

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        futures = {executor.submit(translate_one, seg, i): i
                   for i, seg in enumerate(segments)}

        # 收集结果
        for future in as_completed(futures):
            try:
                result = future.result()
                results[result["idx"]] = result
            except Exception as e:
                idx = futures[future]
                print(f"  - 片段 {idx} 翻译失败: {e}")
                # 使用原文作为回退
                results[idx] = {
                    "idx": idx,
                    "text": segments[idx]["text"],
                    "translated": segments[idx]["text"],
                    "start": segments[idx]["start"],
                    "end": segments[idx]["end"],
                    "duration": segments[idx].get("duration") or segments[idx].get("original_duration")
                }

    elapsed = time.time() - start_time
    print(f"  - 翻译完成: {total} 个片段, 耗时 {elapsed:.1f}s (平均 {elapsed/total:.2f}s/片段)")

    return results


def generate_tts_parallel(segments_data, tts_model, speaker_idx, target_lang,
                          max_workers=3, max_speed_factor=2.0, min_speed_factor=0.5):
    """并行生成 TTS 音频

    由于 TTS 是 GPU 密集型操作，建议使用较小的线程数 (2-3)
    如果显存不足，会自动降级为顺序处理

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

    print(f"  - 开始并行生成 {total} 个 TTS 音频 (线程数: {max_workers})...")
    start_time = time.time()
    completed = [0]

    def generate_one(seg_data, idx):
        """生成单个 TTS 音频"""
        try:
            # 创建临时文件
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
                "success": True
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

    # 尝试并行处理，如果显存不足则降级为顺序处理
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(generate_one, seg, i): i
                       for i, seg in enumerate(segments_data)}

            for future in as_completed(futures):
                try:
                    result = future.result()
                    results[result["idx"]] = result
                    if result["temp_tts_file"]:
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
            if results[i]["temp_tts_file"]:
                temp_files_to_cleanup.append(results[i]["temp_tts_file"])

    return results, temp_files_to_cleanup


def load_coqui_tts_model(voice_config, gpu_is_available=False):
    """加载Coqui TTS模型"""
    model_name = voice_config.get("model_name", "tts_models/en/ljspeech/vits")
    speaker_idx = voice_config.get("speaker_idx", None)
    print(f"  - 加载TTS模型: {model_name}")
    try:
        tts = TTS(model_name=model_name, progress_bar=True, gpu=gpu_is_available)
        return tts, speaker_idx
    except Exception as e:
        print(f"  - 模型加载失败: {e}")
        fallback_model = "tts_models/en/ljspeech/vits"
        tts = TTS(model_name=fallback_model, progress_bar=True, gpu=gpu_is_available)
        return tts, None

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
    
    print(f"    - TTS输入: '{text}' (原: '{original_text}')")
    
    # ===== 2. 生成音频（带重试机制）=====
    max_retries = 2
    for attempt in range(max_retries + 1):
        try:
            # 选择正确的调用方式
            if speaker_idx and hasattr(tts_instance.synthesizer.tts_model, 'speaker_manager'):
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
        estimated_height = 150
    elif video_height >= 720:
        font_size = int(base_font_size * 1.2)
        estimated_height = 130
    else:
        font_size = int(base_font_size * 1.0)
        estimated_height = 110
    img_width = video_width
    # ========== 2. 创建透明背景 ==========
    subtitle_img = Image.new('RGBA', (img_width, estimated_height), color=(0, 0, 0, 0))
    draw = ImageDraw.Draw(subtitle_img)
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
                print(f"    - 尝试加载覆盖字体: {os.path.basename(FONT_PATH_OVERRIDE)}")
                # 对于 .ttc/.ttf 都尝试直接加载
                font = ImageFont.truetype(FONT_PATH_OVERRIDE, size=size)
                print(f"    - 成功加载覆盖字体: {FONT_PATH_OVERRIDE}")
                return font
            except Exception as e:
                print(f"    - 覆盖字体加载失败: {e}")
        
        for font_path in mac_font_paths:
            if os.path.exists(font_path):
                try:
                    print(f"    - 尝试加载: {os.path.basename(font_path)}")
                    if font_path.endswith('.ttc'):
                        # 对于字体集合，尝试不同索引
                        for index in [0, 1, 2]:
                            try:
                                font = ImageFont.truetype(font_path, size=size, index=index)
                                # 测试字体
                                test_text = "Test"
                                bbox = font.getbbox(test_text)
                                if bbox:
                                    print(f"    - 成功加载: {os.path.basename(font_path)} (索引:{index})")
                                    return font
                            except:
                                continue
                    else:
                        font = ImageFont.truetype(font_path, size=size)
                        # 测试字体
                        test_text = "Test"
                        bbox = font.getbbox(test_text)
                        if bbox:
                            print(f"    - 成功加载: {os.path.basename(font_path)}")
                            return font
                except Exception as e:
                    print(f"    - 加载失败 {os.path.basename(font_path)}: {e}")
                    continue
        # 如果所有字体都失败，返回默认字体
        print("    - 使用PIL默认字体")
        return ImageFont.load_default()

    def get_font_for_windows(size):
        """Windows专用字体加载"""
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
                            print(f"    - 尝试加载: {font_name}")
                            if font_name.endswith('.ttc'):
                                font = ImageFont.truetype(font_path, size=size, index=0)
                            else:
                                font = ImageFont.truetype(font_path, size=size)
                            # 测试字体
                            test_text = "Test"
                            bbox = font.getbbox(test_text)
                            if bbox:
                                print(f"    - 成功加载: {font_name}")
                                return font
                        except Exception as e:
                            print(f"    - 加载失败 {font_name}: {e}")
                            continue
        # 如果用户提供了覆盖字体路径，最后再尝试一次（有些系统字体目录不可读）
        if FONT_PATH_OVERRIDE and os.path.exists(FONT_PATH_OVERRIDE):
            try:
                print(f"    - 尝试加载覆盖字体: {os.path.basename(FONT_PATH_OVERRIDE)}")
                font = ImageFont.truetype(FONT_PATH_OVERRIDE, size=size)
                print(f"    - 成功加载覆盖字体: {FONT_PATH_OVERRIDE}")
                return font
            except Exception as e:
                print(f"    - 覆盖字体加载失败: {e}")
        print("    - 使用PIL默认字体")
        return ImageFont.load_default()

    # 根据系统选择字体加载函数
    import sys
    if sys.platform == 'darwin':
        print("    - 系统: macOS")
        font = get_font_for_mac(font_size)
    elif sys.platform.startswith('win'):
        print("    - 系统: Windows")
        font = get_font_for_windows(font_size)
    else:
        print("    - 系统: Linux/其他")
        font = ImageFont.load_default()

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
    max_chars = max(10, int(img_width * 0.7 / avg_char_width))

    # ========== 6. 文本换行 ==========
    def simple_wrap(text, max_chars):
        """简单的文本换行"""
        if not text:
            return []
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

    lines = simple_wrap(text, max_chars)

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

        # 首先绘制文字描边（增强可读性）
        outline_color = (0, 0, 0, 200)
        # 简化描边 - 只绘制主要方向的阴影
        offsets = [(2, 2), (2, -2), (-2, 2), (-2, -2)]
        for dx, dy in offsets:
            try:
                draw.text((text_x + dx, current_y + dy), line,
                          fill=outline_color, font=font)
            except Exception as e:
                print(f"    - 描边绘制失败: {e}")
                # 继续尝试

        # 然后绘制主文字
        try:
            draw.text((text_x, current_y), line,
                      fill=text_color, font=font)
            print(f"    - 成功绘制: {line[:20]}...")
        except Exception as e:
            print(f"    - 主文字绘制失败: {e}")
            # 尝试使用ASCII回退
            try:
                ascii_line = line.encode('ascii', 'ignore').decode()
                if ascii_line:
                    draw.text((text_x, current_y), ascii_line,
                              fill=text_color, font=font)
                    print(f"    - 使用ASCII回退: {ascii_line[:20]}...")
            except:
                print(f"    - ASCII回退也失败")

        # 更新Y位置
        current_y += line_heights[i] + 8

    # ========== 9. 保存和返回 ==========
    temp_img_fd, temp_img_path = tempfile.mkstemp(suffix='.png')
    os.close(temp_img_fd)
    try:
        subtitle_img.save(temp_img_path, format='PNG', optimize=True)
        print(f"    - 字幕图片保存: {temp_img_path}")
    except Exception as e:
        print(f"    - 图片保存失败: {e}")
        # 创建简单的错误图片
        error_img = Image.new('RGBA', (100, 50), color=(255, 0, 0, 128))
        error_img.save(temp_img_path, format='PNG')

    from moviepy.video.VideoClip import ImageClip
    try:
        clip = ImageClip(temp_img_path, duration=duration).set_start(start_time).set_position(('center', 'bottom'))
        clip.temp_path = temp_img_path
        print(f"    - 字幕片段创建成功: {duration:.2f}s")
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
    final_audio_track = None
    video_with_new_audio = None
    final_video_with_subtitles = None
    final_audio_clips_for_composition = []
    subtitle_clips = []
    temp_files_to_cleanup = []
    
    try:
        # 加载原视频
        original_video = VideoFileClip(input_video_path)
        video_duration = original_video.duration
        video_width = original_video.w
        video_height = original_video.h
        
        print(f"视频时长: {video_duration:.2f}s")
        
        # Step 1: 语音识别
        print("\n[1/6] 语音识别...")
        model_size = "small"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "float32"
        print(f"  - 使用设备: {device}, 计算类型: {compute_type}")
        
        try:
            model = WhisperModel(model_size, device=device, compute_type=compute_type)
            segments, info = model.transcribe(input_video_path, language="zh",
                                              task="transcribe", beam_size=10,
                                              best_of=5, patience=1.0)
            
            original_segments_data = []
            for segment in segments:
                if segment.start >= video_duration:
                    continue
                actual_end = min(segment.end, video_duration)
                duration = actual_end - segment.start
                if duration < 0.3:
                    continue
                original_segments_data.append({
                    "text": segment.text.strip(),
                    "start": segment.start,
                    "end": actual_end,
                    "original_duration": duration
                })
            print(f"  - 识别到 {len(original_segments_data)} 个有效语音片段")
            
            if not original_segments_data:
                print("错误: 未识别到任何有效语音。请检查视频或尝试更大的Whisper模型。")
                return False
                
        except Exception as e:
            print(f"  - 语音识别失败: {e}")
            return False
        
        # 释放 ASR 模型内存
        del model
        gc.collect()
        
        # Step 3: 翻译
        print(f"\n[3/6] 翻译为 {target_language}...")
        translated_segments_data = []

        if parallel and len(original_segments_data) > 1:
            # 并行翻译
            translated_results = translate_segments_parallel(
                original_segments_data, target_language, max_workers=workers
            )

            for result in translated_results:
                translated_segments_data.append({
                    "original_text": result["text"],
                    "translated_text": result["translated"],
                    "original_duration": result["duration"],
                    "start": result["start"],
                    "end": result["end"]
                })
        else:
            # 顺序翻译（并行禁用或只有一个片段）
            for i, seg in enumerate(original_segments_data):
                translated_text = translate_text(seg["text"], target_language)
                translated_segments_data.append({
                    "original_text": seg["text"],
                    "translated_text": translated_text,
                    "original_duration": seg["original_duration"],
                    "start": seg["start"],
                    "end": seg["end"]
                })
                if i < 5:
                    print(f"  [{i}] {seg['text'][:40]}... -> {translated_text[:40]}...")

        # Step 4: 加载TTS模型
        print("\n[4/6] 加载TTS模型...")
        voice_config = available_voices.get(selected_voice_key,
                                            {"model_name": "tts_models/en/ljspeech/vits"})
        
        try:
            tts_model, tts_speaker_idx = load_coqui_tts_model(voice_config, gpu_is_available=torch.cuda.is_available())
        except Exception as e:
            print(f"  - TTS模型加载失败: {e}")
            return False
        
        # Step 5: 生成并调整TTS音频
        print("\n[5/6] 生成并调整配音音频...")

        # 5.1 并行生成所有 TTS 音频
        tts_results, tts_temp_files = generate_tts_parallel(
            translated_segments_data, tts_model, tts_speaker_idx, target_language,
            max_workers=tts_workers
        )

        # 5.2 顺序处理音频速度调整和片段创建
        for i, tts_result in enumerate(tts_results):
            if not tts_result or not tts_result.get("success"):
                continue

            seg_data = tts_result["seg_data"]
            temp_tts_file = tts_result["temp_tts_file"]

            print(f"\n--- 处理片段 {i+1}/{len(tts_results)} ---")
            print(f"    原时间: [{seg_data['start']:.2f}s -> {seg_data['end']:.2f}s], 时长: {seg_data['original_duration']:.2f}s")

            # 调整音频速度
            tmp_adj = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            temp_adjusted_file = tmp_adj.name
            tmp_adj.close()
            temp_files_to_cleanup.append(temp_adjusted_file)

            try:
                final_duration, speed_factor = adjust_audio_speed_ffmpeg(
                    temp_tts_file,
                    temp_adjusted_file,
                    seg_data['original_duration'],
                    max_speed_factor=max_speed_factor,
                    min_speed_factor=min_speed_factor
                )
                print(f"    调整后时长: {final_duration:.2f}s, 速度因子: {speed_factor:.2f}x")
            except Exception as e:
                print(f"    - 音频速度调整失败: {e}")
                continue

            # 创建音频片段
            try:
                adjusted_clip = AudioFileClip(temp_adjusted_file).set_start(seg_data['start'])
                final_audio_clips_for_composition.append(adjusted_clip)
            except Exception as e:
                print(f"    - 创建音频片段失败: {e}")
                continue

            # 创建字幕片段
            if final_duration > 0.1:
                try:
                    subtitle_clip = create_subtitle_clip(
                        seg_data["translated_text"],
                        seg_data['start'],
                        min(final_duration, seg_data['original_duration'] * 1.5),
                        video_width,
                        video_height,
                        target_language
                    )
                    subtitle_clips.append(subtitle_clip)
                except Exception as e:
                    print(f"    - 创建字幕片段失败: {e}")

            # 清理当前片段的临时 TTS 文件
            if temp_tts_file and os.path.exists(temp_tts_file):
                try:
                    os.remove(temp_tts_file)
                    if temp_tts_file in tts_temp_files:
                        tts_temp_files.remove(temp_tts_file)
                except:
                    pass

        # 清理剩余的临时 TTS 文件
        for temp_file in tts_temp_files:
            if temp_file and os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass
        
        # 释放 TTS 模型内存
        del tts_model
        gc.collect()
        
        # Step 6: 合成最终视频
        print("\n[6/6] 合成最终视频...")
        print("  - 合成音频轨道 (使用绝对时间对齐)...")
        
        try:
            # 6.1 创建静音背景轨道
            def create_silent_audio(duration, fps=44100):
                n_frames = int(duration * fps)
                silent_array = np.zeros((n_frames, 2), dtype=np.float32)
                silent_audio = AudioArrayClip(silent_array, fps=fps)
                silent_audio.duration = duration
                silent_audio.end = silent_audio.start + duration
                return silent_audio
            
            silent_audio = create_silent_audio(video_duration)
            
            # 6.2 组合音频
            all_audio_clips = [silent_audio] + final_audio_clips_for_composition
            final_audio_track = CompositeAudioClip(all_audio_clips)
            
            # 6.3 应用音频到视频
            video_with_new_audio = original_video.set_audio(final_audio_track)
            
            # 6.4 添加字幕
            print(f"  - 添加 {len(subtitle_clips)} 个字幕片段...")
            if subtitle_clips:
                final_video_with_subtitles = CompositeVideoClip(
                    [video_with_new_audio] + subtitle_clips,
                    size=original_video.size
                )
            else:
                final_video_with_subtitles = video_with_new_audio
            
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
                if hasattr(clip, 'temp_path') and os.path.exists(clip.temp_path):
                    os.remove(clip.temp_path)
            except:
                pass
        
        # 删除所有临时音频文件
        for temp_file in temp_files_to_cleanup:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except:
                pass
        
        # 关闭 MoviePy 对象
        for obj in [final_video_with_subtitles, video_with_new_audio, final_audio_track, original_video]:
            if obj:
                try:
                    obj.close()
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
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        
        if result.returncode != 0:
            # 尝试备用命令（某些FFmpeg版本需要）
            cmd_alt = cmd[:-1] + ['-show_format'] + [video_path]
            result = subprocess.run(cmd_alt, capture_output=True, text=True, timeout=10)
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
                os.remove(filelist_path)
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

def batch_process_videos(input_dir, output_dir, target_language, selected_voice_key, max_speed_factor, min_speed_factor, available_voices, video_extension=".mp4", parallel=True, workers=10):
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
    """
    print(f"\n--- 开始批量处理视频 ---")
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"目标语言: {target_language}")
    print(f"选择声音: {selected_voice_key}")
    print(f"并行翻译: {'启用' if parallel else '禁用'} (线程数: {workers})")

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
            workers=workers
        )
        if not success:
            print(f"警告: 处理 '{input_video_path}' 时失败。跳过此文件。")

    print("\n--- 所有视频处理完成 ---")

#单文件处理:python video_dubbing.py --mode single --input_video input.mp4 --target_lang pt --voice pt_male_001 --output_video output_dubbed.mp4
#多文件处理:python video_dubbing.py --mode batch --input_dir ./input --output_dir ./output --target_lang pt --voice pt_male_001
#多文件处理+合并:python video_dubbing.py --mode batch_merge --input_dir ./input --output_dir ./output --target_lang pt --voice pt_male_001 --merged_filename final_movie.mp4
#仅合并:python video_dubbing.py --mode merge_only --output_dir ./output --merged_filename final_movie.mp4
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
    parser.add_argument("--voice", default="en_sam_tacotron", choices=available_voices.keys(),
                        help="选择配音声音")
    parser.add_argument("--max_speed", type=float, default=2.0,
                        help="最大语速加速倍数 (默认: 1.8)")
    parser.add_argument("--min_speed", type=float, default=0.5,
                        help="最小语速减慢倍数 (默认: 0.7)")
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
                        help="并行 TTS 生成线程数 (默认: 3, 推荐 2-3, GPU 显存受限)")

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
        success = process_single_video(
            args.input_video, args.target_lang, args.voice,
            args.output_video, args.max_speed, args.min_speed, available_voices,
            parallel=args.parallel, workers=args.workers
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
            parallel=args.parallel, workers=args.workers
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
            parallel=args.parallel, workers=args.workers
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
