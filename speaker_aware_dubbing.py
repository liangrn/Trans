#!/usr/bin/env python3
"""
speaker_aware_dubbing.py — 说话人感知配音模块（优化版）

性能优化点：
1. 说话人分离 与 video_dubbing 的 ASR+翻译 并行执行（最大收益）
   - 通过 run_diarization_async() 提前在后台线程启动分离
   - ASR 结束后调用 wait_diarization() 获取结果，几乎零等待
2. 性别识别多线程并行（各说话人互不依赖）
3. Embedding 计算多线程并行
4. 逻辑与 test_diarization.py 完全一致，准确率不变
"""

import os
import warnings
import tempfile
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple, Optional

import numpy as np

warnings.filterwarnings("ignore")


# =====================================================================
# 配置 —— 与 test_diarization.py 完全一致
# =====================================================================
class Config:
    SAMPLE_RATE = 16000
    MIN_SEGMENT_DURATION = 0.5

    DIARIZATION_MODEL = "pyannote/speaker-diarization-community-1"
    CLUSTERING_THRESHOLD = 0.45
    MIN_DURATION_OFF = 0.1

    MIN_SPEAKERS = 2
    MAX_SPEAKERS = 8

    MERGE_COSINE_THRESHOLD = 0.82
    GENDER_CONF_THRESHOLD = 0.55

    # 性别识别并行线程数（说话人数量通常 2~8，全部并行即可）
    GENDER_WORKERS = 4
    # Embedding 并行线程数
    EMBEDDING_WORKERS = 4

    HF_TOKEN = os.environ.get("HF_TOKEN", "")


# =====================================================================
# 优化：异步启动说话人分离（与 ASR 并行）
# =====================================================================

_diarization_future = None      # Future 对象
_diarization_executor = None    # ThreadPoolExecutor，保持引用防止GC
_diarization_waveform = None    # 供后续步骤复用的 waveform
_diarization_sr = None
_diarization_audio_path = None  # 临时文件路径，用完后清理


def run_diarization_async(video_path: str, hf_token: str = None):
    """
    在后台线程启动说话人分离，立即返回。
    在 video_dubbing.py 的 ASR 开始前调用，让分离和 ASR 并行跑。

    用法：
        run_diarization_async(video_path, hf_token)   # ASR 开始前调用
        ... ASR + 翻译 ...
        speaker_map = wait_diarization()              # ASR 结束后取结果
    """
    global _diarization_future, _diarization_waveform, _diarization_sr
    global _diarization_audio_path

    token = hf_token or Config.HF_TOKEN
    if not token:
        return

    global _diarization_executor
    _diarization_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="diarize")
    _diarization_future = _diarization_executor.submit(_run_full_pipeline, video_path, token)
    print("[说话人识别] 已在后台启动，与 ASR 并行运行...")


def wait_diarization() -> Dict[str, Dict]:
    """
    等待后台分离完成并返回结果。
    若 run_diarization_async 未调用，直接返回空字典。
    """
    global _diarization_future
    if _diarization_future is None:
        return {}
    try:
        result = _diarization_future.result()  # 阻塞直到完成
        _diarization_future = None
        return result
    except Exception as e:
        print(f"[说话人识别] 后台任务异常: {e}")
        _diarization_future = None
        return {}
    finally:
        global _diarization_executor
        if _diarization_executor is not None:
            _diarization_executor.shutdown(wait=False)
            _diarization_executor = None


# =====================================================================
# 对外接口 1：同步版（不用异步时的简单调用方式）
# =====================================================================
def analyze_speakers_for_video(
    video_path: str,
    hf_token: str = None
) -> Dict[str, Dict]:
    """同步版：直接阻塞等待结果"""
    token = hf_token or Config.HF_TOKEN
    if not token:
        print("[说话人识别] 未设置 HF_TOKEN，跳过")
        return {}
    return _run_full_pipeline(video_path, token)


# =====================================================================
# 对外接口 2：分配声音
# =====================================================================
def build_speaker_voice_map(
    speaker_map: Dict[str, Dict],
    target_lang: str,
    available_voices: Dict[str, Dict],
    fallback_voice_key: str
) -> Dict[str, str]:
    """按说话人性别顺序分配不同声音"""
    if not speaker_map:
        return {}

    lang_prefix = _get_lang_prefix(target_lang)
    male_voices = sorted([
        k for k in available_voices
        if k.startswith(lang_prefix) and ("_male_" in k or "_m0" in k)
    ])
    female_voices = sorted([
        k for k in available_voices
        if k.startswith(lang_prefix) and ("_female_" in k or "_f0" in k)
    ])

    if not male_voices and not female_voices:
        male_voices = sorted([k for k in available_voices if "_vctk_vits_m" in k])
        female_voices = sorted([k for k in available_voices if "_vctk_vits_f" in k])

    sorted_speakers = sorted(
        speaker_map.items(),
        key=lambda x: x[1].get("total_duration", 0),
        reverse=True
    )

    male_idx = 0
    female_idx = 0
    result = {}

    print("[说话人识别] 声音分配:")
    for speaker_id, info in sorted_speakers:
        gender = info.get("gender", "unknown")
        if gender == "unknown":
            gender = "female" if ("female" in fallback_voice_key or "_f0" in fallback_voice_key) else "male"

        if gender == "male":
            voice_key = male_voices[male_idx % len(male_voices)] if male_voices else fallback_voice_key
            male_idx += 1
        else:
            voice_key = female_voices[female_idx % len(female_voices)] if female_voices else fallback_voice_key
            female_idx += 1

        result[speaker_id] = voice_key
        gz = "男" if gender == "male" else "女"
        print(f"  {speaker_id} ({gz}) → {voice_key}")

    return result


# =====================================================================
# 对外接口 3：按时间点查声音
# =====================================================================
def get_voice_for_segment(
    seg_start: float,
    seg_end: float,
    speaker_map: Dict[str, Dict],
    speaker_voice_map: Dict[str, str],
    fallback_voice_key: str
) -> str:
    """找与该 ASR 片段时间重叠最多的说话人，返回其 voice_key"""
    if not speaker_map or not speaker_voice_map:
        return fallback_voice_key

    best_speaker = None
    best_overlap = 0.0

    for speaker_id, info in speaker_map.items():
        for diar_start, diar_end in info.get("segments", []):
            overlap = max(0.0, min(seg_end, diar_end) - max(seg_start, diar_start))
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = speaker_id

    if best_speaker is None or best_overlap < 0.15:
        return fallback_voice_key

    return speaker_voice_map.get(best_speaker, fallback_voice_key)


# =====================================================================
# 内部流程
# =====================================================================

def _run_full_pipeline(video_path: str, token: str) -> Dict[str, Dict]:
    """完整的分析流程（提取音频→分离→性别→合并）"""
    audio_path = None
    try:
        t0 = time.time()

        # 1. 提取音频
        print("[说话人识别] 提取音频...")
        audio_path = _extract_audio_temp(video_path)
        if not audio_path:
            return {}

        # 2. 加载音频（只加载一次）
        import librosa
        waveform, sr = librosa.load(audio_path, sr=Config.SAMPLE_RATE, mono=True)
        duration = len(waveform) / sr
        print(f"[说话人识别] 音频时长: {duration:.1f}s")

        # 3. 说话人分离
        t1 = time.time()
        diarization = _perform_diarization(audio_path, token)
        print(f"  分离耗时: {time.time()-t1:.1f}s")

        speaker_segments = _parse_diarization(diarization)
        if not speaker_segments:
            print("[说话人识别] 未检测到说话人")
            return {}

        # 4. 性别识别（多线程并行）
        t2 = time.time()
        speaker_info = _identify_genders_parallel(waveform, sr, speaker_segments)
        print(f"  性别识别耗时: {time.time()-t2:.1f}s")

        # 5. 合并过度分割（embedding 并行计算）
        t3 = time.time()
        print("\n[说话人识别] 后处理：检查是否需要合并...")
        speaker_info = _merge_oversplit_speakers_parallel(speaker_info, waveform, sr)
        print(f"  合并检查耗时: {time.time()-t3:.1f}s")

        print(f"[说话人识别] 总耗时: {time.time()-t0:.1f}s")
        _print_summary(speaker_info)
        return speaker_info

    except Exception as e:
        print(f"[说话人识别] 失败: {e}")
        import traceback
        traceback.print_exc()
        return {}
    finally:
        if audio_path and os.path.exists(audio_path):
            os.unlink(audio_path)


def _extract_audio_temp(video_path: str) -> Optional[str]:
    ffmpeg_bin = os.environ.get("FFMPEG_BIN", "ffmpeg")
    try:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        audio_path = tmp.name
        tmp.close()

        ext = os.path.splitext(video_path)[1].lower()
        if ext in (".wav", ".mp3", ".flac", ".m4a", ".ogg"):
            import librosa, soundfile as sf
            wav, sr = librosa.load(video_path, sr=Config.SAMPLE_RATE, mono=True)
            sf.write(audio_path, wav, Config.SAMPLE_RATE)
        else:
            cmd = [
                ffmpeg_bin, "-y", "-i", video_path,
                "-vn", "-acodec", "pcm_s16le",
                "-ar", str(Config.SAMPLE_RATE), "-ac", "1",
                audio_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"  [错误] ffmpeg: {result.stderr.strip()[-300:]}")
                return None
        return audio_path
    except Exception as e:
        print(f"  [错误] 音频提取异常: {e}")
        return None


def _perform_diarization(audio_path: str, token: str):
    """与 test_diarization.py perform_diarization() 完全相同"""
    from pyannote.audio import Pipeline
    import torch, librosa

    print(f"[说话人识别] 运行分离 (模型: {Config.DIARIZATION_MODEL})...")
    pipeline = Pipeline.from_pretrained(Config.DIARIZATION_MODEL, token=token)

    try:
        if torch.cuda.is_available():
            pipeline = pipeline.to(torch.device("cuda"))
            print("  使用 GPU 加速")
    except Exception:
        pass

    try:
        pipeline.instantiate({
            "segmentation": {"min_duration_off": Config.MIN_DURATION_OFF},
            "clustering": {"threshold": Config.CLUSTERING_THRESHOLD}
        })
    except Exception as e:
        print(f"  [提示] 参数微调跳过: {e}")

    waveform, sr = librosa.load(audio_path, sr=Config.SAMPLE_RATE, mono=True)
    wav_tensor = torch.from_numpy(waveform.copy()).float().unsqueeze(0)
    audio_input = {"waveform": wav_tensor, "sample_rate": sr}

    try:
        output = pipeline(audio_input,
                          min_speakers=Config.MIN_SPEAKERS,
                          max_speakers=Config.MAX_SPEAKERS)
    except TypeError:
        output = pipeline(audio_input)

    return output


def _parse_diarization(diarization) -> Dict[str, List[Tuple[float, float]]]:
    """与 test_diarization.py parse_diarization() 完全相同"""
    speaker_segments: Dict[str, List[Tuple[float, float]]] = {}

    def add(start, end, speaker):
        if end - start < Config.MIN_SEGMENT_DURATION:
            return
        spk = (f"SPEAKER_{int(speaker):02d}"
               if str(speaker).lstrip("-").isdigit() else str(speaker))
        speaker_segments.setdefault(spk, []).append((float(start), float(end)))

    if hasattr(diarization, "speaker_diarization"):
        for turn, speaker in diarization.speaker_diarization:
            add(turn.start, turn.end, speaker)
    elif hasattr(diarization, "itertracks"):
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            add(turn.start, turn.end, speaker)
    else:
        raise RuntimeError(f"未知 diarization 类型: {type(diarization)}")

    print(f"  检测到 {len(speaker_segments)} 个说话人:")
    for sp, segs in sorted(speaker_segments.items()):
        total = sum(e - s for s, e in segs)
        print(f"    {sp}: {len(segs)} 段, 共 {total:.1f}s")

    return speaker_segments


def _identify_genders_parallel(
    waveform: np.ndarray,
    sr: int,
    speaker_segments: Dict[str, List[Tuple[float, float]]]
) -> Dict[str, Dict]:
    """
    性别识别 —— 多线程并行版
    逻辑与 test_diarization.py identify_genders() 完全相同，
    只是各说话人改为并行处理。
    注意：GenderClassifier 是单例，模型只加载一次，线程安全（只读推理）。
    """
    from gender_classifier import GenderClassifier, Config as GC

    print("[说话人识别] 性别识别（并行）...")
    # 预先加载模型，避免多线程竞争初始化
    classifier = GenderClassifier()
    classifier.load_models()

    def process_one(speaker, segments):
        total_duration = sum(e - s for s, e in segments)
        valid_count = sum(1 for s, e in segments if e - s >= GC.MIN_DURATION)

        gender, confidence, details = classifier.classify_speaker_segments(
            waveform, sr, segments, speaker_id=speaker
        )
        if confidence < Config.GENDER_CONF_THRESHOLD:
            gender = "unknown"

        return speaker, {
            "gender": gender,
            "confidence": confidence,
            "segments": segments,
            "total_duration": total_duration,
            "segment_count": len(segments),
            "details": details,
        }

    speaker_info = {}
    n_speakers = len(speaker_segments)
    # 说话人数量通常 ≤ 8，全部并行
    workers = min(Config.GENDER_WORKERS, n_speakers)

    with ThreadPoolExecutor(max_workers=workers,
                            thread_name_prefix="gender") as executor:
        futures = {
            executor.submit(process_one, sp, segs): sp
            for sp, segs in speaker_segments.items()
        }
        for future in as_completed(futures):
            try:
                speaker, info = future.result()
                gz = {"male": "男", "female": "女", "unknown": "未知"}[info["gender"]]
                print(f"  {speaker}: {gz} (置信度={info['confidence']:.2f}, "
                      f"方法={info['details'].get('method', '?')})")
                speaker_info[speaker] = info
            except Exception as e:
                sp = futures[future]
                print(f"  {sp}: 性别识别失败 ({e})")
                speaker_info[sp] = {
                    "gender": "unknown", "confidence": 0.0,
                    "segments": speaker_segments[sp],
                    "total_duration": sum(e - s for s, e in speaker_segments[sp]),
                    "segment_count": len(speaker_segments[sp]),
                    "details": {"error": str(e)},
                }

    return speaker_info


def _merge_oversplit_speakers_parallel(
    speaker_info: Dict[str, Dict],
    waveform: np.ndarray,
    sr: int
) -> Dict[str, Dict]:
    """
    合并过度分割的说话人 —— Embedding 计算并行版
    逻辑与 test_diarization.py merge_oversplit_speakers() 完全相同
    """
    if len(speaker_info) <= 1:
        return speaker_info

    print("  [合并检查] 并行计算 embedding...")

    from gender_classifier import GenderClassifier
    classifier = GenderClassifier()

    if not classifier.load_models() or classifier._ecapa_model is None:
        print("  ECAPA 不可用，跳过合并")
        return speaker_info

    def compute_embedding(sp, info):
        segs = sorted(info["segments"], key=lambda x: x[1]-x[0], reverse=True)[:3]
        embs = []
        for start, end in segs:
            if end - start < 1.0:
                continue
            seg = waveform[int(start*sr):int(end*sr)]
            emb = classifier.get_embedding(seg, sr)
            if emb is not None:
                embs.append(emb)
        if embs:
            return sp, np.mean(embs, axis=0)
        return sp, None

    # 并行计算所有说话人的 embedding
    speaker_embeddings = {}
    workers = min(Config.EMBEDDING_WORKERS, len(speaker_info))
    with ThreadPoolExecutor(max_workers=workers,
                            thread_name_prefix="embed") as executor:
        futures = {
            executor.submit(compute_embedding, sp, info): sp
            for sp, info in speaker_info.items()
        }
        for future in as_completed(futures):
            sp, emb = future.result()
            if emb is not None:
                speaker_embeddings[sp] = emb

    # 以下合并逻辑与原版完全相同
    speakers = list(speaker_info.keys())
    merged = {}

    for i, sp1 in enumerate(speakers):
        if sp1 in merged:
            continue
        for sp2 in speakers[i+1:]:
            if sp2 in merged:
                continue
            g1 = speaker_info[sp1]["gender"]
            g2 = speaker_info[sp2]["gender"]
            if g1 != g2 and "unknown" not in (g1, g2):
                continue
            if sp1 not in speaker_embeddings or sp2 not in speaker_embeddings:
                continue
            e1, e2 = speaker_embeddings[sp1], speaker_embeddings[sp2]
            cos = float(np.dot(e1, e2) / (np.linalg.norm(e1)*np.linalg.norm(e2) + 1e-8))
            if cos >= Config.MERGE_COSINE_THRESHOLD:
                print(f"  合并 {sp2} → {sp1} (余弦={cos:.3f})")
                merged[sp2] = sp1

    if not merged:
        print("  无需合并")
        return speaker_info

    new_info = {}
    for sp, info in speaker_info.items():
        canonical = merged.get(sp, sp)
        if canonical not in new_info:
            new_info[canonical] = {**info,
                                   "segments": list(info["segments"]),
                                   "merged_from": []}
        else:
            new_info[canonical]["segments"].extend(info["segments"])
            new_info[canonical]["total_duration"] += info["total_duration"]
            new_info[canonical]["segment_count"] += info["segment_count"]
            new_info[canonical]["merged_from"].append(sp)
            if info["confidence"] > new_info[canonical]["confidence"]:
                new_info[canonical]["gender"] = info["gender"]
                new_info[canonical]["confidence"] = info["confidence"]

    final = {}
    for i, (sp, info) in enumerate(sorted(new_info.items())):
        name = f"SPEAKER_{i:02d}"
        final[name] = info
        if info.get("merged_from"):
            print(f"  {name} ← {sp} + {', '.join(info['merged_from'])}")

    print(f"  合并结果: {len(speaker_info)} → {len(final)} 个说话人")
    return final


def _get_lang_prefix(target_lang: str) -> str:
    lang_map = {
        "en": "en", "ja": "ja", "ko": "ko", "id": "id",
        "vi": "vi", "es": "es", "tr": "tr", "pt": "pt",
        "hi": "hi", "ar": "ar", "th": "th", "fr": "fr",
        "de": "de", "it": "it", "zh": "zh", "ru": "ru",
    }
    return lang_map.get(target_lang.lower(), "en")


def _print_summary(speaker_map: Dict[str, Dict]):
    print("\n[说话人识别] 结果:")
    print("-" * 50)
    for sp, info in sorted(speaker_map.items()):
        gz = {"male": "男", "female": "女", "unknown": "未知"}.get(info["gender"], "?")
        print(f"  {sp}: {gz}  时长={info['total_duration']:.1f}s  "
              f"片段={len(info['segments'])}")
    print("-" * 50)
