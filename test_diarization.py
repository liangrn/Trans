#!/usr/bin/env python3
"""
说话人分离 + 性别识别 - 影视场景优化版 v3

优化点：
1. 音频只加载一次，传给所有后续步骤
2. 模型只初始化一次（GenderClassifier 单例）
3. 短片段过滤：时间线输出默认只显示 >= 1s 的片段
4. 说话人合并使用已有的 ECAPA embedding，不重复加载模型
5. --min-display-duration 控制时间线最小显示时长

环境: conda activate iai
依赖: pip install pyannote.audio speechbrain librosa scipy torch soundfile huggingface_hub
"""

import os
import sys
import json
import tempfile
import subprocess
import warnings
from typing import Dict, List, Tuple, Optional

import numpy as np

warnings.filterwarnings("ignore")


# =================== 配置 ===================
class Config:
    SAMPLE_RATE = 16000
    MIN_SEGMENT_DURATION = 0.5       # pyannote 片段最短时长（过滤噪声）
    MIN_DISPLAY_DURATION = 1.0       # 时间线显示最短时长（可通过参数覆盖）

    # pyannote 4.x
    DIARIZATION_MODEL = "pyannote/speaker-diarization-community-1"
    CLUSTERING_THRESHOLD = 0.45
    MIN_DURATION_OFF = 0.1

    # 说话人数量
    MIN_SPEAKERS = 2
    MAX_SPEAKERS = 8

    # 说话人合并：余弦相似度高于此值合并（相同性别）
    MERGE_COSINE_THRESHOLD = 0.82

    # 性别置信度低于此值标记为 unknown
    GENDER_CONF_THRESHOLD = 0.55

    HF_TOKEN = os.environ.get("HF_TOKEN", "")


# =================== 音频工具 ===================

def extract_audio(video_path: str, audio_path: str) -> bool:
    """从视频提取 16kHz mono WAV"""
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le",
        "-ar", str(Config.SAMPLE_RATE), "-ac", "1",
        audio_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [错误] ffmpeg: {result.stderr.strip()[-300:]}")
        return False
    return os.path.exists(audio_path)


def load_audio(audio_path: str) -> Tuple[np.ndarray, int]:
    import librosa
    return librosa.load(audio_path, sr=Config.SAMPLE_RATE, mono=True)


# =================== 说话人分离 ===================

def perform_diarization(audio_path: str, hf_token: str,
                        num_speakers: Optional[int] = None):
    """pyannote 4.x 说话人分离"""
    from pyannote.audio import Pipeline
    import torch

    token = hf_token or Config.HF_TOKEN
    if not token:
        raise ValueError("请设置 HF_TOKEN 环境变量")

    print(f"[3/5] 说话人分离 (模型: {Config.DIARIZATION_MODEL})...")
    if num_speakers:
        print(f"  指定说话人数: {num_speakers}")
    else:
        print(f"  自动检测 ({Config.MIN_SPEAKERS}~{Config.MAX_SPEAKERS} 人)")

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

    # 传 waveform tensor 避免 pyannote 4.x AudioDecoder bug
    import librosa, torch
    waveform, sr = librosa.load(audio_path, sr=Config.SAMPLE_RATE, mono=True)
    wav_tensor = torch.from_numpy(waveform.copy()).float().unsqueeze(0)
    audio_input = {"waveform": wav_tensor, "sample_rate": sr}

    try:
        if num_speakers:
            output = pipeline(audio_input, num_speakers=num_speakers)
        else:
            output = pipeline(audio_input,
                              min_speakers=Config.MIN_SPEAKERS,
                              max_speakers=Config.MAX_SPEAKERS)
    except TypeError:
        output = pipeline(audio_input)

    return output


def parse_diarization(diarization) -> Dict[str, List[Tuple[float, float]]]:
    """解析 pyannote 4.x DiarizeOutput 或 3.x Annotation"""
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
        raise RuntimeError(
            f"未知 diarization 类型: {type(diarization)}\n"
            f"属性: {[a for a in dir(diarization) if not a.startswith('_')]}"
        )

    print(f"  检测到 {len(speaker_segments)} 个说话人:")
    for sp, segs in sorted(speaker_segments.items()):
        total = sum(e - s for s, e in segs)
        print(f"    {sp}: {len(segs)} 段, 共 {total:.1f}s")

    return speaker_segments


# =================== 性别识别 ===================

def identify_genders(
    waveform: np.ndarray,
    sr: int,
    speaker_segments: Dict[str, List[Tuple[float, float]]],
    per_segment: bool = False
) -> Dict[str, Dict]:
    """
    性别识别：传入已加载的 waveform，避免重复 IO

    Args:
        waveform: 音频波形
        sr: 采样率
        speaker_segments: {speaker: [(start, end), ...]}
        per_segment: 是否对每个片段独立判断性别（True=不投票聚合）
    """
    from gender_classifier import GenderClassifier, Config as GC

    mode_str = "（按片段）" if per_segment else "（按说话人投票）"
    print(f"\n[4/5] 性别识别{mode_str}...")
    classifier = GenderClassifier()

    speaker_info = {}
    for speaker, segments in sorted(speaker_segments.items()):
        total_duration = sum(e - s for s, e in segments)
        valid_count = sum(1 for s, e in segments if e - s >= GC.MIN_DURATION)
        print(f"\n  {speaker} ({total_duration:.1f}s, {len(segments)} 段, "
              f"有效 {valid_count} 段):")

        if per_segment:
            # 每个片段独立判断
            segment_results = classifier.classify_each_segment(
                waveform, sr, segments, speaker_id=speaker
            )

            # 统计投票（用于显示）
            votes = {"male": 0, "female": 0, "unknown": 0}
            for seg_info in segment_results:
                votes[seg_info["gender"]] += 1

            # 主性别（多数投票）
            main_gender = max(votes, key=votes.get)
            confidence = votes[main_gender] / len(segment_results) if segment_results else 0

            gender_zh = {"male": "男", "female": "女", "unknown": "未知"}[main_gender]
            print(f"  → 主性别: {gender_zh} (male={votes['male']}, female={votes['female']}, unknown={votes['unknown']})")

            speaker_info[speaker] = {
                "gender": main_gender,
                "confidence": confidence,
                "segments": segments,
                "segment_genders": segment_results,  # 新增：每个片段的性别
                "total_duration": total_duration,
                "segment_count": len(segments),
                "details": {"method": "per_segment", "votes": votes},
            }
        else:
            # 按说话人投票聚合（原逻辑）
            gender, confidence, details = classifier.classify_speaker_segments(
                waveform, sr, segments, speaker_id=speaker
            )

            if confidence < Config.GENDER_CONF_THRESHOLD:
                gender = "unknown"

            gender_zh = {"male": "男", "female": "女", "unknown": "未知"}[gender]
            print(f"  → {gender_zh} (置信度={confidence:.2f}, 方法={details.get('method', '?')})")

            speaker_info[speaker] = {
                "gender": gender,
                "confidence": confidence,
                "segments": segments,
                "total_duration": total_duration,
                "segment_count": len(segments),
                "details": details,
            }

    return speaker_info


# =================== 说话人合并 ===================

def merge_oversplit_speakers(
    speaker_info: Dict[str, Dict],
    waveform: np.ndarray,
    sr: int
) -> Dict[str, Dict]:
    """
    用 ECAPA embedding 检测并合并被过度分割的说话人
    复用 GenderClassifier 里已加载的 ecapa_model，不重复初始化
    """
    if len(speaker_info) <= 1:
        return speaker_info

    print("\n  [合并检查] 计算说话人 embedding 相似度...")

    from gender_classifier import GenderClassifier
    classifier = GenderClassifier()

    # 确保模型已加载
    if not classifier.load_models() or classifier._ecapa_model is None:
        print("  ECAPA 不可用，跳过合并")
        return speaker_info

    # 每个说话人取最长的 3 段算平均 embedding
    speaker_embeddings = {}
    for sp, info in speaker_info.items():
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
            speaker_embeddings[sp] = np.mean(embs, axis=0)

    speakers = list(speaker_info.keys())
    merged = {}  # sp -> canonical

    for i, sp1 in enumerate(speakers):
        if sp1 in merged:
            continue
        for sp2 in speakers[i+1:]:
            if sp2 in merged:
                continue
            g1 = speaker_info[sp1]["gender"]
            g2 = speaker_info[sp2]["gender"]
            # 只合并性别相同（或其中一个 unknown）的说话人
            if g1 != g2 and "unknown" not in (g1, g2):
                continue
            if sp1 not in speaker_embeddings or sp2 not in speaker_embeddings:
                continue
            e1, e2 = speaker_embeddings[sp1], speaker_embeddings[sp2]
            cos = float(np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-8))
            if cos >= Config.MERGE_COSINE_THRESHOLD:
                print(f"  合并 {sp2} → {sp1} (余弦={cos:.3f})")
                merged[sp2] = sp1

    if not merged:
        print("  无需合并")
        return speaker_info

    # 执行合并
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

    # 重新编号
    final = {}
    for i, (sp, info) in enumerate(sorted(new_info.items())):
        name = f"SPEAKER_{i:02d}"
        final[name] = info
        if info.get("merged_from"):
            print(f"  {name} ← {sp} + {', '.join(info['merged_from'])}")

    print(f"  合并结果: {len(speaker_info)} → {len(final)} 个说话人")
    return final


# =================== 输出 ===================

def print_results(speaker_info: Dict[str, Dict],
                  min_display_duration: float = 1.0):
    print("\n" + "=" * 65)
    print("说话人分析结果")
    print("=" * 65)
    print(f"\n检测到 {len(speaker_info)} 个说话人:\n")

    for sp, info in sorted(speaker_info.items()):
        g = info.get("gender", "unknown")
        gz = {"male": "男", "female": "女", "unknown": "未知"}.get(g, "?")
        print(f"  {sp}: {gz}  置信度={info.get('confidence',0):.2f}  "
              f"时长={info.get('total_duration',0):.1f}s  "
              f"片段={info.get('segment_count',0)}")
        if info.get("merged_from"):
            print(f"    (合并自: {', '.join(info['merged_from'])})")

    print(f"\n  * 时间线仅显示 >= {min_display_duration}s 的片段")
    print("-" * 65)
    print("时间线:")
    print("-" * 65)

    all_segs = []
    for sp, info in speaker_info.items():
        # 检查是否有按片段的性别信息
        segment_genders = info.get("segment_genders")

        if segment_genders:
            # 按片段性别显示
            for seg_info in segment_genders:
                start, end = seg_info["start"], seg_info["end"]
                if end - start >= min_display_duration:
                    all_segs.append((start, end, sp, seg_info["gender"]))
        else:
            # 原逻辑：按说话人性别显示
            g = info.get("gender", "unknown")
            for start, end in info.get("segments", []):
                if end - start >= min_display_duration:
                    all_segs.append((start, end, sp, g))
    all_segs.sort()

    for start, end, sp, g in all_segs:
        gz = {"male": "男", "female": "女", "unknown": "?"}.get(g, "?")
        sm, ss = int(start // 60), start % 60
        em, es = int(end // 60), end % 60
        print(f"  [{sm:3d}:{ss:05.2f} - {em:3d}:{es:05.2f}] "
              f"({end-start:4.1f}s) {sp} ({gz})")

    total_segs = sum(len(info["segments"]) for info in speaker_info.values())
    print(f"\n  显示 {len(all_segs)} / 共 {total_segs} 个片段")


def export_json(speaker_info: Dict[str, Dict], output_path: str):
    out = {sp: {**info, "segments": [list(s) for s in info["segments"]]}
           for sp, info in speaker_info.items()}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n[输出] JSON: {output_path}")


# =================== 主流程 ===================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="说话人分离+性别识别 (影视优化版)")
    parser.add_argument("video", help="输入视频/音频")
    parser.add_argument("--output-json", type=str)
    parser.add_argument("--num-speakers", type=int, default=None,
                        help="已知说话人数（指定后更准确）")
    parser.add_argument("--hf-token", type=str, default=None)
    parser.add_argument("--clustering-threshold", type=float, default=None,
                        help=f"聚类阈值 默认={Config.CLUSTERING_THRESHOLD}；"
                             "同一人被分多人→调大，不同人被合并→调小")
    parser.add_argument("--min-display-duration", type=float, default=1.0,
                        help="时间线最小显示时长(秒)，默认=1.0，设0显示全部")
    parser.add_argument("--per-segment", action="store_true",
                        help="对每个片段独立判断性别（不投票聚合）")
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"[错误] 文件不存在: {args.video}")
        sys.exit(1)

    if args.clustering_threshold:
        Config.CLUSTERING_THRESHOLD = args.clustering_threshold

    hf_token = args.hf_token or os.environ.get("HF_TOKEN", Config.HF_TOKEN)
    if not hf_token:
        print("[错误] 请设置 HF_TOKEN 环境变量")
        sys.exit(1)

    print(f"[开始] {args.video}")
    print("-" * 65)

    audio_path = None
    try:
        # 1. 提取音频
        print("[1/5] 提取音频...")
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            audio_path = tmp.name

        ext = os.path.splitext(args.video)[1].lower()
        if ext in (".wav", ".mp3", ".flac", ".m4a", ".ogg"):
            import librosa, soundfile as sf
            wav, sr = librosa.load(args.video, sr=Config.SAMPLE_RATE, mono=True)
            sf.write(audio_path, wav, Config.SAMPLE_RATE)
        else:
            if not extract_audio(args.video, audio_path):
                sys.exit(1)

        # 2. 加载音频（只加载一次，后续复用）
        print("[2/5] 加载音频...")
        waveform, sr = load_audio(audio_path)
        duration = len(waveform) / sr
        print(f"  时长: {duration:.1f}s ({duration/60:.1f}min)")

        # 3. 说话人分离
        diarization = perform_diarization(audio_path, hf_token, args.num_speakers)
        speaker_segments = parse_diarization(diarization)

        if not speaker_segments:
            print("[错误] 未检测到说话人")
            sys.exit(1)

        # 4. 性别识别（传入已加载的 waveform）
        speaker_info = identify_genders(waveform, sr, speaker_segments,
                                        per_segment=args.per_segment)

        # 5. 说话人合并（复用已加载的 ECAPA 模型）
        print("\n[5/5] 后处理...")
        speaker_info = merge_oversplit_speakers(speaker_info, waveform, sr)

        # 输出
        print_results(speaker_info, min_display_duration=args.min_display_duration)

        if args.output_json:
            export_json(speaker_info, args.output_json)

    except KeyboardInterrupt:
        print("\n[中断]")
        sys.exit(130)
    except Exception as e:
        print(f"\n[错误] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        if audio_path and os.path.exists(audio_path):
            os.unlink(audio_path)


if __name__ == "__main__":
    main()
