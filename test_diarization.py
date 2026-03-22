#!/usr/bin/env python3
"""
说话人分离 + 性别识别 - 影视场景优化版

主要改进：
1. pyannote 参数针对影视多人场景调优（放宽说话人数、细化聚类）
2. 性别识别改为"先聚合所有片段，再统一判断"，准确率大幅提升
3. 增加说话人合并后处理（解决同一人被识别成多人问题）
4. 增加 --num-speakers 参数，已知人数时可强制指定

环境: conda activate iai
依赖: pip install pyannote.audio speechbrain librosa scipy torch soundfile
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
    MIN_SEGMENT_DURATION = 0.5      # 有效片段最短时长（秒）

    # pyannote 影视场景参数
    DIARIZATION_MODEL = "pyannote/speaker-diarization-community-1"
    # 聚类阈值：越小越容易把两人分开（影视建议 0.4~0.6）
    # 如果同一人被分成多人 → 调大；不同人被合并 → 调小
    CLUSTERING_THRESHOLD = 0.45
    MIN_CLUSTER_SIZE = 2
    MIN_DURATION_OFF = 0.1          # 静音间隔阈值（影视场景说话切换快）

    # 说话人数量限制（影视对话通常 2~8 人）
    MIN_SPEAKERS = 2
    MAX_SPEAKERS = 8                # 设置上限避免过度分割

    # 说话人合并：两个说话人 embedding 余弦相似度超过此值则合并
    MERGE_COSINE_THRESHOLD = 0.75

    # 性别识别
    GENDER_CONF_THRESHOLD = 0.55    # 低于此值标记为 unknown

    # HuggingFace Token（建议用环境变量替代）
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
        print(f"    [错误] ffmpeg 失败: {result.stderr.strip()[-200:]}")
        return False
    return os.path.exists(audio_path)


def load_audio(audio_path: str) -> Tuple[np.ndarray, int]:
    import librosa
    return librosa.load(audio_path, sr=Config.SAMPLE_RATE, mono=True)


# =================== 说话人分离 ===================

def perform_diarization(audio_path: str, hf_token: str,
                        num_speakers: Optional[int] = None):
    """
    使用 pyannote 3.1 进行说话人分离

    Args:
        num_speakers: 已知说话人数时传入，可大幅提升准确率
    """
    from pyannote.audio import Pipeline
    import librosa
    import torch

    print(f"[3/5] 说话人分离...")
    print(f"    模型: {Config.DIARIZATION_MODEL}")
    if num_speakers:
        print(f"    指定说话人数: {num_speakers}")
    else:
        print(f"    自动检测说话人数 (范围: {Config.MIN_SPEAKERS}~{Config.MAX_SPEAKERS})")

    token = hf_token or Config.HF_TOKEN
    if not token:
        raise ValueError("未设置 HF_TOKEN，请设置环境变量或在 Config 中填入")

    pipeline = Pipeline.from_pretrained(Config.DIARIZATION_MODEL, token=token)

    # 移动到 GPU（如果可用）
    try:
        import torch
        if torch.cuda.is_available():
            pipeline = pipeline.to(torch.device("cuda"))
            print("    使用 GPU 加速")
    except Exception:
        pass

    # 调整内部参数
    pipeline.instantiate({"segmentation": {"min_duration_off": Config.MIN_DURATION_OFF}, "clustering": {"threshold": Config.CLUSTERING_THRESHOLD}})

    waveform, sr = librosa.load(audio_path, sr=Config.SAMPLE_RATE, mono=True)
    waveform_tensor = torch.from_numpy(waveform.copy()).float().unsqueeze(0)

    # 根据是否已知说话人数选择调用方式
    kwargs = {"waveform": waveform_tensor, "sample_rate": sr}
    if num_speakers:
        diarization = pipeline(kwargs, num_speakers=num_speakers)
    else:
        diarization = pipeline(
            kwargs,
            min_speakers=Config.MIN_SPEAKERS,
            max_speakers=Config.MAX_SPEAKERS
        )

    return diarization


def parse_diarization(diarization) -> Dict[str, List[Tuple[float, float]]]:
    """解析 pyannote 4.x DiarizeOutput 或 3.x Annotation"""
    speaker_segments: Dict[str, List[Tuple[float, float]]] = {}

    def add(start, end, speaker):
        if end - start < Config.MIN_SEGMENT_DURATION:
            return
        spk = f"SPEAKER_{int(speaker):02d}" if str(speaker).lstrip("-").isdigit() else str(speaker)
        speaker_segments.setdefault(spk, []).append((float(start), float(end)))

    # 打印所有可用属性，帮助调试
    attrs = [a for a in dir(diarization) if not a.startswith('_')]
    print(f"    [debug] DiarizeOutput 属性: {attrs}")

    if hasattr(diarization, 'speaker_diarization'):
        print("    [debug] 使用 4.x speaker_diarization 接口")
        for turn, speaker in diarization.speaker_diarization:
            add(turn.start, turn.end, speaker)
    elif hasattr(diarization, 'itertracks'):
        print("    [debug] 使用 3.x itertracks 接口")
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            add(turn.start, turn.end, speaker)
    else:
        raise RuntimeError(
            f"未知类型: {type(diarization)}, 属性: {attrs}"
        )

    print(f"    检测到 {len(speaker_segments)} 个说话人")
    for sp, segs in sorted(speaker_segments.items()):
        total = sum(e - s for s, e in segs)
        print(f"      {sp}: {len(segs)} 段, 共 {total:.1f}s")
    return speaker_segments

def identify_genders(
    audio_path: str,
    speaker_segments: Dict[str, List[Tuple[float, float]]]
) -> Dict[str, Dict]:
    """
    对每个说话人做性别识别

    改进：聚合该说话人所有片段，统一判断，而非逐段投票
    """
    from gender_classifier import GenderClassifier

    print("\n[4/5] 性别识别...")
    classifier = GenderClassifier()

    speaker_info = {}
    for speaker, segments in sorted(speaker_segments.items()):
        total_duration = sum(e - s for s, e in segments)
        print(f"\n  {speaker} (共 {total_duration:.1f}s, {len(segments)} 段):")

        gender, confidence, details = classifier.classify_speaker_segments(
            audio_path, segments, speaker_id=speaker
        )

        # 置信度不足时标记为 unknown
        if confidence < Config.GENDER_CONF_THRESHOLD:
            print(f"    → 置信度不足 ({confidence:.2f})，标记为未知")
            gender = "unknown"

        gender_zh = {"male": "男", "female": "女", "unknown": "未知"}[gender]
        print(f"    → 最终: {gender_zh} (置信度={confidence:.2f})")

        speaker_info[speaker] = {
            "gender": gender,
            "confidence": confidence,
            "segments": segments,
            "total_duration": total_duration,
            "segment_count": len(segments),
            "details": details,
        }

    return speaker_info


# =================== 说话人合并后处理 ===================

def merge_oversplit_speakers(
    speaker_info: Dict[str, Dict],
    audio_path: str
) -> Dict[str, Dict]:
    """
    检测并合并被过度分割的说话人（同一人被识别成多人）

    策略：对所有说话人对，如果性别相同且 embedding 余弦相似度高，则合并。
    """
    speakers = list(speaker_info.keys())
    if len(speakers) <= 1:
        return speaker_info

    print("\n  [后处理] 检查过度分割...")

    # 尝试用 SpeechBrain 计算 embedding 相似度
    try:
        from gender_classifier import GenderClassifier
        import librosa, torch

        classifier = GenderClassifier()
        if not classifier._try_load_speechbrain():
            print("    SpeechBrain 不可用，跳过合并检查")
            return speaker_info

        waveform_full, sr = librosa.load(audio_path, sr=16000, mono=True)

        # 为每个说话人计算平均 embedding（取前 3 段中最长的）
        speaker_embeddings = {}
        for sp, info in speaker_info.items():
            segs = sorted(info["segments"], key=lambda x: x[1]-x[0], reverse=True)[:3]
            embs = []
            for start, end in segs:
                if end - start < 0.5:
                    continue
                seg = waveform_full[int(start*sr):int(end*sr)]
                emb = classifier._get_ecapa_embedding(seg, sr)
                if emb is not None:
                    embs.append(emb)
            if embs:
                speaker_embeddings[sp] = np.mean(embs, axis=0)

        # 找需要合并的说话人对
        merged = {}  # old_speaker -> canonical_speaker
        for i, sp1 in enumerate(speakers):
            if sp1 in merged:
                continue
            for sp2 in speakers[i+1:]:
                if sp2 in merged:
                    continue
                # 只合并相同性别（或其中一个 unknown）
                g1 = speaker_info[sp1]["gender"]
                g2 = speaker_info[sp2]["gender"]
                if g1 != g2 and g1 != "unknown" and g2 != "unknown":
                    continue

                if sp1 not in speaker_embeddings or sp2 not in speaker_embeddings:
                    continue

                emb1 = speaker_embeddings[sp1]
                emb2 = speaker_embeddings[sp2]
                cosine = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8)

                if cosine >= Config.MERGE_COSINE_THRESHOLD:
                    print(f"    合并 {sp2} → {sp1} (余弦相似度={cosine:.3f})")
                    merged[sp2] = sp1

        if not merged:
            print("    无需合并")
            return speaker_info

        # 执行合并
        new_info = {}
        for sp, info in speaker_info.items():
            canonical = merged.get(sp, sp)
            if canonical not in new_info:
                new_info[canonical] = {
                    "gender": info["gender"],
                    "confidence": info["confidence"],
                    "segments": list(info["segments"]),
                    "total_duration": info["total_duration"],
                    "segment_count": info["segment_count"],
                    "details": info["details"],
                    "merged_from": [],
                }
            else:
                # 合并
                new_info[canonical]["segments"].extend(info["segments"])
                new_info[canonical]["total_duration"] += info["total_duration"]
                new_info[canonical]["segment_count"] += info["segment_count"]
                new_info[canonical]["merged_from"].append(sp)
                # 取置信度更高的性别结果
                if info["confidence"] > new_info[canonical]["confidence"]:
                    new_info[canonical]["gender"] = info["gender"]
                    new_info[canonical]["confidence"] = info["confidence"]

        # 重命名（SPEAKER_00, SPEAKER_01, ...）
        final_info = {}
        for i, (sp, info) in enumerate(sorted(new_info.items())):
            new_name = f"SPEAKER_{i:02d}"
            final_info[new_name] = info
            if info.get("merged_from"):
                print(f"    {new_name} (原 {sp} + {', '.join(info['merged_from'])})")

        print(f"    合并后: {len(speaker_info)} → {len(final_info)} 个说话人")
        return final_info

    except Exception as e:
        print(f"    合并检查失败: {e}")
        return speaker_info


# =================== 结果输出 ===================

def print_results(speaker_info: Dict[str, Dict]):
    print("\n" + "=" * 65)
    print("说话人分析结果")
    print("=" * 65)
    print(f"\n检测到 {len(speaker_info)} 个说话人:\n")

    for speaker, info in sorted(speaker_info.items()):
        gender = info.get("gender", "unknown")
        gender_zh = {"male": "男", "female": "女", "unknown": "未知"}.get(gender, "?")
        conf = info.get("confidence", 0)
        duration = info.get("total_duration", 0)
        seg_count = info.get("segment_count", 0)
        method = info.get("details", {}).get("method", "?")

        print(f"  {speaker}:")
        print(f"    性别: {gender_zh}  (置信度={conf:.2f}, 方法={method})")
        print(f"    总时长: {duration:.1f}s  片段数: {seg_count}")
        if info.get("merged_from"):
            print(f"    合并自: {', '.join(info['merged_from'])}")
        print()

    # 时间线
    print("-" * 65)
    print("时间线:")
    print("-" * 65)

    all_segs = []
    for sp, info in speaker_info.items():
        gender = info.get("gender", "unknown")
        for start, end in info.get("segments", []):
            all_segs.append((start, end, sp, gender))
    all_segs.sort()

    for start, end, sp, gender in all_segs:
        symbol = {"male": "男", "female": "女", "unknown": "?"}.get(gender, "?")
        sm, ss = int(start // 60), start % 60
        em, es = int(end // 60), end % 60
        dur = end - start
        print(f"  [{sm:3d}:{ss:05.2f} - {em:3d}:{es:05.2f}] ({dur:4.1f}s) {sp} ({symbol})")

    print(f"\n  共 {len(all_segs)} 个片段")


def export_json(speaker_info: Dict[str, Dict], output_path: str):
    # segments 是 tuple，需要转成 list
    serializable = {}
    for sp, info in speaker_info.items():
        serializable[sp] = {
            **info,
            "segments": [list(seg) for seg in info["segments"]],
        }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)
    print(f"\n[输出] JSON 已保存: {output_path}")


# =================== 主流程 ===================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="说话人分离 + 性别识别（影视场景优化）")
    parser.add_argument("video", help="输入视频/音频文件")
    parser.add_argument("--output-json", type=str, help="导出 JSON 结果")
    parser.add_argument("--num-speakers", type=int, default=None,
                        help="已知说话人数（指定后准确率更高）")
    parser.add_argument("--hf-token", type=str, default=None,
                        help="HuggingFace Token（也可设置 HF_TOKEN 环境变量）")
    parser.add_argument("--clustering-threshold", type=float, default=None,
                        help=f"聚类阈值（默认={Config.CLUSTERING_THRESHOLD}）"
                             "；同一人被分多人→调大，不同人被合并→调小")
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"[错误] 文件不存在: {args.video}")
        sys.exit(1)

    if args.clustering_threshold:
        Config.CLUSTERING_THRESHOLD = args.clustering_threshold

    hf_token = args.hf_token or os.environ.get("HF_TOKEN", Config.HF_TOKEN)
    if not hf_token:
        print("[错误] 请设置 HF_TOKEN 环境变量，或用 --hf-token 传入")
        print("  获取: https://huggingface.co/settings/tokens")
        sys.exit(1)

    print(f"[开始] 处理: {args.video}")
    if args.num_speakers:
        print(f"[配置] 指定说话人数: {args.num_speakers}")
    print("-" * 65)

    audio_path = None
    try:
        # 1. 提取音频
        print("[1/5] 提取音频...")
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            audio_path = tmp.name

        # 判断输入是视频还是音频
        ext = os.path.splitext(args.video)[1].lower()
        if ext in (".wav", ".mp3", ".flac", ".m4a", ".ogg"):
            # 直接重采样
            import librosa, soundfile as sf
            waveform, sr = librosa.load(args.video, sr=Config.SAMPLE_RATE, mono=True)
            sf.write(audio_path, waveform, Config.SAMPLE_RATE)
            print(f"    音频文件直接加载，时长: {len(waveform)/Config.SAMPLE_RATE:.1f}s")
        else:
            if not extract_audio(args.video, audio_path):
                print("[错误] 音频提取失败")
                sys.exit(1)

        # 2. 时长检查
        waveform_check, sr_check = load_audio(audio_path)
        duration_total = len(waveform_check) / sr_check
        print(f"[2/5] 音频时长: {duration_total:.1f}s ({duration_total/60:.1f}min)")
        del waveform_check

        # 3. 说话人分离
        diarization = perform_diarization(
            audio_path, hf_token,
            num_speakers=args.num_speakers
        )

        # 4. 解析分离结果
        speaker_segments = parse_diarization(diarization)

        if not speaker_segments:
            print("[错误] 未检测到说话人")
            sys.exit(1)

        # 5. 性别识别（聚合判断）
        speaker_info = identify_genders(audio_path, speaker_segments)

        # 6. 说话人合并后处理
        print("\n[5/5] 后处理...")
        speaker_info = merge_oversplit_speakers(speaker_info, audio_path)

        # 7. 输出结果
        print_results(speaker_info)

        if args.output_json:
            export_json(speaker_info, args.output_json)

    except KeyboardInterrupt:
        print("\n[中断] 用户取消")
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
