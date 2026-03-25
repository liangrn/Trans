#!/usr/bin/env python3
"""
性别识别模块 - 优化版

改进：
1. 模型只加载一次（单例）
2. 直接用 waveform tensor 推理，不写临时文件
3. ECAPA embedding 复用用于性别分类和说话人合并
4. 接口统一，消除重复加载
"""

import os
import warnings
from typing import Tuple, Dict, List, Optional
import numpy as np

warnings.filterwarnings("ignore")


class Config:
    SAMPLE_RATE = 16000
    MIN_DURATION = 0.5
    GENDER_CONF_THRESHOLD = 0.55
    MAX_SEGMENTS_FOR_GENDER = 10


class GenderClassifier:
    """
    性别识别器（单例模式，模型只加载一次）
    """
    _instance = None

    def __new__(cls, device: str = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, device: str = None):
        if self._initialized:
            return
        self._initialized = True
        self.device = device or self._detect_device()
        self._ecapa_model = None
        self._gender_model = None
        self._gender_head = None
        self._use_full_model = False
        self._model_ready = None

    def _detect_device(self) -> str:
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    @staticmethod
    def _patch_hf_hub():
        """修复 speechbrain 与新版 huggingface_hub 的兼容性问题
        新版 hub 废弃了 use_auth_token 参数，改为 token"""
        try:
            import huggingface_hub
            _orig = huggingface_hub.hf_hub_download
            def _patched(*args, **kwargs):
                if 'use_auth_token' in kwargs:
                    kwargs['token'] = kwargs.pop('use_auth_token')
                return _orig(*args, **kwargs)
            huggingface_hub.hf_hub_download = _patched
            # 同时 patch speechbrain 内部各模块的引用
            for _mod_name in [
                'speechbrain.utils.fetching',
                'speechbrain.utils.parameter_transfer',
                'speechbrain.pretrained.fetching',
            ]:
                try:
                    import importlib
                    _mod = importlib.import_module(_mod_name)
                    if hasattr(_mod, 'hf_hub_download'):
                        _mod.hf_hub_download = _patched
                except Exception:
                    pass
        except Exception:
            pass

    def load_models(self) -> bool:
        """加载所有模型，幂等"""
        if self._model_ready is not None:
            return self._model_ready

        try:
            import torch
            import torch.nn as nn

            # 在加载任何 speechbrain 模型前先打补丁
            self._patch_hf_hub()

            from speechbrain.inference.classifiers import EncoderClassifier

            print(f"  [模型] 加载 ECAPA embedding ({self.device})...")
            # 预创建 custom.py 占位文件，避免新版仓库结构变化导致 404
            import pathlib
            _savedir = pathlib.Path("pretrained_models/spkrec-ecapa-voxceleb")
            _savedir.mkdir(parents=True, exist_ok=True)
            (_savedir / "custom.py").write_text("# placeholder\n")
            self._ecapa_model = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir=str(_savedir),
                run_opts={"device": self.device}
            )

            # 尝试加载专用性别分类模型
            import sys
            sys.path.insert(0, "voice-gender-classifier")
            try:
                from model import ECAPA_gender
                print(f"  [模型] 加载 ECAPA-gender 分类模型...")
                self._gender_model = ECAPA_gender.from_pretrained("JaesungHuh/ecapa-gender")
                self._gender_model.eval()
                self._gender_model.to(torch.device(self.device))
                self._use_full_model = True
                print(f"  [模型] ECAPA-gender 加载成功 ✓")
            except Exception as e1:
                print(f"  [模型] 完整模型失败({e1})，尝试分类头...")
                try:
                    from huggingface_hub import hf_hub_download
                    model_path = hf_hub_download(
                        repo_id="JaesungHuh/ecapa-gender",
                        filename="model.pt"
                    )
                    state = torch.load(model_path, map_location=self.device)
                    self._gender_head = nn.Linear(192, 2).to(self.device)
                    if isinstance(state, dict) and 'weight' in state:
                        self._gender_head.load_state_dict(state)
                    else:
                        vals = list(state.values())
                        self._gender_head.weight = nn.Parameter(vals[-2].to(self.device))
                        self._gender_head.bias = nn.Parameter(vals[-1].to(self.device))
                    self._gender_head.eval()
                    print(f"  [模型] 分类头加载成功 ✓")
                except Exception as e2:
                    print(f"  [模型] 分类头失败({e2})，使用 embedding 统计")

            self._model_ready = True

        except Exception as e:
            print(f"  [模型] 加载失败，降级到 F0: {e}")
            self._model_ready = False

        return self._model_ready

    def get_embedding(self, waveform: np.ndarray, sr: int) -> Optional[np.ndarray]:
        """提取 ECAPA speaker embedding（用于说话人合并）"""
        if self._ecapa_model is None:
            return None
        try:
            import torch
            import librosa
            if sr != Config.SAMPLE_RATE:
                waveform = librosa.resample(waveform, orig_sr=sr, target_sr=Config.SAMPLE_RATE)
            wav_tensor = torch.from_numpy(waveform.copy()).float().unsqueeze(0).to(self.device)
            with torch.no_grad():
                emb = self._ecapa_model.encode_batch(wav_tensor)
            return emb.squeeze().cpu().numpy()
        except Exception:
            return None

    def _predict_one(self, waveform: np.ndarray) -> Tuple[Optional[str], float]:
        """单段 waveform 预测（已假设 sr=16000）"""
        import torch

        wav_tensor = torch.from_numpy(waveform.copy()).float().unsqueeze(0).to(self.device)

        try:
            if self._use_full_model and self._gender_model is not None:
                import tempfile, soundfile as sf, os
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp_path = tmp.name
                sf.write(tmp_path, waveform, Config.SAMPLE_RATE)
                with torch.no_grad():
                    result = self._gender_model.predict(tmp_path, torch.device(self.device))
                os.unlink(tmp_path)
                gender = "female" if "female" in str(result).lower() else "male"
                return gender, 0.90

            elif self._ecapa_model is not None and self._gender_head is not None:
                import torch.nn.functional as F
                with torch.no_grad():
                    emb = self._ecapa_model.encode_batch(wav_tensor).squeeze()
                    logits = self._gender_head(emb)
                    probs = F.softmax(logits, dim=0)
                    pred = int(torch.argmax(probs))
                    score = float(probs[pred])
                return ("male" if pred == 1 else "female"), score

            elif self._ecapa_model is not None:
                with torch.no_grad():
                    emb = self._ecapa_model.encode_batch(wav_tensor).squeeze().cpu().numpy()
                neg_ratio = float(np.mean(emb < -0.02))
                return ("male" if neg_ratio > 0.52 else "female"), 0.65

        except Exception:
            pass

        return None, 0.0

    def classify_speaker_segments(
        self,
        waveform_full: np.ndarray,
        sr: int,
        segments: List[Tuple[float, float]],
        speaker_id: str = ""
    ) -> Tuple[str, float, Dict]:
        """
        对同一说话人多段音频做性别识别

        Args:
            waveform_full: 完整音频（避免重复 IO）
            sr: 采样率
            segments: [(start, end), ...]
            speaker_id: 日志前缀
        """
        import librosa

        prefix = f"[{speaker_id}]" if speaker_id else "[]"

        # 过滤短片段，取最长的前 N 段
        valid = sorted(
            [(s, e) for s, e in segments if e - s >= Config.MIN_DURATION],
            key=lambda x: x[1] - x[0], reverse=True
        )[:Config.MAX_SEGMENTS_FOR_GENDER]

        if not valid:
            return "unknown", 0.0, {"error": "all_segments_too_short"}

        model_ok = self.load_models()

        if model_ok:
            votes = {"male": 0, "female": 0}
            scores = []
            for start, end in valid:
                seg = waveform_full[int(start * sr):int(end * sr)]
                # 确保 16kHz
                if sr != Config.SAMPLE_RATE:
                    seg = librosa.resample(seg, orig_sr=sr, target_sr=Config.SAMPLE_RATE)
                g, score = self._predict_one(seg)
                if g:
                    votes[g] += 1
                    scores.append(score)

            total = votes["male"] + votes["female"]
            if total > 0:
                gender = "male" if votes["male"] >= votes["female"] else "female"
                base_conf = float(np.mean(scores)) if scores else 0.5
                vote_ratio = abs(votes["male"] - votes["female"]) / total
                conf = base_conf * (0.7 + 0.3 * vote_ratio)
                gender_zh = "男" if gender == "male" else "女"
                print(f"  {prefix} male={votes['male']} female={votes['female']} "
                      f"→ {gender_zh} (conf={conf:.2f})")
                return gender, min(conf, 0.99), {
                    "method": "ecapa_gender",
                    "votes": votes,
                    "segments_used": total
                }

        return self._classify_by_f0(waveform_full, sr, valid)

    def _classify_by_f0(self, waveform_full, sr, segments):
        import librosa
        p25_list = []
        for start, end in segments[:5]:
            seg = waveform_full[int(start * sr):int(end * sr)]
            try:
                f0, _, vp = librosa.pyin(seg, fmin=60, fmax=500, sr=sr, fill_na=np.nan)
                valid = f0[~np.isnan(f0) & (vp > 0.4)]
                if len(valid) >= 5:
                    iqr = float(np.percentile(valid, 75) - np.percentile(valid, 25))
                    if iqr < 80:
                        p25_list.append(float(np.percentile(valid, 25)))
            except Exception:
                pass
        if not p25_list:
            return "unknown", 0.0, {"method": "f0", "error": "no_valid_f0"}
        mp25 = float(np.median(p25_list))
        if mp25 < 155:
            return "male", 0.70, {"method": "f0", "median_p25": mp25}
        elif mp25 > 195:
            return "female", 0.70, {"method": "f0", "median_p25": mp25}
        g = "male" if mp25 < 175 else "female"
        return g, 0.55, {"method": "f0", "median_p25": mp25}


# =================== 兼容旧接口 ===================

def detect_gender_from_audio(audio_path, start, end, hf_token=None):
    import librosa
    if end - start < 0.5:
        return "unknown", {}
    waveform, sr = librosa.load(audio_path, sr=Config.SAMPLE_RATE, mono=True,
                                offset=start, duration=end - start)
    c = GenderClassifier()
    gender, _, stats = c.classify_speaker_segments(waveform, sr, [(0, end - start)])
    return gender, stats


def classify_gender(audio_path, start=None, end=None):
    import librosa
    offset = start or 0
    duration = (end - start) if start is not None and end is not None else None
    waveform, sr = librosa.load(audio_path, sr=Config.SAMPLE_RATE, mono=True,
                                offset=offset, duration=duration)
    c = GenderClassifier()
    gender, conf, _ = c.classify_speaker_segments(waveform, sr, [(0, len(waveform) / sr)])
    return gender, conf
