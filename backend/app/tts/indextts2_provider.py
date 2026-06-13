"""
IndexTTS2 TTS Provider — Bilibili 开源零样本语音合成

依赖:
  - indextts 包（pip install -e /path/to/index-tts2）
  - PyTorch + CUDA
  - 模型文件位于 checkpoints/IndexTTS-2/ 目录

使用前先加载模型（约 25-30 秒），之后保持单例复用。
"""
import os
import io
import json
import sys
import uuid
import time
import asyncio
from pathlib import Path
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor

from app.tts.base import TTSProvider, VoiceInfo, VoiceProfile


class IndexTTS2Provider(TTSProvider):
    """IndexTTS2 音色克隆 Provider"""

    def __init__(
        self,
        model_dir: str = None,
        voice_dir: str = "./data/voices",
    ):
        # 默认使用本地 models 目录
        if model_dir is None:
            model_dir = str(Path(__file__).parent.parent.parent / "models" / "checkpoints" / "IndexTTS-2")
        self.model_dir = model_dir
        self.voice_dir = Path(voice_dir)
        self.voice_dir.mkdir(parents=True, exist_ok=True)
        self._mapping_file = self.voice_dir / "voice_mapping.json"
        self._mapping: dict[str, VoiceProfile] = {}
        self._tts = None
        self._load_mapping()

        # 使用线程池运行同步的 IndexTTS2 推理（避免阻塞 FastAPI 事件循环）
        self._executor = ThreadPoolExecutor(max_workers=1)

    @property
    def name(self) -> str:
        return "indextts2"

    def _ensure_model(self):
        """懒加载模型（仅首次调用时初始化）"""
        if self._tts is None:
            # 添加本地包路径（指向 models/ 目录，Python 会从中查找 indextts 包）
            models_dir = str(Path(__file__).parent.parent.parent / "models")
            if models_dir not in sys.path:
                sys.path.insert(0, models_dir)

            from indextts.infer_v2 import IndexTTS2
            print(f"[IndexTTS2] Loading model from {self.model_dir}...")
            t0 = time.time()
            self._tts = IndexTTS2(
                model_dir=self.model_dir,
                cfg_path=os.path.join(self.model_dir, "config.yaml"),
                use_fp16=True,
                use_cuda_kernel=False,
                use_deepspeed=False,
            )
            print(f"[IndexTTS2] Model loaded in {time.time() - t0:.1f}s")

    def _emotion_to_vector(self, emotion: Optional[str], tone: Optional[str]) -> Optional[list]:
        """将 emotion/tone 标签映射为 IndexTTS2 8维情感向量。

        8 维顺序：[高兴, 愤怒, 悲伤, 恐惧, 厌恶, 低落, 惊喜, 平静]
        """
        # 默认情感向量（平静）
        base = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

        emotion_map = {
            "happy":     [0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2],
            "sad":       [0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.2],
            "angry":     [0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2],
            "surprised": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.2],
            "calm":      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            "nervous":   [0.0, 0.0, 0.0, 0.6, 0.0, 0.3, 0.0, 0.1],
            "cold":      [0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.0, 0.7],
        }

        if emotion and emotion in emotion_map:
            vec = emotion_map[emotion].copy()
        else:
            vec = base.copy()

        # tone 微调
        tone_adjust = {
            "loud":      {1: +0.2},
            "soft":      {6: +0.3, 7: -0.2},
            "whisper":   {6: +0.4, 7: -0.3},
            "gentle":    {0: +0.3, 7: +0.2},
            "serious":   {4: +0.3, 7: -0.2},
            "sarcastic": {0: +0.2, 4: +0.3},
        }

        if tone and tone in tone_adjust:
            for idx, delta in tone_adjust[tone].items():
                vec[idx] = max(0.0, min(1.0, vec[idx] + delta))

        # 归一化（与 WebUI 保持一致）
        import numpy as np
        k_vec = [0.75, 0.70, 0.80, 0.80, 0.75, 0.75, 0.55, 0.45]
        tmp = np.array(k_vec) * np.array(vec)
        if np.sum(tmp) > 0.8:
            tmp = tmp * 0.8 / np.sum(tmp)
        return tmp.tolist()

    async def synthesize(
        self,
        text: str,
        voice_id: str,
        emotion: Optional[str] = None,
        tone: Optional[str] = None,
        **params,
    ) -> bytes:
        """用克隆音色合成语音"""
        self._ensure_model()

        profile = self._mapping.get(voice_id)
        if not profile:
            raise ValueError(f"Voice not found: {voice_id}")

        # 将 emotion/tone 映射为情感向量
        emo_vector = self._emotion_to_vector(emotion, tone)

        # 在线程池中运行同步推理
        loop = asyncio.get_event_loop()
        output_path = f"/tmp/indextts2_out_{uuid.uuid4().hex}.wav"
        try:
            await loop.run_in_executor(
                self._executor,
                lambda: self._tts.infer(
                    spk_audio_prompt=profile.audio_path,
                    text=text,
                    output_path=output_path,
                    emo_vector=emo_vector,
                    emo_alpha=params.get("emo_alpha", 0.8),
                    verbose=False,
                )
            )

            # 读取结果
            import soundfile as sf
            data, sr = sf.read(output_path)
            buf = io.BytesIO()
            sf.write(buf, data, sr, format="WAV")
            return buf.getvalue()

        finally:
            if os.path.exists(output_path):
                os.remove(output_path)

    async def clone_voice(self, audio_path: str, voice_name: str) -> VoiceProfile:
        """克隆音色——IndexTTS2 是零样本，只需保存音频路径映射。"""
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Reference audio not found: {audio_path}")

        voice_id = f"indextts2_{uuid.uuid4().hex[:8]}"

        import shutil
        ext = Path(audio_path).suffix or ".wav"
        dest_path = self.voice_dir / f"{voice_id}{ext}"
        shutil.copy2(audio_path, str(dest_path))

        profile = VoiceProfile(
            voice_id=voice_id,
            name=voice_name,
            audio_path=str(dest_path),
            metadata={"source": "indextts2", "created_at": time.time()},
        )
        self._mapping[voice_id] = profile
        self._save_mapping()
        return profile

    async def get_voices(self) -> List[VoiceInfo]:
        """获取已克隆的音色列表"""
        return [
            VoiceInfo(voice_id=p.voice_id, name=p.name, provider="indextts2", is_preset=False)
            for p in self._mapping.values()
        ]

    async def check_available(self) -> bool:
        """检查 IndexTTS2 是否可用"""
        try:
            models_dir = str(Path(__file__).parent.parent.parent / "models")
            if models_dir not in sys.path:
                sys.path.insert(0, models_dir)
            from indextts.infer_v2 import IndexTTS2
            return os.path.exists(self.model_dir)
        except ImportError:
            return False

    def _load_mapping(self):
        """从文件加载音色映射"""
        if self._mapping_file.exists():
            with open(self._mapping_file, "r", encoding="utf-8") as f:
                for item in json.load(f):
                    self._mapping[item["voice_id"]] = VoiceProfile(
                        voice_id=item["voice_id"],
                        name=item["voice_name"],
                        audio_path=item["ref_audio_path"],
                        metadata=item.get("metadata", {}),
                    )

    def _save_mapping(self):
        """保存音色映射到文件"""
        data = [
            {
                "voice_id": p.voice_id,
                "voice_name": p.name,
                "ref_audio_path": p.audio_path,
                "metadata": p.metadata,
            }
            for p in self._mapping.values()
        ]
        with open(self._mapping_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
