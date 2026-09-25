"""验证：采样参数（top_p / top_k）能否减少控制词泄漏。

关键设计
--------
必须用【原始散文 instruct】（已知会泄漏的那批），而不是已改写过的元指令，
否则分不清是「采样参数起作用」还是「提示词已改」。

对照组（同一批散文 instruct）：
  - 默认 top_p=0.8, top_k=25
  - 收紧 top_p=0.6, top_k=10
  - 很紧 top_p=0.3, top_k=3
  - 贪心 top_p=0.01, top_k=1

每组重复 3 次，用 ASR 判定是否泄漏（出现导演术语即算泄漏）。
"""

import json
import os
import sys

sys.path.insert(0, "E:/projects/cosyvoice")
sys.path.insert(0, "E:/projects/cosyvoice/third_party/Matcha-TTS")

import torch
import torchaudio
from cosyvoice.cli.cosyvoice import AutoModel
from cosyvoice.utils import common

OUT = "E:/projects/novel-voice-cast/output/_sampling_test2"
os.makedirs(OUT, exist_ok=True)

# 原始散文 instruct（来自 performance_directions.json，已知会泄漏）
CASES = [
    (20, "不过，我也因此拿到上等的皮草啊，我会再来的。",
     "前半随口带过，重音轻落「上等皮草」；「啊」后顿半拍、轻吸一气，随即语速放稳，重音落「再来」，音量平和，四字短促笃定收尾。"),
    (21, "结束一如往常的对话，离开深山里的村落已过了五个小时。太阳升起后就立刻动身，下山来到这片草原时已过了中午。",
     "轻吸一口气从容起句，语速不疾不徐、音量平稳；句号处停半拍标出时间跳跃；「五个小时」放缓匀开让时光流过；「立刻动身」稍提即收；末句「过了中午」沉稳放平，如马车匀速行进。"),
    (525, "不过，这么一来正好可以避寒取暖。",
     "缓吸定气、音量放平；不过轻转半档、轻咬不抬调；避寒取暖一带而过，取暖前轻屏半拍，紧字放缓，放软尾音轻垂，收定暖意，句末留半拍静默交棒五寸。"),
]

VARIANTS = [
    ("A_default_p0.8_k25", 0.8, 25),
    ("B_tight_p0.6_k10", 0.6, 10),
    ("C_verytight_p0.3_k3", 0.3, 3),
    ("D_greedy_p0.01_k1", 0.01, 1),
]

_ORIG = common.ras_sampling


def make_patched(top_p: float, top_k: int):
    def patched(weighted_scores, decoded_tokens, sampling, win_size=10, tau_r=0.1):
        return _ORIG(weighted_scores, decoded_tokens, sampling,
                     top_p=top_p, top_k=top_k, win_size=win_size, tau_r=tau_r)
    return patched


model = AutoModel(
    model_dir="E:/projects/cosyvoice/pretrained_models/Fun-CosyVoice3-0.5B", fp16=True
)
REF = "E:/projects/novel-voice-cast/backend/data/presets/design_male_deep.wav"
REPEATS = 3

manifest = []
for tag, top_p, top_k in VARIANTS:
    model.model.llm.sampling = make_patched(top_p, top_k)
    print(f"\n--- {tag} (top_p={top_p}, top_k={top_k}) ---", flush=True)
    for idx, text, instruct in CASES:
        full = f"You are a helpful assistant. {instruct}<|endofprompt|>"
        for r in range(1, REPEATS + 1):
            try:
                chunks = []
                for item in model.inference_instruct2(text, full, REF, stream=False):
                    chunks.append(item["tts_speech"])
                speech = chunks[0] if len(chunks) == 1 else torch.cat(chunks, dim=-1)
                dur = round(speech.shape[-1] / model.sample_rate, 2)
                path = f"{OUT}/{tag}_{idx}_r{r}.wav"
                torchaudio.save(path, speech, model.sample_rate)
                manifest.append({"tag": tag, "idx": idx, "run": r, "dur": dur,
                                 "expected": round(len(text) / 4.5, 1), "path": path})
                print(f"  idx {idx:>3} r{r}: {dur}s (预期 {len(text)/4.5:.1f}s)", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"  idx {idx:>3} r{r}: 失败 {type(exc).__name__}", flush=True)

json.dump(manifest, open(f"{OUT}/manifest.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
print(f"\n音频 -> {OUT}", flush=True)
