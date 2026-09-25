"""验证：降低采样随机性（top_p / top_k）能否减少控制词泄漏。

背景
----
CosyVoice 没有 temperature 参数，真正的采样控制是 `ras_sampling`：
    ras_sampling(scores, tokens, sampling, top_p=0.8, top_k=25, win_size=10, tau_r=0.1)
      → nucleus_sampling(top_p, top_k) 做 top-p/top-k 截断，再 multinomial 随机抽取
      → 若检测到重复，退回 random_sampling

推论：把 top_p / top_k 调小 → 候选集变小 → 采样更确定 → 泄漏概率可能下降。

本实验对同一批已知泄漏的句子，用 4 组采样参数各生成 2 次，统计泄漏率。

判定：ASR 转录中是否出现导演术语，或覆盖率是否异常。
"""

import json
import os
import sys
import time

sys.path.insert(0, "E:/projects/cosyvoice")
sys.path.insert(0, "E:/projects/cosyvoice/third_party/Matcha-TTS")

import torch
import torchaudio
from cosyvoice.cli.cosyvoice import AutoModel
from cosyvoice.utils import common

OUT = "E:/projects/novel-voice-cast/output/_sampling_test"
os.makedirs(OUT, exist_ok=True)

# 已知会泄漏的句子（来自 tts_quality_report.json）
CASES = [
    (17, "这是最后一件了吧？", "请以朴实温暖、随口确认的语气说这句话。语速从容放缓，开头两字后稍作停顿，中间四字稳稳落地，带一点清点的分量，句末自然轻轻上扬，化作家常的疑问尾音，不推销、不拿腔。"),
    (20, "不过，我也因此拿到上等的皮草啊，我会再来的。", "请用随口带过、结尾笃定的语气说这句话。前半句语速轻快、重音轻落；句中顿半拍并轻吸一口气，随后语速放稳、音量平和，重音移向句尾，句末短促有力、笃定收尾。"),
    (21, "结束一如往常的对话，离开深山里的村落已过了五个小时。太阳升起后就立刻动身，下山来到这片草原时已过了中午。", "请以从容平稳的语气说这句话。语速不疾不徐，音量平稳；句号处稍停半拍；五个小时放缓匀开；立刻动身稍提即收；末句沉稳放平，如马车匀速行进。"),
]

# 采样参数组：(标签, top_p, top_k)
VARIANTS = [
    ("默认_top_p0.8_k25", 0.8, 25),
    ("收紧_top_p0.6_k10", 0.6, 10),
    ("很紧_top_p0.3_k3", 0.3, 3),
    ("贪心_top_p0.0_k1", 0.01, 1),
]

_ORIG_RAS = common.ras_sampling


def make_patched(top_p: float, top_k: int):
    """返回一个把 top_p/top_k 固定为指定值的 ras_sampling。"""

    def patched(weighted_scores, decoded_tokens, sampling, win_size=10, tau_r=0.1):
        return _ORIG_RAS(
            weighted_scores, decoded_tokens, sampling,
            top_p=top_p, top_k=top_k, win_size=win_size, tau_r=tau_r,
        )

    return patched


model = AutoModel(
    model_dir="E:/projects/cosyvoice/pretrained_models/Fun-CosyVoice3-0.5B", fp16=True
)
REF = "E:/projects/novel-voice-cast/backend/data/presets/design_male_deep.wav"
REPEATS = 2

results = []
for tag, top_p, top_k in VARIANTS:
    # 运行时替换采样函数
    model.model.llm.sampling = make_patched(top_p, top_k)
    print(f"\n--- {tag} (top_p={top_p}, top_k={top_k}) ---", flush=True)
    for idx, text, instruct in CASES:
        full = f"You are a helpful assistant. {instruct}<|endofprompt|>"
        durations = []
        for r in range(1, REPEATS + 1):
            try:
                chunks = []
                for item in model.inference_instruct2(text, full, REF, stream=False):
                    chunks.append(item["tts_speech"])
                speech = chunks[0] if len(chunks) == 1 else torch.cat(chunks, dim=-1)
                dur = speech.shape[-1] / model.sample_rate
                durations.append(round(dur, 2))
                torchaudio.save(f"{OUT}/{tag}_{idx}_r{r}.wav", speech, model.sample_rate)
            except Exception as exc:  # noqa: BLE001
                durations.append(f"ERR:{type(exc).__name__}")
        exp = round(len(text) / 4.5, 1)
        print(f"  idx {idx:>3} 预期{exp}s 实际 {durations}", flush=True)
        results.append({"tag": tag, "idx": idx, "expected": exp, "durations": durations})

json.dump(results, open(f"{OUT}/durations.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(f"\n音频 -> {OUT}", flush=True)
