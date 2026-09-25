"""决定性实验：对真泄漏的句子，只做「原样重试」能否解决？

背景
----
已确认 111 条 LEAK 中相当部分是【真泄漏】（instruct 被完整念出，
ASR 覆盖率 2.7-3.9x，不是误判）。同时同一条 instruct 在别的实验里
重跑多次却干净 —— 说明是随机的。

问题：随机 → 重试能否解决？重试几次能过？

方法
----
取若干条真泄漏句子（覆盖率 > 1.5），**保持 instruct 不变**，
重复生成 N 次，每次 ASR 判定是否泄漏。
统计：多少条能靠重试解决，平均需要几次。

这直接回答「是否必须改 instruct」——若重试 3 次内 90% 能过，
就不必动 instruct，角色表演层次得以保留。
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, "E:/projects/cosyvoice")
sys.path.insert(0, "E:/projects/cosyvoice/third_party/Matcha-TTS")

import torch
import torchaudio
from cosyvoice.cli.cosyvoice import AutoModel
from cosyvoice.utils import common  # noqa: F401

PROJECT_ROOT = Path("E:/projects/novel-voice-cast")
OUT = PROJECT_ROOT / "output" / "_retry_test"
OUT.mkdir(parents=True, exist_ok=True)

REPEATS = 4
REF = str(PROJECT_ROOT / "backend/data/presets/design_male_deep.wav")


def norm(s: str) -> str:
    """归一化：去标点空白。繁简差异不影响覆盖率判定，故不做转换。"""
    return re.sub(r"[^\u4e00-\u9fff0-9a-zA-Z]", "", str(s or ""))


def main() -> int:
    report = json.loads((PROJECT_ROOT / "output" / "tts_quality_report.json").read_text(encoding="utf-8"))
    segs = json.loads(
        (PROJECT_ROOT / "output" / "streaming_tts.checkpoint.json").read_text(encoding="utf-8")
    )["segments"]
    by = {int(v["index"]): v for v in segs.values()}

    # 选覆盖率 > 1.5 的真泄漏（内容明显多出来的）
    leak = []
    for r in report["records"]:
        if r["verdict"] != "LEAK":
            continue
        o, a = norm(r["original"]), norm(r["transcript"])
        if len(o) >= 10 and len(a) / len(o) > 1.5:
            leak.append(r)
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    leak = leak[:limit]
    print(f"选取真泄漏 {len(leak)} 条，每条重试 {REPEATS} 次\n", flush=True)

    model = AutoModel(
        model_dir="E:/projects/cosyvoice/pretrained_models/Fun-CosyVoice3-0.5B", fp16=True
    )
    print("CosyVoice 就绪（仅生成音频，ASR 由另一脚本处理）\n", flush=True)

    results = []
    for r in leak:
        idx = r["index"]
        instruct = f"You are a helpful assistant. {r['control']}<|endofprompt|>"
        o = norm(r["original"])
        row = {"index": idx, "speaker": r["speaker"], "original_len": len(o),
               "original": r["original"], "control": r["control"], "runs": []}
        print(f"--- idx {idx} [{r['speaker']}] 原文{len(o)}字 ---", flush=True)
        for k in range(1, REPEATS + 1):
            try:
                chunks = []
                for item in model.inference_instruct2(r["original"], instruct, REF, stream=False):
                    chunks.append(item["tts_speech"])
                speech = chunks[0] if len(chunks) == 1 else torch.cat(chunks, dim=-1)
                path = OUT / f"{idx}_r{k}.wav"
                torchaudio.save(str(path), speech, model.sample_rate)
                row["runs"].append({"run": k, "path": str(path),
                                    "dur": round(speech.shape[-1] / model.sample_rate, 2)})
                print(f"  r{k}: {row['runs'][-1]['dur']}s", flush=True)
            except Exception as exc:  # noqa: BLE001
                row["runs"].append({"run": k, "err": type(exc).__name__})
                print(f"  r{k}: 失败 {type(exc).__name__}", flush=True)
        results.append(row)

    (OUT / "gen.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n生成完成 -> {OUT}/gen.json", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
