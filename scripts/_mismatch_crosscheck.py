"""对 MISMATCH 条目做多参数 ASR 交叉验证：区分「音频问题」与「ASR 误判」。

方法：用不同解码参数转录同一音频，若结果与原文的相似度大幅波动，
且某些参数下能还原原文 → ASR 侧问题（音频本身正确）。
"""

from __future__ import annotations

import difflib
import json
import os
import re
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

from faster_whisper import WhisperModel  # noqa: E402
from opencc import OpenCC  # noqa: E402

ROOT = Path("E:/projects/novel-voice-cast")
OUT = ROOT / "output"
cc = OpenCC("t2s")


def n(s: str) -> str:
    return re.sub(r"[^\u4e00-\u9fff0-9a-zA-Z]", "", cc.convert(str(s or "")))


def main() -> int:
    report = json.loads((OUT / "tts_quality_report.json").read_text(encoding="utf-8"))
    ckpt = json.loads((OUT / "streaming_tts.checkpoint.json").read_text(encoding="utf-8"))
    by = {int(v["index"]): v for v in ckpt["segments"].values()}
    targets = [r for r in report["records"] if r["verdict"] == "MISMATCH"]

    model = WhisperModel(
        str(Path("E:/huggingface_models/hub/models--Systran--faster-whisper-large-v2")
            / "snapshots/f0fe81560cb8b68660e564f55dd99207059c092e"),
        device="cuda", compute_type="float16",
    )

    configs = [
        ("beam5", dict(beam_size=5)),
        ("beam1", dict(beam_size=1)),
        ("beam5_noprev", dict(beam_size=5, condition_on_previous_text=False)),
        ("beam10", dict(beam_size=10)),
    ]

    rows = []
    for r in targets:
        i = r["index"]
        orig = n(r["original"])
        p = Path(by[i]["audio_path"])
        if not p.is_file():
            continue
        print(f"\n=== idx {i} [{r['speaker']}] 原文: {r['original']} ===", flush=True)
        outs = {}
        for name, kw in configs:
            segs, _ = model.transcribe(str(p), language="zh", **kw)
            t = n("".join(s.text for s in segs))
            sim = difflib.SequenceMatcher(None, orig, t).ratio()
            outs[name] = {"text": t, "sim": round(sim, 3)}
            print(f"  {name:<14} 相似{sim:.2f}  {t[:60]}", flush=True)
        best = max(v["sim"] for v in outs.values())
        rows.append({"index": i, "speaker": r["speaker"], "original": r["original"],
                     "best_sim": best, "outs": outs})

    print("\n=== 汇总：各参数下的最高相似度 ===", flush=True)
    for x in rows:
        verdict = "ASR 侧问题（多参数下可还原）" if x["best_sim"] >= 0.85 else "疑似音频问题"
        print(f"  idx {x['index']:>4} 最高相似 {x['best_sim']:.2f}  → {verdict}", flush=True)

    (OUT / "_mismatch_crosscheck.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
