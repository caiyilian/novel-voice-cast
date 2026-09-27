"""对残留条目做二次 ASR 稳定性检验：区分「真泄漏」与「ASR 幻觉」。

判据：
- 两次转录差异大 → ASR 不稳定（幻觉）
- 两次转录都含 instruct 片段 → 真泄漏
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

from faster_whisper import WhisperModel  # noqa: E402
from opencc import OpenCC  # noqa: E402

ROOT = Path("E:/projects/novel-voice-cast")
OUT = ROOT / "output"

cc = OpenCC("t2s")


def n(s: str) -> str:
    return re.sub(r"[^\u4e00-\u9fff0-9a-zA-Z]", "", cc.convert(str(s or "")))


def max_instr_frag(ctrl: str, asr: str, orig: str, min_len: int = 3) -> list[str]:
    """转录里出现的、且原文没有的 instruct 连续片段。"""
    c, t, o = n(ctrl), n(asr), n(orig)
    hits: list[str] = []
    for ln in range(8, min_len - 1, -1):
        for j in range(len(c) - ln + 1):
            frag = c[j : j + ln]
            if frag in t and frag not in o and not any(frag in h for h in hits):
                hits.append(frag)
    return hits[:5]


def main() -> int:
    report = json.loads((OUT / "tts_quality_report.json").read_text(encoding="utf-8"))
    ckpt = json.loads((OUT / "streaming_tts.checkpoint.json").read_text(encoding="utf-8"))
    by = {int(v["index"]): v for v in ckpt["segments"].values()}
    bad = [r for r in report["records"] if r["verdict"] != "OK"]

    model = WhisperModel(
        str(Path("E:/huggingface_models/hub/models--Systran--faster-whisper-large-v2")
            / "snapshots/f0fe81560cb8b68660e564f55dd99207059c092e"),
        device="cuda", compute_type="float16",
    )

    rows = []
    for r in sorted(bad, key=lambda x: x["index"]):
        i = r["index"]
        p = Path(by[i]["audio_path"])
        ctrl = str(by[i].get("instruct_text", ""))
        orig = r["original"]
        if not p.is_file():
            print(f"idx {i}: 音频缺失", flush=True)
            continue
        texts = []
        for k in (1, 2):
            segs, _ = model.transcribe(str(p), language="zh", beam_size=5)
            texts.append(n("".join(s.text for s in segs)))
        t1, t2 = texts
        cov1 = len(t1) / max(1, len(n(orig)))
        stable = (t1 == t2)
        frags = max_instr_frag(ctrl, t1, orig)
        # 内容完整性：原文覆盖率（用 difflib 比稳）
        import difflib

        sim = difflib.SequenceMatcher(None, n(orig), t1).ratio()
        rows.append(
            {
                "index": i, "verdict": r["verdict"], "speaker": r["speaker"],
                "orig_len": len(n(orig)), "ctrl_len": len(n(ctrl)),
                "cov": round(cov1, 2), "sim": round(sim, 3),
                "stable": stable, "frags": frags,
                "t1": t1[:120], "t2": t2[:120],
            }
        )
        print(
            f"idx {i:>4} [{r['speaker']:<4}] {r['verdict']:<8} 覆盖{cov1:>5.2f} "
            f"相似{sim:.2f} 稳定={'Y' if stable else 'N'} 片段={len(frags)}",
            flush=True,
        )

    (OUT / "_residual_verify.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n=== 分类汇总 ===", flush=True)
    real = [x for x in rows if x["frags"] and x["stable"]]
    unstable = [x for x in rows if not x["stable"] and not x["frags"]]
    lowsim = [x for x in rows if x["stable"] and not x["frags"] and x["sim"] < 0.7]
    other = [x for x in rows if x not in real and x not in unstable and x not in lowsim]
    print(f"  真泄漏（稳定 + 含 instruct 片段）: {len(real)} -> {[x['index'] for x in real]}")
    print(f"  ASR 不稳定（两次不同，无片段）: {len(unstable)} -> {[x['index'] for x in unstable]}")
    print(f"  内容缺失（稳定但相似度低）: {len(lowsim)} -> {[x['index'] for x in lowsim]}")
    print(f"  其他: {len(other)} -> {[x['index'] for x in other]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
