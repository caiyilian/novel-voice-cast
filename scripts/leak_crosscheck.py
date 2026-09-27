"""独立交叉验证：完全绕开 LLM 判定，用客观指标扫全量，找潜在漏报。

两个独立指标：
  A. 覆盖率（转录字数/原文字数）—— 泄漏会显著 >1.35
  B. instruct 片段命中 —— 转录中出现「instruct 里有、原文里没有」的连续 4+ 字

与 LLM 判定交叉比对：
  - LLM 判 OK 但 A 或 B 异常 → 潜在漏报，需人工复核
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from opencc import OpenCC

ROOT = Path("E:/projects/novel-voice-cast")
OUT = ROOT / "output"
cc = OpenCC("t2s")

COV_WARN = 1.35
FRAG_MIN = 4


def n(s: str) -> str:
    return re.sub(r"[^\u4e00-\u9fff0-9a-zA-Z]", "", cc.convert(str(s or "")))


def instr_frags(ctrl: str, asr: str, orig: str, min_len: int = FRAG_MIN) -> list[str]:
    """转录里出现的、属于 instruct 但不属于原文的连续片段。"""
    c, t, o = n(ctrl), n(asr), n(orig)
    hits: list[str] = []
    for ln in range(8, min_len - 1, -1):
        for j in range(len(c) - ln + 1):
            frag = c[j : j + ln]
            if frag in t and frag not in o and not any(frag in h for h in hits):
                hits.append(frag)
    return hits[:4]


def main() -> int:
    report = json.loads((OUT / "tts_quality_report.json").read_text(encoding="utf-8"))
    asr = json.loads((OUT / "asr_transcripts.json").read_text(encoding="utf-8"))["transcripts"]
    ckpt = json.loads((OUT / "streaming_tts.checkpoint.json").read_text(encoding="utf-8"))
    by = {int(v["index"]): v for v in ckpt["segments"].values()}
    recs = report["records"]

    # --- 指标 A：覆盖率分布 ---
    covs = []
    for r in recs:
        i = r["index"]
        o = n(r["original"])
        t = n(asr.get(str(i), {}).get("text", "") or r["transcript"])
        if o:
            covs.append((i, len(t) / len(o), len(o)))

    import statistics

    print("=" * 68)
    print("A. 覆盖率分布（全量，独立于 LLM 判定）")
    print("=" * 68)
    vals = [c for _, c, _ in covs]
    print(f"  平均 {statistics.mean(vals):.3f}  中位 {statistics.median(vals):.3f}")
    buckets = [(0, 0.5), (0.5, 0.8), (0.8, 0.95), (0.95, 1.05), (1.05, 1.35), (1.35, 99)]
    for lo, hi in buckets:
        c = sum(1 for v in vals if lo <= v < hi)
        flag = "  ⚠ 泄漏区" if lo >= 1.35 else ""
        print(f"  {lo:.2f}-{hi:<5} {c:>5} ({c/len(vals)*100:>5.1f}%){flag}")

    # --- 指标 B：instruct 片段命中 ---
    print()
    print("=" * 68)
    print(f"B. instruct 片段命中（连续 {FRAG_MIN}+ 字，独立于 LLM 判定）")
    print("=" * 68)
    hits_all = []
    for r in recs:
        i = r["index"]
        ctrl = str(by[i].get("instruct_text", ""))
        t = asr.get(str(i), {}).get("text", "") or r["transcript"]
        fr = instr_frags(ctrl, t, r["original"])
        if fr:
            hits_all.append((i, r["verdict"], fr, len(n(r["original"]))))
    print(f"  命中片段: {len(hits_all)} 条")
    ok_but_hit = [x for x in hits_all if x[1] == "OK"]
    print(f"  其中 LLM 判 OK 的: {len(ok_but_hit)}  ← 潜在漏报，需复核")

    # --- 交叉异常 ---
    print()
    print("=" * 68)
    print("C. LLM 判 OK 但客观指标异常（潜在漏报清单）")
    print("=" * 68)
    vmap = {r["index"]: r for r in recs}
    sus = []
    for i, cov, olen in covs:
        r = vmap[i]
        if r["verdict"] != "OK":
            continue
        t = asr.get(str(i), {}).get("text", "") or r["transcript"]
        fr = instr_frags(str(by[i].get("instruct_text", "")), t, r["original"])
        if cov >= COV_WARN or fr:
            sus.append((i, round(cov, 2), olen, fr[:2]))
    print(f"  疑似 {len(sus)} 条:")
    for i, cov, olen, fr in sus[:30]:
        r = vmap[i]
        print(f"    idx {i:>4} [{r['speaker']:<4}] 覆盖{cov:>5} 原文{olen:>4}字 片段{fr}")
        print(f"      原文: {r['original'][:52]}")
        print(f"      转录: {str(r['transcript'])[:62]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
