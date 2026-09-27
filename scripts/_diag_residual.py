"""诊断残留泄漏的两个候选假设：

H1「超短正文 + 长 instruct」：正文 <20 字时，instruct 的绝对长度仍会压倒正文。
H2「位置描述里嵌正文原字」：即使无 3+ 字片段，1-2 字的原字仍会成为续写引导。
"""

from __future__ import annotations

import json
import re
import statistics
from pathlib import Path

from opencc import OpenCC

ROOT = Path("E:/projects/novel-voice-cast")
cc = OpenCC("t2s")


def n(s: str) -> str:
    return re.sub(r"[^\u4e00-\u9fff0-9a-zA-Z]", "", cc.convert(str(s or "")))


def main() -> int:
    report = json.loads((ROOT / "output" / "tts_quality_report.json").read_text(encoding="utf-8"))
    ckpt = json.loads((ROOT / "output" / "streaming_tts.checkpoint.json").read_text(encoding="utf-8"))
    by = {int(v["index"]): v for v in ckpt["segments"].values()}
    recs = report["records"]

    def ilen(i: int) -> int:
        return len(n(by[i].get("instruct_text", "")))

    def olen(r: dict) -> int:
        return len(n(r["original"]))

    leak = [r for r in recs if r["verdict"] == "LEAK"]
    ok = [r for r in recs if r["verdict"] == "OK"]

    print("=" * 70)
    print("H1 超短正文（<20 字）下，instruct 长度的泄漏率")
    print("=" * 70)
    short_leak = [r for r in leak if olen(r) < 20]
    short_ok = [r for r in ok if olen(r) < 20]
    tot = len(short_leak) + len(short_ok)
    print(f"全量短正文（<20字）: {tot} 条，泄漏 {len(short_leak)} ({len(short_leak)/max(1,tot)*100:.1f}%)")
    print()
    print(f"{'instruct 长度':<16}{'条数':>6}{'泄漏':>6}{'泄漏率':>9}")
    for lo, hi in [(0, 30), (30, 40), (40, 50), (50, 200)]:
        sl = [r for r in short_leak if lo <= ilen(r["index"]) < hi]
        so = [r for r in short_ok if lo <= ilen(r["index"]) < hi]
        t = len(sl) + len(so)
        if t:
            print(f"  {lo}-{hi:<12}{t:>6}{len(sl):>6}{len(sl)/t*100:>8.1f}%")

    print()
    print("对照：长正文（>=20 字）")
    lg_leak = [r for r in leak if olen(r) >= 20]
    lg_ok = [r for r in ok if olen(r) >= 20]
    t2 = len(lg_leak) + len(lg_ok)
    print(f"  长正文 {t2} 条，泄漏 {len(lg_leak)} ({len(lg_leak)/max(1,t2)*100:.1f}%)")

    print()
    print("=" * 70)
    print("H2 instruct 与正文的字符重叠（1-2 字级）")
    print("=" * 70)

    def char_overlap(ctrl: str, orig: str) -> tuple[int, int]:
        """返回（重叠的 2-gram 数, 重叠的单字数）"""
        c, o = n(ctrl), n(orig)
        bigrams = {o[i : i + 2] for i in range(len(o) - 1)}
        hit2 = sum(1 for i in range(len(c) - 1) if c[i : i + 2] in bigrams)
        oset = set(o)
        hit1 = sum(1 for ch in set(c) if ch in oset)
        return hit2, hit1

    print(f"{'组':<10}{'条数':>6}{'平均2-gram重叠':>16}{'平均单字重叠':>14}")
    for name, grp in (("LEAK", leak), ("OK", ok)):
        b2 = [char_overlap(str(by[r["index"]].get("instruct_text", "")), r["original"]) for r in grp]
        a2 = statistics.mean(x[0] for x in b2) if b2 else 0
        a1 = statistics.mean(x[1] for x in b2) if b2 else 0
        print(f"  {name:<8}{len(grp):>6}{a2:>16.2f}{a1:>14.1f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
