"""TTS 质量闭环（用户设计的流程）。

流程
----
    1. ASR 转录全部音频
    2. LLM 判断每条：OK / LEAK / MISMATCH
    3. 对有问题的句子：让 LLM 改提示词 → 重新生成 → 重新 ASR → 重新判断
    4. 通过就过，不通过就再来一轮（最多 N 轮）

用法：
    # 完整跑一遍（转录 → 判断 → 修复循环）
    python scripts/tts_fix_loop.py --max-rounds 3

    # 只对已有报告里的问题句做修复循环（跳过全量转录）
    python scripts/tts_fix_loop.py --report output/tts_quality_report.json --max-rounds 3
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PY = str(PROJECT_ROOT / ".venv" / "Scripts" / "python.exe")
QUALITY = str(PROJECT_ROOT / "scripts" / "tts_quality_loop.py")


def run(args: list[str], label: str) -> int:
    print(f"\n{'='*70}\n▶ {label}\n{'='*70}", flush=True)
    cmd = [PY, "-u"] + args
    return subprocess.call(cmd, cwd=str(PROJECT_ROOT))


def read_report(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def bad_indices(report: dict) -> list[int]:
    """未通过的句子：LEAK / MISMATCH / ERROR 都算。

    ERROR（LLM 调用失败、转录失败）绝不能当作通过——否则异常会被静默吞掉，
    表面上「全部 OK」实则从未被验证过。
    """
    bad = [r["index"] for r in report["records"] if r["verdict"] != "OK"]
    bad += [item["index"] for item in report.get("untranscribed") or []]
    return sorted(set(bad))


def main() -> int:
    parser = argparse.ArgumentParser(description="TTS 质量闭环")
    parser.add_argument("--report", default=str(PROJECT_ROOT / "output" / "tts_quality_report.json"))
    parser.add_argument("--transcripts", default=str(PROJECT_ROOT / "output" / "asr_transcripts.json"))
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--skip-transcribe", action="store_true", help="跳过全量转录（报告已存在时）")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    report_path = Path(args.report)

    # ── 1. 全量 ASR 转录 ──
    if not args.skip_transcribe:
        rc = run(
            [QUALITY, "transcribe", "--transcripts", args.transcripts],
            "步骤 1/3：ASR 转录全部音频",
        )
        if rc != 0:
            print("转录失败", flush=True)
            return rc

    # ── 2. 全量 LLM 判断 ──
    rc = run(
        [QUALITY, "audit", "--report", str(report_path), "--transcripts", args.transcripts,
         "--workers", str(args.workers)],
        "步骤 2/3：LLM 判断全部句子",
    )
    if rc != 0:
        print("判定失败", flush=True)
        return rc

    report = read_report(report_path)
    counts = report["counts"]
    print(f"\n首轮判定：OK={counts.get('OK', 0)}  LEAK={counts.get('LEAK', 0)}  "
          f"MISMATCH={counts.get('MISMATCH', 0)}", flush=True)

    # ── 3. 修复循环 ──
    for round_no in range(1, args.max_rounds + 1):
        todo = bad_indices(report)
        if not todo:
            print(f"\n✓ 全部通过，无需修复", flush=True)
            break

        print(f"\n{'#'*70}\n# 第 {round_no}/{args.max_rounds} 轮修复：{len(todo)} 条\n{'#'*70}", flush=True)

        # 3a. 让 LLM 改提示词（--regenerate 会立即重生成音频）
        plan_path = PROJECT_ROOT / "output" / f"repair_plan_r{round_no}.json"
        rc = run(
            [QUALITY, "repair", "--report", str(report_path), "--plan", str(plan_path),
             "--regenerate"],
            f"第 {round_no} 轮：LLM 改提示词 + 重生成",
        )
        if rc != 0:
            print("修复失败，中断", flush=True)
            return rc

        # 3b. 重新 ASR（只跑修过的句子）
        rc = run(
            [QUALITY, "transcribe", "--transcripts", args.transcripts,
             "--indices", ",".join(str(i) for i in todo), "--force"],
            f"第 {round_no} 轮：重新 ASR 复检",
        )
        if rc != 0:
            print("复检转录失败，中断", flush=True)
            return rc

        # 3c. 重新判断（只审修过的句子）
        rc = run(
            [QUALITY, "audit", "--report", str(report_path), "--transcripts", args.transcripts,
             "--indices", ",".join(str(i) for i in todo), "--workers", str(args.workers)],
            f"第 {round_no} 轮：重新判断",
        )
        if rc != 0:
            print("复检判定失败，中断", flush=True)
            return rc

        report = read_report(report_path)
        remain = bad_indices(report)
        print(f"\n第 {round_no} 轮结束：仍有问题 {len(remain)} 条"
              f"（本轮修复 {len(todo) - len(remain)} 条）", flush=True)

    # ── 汇总 ──
    report = read_report(report_path)
    counts = report["counts"]
    print(f"\n{'='*70}\n最终结果\n{'='*70}", flush=True)
    print(f"  OK={counts.get('OK', 0)}  LEAK={counts.get('LEAK', 0)}  "
          f"MISMATCH={counts.get('MISMATCH', 0)}", flush=True)
    remain = bad_indices(report)
    if remain:
        print(f"  仍未通过 {len(remain)} 条: {remain[:30]}{'...' if len(remain) > 30 else ''}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
