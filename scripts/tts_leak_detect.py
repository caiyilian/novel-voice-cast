"""TTS 泄漏检测器：基于字符级相似度，不依赖硬编码术语表。

设计原则
--------
之前用「导演术语命中」初筛有两个盲区：
  1. 只覆盖硬编码的词，换种措辞就漏（假阴性）
  2. ASR 误识别出的近音字可能凑巧命中（假阳性）

本模块改用**字符级比对**（difflib.SequenceMatcher），直接回答一个更本质的问题：
「ASR 转录里有多少内容是不属于原文的？」

分类逻辑：
  - extra_ratio  = 转录中「非原文部分」占原文长度的比例
  - missing_ratio = 原文中「未被转录覆盖」的比例
  - 泄漏的典型特征是 extra_ratio 显著 > 0（念了原文没有的内容）

判定：
  - clean     : extra 极低（仅 ASR 识别误差）
  - suspect   : extra 中等，需 LLM 精判
  - leak      : extra 高（明确念了额外内容）

用法：
    python scripts/tts_leak_detect.py scan --report output/leak_scan.json
    python scripts/tts_leak_detect.py scan --explain 525   # 看单条细节
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

DEFAULT_TTS_CHECKPOINT = PROJECT_ROOT / "output" / "streaming_tts.checkpoint.json"
DEFAULT_TRANSCRIPTS = PROJECT_ROOT / "output" / "asr_transcripts.json"
DEFAULT_DIRECTIONS = PROJECT_ROOT / "backend" / "data" / "performance_directions.json"
DEFAULT_SUPPLEMENTAL = PROJECT_ROOT / "backend" / "data" / "performance_directions_supplemental.json"
DEFAULT_REPORT = PROJECT_ROOT / "output" / "leak_scan.json"

# 判定阈值（extra_ratio）
CLEAN_MAX = 0.15
SUSPECT_MAX = 0.45


def _norm(text: str) -> str:
    """归一化：繁转简、去标点空白、统一异体字，便于字符级比对。"""
    from opencc import OpenCC

    cc = OpenCC("t2s")
    value = cc.convert(str(text or ""))
    value = re.sub(r"[^\u4e00-\u9fff0-9a-zA-Z]", "", value)
    return value


def compare(original: str, transcript: str) -> dict[str, Any]:
    """比对原文与转录，返回覆盖率与额外内容指标。"""
    a, b = _norm(original), _norm(transcript)
    if not a:
        return {"original_len": 0, "transcript_len": len(b), "extra_ratio": 0.0,
                "missing_ratio": 0.0, "similarity": 1.0, "extra_text": "", "missing_text": ""}
    if not b:
        return {"original_len": len(a), "transcript_len": 0, "extra_ratio": 0.0,
                "missing_ratio": 1.0, "similarity": 0.0, "extra_text": "", "missing_text": a}

    matcher = SequenceMatcher(None, a, b, autojunk=False)
    matched = sum(block.size for block in matcher.get_matching_blocks())

    # 转录中不属于原文的片段（泄漏嫌疑）
    extra_parts: list[str] = []
    missing_parts: list[str] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag in ("replace", "insert"):
            extra_parts.append(b[j1:j2])
        if tag in ("replace", "delete"):
            missing_parts.append(a[i1:i2])

    return {
        "original_len": len(a),
        "transcript_len": len(b),
        "extra_ratio": round(sum(len(p) for p in extra_parts) / len(a), 4),
        "missing_ratio": round(sum(len(p) for p in missing_parts) / len(a), 4),
        "similarity": round(matched / len(a), 4),
        "extra_text": "|".join(p for p in extra_parts if p)[:200],
        "missing_text": "|".join(p for p in missing_parts if p)[:200],
    }


def classify(metrics: dict[str, Any], duration_ratio: float | None = None) -> str:
    """按指标判定等级。"""
    extra = metrics["extra_ratio"]
    missing = metrics["missing_ratio"]
    # 时长显著超长也是强信号（念了额外内容必然更久）
    if duration_ratio is not None and duration_ratio >= 2.2:
        return "leak"
    if extra >= SUSPECT_MAX:
        return "leak"
    if extra >= CLEAN_MAX or missing >= 0.35:
        return "suspect"
    return "clean"


def load_rows(
    tts_path: Path, transcripts_path: Path, directions_paths: list[Path]
) -> list[dict[str, Any]]:
    import wave

    segments = json.loads(tts_path.read_text(encoding="utf-8"))["segments"]
    transcripts = json.loads(transcripts_path.read_text(encoding="utf-8"))["transcripts"]
    directions: dict[int, dict[str, Any]] = {}
    for path in directions_paths:
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        for value in (payload.get("results") or {}).values():
            directions[int(value["dialogue_index"])] = value

    rows: list[dict[str, Any]] = []
    for key, rec in segments.items():
        idx = int(rec["index"])
        audio = Path(rec.get("audio_path", ""))
        if not audio.is_file():
            continue
        try:
            with wave.open(str(audio)) as handle:
                duration = handle.getnframes() / handle.getframerate()
        except Exception:  # noqa: BLE001
            duration = 0.0
        text = str(directions.get(idx, {}).get("text") or "")
        asr = str(transcripts.get(str(idx), {}).get("text") or "")
        metrics = compare(text, asr)
        # 中文正常语速约 4.5 字/秒
        semantic = len(_norm(text))
        expected = semantic / 4.5 if semantic else 0.0
        dur_ratio = round(duration / expected, 2) if expected > 0.5 else None
        rows.append(
            {
                "index": idx,
                "speaker": rec.get("speaker"),
                "text": text,
                "instruct": str(rec.get("instruct_text") or ""),
                "asr": asr,
                "duration": round(duration, 2),
                "duration_ratio": dur_ratio,
                **metrics,
                "level": classify(metrics, dur_ratio),
            }
        )
    rows.sort(key=lambda r: -r["extra_ratio"])
    return rows


def cmd_scan(args: argparse.Namespace) -> int:
    rows = load_rows(
        Path(args.tts_checkpoint),
        Path(args.transcripts),
        [Path(args.directions), Path(args.supplemental)],
    )
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["level"]] = counts.get(row["level"], 0) + 1

    total = len(rows)
    print(f"总 {total} 条")
    for level in ("clean", "suspect", "leak"):
        n = counts.get(level, 0)
        print(f"  {level:<8} {n:>5} ({n/total*100:.2f}%)")

    report = {
        "total": total,
        "counts": counts,
        "thresholds": {"clean_max": CLEAN_MAX, "suspect_max": SUSPECT_MAX},
        "rows": rows,
    }
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n报告 -> {args.report}")

    print("\n=== extra_ratio 最高的 15 条 ===")
    for row in rows[:15]:
        print(f"  idx {row['index']:>4} [{row['speaker']}] {row['level']:<7} "
              f"extra={row['extra_ratio']:.2f} dur={row['duration']}s")
        print(f"      原文: {row['text'][:45]}")
        print(f"      额外: {row['extra_text'][:80]}")
    return 0


def cmd_explain(args: argparse.Namespace) -> int:
    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    target = next((r for r in report["rows"] if r["index"] == args.explain), None)
    if not target:
        print(f"报告里没有 idx={args.explain}")
        return 1
    for key in ("index", "speaker", "level", "extra_ratio", "missing_ratio", "similarity",
                "duration", "duration_ratio"):
        print(f"  {key}: {target.get(key)}")
    print(f"\n原文:\n  {target['text']}")
    print(f"\ninstruct:\n  {target['instruct']}")
    print(f"\nASR:\n  {target['asr']}")
    print(f"\n额外内容:\n  {target['extra_text']}")
    print(f"\n缺失内容:\n  {target['missing_text']}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="TTS 泄漏检测（字符级比对）")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("scan", help="全量扫描并出报告")
    p.add_argument("--tts-checkpoint", default=str(DEFAULT_TTS_CHECKPOINT))
    p.add_argument("--transcripts", default=str(DEFAULT_TRANSCRIPTS))
    p.add_argument("--directions", default=str(DEFAULT_DIRECTIONS))
    p.add_argument("--supplemental", default=str(DEFAULT_SUPPLEMENTAL))
    p.add_argument("--report", default=str(DEFAULT_REPORT))
    p.set_defaults(func=cmd_scan)

    p2 = sub.add_parser("explain", help="看单条详情")
    p2.add_argument("explain", type=int, help="dialogue index")
    p2.add_argument("--report", default=str(DEFAULT_REPORT))
    p2.set_defaults(func=cmd_explain)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
