"""复核可疑句：对判定为 LEAK/MISMATCH 的句子重新 ASR 一次。

目的
----
单次 ASR 可能因音频较长/语速较快而漏识别，导致 MISMATCH 误判。
对可疑句重跑一次 ASR，取两次结果中「更完整」的那份，再让 LLM 复核。

用法：
    python scripts/tts_recheck.py --report output/tts_quality_report.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

DEFAULT_WHISPER = Path(
    os.environ.get(
        "WHISPER_MODEL_DIR",
        "E:/huggingface_models/hub/models--Systran--faster-whisper-large-v2"
        "/snapshots/f0fe81560cb8b68660e564f55dd99207059c092e",
    )
)


def main() -> int:
    parser = argparse.ArgumentParser(description="对可疑句重新 ASR 复核")
    parser.add_argument("--report", default=str(PROJECT_ROOT / "output" / "tts_quality_report.json"))
    parser.add_argument("--tts-checkpoint", default=str(PROJECT_ROOT / "output" / "streaming_tts.checkpoint.json"))
    parser.add_argument("--out", default=str(PROJECT_ROOT / "output" / "tts_recheck.json"))
    parser.add_argument("--whisper-model", default=str(DEFAULT_WHISPER))
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    bad = [r for r in report["records"] if r["verdict"] in ("LEAK", "MISMATCH")]
    if args.limit:
        bad = bad[: args.limit]
    print(f"待复核: {len(bad)} 条", flush=True)

    segments = json.loads(Path(args.tts_checkpoint).read_text(encoding="utf-8"))["segments"]
    by_index = {int(v["index"]): v for v in segments.values()}

    from faster_whisper import WhisperModel

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    model = WhisperModel(str(args.whisper_model), device="cuda", compute_type="float16")
    print("Whisper 就绪", flush=True)

    out: dict[str, object] = {"rechecked": {}}
    for n, rec in enumerate(bad, 1):
        idx = rec["index"]
        seg = by_index.get(idx)
        if not seg:
            continue
        audio = Path(seg["audio_path"])
        if not audio.is_file():
            continue
        try:
            # 用更大 beam 与更宽松的参数重跑，尽量拿到完整转录
            pieces, info = model.transcribe(
                str(audio),
                language="zh",
                beam_size=10,
                vad_filter=False,
                condition_on_previous_text=False,
            )
            text = "".join(p.text for p in pieces).strip()
            out["rechecked"][str(idx)] = {
                "text": text,
                "duration": round(float(getattr(info, "duration", 0.0) or 0.0), 2),
                "first_text": rec["transcript"],
            }
        except Exception as exc:  # noqa: BLE001
            out["rechecked"][str(idx)] = {"error": str(exc)[:150]}
        if n % 20 == 0:
            Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"  {n}/{len(bad)}", flush=True)

    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"完成 -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
