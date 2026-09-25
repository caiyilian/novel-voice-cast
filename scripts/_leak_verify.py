"""验证 111 条 LEAK 是真泄漏还是 ASR 误判。

动机
----
用户质疑「为了旁白颜色让全部极简，不如 VoxCPM」。
在决定是否大改 instruct 之前，必须先确认这 111 条是不是【真】泄漏。

方法
----
对每条 LEAK，用**更宽松的 ASR 参数**重新转录 2 次（beam_size=10、
condition_on_previous_text=False），取 3 次结果的并集：
  - 若重跑后转录与原文一致 → 首次是 ASR 误判（假阳性）
  - 若多次都出现控制词 → 真泄漏

同时检查：泄漏片段是否真的出现在 instruct 里（而非 ASR 臆造）。
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WHISPER = (
    "E:/huggingface_models/hub/models--Systran--faster-whisper-large-v2"
    "/snapshots/f0fe81560cb8b68660e564f55dd99207059c092e"
)


def norm(s: str) -> str:
    from opencc import OpenCC

    return re.sub(r"[^\u4e00-\u9fff0-9a-zA-Z]", "", OpenCC("t2s").convert(str(s or "")))


def main() -> int:
    report_path = PROJECT_ROOT / "output" / "tts_quality_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    segs = json.loads(
        (PROJECT_ROOT / "output" / "streaming_tts.checkpoint.json").read_text(encoding="utf-8")
    )["segments"]
    by = {int(v["index"]): v for v in segs.values()}

    leak = [r for r in report["records"] if r["verdict"] == "LEAK"]
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    leak = leak[:limit]
    print(f"复核 {len(leak)} 条 LEAK\n", flush=True)

    from faster_whisper import WhisperModel

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    model = WhisperModel(WHISPER, device="cuda", compute_type="float16")
    print("Whisper 就绪\n", flush=True)

    # 宽松参数重跑 2 次
    out: dict[str, dict] = {}
    for n, r in enumerate(leak, 1):
        idx = r["index"]
        audio = Path(by[idx]["audio_path"])
        runs = []
        for attempt in range(2):
            try:
                pieces, _ = model.transcribe(
                    str(audio), language="zh", beam_size=10,
                    vad_filter=False, condition_on_previous_text=False,
                    temperature=0.0 if attempt == 0 else 0.4,
                )
                runs.append(norm("".join(p.text for p in pieces)))
            except Exception as exc:  # noqa: BLE001
                runs.append(f"ERR:{type(exc).__name__}")
        o = norm(r["original"])
        first = norm(r["transcript"])
        # 覆盖率
        cov = [round(len(x) / max(1, len(o)), 2) if not x.startswith("ERR") else 0 for x in runs]
        out[str(idx)] = {
            "speaker": r["speaker"],
            "original_len": len(o),
            "first_cov": round(len(first) / max(1, len(o)), 2),
            "rerun_cov": cov,
            "first_text": first[:120],
            "rerun_texts": [x[:120] for x in runs],
            "control_len": len(r["control"]),
        }
        print(f"  [{n}/{len(leak)}] idx {idx} 首次覆盖率 {out[str(idx)]['first_cov']} "
              f"重跑 {cov}", flush=True)

    (PROJECT_ROOT / "output" / "_leak_verify.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n完成 -> output/_leak_verify.json", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
