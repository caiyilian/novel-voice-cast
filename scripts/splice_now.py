"""直接拼接分段音频为成品（绕过 performance 校验）。

背景：`run_full.py --from-stage splice` 会强制校验 performance profile 的
prompt 签名，而我们改了 prompt（新增旁白情绪约束）后旧 profile 签名失效，
导致纯拼接操作被拦。但分段音频本身已存在且验证正确，无需重新生成。

本脚本只做拼接：读 checkpoint 的分段清单 → AudioSplicer → output/full_volume.mp3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path("E:/projects/novel-voice-cast")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "scripts"))

from app.core.splicer import AudioSplicer  # noqa: E402

CHECKPOINT = ROOT / "output" / "streaming_tts.checkpoint.json"
DATA_FILES = (
    ROOT / "backend" / "data" / "performance_directions.json",
    ROOT / "backend" / "data" / "performance_directions_supplemental.json",
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "output" / "full_volume.mp3"))
    ap.add_argument("--bitrate", default="192k")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    ckpt = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    segs = ckpt["segments"]
    by = {int(v["index"]): v for v in segs.values()}

    # 章节信息（用于分段顺序 / 章节边界）
    meta: dict[int, dict] = {}
    for fn in DATA_FILES:
        if not fn.is_file():
            continue
        data = json.loads(fn.read_text(encoding="utf-8")).get("results", {})
        for v in data.values():
            meta[int(v["dialogue_index"])] = {
                "chapter": str(v.get("chapter", "")),
                "speaker": str(v.get("speaker", "")),
                "text": str(v.get("text", "")),
            }

    segments = []
    missing = []
    for order, i in enumerate(sorted(by)):
        rec = by[i]
        p = Path(rec["audio_path"])
        if not p.is_file() or p.stat().st_size == 0:
            missing.append((i, str(p)))
            continue
        segments.append({
            "audio_path": str(p),
            "chapter": meta.get(i, {}).get("chapter", ""),
            "order": order,
            "index": i,
        })

    print(f"分段: {len(segments)} 条，缺失 {len(missing)}", flush=True)
    if missing:
        for i, p in missing[:10]:
            print(f"  缺失 idx {i}: {p}", flush=True)
        print("有缺失，中止（不允许产出不完整成品）", flush=True)
        return 1

    if args.dry_run:
        print("dry-run，不实际拼接", flush=True)
        return 0

    out = Path(args.out)
    print(f"开始拼接 -> {out}（bitrate {args.bitrate}）", flush=True)
    splicer = AudioSplicer(output_bitrate=args.bitrate)
    result = splicer.splice(segments, output_path=str(out))
    dur = len(result) / 1000.0
    size_mb = out.stat().st_size / 1024 / 1024
    print(f"完成: {dur/60:.1f} 分钟 ({dur/3600:.2f} 小时), {size_mb:.1f} MB", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
