"""把 output/segments 的 wav 统一为 PCM_16，并按当前 task 元数据回填 streaming TTS checkpoint。

用途：分布式脚本在服务器上生成的 segments 不走 run_full 的 bookkeeping，
直接跑 run_full --to-stage splice 会因 checkpoint 为空而**重生成全部**。
本脚本把已就绪的 segments 元数据（fingerprint + wav 指纹）写进 checkpoint，
run_full 的 reusable_tts_entry 即判定可复用，从而跳过 TTS 直接进入 splice。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def normalize_wav_to_pcm16(path: Path) -> bool:
    import soundfile as sf

    try:
        info = sf.info(str(path))
    except Exception:
        return False
    if info.subtype == "PCM_16":
        return False
    data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    sf.write(str(path), data, sr, subtype="PCM_16")
    return True


def main() -> int:
    import yaml
    from scripts import run_full as rf

    config = yaml.safe_load(open(ROOT / "config" / "config.yaml", encoding="utf-8"))
    segments = rf.output_dir(config) / "segments"

    # 1) 统一 PCM_16
    converted = 0
    for f in sorted(segments.glob("*.wav")):
        if normalize_wav_to_pcm16(f):
            converted += 1
    print(f"格式归一化: {converted} 个转为 PCM_16", flush=True)

    # 2) 回填 checkpoint
    dialogues, _c, _n = rf.step_parse(config)
    gender_results = rf.read_json(rf.ROOT / "backend" / "data" / "gender_results.json", {})
    # 表演指导分主组 + 补充组两个文件，必须合并（与 pipeline 的 load_partial_performance_results 一致），
    # 否则补充组那批句子的 performance 为空、fingerprint 不匹配而被重生成。
    perf_results: dict[str, object] = {}
    for name in ("performance_directions.json", "performance_directions_supplemental.json"):
        payload_file = rf.read_json(rf.ROOT / "backend" / "data" / name, {})
        part = payload_file.get("results", {}) if isinstance(payload_file, dict) else {}
        if isinstance(part, dict):
            perf_results.update(part)
    print(f"performance 结果合并: {len(perf_results)} 条", flush=True)

    entries: dict[str, object] = {}
    bad: list[tuple[int, str]] = []
    for index, dialogue in enumerate(dialogues):
        speaker = rf.effective_speaker(dialogue.get("speaker", ""))
        gender = gender_results.get(speaker, {}).get("gender", "male")
        if gender not in {"male", "female"}:
            gender = "male"
        task = rf.make_tts_task(config, index, dialogue, gender, perf_results.get(str(index), {}))
        try:
            entries[str(index)] = rf.completed_tts_entry(task)
        except Exception as exc:
            bad.append((index, str(exc)))

    payload = {
        "version": rf.STREAMING_TTS_CHECKPOINT_VERSION,
        "tts_pipeline_version": rf.TTS_PIPELINE_VERSION,
        "total_segments": len(dialogues),
        "segments": entries,
        "active_batch": [],
    }
    rf.write_json(rf.streaming_tts_checkpoint_path(config), payload)
    print(f"checkpoint 写入: {len(entries)} / {len(dialogues)}", flush=True)
    if bad:
        print(f"⚠️ 无法入库 {len(bad)} 个: {bad[:10]}", flush=True)
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
