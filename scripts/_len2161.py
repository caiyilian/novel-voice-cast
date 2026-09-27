"""针对顽固句 2161 的长度梯度实验：验证「更短是否更安全」。

对同一句正文，用 4 个长度档（20/28/38/50 字）各生成 5 次，
ASR 判定干净率，找出安全阈值。
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path("E:/projects/novel-voice-cast")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from run_full import load_config, resolve_cosyvoice_python, resolve_path  # noqa: E402

OUT = ROOT / "output"
VAR = OUT / "_len2161"
VAR.mkdir(parents=True, exist_ok=True)

TARGET = 2161
REPEATS = 5

# 4 个长度档（20 / 28 / 38 / 50 字）
VARIANTS = {
    "L20": "惊讶平读，次句定论转稳，末尾两处放慢加重",
    "L28": "惊讶平读顿半拍接次句；定论放缓转稳；末尾名词逐字加重",
    "L38": "惊讶平读顿半拍接次句了然；定论放缓转稳留半拍；末尾名词逐字放慢加重",
    "L50": "惊讶平读顿半拍接次句首了然；定论放缓转稳句后留半拍；小国君主逐字放慢加重，句尾屏气垂落",
}


def n(s: str) -> str:
    from opencc import OpenCC

    return re.sub(r"[^\u4e00-\u9fff0-9a-zA-Z]", "", OpenCC("t2s").convert(str(s or "")))


def gen() -> list[dict]:
    cfg = load_config(str(ROOT / "config" / "config.yaml"))
    cv = cfg.get("cosyvoice", {})
    voice: dict[str, str] = {}
    for key in ("characters", "voice_assignments"):
        sec = cfg.get(key) or {}
        if isinstance(sec, dict):
            for sp, val in sec.items():
                voice[str(sp)] = val if isinstance(val, str) else (val or {}).get("reference_audio", "")

    ckpt = json.loads((OUT / "streaming_tts.checkpoint.json").read_text(encoding="utf-8"))
    by = {int(v["index"]): v for v in ckpt["segments"].values()}
    rec = by[TARGET]

    texts: dict[int, str] = {}
    for fn in ("performance_directions.json", "performance_directions_supplemental.json"):
        d = json.loads((ROOT / "backend" / "data" / fn).read_text(encoding="utf-8"))["results"]
        for v in d.values():
            texts[int(v["dialogue_index"])] = str(v.get("text", ""))

    ref = voice.get(rec.get("speaker", "")) or voice.get("旁白") or ""
    rp = Path(ref)
    tasks = []
    for tag, instr in VARIANTS.items():
        for k in range(1, REPEATS + 1):
            tasks.append({
                "index": TARGET, "task_key": f"{tag}-{k}",
                "text": texts.get(TARGET, ""),
                "output_path": str(VAR / f"{tag}_r{k}.wav"),
                "fingerprint": f"len2161-{tag}-{k}",
                "reference_audio": str(rp if rp.is_absolute() else ROOT / rp),
                "instruct_text": instr,
            })
    spec = OUT / "_len2161_spec.json"
    res = OUT / "_len2161_res.json"
    spec.write_text(json.dumps({
        "tasks": tasks, "repo_path": str(resolve_path(cv["repo_path"])),
        "model_path": str(resolve_path(cv["model_path"])), "results_path": str(res),
        "task_attempts": 3, "fp16": bool(cv.get("fp16", False)),
    }, ensure_ascii=False), encoding="utf-8")
    rc = subprocess.call([str(resolve_cosyvoice_python(cfg)), str(ROOT / "backend" / "cosyvoice_worker.py"), str(spec)])
    print(f"生成 exit={rc}", flush=True)
    return tasks


def judge(tasks: list[dict]) -> None:
    from faster_whisper import WhisperModel

    orig_rec = json.loads((OUT / "tts_quality_report.json").read_text(encoding="utf-8"))
    orig = next(r["original"] for r in orig_rec["records"] if r["index"] == TARGET)
    o = n(orig)
    model = WhisperModel(
        str(Path("E:/huggingface_models/hub/models--Systran--faster-whisper-large-v2")
            / "snapshots/f0fe81560cb8b68660e564f55dd99207059c092e"),
        device="cuda", compute_type="float16",
    )
    stat: dict[str, list[int]] = {}
    import difflib

    for t in tasks:
        tag = t["task_key"].split("-")[0]
        p = Path(t["output_path"])
        if not p.is_file():
            continue
        segs, _ = model.transcribe(str(p), language="zh", beam_size=5)
        txt = n("".join(s.text for s in segs))
        instr = n(t["instruct_text"])
        leak = any(instr[j : j + 4] in txt and instr[j : j + 4] not in o for j in range(len(instr) - 3))
        sim = difflib.SequenceMatcher(None, o, txt).ratio()
        ok = (not leak) and sim >= 0.85
        st = stat.setdefault(tag, [0, 0])
        st[0] += 1 if ok else 0
        st[1] += 1
        print(f"  {t['task_key']:<8} 相似{sim:.2f} 泄漏={'Y' if leak else 'N'} {'✓' if ok else '✗'}", flush=True)

    print("\n=== 各长度档干净率 ===", flush=True)
    for tag in ("L20", "L28", "L38", "L50"):
        if tag in stat:
            c, t = stat[tag]
            print(f"  {tag}（{len(VARIANTS[tag])}字）: {c}/{t} = {c/t*100:.0f}%", flush=True)


if __name__ == "__main__":
    tasks = gen()
    judge(tasks)
