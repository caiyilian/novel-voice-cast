"""收尾：修复最后 3 条（2161 换用 L38 达标版；981/2353 重生成并做同音分析）。"""

from __future__ import annotations

import difflib
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

# 2161 用实验验证 100% 干净的 L38 版本（33 字）
NEW_INSTR = {
    2161: "惊讶平读顿半拍接次句了然；定论放缓转稳留半拍；末尾名词逐字放慢加重",
}


def load_inputs():
    ckpt_path = OUT / "streaming_tts.checkpoint.json"
    ckpt = json.loads(ckpt_path.read_text(encoding="utf-8"))
    by = {int(v["index"]): v for v in ckpt["segments"].values()}
    report = json.loads((OUT / "tts_quality_report.json").read_text(encoding="utf-8"))
    rmap = {r["index"]: r for r in report["records"]}
    texts: dict[int, str] = {}
    for fn in ("performance_directions.json", "performance_directions_supplemental.json"):
        d = json.loads((ROOT / "backend" / "data" / fn).read_text(encoding="utf-8"))["results"]
        for v in d.values():
            texts[int(v["dialogue_index"])] = str(v.get("text", ""))
    return ckpt_path, ckpt, by, rmap, texts


def gen(indices: list[int], ckpt_path, ckpt, by, texts) -> None:
    cfg = load_config(str(ROOT / "config" / "config.yaml"))
    cv = cfg.get("cosyvoice", {})
    voice: dict[str, str] = {}
    for key in ("characters", "voice_assignments"):
        sec = cfg.get(key) or {}
        if isinstance(sec, dict):
            for sp, val in sec.items():
                voice[str(sp)] = val if isinstance(val, str) else (val or {}).get("reference_audio", "")
    tasks = []
    for i in indices:
        rec = by[i]
        ref = voice.get(rec.get("speaker", "")) or voice.get("旁白") or ""
        rp = Path(ref)
        tasks.append({
            "index": i, "task_key": f"final-{i}-1", "text": texts.get(i, ""),
            "output_path": rec["audio_path"], "fingerprint": f"final-{i}-1",
            "reference_audio": str(rp if rp.is_absolute() else ROOT / rp),
            "instruct_text": str(rec.get("instruct_text", "")),
        })
    spec = OUT / "_final_spec.json"
    res = OUT / "_final_res.json"
    spec.write_text(json.dumps({
        "tasks": tasks, "repo_path": str(resolve_path(cv["repo_path"])),
        "model_path": str(resolve_path(cv["model_path"])), "results_path": str(res),
        "task_attempts": 3, "fp16": bool(cv.get("fp16", False)),
    }, ensure_ascii=False), encoding="utf-8")
    rc = subprocess.call([str(resolve_cosyvoice_python(cfg)), str(ROOT / "backend" / "cosyvoice_worker.py"), str(spec)])
    got = json.loads(res.read_text(encoding="utf-8")).get("results", {}) if res.is_file() else {}
    ok = sum(1 for v in got.values() if v.get("status") == "ok")
    print(f"重生成 exit={rc} 成功 {ok}/{len(tasks)}", flush=True)


def homophone_report(index: int, orig: str, transcript: str) -> None:
    """给出拼音级对比，判断是否同音误判。"""
    try:
        from pypinyin import Style, lazy_pinyin
    except ImportError:
        print(f"  （未安装 pypinyin，跳过拼音对比）", flush=True)
        return
    o = re.sub(r"[^\u4e00-\u9fff]", "", orig)
    t = re.sub(r"[^\u4e00-\u9fff]", "", transcript)
    po = lazy_pinyin(o, style=Style.NORMAL)
    pt = lazy_pinyin(t, style=Style.NORMAL)
    print(f"  原文拼音: {' '.join(po)}", flush=True)
    print(f"  转录拼音: {' '.join(pt)}", flush=True)
    sm = difflib.SequenceMatcher(None, po, pt)
    print(f"  拼音相似度: {sm.ratio():.2f}", flush=True)
    return sm.ratio()


if __name__ == "__main__":
    ckpt_path, ckpt, by, rmap, texts = load_inputs()

    # 1) 应用 2161 的 L38 版本
    for i, txt in NEW_INSTR.items():
        by[i]["instruct_text"] = txt
    ckpt_path.write_text(json.dumps(ckpt, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已更新 instruct: {list(NEW_INSTR)}", flush=True)

    # 2) 重生成 3 条
    gen([2161, 981, 2353], ckpt_path, ckpt, by, texts)
