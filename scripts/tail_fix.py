"""收尾：对残留条目做定向修复（强制压缩 + 去片段），然后重生成。

策略：
- 真泄漏（稳定 + 含 instruct 片段）→ instruct 强化压缩至 ≤40 字且无 3+ 字片段
- 内容缺失 / ASR 不稳定 / 其他 → 原样重生成（换一次采样）
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path("E:/projects/novel-voice-cast")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "scripts"))

from app.core.llm_client import LLMClient  # noqa: E402
from opencc import OpenCC  # noqa: E402
from run_full import load_config, resolve_cosyvoice_python, resolve_path  # noqa: E402

OUT = ROOT / "output"
cc = OpenCC("t2s")


def n(s: str) -> str:
    return re.sub(r"[^\u4e00-\u9fff0-9a-zA-Z]", "", cc.convert(str(s or "")))


def max_frag(ctrl: str, orig: str, min_len: int = 3) -> int:
    c, o = n(ctrl), n(orig)
    best = 0
    for i in range(len(o) - min_len + 1):
        for ln in range(min_len, min(12, len(o) - i) + 1):
            if o[i : i + ln] in c:
                best = max(best, ln)
            else:
                break
    return best


SYSTEM = (
    "你是 CosyVoice 3 的表演控制词压缩专家。控制词会与正文拼成同一序列送入 TTS，"
    "若控制词过长或含正文片段，模型会继续念控制词（泄漏）。\n"
    "硬约束：\n"
    "1. 目标长度 ≤40 字（汉字数，不含标点），越短越好，但必须保留可执行的表演细节\n"
    "2. 【禁止出现正文的任何连续 3 个字】——用位置描述代替（句首/句尾/第一分句/名词短语/动词短语/叹词/数量词）\n"
    "3. 保留：语速、停顿位置、重音落点、气息、音量、句末收法\n"
    "4. 删掉：比喻、意象、重复修饰、非必要的心理描写\n"
    "5. 输出纯中文，无括号、无换行"
)

TOOL = [
    {
        "type": "function",
        "function": {
            "name": "submit",
            "description": "提交压缩后的控制词",
            "parameters": {
                "type": "object",
                "properties": {
                    "compacted": {"type": "string", "description": "压缩后的控制词（≤40 字）"},
                },
                "required": ["compacted"],
            },
        },
    }
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--indices", type=str, required=True)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    idx = [int(x) for x in args.indices.split(",") if x.strip()]

    rows = json.loads((OUT / "_residual_verify.json").read_text(encoding="utf-8"))
    verdict = {r["index"]: r for r in rows}
    real_leak = {r["index"] for r in rows if r["frags"] and r["stable"]}

    ckpt_path = OUT / "streaming_tts.checkpoint.json"
    ckpt = json.loads(ckpt_path.read_text(encoding="utf-8"))
    by = {int(v["index"]): v for v in ckpt["segments"].values()}
    report = json.loads((OUT / "tts_quality_report.json").read_text(encoding="utf-8"))
    rmap = {r["index"]: r for r in report["records"]}

    client = LLMClient.for_flash_lite("tts_tail_fix")

    def work(i: int):
        rec = by[i]
        orig = rmap[i]["original"]
        ctrl = str(rec.get("instruct_text", ""))
        if i not in real_leak:
            return i, None  # 原样重生成，不改 instruct
        user = (
            f"正文：\n{orig}\n\n原控制词（{len(n(ctrl))} 字）：\n{ctrl}\n\n"
            f"请压缩到 ≤40 字，并确保【不含正文任何连续 3 个字】。"
        )
        for _ in range(3):
            try:
                res = client.chat(
                    messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
                    tools=TOOL, tool_choice="required", temperature=0.2,
                )
                if res.tool_calls:
                    txt = res.tool_calls[0].arguments.get("compacted", "")
                    if txt:
                        return i, txt
            except Exception:
                pass
        return i, None

    new_instr: dict[int, str] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for f in as_completed([pool.submit(work, i) for i in idx]):
            i, txt = f.result()
            if txt:
                new_instr[i] = txt

    print(f"改写 {len(new_instr)} 条（真泄漏 {len([i for i in idx if i in real_leak])} 条）", flush=True)
    for i, txt in list(new_instr.items())[:20]:
        frag = max_frag(txt, rmap[i]["original"])
        print(f"  idx {i}: {len(n(txt))} 字, 最长片段 {frag} -> {txt[:60]}", flush=True)

    # 写回 checkpoint
    for i, txt in new_instr.items():
        by[i]["instruct_text"] = txt
    ckpt_path.write_text(json.dumps(ckpt, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已写回 checkpoint", flush=True)

    # 重生成
    texts: dict[int, str] = {}
    for fn in ("performance_directions.json", "performance_directions_supplemental.json"):
        d = json.loads((ROOT / "backend" / "data" / fn).read_text(encoding="utf-8"))["results"]
        for v in d.values():
            texts[int(v["dialogue_index"])] = str(v.get("text", ""))

    cfg = load_config(str(ROOT / "config" / "config.yaml"))
    cv = cfg.get("cosyvoice", {})
    voice: dict[str, str] = {}
    for key in ("characters", "voice_assignments"):
        sec = cfg.get(key) or {}
        if isinstance(sec, dict):
            for sp, val in sec.items():
                voice[str(sp)] = val if isinstance(val, str) else (val or {}).get("reference_audio", "")

    tasks = []
    for i in idx:
        rec = by[i]
        p = Path(rec["audio_path"])
        ref = voice.get(rec.get("speaker", "")) or voice.get("旁白") or ""
        rp = Path(ref)
        tasks.append({
            "index": i, "text": texts.get(i, ""), "output_path": str(p),
            "fingerprint": f"tailfix-{i}", "reference_audio": str(rp if rp.is_absolute() else ROOT / rp),
            "instruct_text": str(rec.get("instruct_text", "")),
        })
    spec = ROOT / "output" / "_tail_fix_spec.json"
    res = ROOT / "output" / "_tail_fix_res.json"
    spec.write_text(json.dumps({
        "tasks": tasks, "repo_path": str(resolve_path(cv["repo_path"])),
        "model_path": str(resolve_path(cv["model_path"])), "results_path": str(res),
        "task_attempts": 3, "fp16": bool(cv.get("fp16", False)),
    }, ensure_ascii=False), encoding="utf-8")
    rc = subprocess.call([str(resolve_cosyvoice_python(cfg)), str(ROOT / "backend" / "cosyvoice_worker.py"), str(spec)])
    print(f"重生成 exit={rc}", flush=True)
    got = json.loads(res.read_text(encoding="utf-8")).get("results", {}) if res.is_file() else {}
    ok = sum(1 for v in got.values() if v.get("status") == "ok")
    print(f"成功 {ok}/{len(tasks)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
