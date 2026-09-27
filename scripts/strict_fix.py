"""严格迭代修复：强制满足「≤38 字 + 无正文 3+ 字片段」，带自动校验重试。

与前版 tail_fix.py 的差别：
- 每次 LLM 输出后立即校验（长度 + 片段），不合格则把违规详情反馈给 LLM 重写
- 最多 6 轮，直到同时满足两个硬约束
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
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

MAX_LEN = 38
MIN_FRAG = 3


def n(s: str) -> str:
    return re.sub(r"[^\u4e00-\u9fff0-9a-zA-Z]", "", cc.convert(str(s or "")))


def find_frags(ctrl: str, orig: str, min_len: int = MIN_FRAG) -> list[str]:
    """ctrl 中出现的、属于 orig 的连续 min_len+ 字片段（去重、按长度降序）。"""
    c, o = n(ctrl), n(orig)
    hits: list[str] = []
    for ln in range(10, min_len - 1, -1):
        for j in range(len(c) - ln + 1):
            frag = c[j : j + ln]
            if frag in o and not any(frag in h for h in hits):
                hits.append(frag)
    return hits[:6]


SYSTEM = (
    "你是 CosyVoice 3 表演控制词的精简专家。控制词会与台词拼成同一序列送入 TTS，"
    "任何与台词重合的用词都会让模型顺着念下去（泄漏）。\n"
    "硬约束（必须同时满足，违反即失败）：\n"
    "1. 长度 ≤38 个汉字（不含标点）\n"
    "2. 不得包含台词的任何连续 3 个字。指代某一处时用位置描述："
    "句首/句尾/第一分句/第二分句/名词短语/动词短语/叹词/数量词/时间短语/转折词\n"
    "3. 保留可执行的表演信息：语速、停顿、重音落点、气息、音量、句末收法\n"
    "4. 删除比喻、意象、心理描写、重复修饰\n"
    "5. 输出纯中文，无括号、无换行、无标点以外的符号"
)

TOOL = [
    {
        "type": "function",
        "function": {
            "name": "submit",
            "description": "提交精简后的控制词",
            "parameters": {
                "type": "object",
                "properties": {
                    "compacted": {"type": "string", "description": "精简后的控制词（≤38 汉字，不含台词 3 连续字）"},
                },
                "required": ["compacted"],
            },
        },
    }
]


def rewrite_one(client, idx: int, orig: str, ctrl: str) -> tuple[str, list[str]]:
    """迭代改写直到满足硬约束；返回 (最终文本, 最优候选的违规片段)。"""
    best = ""
    best_frags = None
    for attempt in range(6):
        feedback = ""
        if attempt and best:
            bad = find_frags(best, orig)
            msgs = []
            if len(n(best)) > MAX_LEN:
                msgs.append(f"上一次 {len(n(best))} 字，超过 {MAX_LEN} 字上限")
            if bad:
                msgs.append(f"上一次包含台词片段 {bad}，必须换掉这些词")
            feedback = (
                "\n\n【上一次尝试不合格】" + "；".join(msgs) + f"\n上一次输出：{best}\n"
                "请重写，务必同时满足两条硬约束。"
            )
        user = f"台词：\n{orig}\n\n原控制词（{len(n(ctrl))} 字）：\n{ctrl}\n\n请精简到 ≤{MAX_LEN} 字。{feedback}"
        try:
            res = client.chat(
                messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
                tools=TOOL, tool_choice="required", temperature=0.3,
            )
            if res.tool_calls:
                cand = str(res.tool_calls[0].arguments.get("compacted", "")).strip()
                if cand:
                    frags = find_frags(cand, orig)
                    ok = len(n(cand)) <= MAX_LEN and not frags
                    print(
                        f"    idx {idx} 第{attempt+1}次: {len(n(cand))}字 片段{len(frags)} "
                        f"{'✓通过' if ok else '✗'}",
                        flush=True,
                    )
                    if ok:
                        return cand, []
                    # 记录最优（先比片段数，再比长度）
                    score = (len(frags), len(n(cand)))
                    if best_frags is None or score < (len(best_frags), len(n(best))):
                        best, best_frags = cand, frags
        except Exception as e:  # noqa: BLE001
            print(f"    idx {idx} 第{attempt+1}次异常 {type(e).__name__}", flush=True)
    return best, (best_frags or [])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--indices", type=str, required=True)
    ap.add_argument("--gen", action="store_true", help="改写后立即重生成")
    args = ap.parse_args()

    idx = [int(x) for x in args.indices.split(",") if x.strip()]

    ckpt_path = OUT / "streaming_tts.checkpoint.json"
    ckpt = json.loads(ckpt_path.read_text(encoding="utf-8"))
    by = {int(v["index"]): v for v in ckpt["segments"].values()}
    report = json.loads((OUT / "tts_quality_report.json").read_text(encoding="utf-8"))
    rmap = {r["index"]: r for r in report["records"]}

    client = LLMClient.for_flash_lite("tts_strict_fix")
    changed: dict[int, str] = {}
    for i in idx:
        orig = rmap[i]["original"]
        ctrl = str(by[i].get("instruct_text", ""))
        print(f"--- idx {i} 原 {len(n(ctrl))} 字, 现有片段 {find_frags(ctrl, orig)} ---", flush=True)
        txt, left = rewrite_one(client, i, orig, ctrl)
        if txt:
            changed[i] = txt
            print(f"  → {len(n(txt))} 字, 残留片段 {left}: {txt}", flush=True)
        else:
            print("  → 未产出", flush=True)

    for i, txt in changed.items():
        by[i]["instruct_text"] = txt
    ckpt_path.write_text(json.dumps(ckpt, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写回 checkpoint（{len(changed)} 条）", flush=True)

    if args.gen:
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
        for i in changed:
            rec = by[i]
            ref = voice.get(rec.get("speaker", "")) or voice.get("旁白") or ""
            rp = Path(ref)
            tasks.append({
                "index": i, "text": texts.get(i, ""), "output_path": rec["audio_path"],
                "fingerprint": f"strictfix-{i}", "reference_audio": str(rp if rp.is_absolute() else ROOT / rp),
                "instruct_text": str(rec.get("instruct_text", "")),
            })
        spec = OUT / "_strict_fix_spec.json"
        res = OUT / "_strict_fix_res.json"
        spec.write_text(json.dumps({
            "tasks": tasks, "repo_path": str(resolve_path(cv["repo_path"])),
            "model_path": str(resolve_path(cv["model_path"])), "results_path": str(res),
            "task_attempts": 3, "fp16": bool(cv.get("fp16", False)),
        }, ensure_ascii=False), encoding="utf-8")
        rc = subprocess.call([str(resolve_cosyvoice_python(cfg)), str(ROOT / "backend" / "cosyvoice_worker.py"), str(spec)])
        got = json.loads(res.read_text(encoding="utf-8")).get("results", {}) if res.is_file() else {}
        ok = sum(1 for v in got.values() if v.get("status") == "ok")
        print(f"重生成 exit={rc}, 成功 {ok}/{len(tasks)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
