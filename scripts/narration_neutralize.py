"""清除旁白 instruct 中的情绪色彩词，改为平直陈述 + 保表演细节。

背景：performance_profiles.json 的 emotional_range 字段含「莞尔/含笑」等词，
被下游 line-direction 当作逐句表演指令照搬进 performance_control，
导致 21.4%（360/1686）旁白听起来戏谑俏皮，与「不戏剧化、不渲染」的定位冲突。

做法：
- 只改旁白（对话角色不动）
- 把情绪标签替换为等价的声音层指令（语速/停顿/重音/气息/音量/句末收法）
- 长度不得增加（宁短不长，泄漏风险与长度正相关）
- 不含正文 3+ 字连续片段（沿用既有约束）
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

BANNED = [
    "会心", "莞尔", "含笑", "带笑", "笑意", "促狭", "俏皮", "玩味", "戏谑",
    "打趣", "欣然", "得意", "狡黠", "俏", "好笑", "好笑地", "调皮", "逗趣",
]


def n(s: str) -> str:
    return re.sub(r"[^\u4e00-\u9fff0-9a-zA-Z]", "", cc.convert(str(s or "")))


def find_banned(txt: str) -> list[str]:
    return [w for w in BANNED if w in str(txt or "")]


def find_frags(ctrl: str, orig: str, min_len: int = 3) -> list[str]:
    c, o = n(ctrl), n(orig)
    hits: list[str] = []
    for ln in range(8, min_len - 1, -1):
        for j in range(len(c) - ln + 1):
            frag = c[j : j + ln]
            if frag in o and not any(frag in h for h in hits):
                hits.append(frag)
    return hits[:4]


SYSTEM = (
    "你是 CosyVoice 3 有声书旁白的表演控制词编辑。\n"
    "本书旁白是冷静克制的文学叙述者，绝不能演得戏谑、俏皮或卖弄笑意。\n"
    "任务：删除控制词中的情绪色彩词，改为等价的「声音层」指令。\n\n"
    "必须删除的词（任何位置）：会心 / 莞尔 / 含笑 / 带笑 / 笑意 / 促狭 / 俏皮 /\n"
    "玩味 / 戏谑 / 打趣 / 欣然 / 得意 / 狡黠 / 俏 / 好笑 / 调皮 / 逗趣\n\n"
    "替换原则：\n"
    "- 把「带会心」「带笑意」等，改成具体的声音处理：平直陈述 / 语速平缓 /\n"
    "  停半拍 / 句末放平 / 音量不扬 / 不拖不叹 / 平平说出\n"
    "- 如果原句的意味是「温和的宽容」，写成 「平直口吻，不渲染」而非「带莞尔」\n"
    "- 如果原句是「反讽」，写成「平平说出，不着重」而非「带戏谑」\n\n"
    "硬约束：\n"
    "1. 长度不得超过原控制词（宁短不长）\n"
    "2. 不得出现正文的任何连续 3 个字 —— 用位置描述代替（句首/句尾/第N分句/\n"
    "   名词短语/动词短语/叹词/数量词/时间短语/转折词）\n"
    "3. 保留原有的语速、停顿、重音落点、气息、音量、句末收法\n"
    "4. 输出纯中文，无括号、无换行、无引号"
)

TOOL = [
    {
        "type": "function",
        "function": {
            "name": "submit",
            "description": "提交去情绪化后的控制词",
            "parameters": {
                "type": "object",
                "properties": {
                    "rewritten": {"type": "string", "description": "改写后的控制词（不含情绪色彩词，长度≤原文）"},
                },
                "required": ["rewritten"],
            },
        },
    }
]


def rewrite_one(client, orig: str, ctrl: str, max_rounds: int = 4) -> str:
    best, best_bad = "", None
    for attempt in range(max_rounds):
        fb = ""
        if attempt and best:
            bad = find_banned(best)
            frags = find_frags(best, orig)
            msgs = []
            if bad:
                msgs.append(f"还残留情绪词 {bad}")
            if frags:
                msgs.append(f"含正文片段 {frags}")
            if len(n(best)) > len(n(ctrl)):
                msgs.append(f"超长 {len(n(best))}>{len(n(ctrl))}")
            fb = "\n\n【上次不合格】" + "；".join(msgs) + f"\n上次输出：{best}\n请重写。"
        user = f"台词：\n{orig}\n\n原控制词（{len(n(ctrl))} 字）：\n{ctrl}\n\n请删除情绪色彩词并改写。{fb}"
        try:
            res = client.chat(
                messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
                tools=TOOL, tool_choice="required", temperature=0.2,
            )
            if res.tool_calls:
                cand = str(res.tool_calls[0].arguments.get("rewritten", "")).strip()
                if cand:
                    bad = find_banned(cand)
                    frags = find_frags(cand, orig)
                    ok = not bad and not frags and len(n(cand)) <= len(n(ctrl))
                    if ok:
                        return cand
                    score = (len(bad), len(frags), len(n(cand)))
                    if best_bad is None or score < best_bad:
                        best, best_bad = cand, score
        except Exception:
            pass
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default="output/_narr_neutral.json")
    ap.add_argument("--gen", action="store_true", help="改写后重生成")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    ckpt_path = OUT / "streaming_tts.checkpoint.json"
    ckpt = json.loads(ckpt_path.read_text(encoding="utf-8"))
    by = {int(v["index"]): v for v in ckpt["segments"].values()}
    report = json.loads((OUT / "tts_quality_report.json").read_text(encoding="utf-8"))
    rmap = {r["index"]: r for r in report["records"]}

    targets = [
        i for i, v in sorted(by.items())
        if v.get("speaker") == "旁白" and find_banned(str(v.get("instruct_text", "")))
    ]
    if args.limit:
        targets = targets[: args.limit]
    print(f"旁白含情绪词: {len(targets)} 条", flush=True)

    client = LLMClient.for_flash_lite("tts_narration_neutralizer")
    done: dict[int, dict] = {}

    def work(i: int):
        ctrl = str(by[i].get("instruct_text", ""))
        txt = rewrite_one(client, rmap[i]["original"], ctrl)
        return i, ctrl, txt

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(work, i) for i in targets]
        for k, f in enumerate(as_completed(futs), 1):
            i, ctrl, txt = f.result()
            if txt and txt != ctrl:
                done[i] = {
                    "before": ctrl, "after": txt,
                    "len_before": len(n(ctrl)), "len_after": len(n(txt)),
                    "banned_after": find_banned(txt),
                    "frags_after": find_frags(txt, rmap[i]["original"]),
                }
            if k % 50 == 0:
                print(f"  进度 {k}/{len(targets)}", flush=True)

    bad_left = sum(1 for v in done.values() if v["banned_after"] or v["frags_after"])
    print(f"\n改写完成 {len(done)} 条，仍违规 {bad_left}", flush=True)

    Path(args.out).write_text(json.dumps({"results": {str(k): v for k, v in done.items()}},
                                        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已写入 {args.out}", flush=True)

    # 抽样展示
    for k in list(done)[:4]:
        v = done[k]
        print(f"\nidx {k} ({v['len_before']}→{v['len_after']} 字)", flush=True)
        print(f"  前: {v['before']}", flush=True)
        print(f"  后: {v['after']}", flush=True)

    if args.gen and done:
        for i, v in done.items():
            by[i]["instruct_text"] = v["after"]
        ckpt_path.write_text(json.dumps(ckpt, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n已写回 checkpoint {len(done)} 条", flush=True)

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
        for i in done:
            rec = by[i]
            ref = voice.get(rec.get("speaker", "")) or voice.get("旁白") or ""
            rp = Path(ref)
            tasks.append({
                "index": i, "task_key": f"narr-{i}", "text": texts.get(i, ""),
                "output_path": rec["audio_path"], "fingerprint": f"narr-{i}",
                "reference_audio": str(rp if rp.is_absolute() else ROOT / rp),
                "instruct_text": str(rec.get("instruct_text", "")),
            })
        py = str(resolve_cosyvoice_python(cfg))
        BATCH = 40
        for s in range(0, len(tasks), BATCH):
            chunk = tasks[s : s + BATCH]
            spec = OUT / f"_narr_spec{chunk[0]['index']}.json"
            res = OUT / f"_narr_res{chunk[0]['index']}.json"
            spec.write_text(json.dumps({
                "tasks": chunk, "repo_path": str(resolve_path(cv["repo_path"])),
                "model_path": str(resolve_path(cv["model_path"])), "results_path": str(res),
                "task_attempts": 3, "fp16": bool(cv.get("fp16", False)),
            }, ensure_ascii=False), encoding="utf-8")
            print(f"--- 批次 {s//BATCH+1}: {chunk[0]['index']}-{chunk[-1]['index']} ---", flush=True)
            subprocess.call([py, str(ROOT / "backend" / "cosyvoice_worker.py"), str(spec)])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
