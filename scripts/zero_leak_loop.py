"""零泄漏收敛闭环：迭代改写+重生成，直到 LEAK 归零。

原理
----
泄漏的必要条件是 instruct 含正文连续片段：
  无片段 514 条 → 零泄漏
  含片段 2453 条 → 42 条泄漏（1.7%）

所以修复分两步：
  1. 消除片段（治本，把系统性原因去掉）
  2. 重生成后复检（兜住残留的随机性）
迭代直到归零。

相比全量改写（2453 条 ≈17 小时），本方案只动实际泄漏的句子，
每轮约 33 分钟，通常 2-3 轮收敛。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "backend"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

WHISPER = (
    "E:/huggingface_models/hub/models--Systran--faster-whisper-large-v2"
    "/snapshots/f0fe81560cb8b68660e564f55dd99207059c092e"
)

# 严禁出现正文任何连续 3 字的改写提示词
SYSTEM = """你是语音合成控制词改写专家。任务：改写控制词，使其**不包含正文的任何连续 3 个字**。

## 背景

控制词通过独立参数传给 CosyVoice 3，与正文拼成同一序列由模型续写。
实测发现：控制词里出现正文的连续片段时，模型会误以为正文还没结束，
从而把控制词念出来（泄漏）。

实测数据（按控制词中最长的正文片段长度）：
  无片段    514 条 → 泄漏率 0.0%
  3-4 字    938 条 → 1.1-1.3%
  5-8+ 字  1001 条 → 1.6-3.3%
片段越长，泄漏率越高。**必须彻底消除片段。**

## 改写方法：用「位置 / 类型」代替「具体词」

| 原写法 | 改写后 |
|---|---|
| 重音落「迎风摇曳」与「狼」 | 重音落在句首词组与句尾单字 |
| 「好几百年」放慢放轻 | 第一分句的时间短语放慢放轻 |
| 「但是」前轻吸一气 | 转折词前轻吸一气 |
| 句尾「全变了样」平稳沉下 | 句尾四字短语平稳沉下 |
| 「司空见惯」加重、「这点小事」放轻 | 前半句成语加重、后半句短语放轻 |
| 「啊」借呼气单点脱口 | 句首叹词借呼气单点脱口 |

## 硬约束

1. **绝不出现正文的任何连续 3 个字** —— 这是唯一目的，务必逐字检查
2. 用「句首/句尾/第一分句/动作短语/叹词/名词短语/成语/数量词/时间短语/转折词」等
3. 保留原有语速、停顿、音量、气息描述，长度不增（±10%）
4. 语气一致，不新增表演意图

## 输出

只输出改写后的控制词文本。"""

TOOL = [
    {
        "type": "function",
        "function": {
            "name": "submit_rewrite",
            "description": "提交改写后的控制词（不含正文任何连续 3 字）",
            "parameters": {
                "type": "object",
                "properties": {
                    "rewritten": {"type": "string"},
                    "note": {"type": "string", "description": "一句话说明"},
                },
                "required": ["rewritten", "note"],
            },
        },
    }
]


def norm(s: str) -> str:
    from opencc import OpenCC

    return re.sub(r"[^\u4e00-\u9fff0-9a-zA-Z]", "", OpenCC("t2s").convert(str(s or "")))


def max_fragment(ctrl: str, orig: str, min_len: int = 3) -> int:
    """返回 instruct 中出现的最长正文连续片段长度（<min_len 记 0）。"""
    c, o = norm(ctrl), norm(orig)
    best = 0
    for i in range(len(o) - min_len + 1):
        for ln in range(min_len, min(14, len(o) - i) + 1):
            if o[i : i + ln] in c:
                best = max(best, ln)
            else:
                break
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--report", default="output/tts_quality_report.json")
    args = ap.parse_args()

    from app.core.llm_client import LLMClient
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from run_full import load_config, resolve_cosyvoice_python, resolve_path

    report_path = PROJECT_ROOT / args.report
    ckpt_path = PROJECT_ROOT / "output" / "streaming_tts.checkpoint.json"

    cfg = load_config(str(PROJECT_ROOT / "config" / "config.yaml"))
    cv = cfg.get("cosyvoice", {})
    py = str(resolve_cosyvoice_python(cfg))
    repo = str(resolve_path(cv["repo_path"]))
    model_dir = str(resolve_path(cv["model_path"]))

    # 说话人 → 参考音频
    voice: dict[str, str] = {}
    for key in ("characters", "voice_assignments"):
        sec = cfg.get(key) or {}
        if isinstance(sec, dict):
            for sp, val in sec.items():
                voice[str(sp)] = val if isinstance(val, str) else (val or {}).get("reference_audio", "")

    # 正文（唯一权威来源，绝不修改）
    texts: dict[int, str] = {}
    for fn in ("performance_directions.json", "performance_directions_supplemental.json"):
        dd = json.loads((PROJECT_ROOT / "backend" / "data" / fn).read_text(encoding="utf-8"))["results"]
        for v in dd.values():
            texts[int(v["dialogue_index"])] = str(v.get("text", ""))

    client = LLMClient.for_flash_lite("tts_zero_leak")

    for rnd in range(1, args.rounds + 1):
        report = json.loads(report_path.read_text(encoding="utf-8"))
        leak = [r for r in report["records"] if r["verdict"] == "LEAK"]
        if not leak:
            print(f"\n✅ 第 {rnd} 轮前检查：零 LEAK，任务完成", flush=True)
            return 0
        print(f"\n{'='*66}\n第 {rnd} 轮：当前 LEAK {len(leak)} 条\n{'='*66}", flush=True)

        ckpt = json.loads(ckpt_path.read_text(encoding="utf-8"))
        segs = ckpt["segments"]
        by = {int(v["index"]): v for v in segs.values()}

        # 1) 改写 instruct：消除正文片段
        def rewrite(item: dict) -> tuple[int, dict | None]:
            idx = item["index"]
            orig = texts.get(idx, item["original"])
            cur = str(by[idx].get("instruct_text", ""))
            frag = max_fragment(cur, orig)
            user = (
                f"正文：\n{orig}\n\n"
                f"当前控制词：\n{cur}\n\n"
                f"当前控制词中出现了正文的连续 {frag} 字片段，这是泄漏的直接原因。\n"
                f"请改写，确保控制词中【不出现正文的任何连续 3 个字】。"
            )
            for _ in range(3):
                try:
                    res = client.chat(
                        messages=[{"role": "system", "content": SYSTEM},
                                  {"role": "user", "content": user}],
                        tools=TOOL, tool_choice="required", temperature=0.2,
                    )
                    if res.tool_calls:
                        return idx, res.tool_calls[0].arguments
                except Exception:
                    pass
            return idx, None

        print("  改写控制词…", flush=True)
        new_instr: dict[int, str] = {}
        with ThreadPoolExecutor(max_workers=6) as pool:
            futs = [pool.submit(rewrite, r) for r in leak]
            for f in as_completed(futs):
                idx, data = f.result()
                if data:
                    new_instr[idx] = data["rewritten"]

        # 校验：改写后是否真的无片段
        still_frag = []
        for idx, txt in new_instr.items():
            if max_fragment(txt, texts.get(idx, "")) >= 3:
                still_frag.append(idx)
        print(f"  改写 {len(new_instr)} 条，其中仍有片段 {len(still_frag)} 条", flush=True)

        # 写回 checkpoint
        for idx, txt in new_instr.items():
            if idx in by:
                by[idx]["instruct_text"] = txt
        ckpt_path.write_text(json.dumps(ckpt, ensure_ascii=False, indent=2), encoding="utf-8")

        # 2) 重生成
        print("  重生成音频…", flush=True)
        tasks = []
        for idx in new_instr:
            rec = by[idx]
            p = Path(rec["audio_path"])
            p.unlink(missing_ok=True)
            ref = voice.get(rec.get("speaker", "")) or voice.get("旁白") or ""
            rp = Path(ref)
            tasks.append({
                "index": idx,
                "text": texts.get(idx, ""),
                "output_path": str(p),
                "fingerprint": rec.get("fingerprint", ""),
                "reference_audio": str(rp if rp.is_absolute() else PROJECT_ROOT / rp),
                "instruct_text": rec["instruct_text"],
            })
        spec = {
            "tasks": tasks, "repo_path": repo, "model_path": model_dir,
            "results_path": f"output/_zl_gen_r{rnd}.json",
            "task_attempts": 3, "fp16": bool(cv.get("fp16", False)),
        }
        spec_path = PROJECT_ROOT / "output" / f"_zl_spec_r{rnd}.json"
        spec_path.write_text(json.dumps(spec, ensure_ascii=False), encoding="utf-8")
        subprocess.call([py, str(PROJECT_ROOT / "backend" / "cosyvoice_worker.py"), str(spec_path)])

        # 3) 重新转录 + 判定（用主脚本，保证一致）
        indices = ",".join(str(i) for i in sorted(new_instr))
        print("  重新转录…", flush=True)
        subprocess.call([
            sys.executable, "-u", str(PROJECT_ROOT / "scripts" / "tts_quality_loop.py"),
            "transcribe", "--transcripts", str(PROJECT_ROOT / "output" / "asr_transcripts.json"),
            "--indices", indices, "--force",
        ])
        print("  重新判定…", flush=True)
        subprocess.call([
            sys.executable, "-u", str(PROJECT_ROOT / "scripts" / "tts_quality_loop.py"),
            "audit", "--report", str(report_path), "--indices", indices, "--workers", "6",
        ])

    report = json.loads(report_path.read_text(encoding="utf-8"))
    left = [r["index"] for r in report["records"] if r["verdict"] == "LEAK"]
    print(f"\n达到轮次上限，剩余 LEAK {len(left)}: {left[:20]}", flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
