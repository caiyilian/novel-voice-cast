"""全量无片段化：把 2453 条含正文片段的 instruct 改写为「位置描述」版本。

方案 B 的执行脚本。分三个可独立续跑的阶段：

  stage1 rewrite  LLM 改写（消除正文连续 3+ 字片段），带校验与重试
  stage2 gen      分批重生成音频（每批 40 条，抗中断）
  stage3 verify   转录 + 判定

每个阶段都可重复执行，已完成的部分自动跳过（断点续跑）。

背景
----
泄漏的必要条件是 instruct 含正文连续片段（实测）：
  无片段    514 条 → 泄漏率 0.0%
  3-4 字    938 条 → 1.1-1.3%
  5-8+ 字  1001 条 → 1.6-3.3%

改写手法（保留表演细节，只换指代方式）：
  重音落「迎风摇曳」与「狼」  →  重音落在句首词组与句尾单字
  「好几百年」放慢放轻        →  第一分句的时间短语放慢放轻
  「但是」前轻吸一气          →  转折词前轻吸一气
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "backend"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

STATE = PROJECT_ROOT / "output" / "_nofrag_state.json"
REWRITES = PROJECT_ROOT / "output" / "_nofrag_rewrites.json"
GEN_RESULTS = PROJECT_ROOT / "output" / "_nofrag_gen.json"

MIN_FRAG = 3

SYSTEM = """你是语音合成控制词改写专家。任务：改写控制词，使其**不包含正文的任何连续 3 个字**。

## 为什么

控制词与正文会被拼接成同一序列交给 TTS 模型续写。若控制词里出现正文的连续片段，
模型会误判「正文还没开始」而把控制词念出来（泄漏）。实测泄漏率随最长共享片段递增：

  无共享片段   泄漏率 0.0%
  共享 3-4 字  泄漏率 1.1-1.3%
  共享 5-6 字  泄漏率 1.6-1.9%
  共享 7-8 字  泄漏率 1.9-3.3%

## 怎么改：用「位置 / 类型」指代，不引用原词

| 原写法 | 改写后 |
|---|---|
| 重音落「迎风摇曳」与「狼」 | 重音落在句首词组与句尾单字 |
| 「好几百年」放慢放轻 | 第一分句的时间短语放慢放轻 |
| 「但是」前轻吸一气顿半拍 | 转折词前轻吸一气顿半拍 |
| 句尾「全变了样」平稳沉下 | 句尾四字短语平稳沉下 |
| 「司空见惯」加重、「这点小事」放轻 | 前半句成语加重、后半句短语放轻 |
| 「啊」借呼气单点脱口 | 句首叹词借呼气单点脱口 |
| 「结算日」「三天前」放慢咬清 | 句中名词与时间词放慢咬清 |

## 硬约束

1. **绝不出现正文的任何连续 3 个字**（务必逐字核对）—— 唯一目的
2. 不使用「」『』引号
3. 保留全部表演细节：语速、停顿、重音位置、气息、音量、句内变化
4. 长度不增（±10% 以内），语气一致，不新增表演意图

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
                    "rewritten": {"type": "string", "description": "改写后的控制词"},
                    "note": {"type": "string", "description": "一句话说明改了什么"},
                },
                "required": ["rewritten", "note"],
            },
        },
    }
]


def norm(s: str) -> str:
    from opencc import OpenCC

    return re.sub(r"[^\u4e00-\u9fff0-9a-zA-Z]", "", OpenCC("t2s").convert(str(s or "")))


def max_fragment(ctrl: str, orig: str, min_len: int = MIN_FRAG) -> int:
    """instruct 中出现的最长正文连续片段长度；小于 min_len 记 0。"""
    c, o = norm(ctrl), norm(orig)
    best = 0
    for i in range(len(o) - min_len + 1):
        for ln in range(min_len, min(14, len(o) - i) + 1):
            if o[i : i + ln] in c:
                best = max(best, ln)
            else:
                break
    return best


def load_inputs() -> tuple[dict[int, str], dict[int, dict], dict[int, str]]:
    """返回 (正文, checkpoint 段, 参考音频映射)。"""
    texts: dict[int, str] = {}
    for fn in ("performance_directions.json", "performance_directions_supplemental.json"):
        dd = json.loads((PROJECT_ROOT / "backend" / "data" / fn).read_text(encoding="utf-8"))["results"]
        for v in dd.values():
            texts[int(v["dialogue_index"])] = str(v.get("text", ""))

    ckpt = json.loads((PROJECT_ROOT / "output" / "streaming_tts.checkpoint.json").read_text(encoding="utf-8"))
    segs = ckpt["segments"]
    by = {int(v["index"]): v for v in segs.values()}

    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from run_full import load_config

    cfg = load_config(str(PROJECT_ROOT / "config" / "config.yaml"))
    voice: dict[str, str] = {}
    for key in ("characters", "voice_assignments"):
        sec = cfg.get(key) or {}
        if isinstance(sec, dict):
            for sp, val in sec.items():
                voice[str(sp)] = val if isinstance(val, str) else (val or {}).get("reference_audio", "")
    return texts, by, voice


def stage_rewrite(workers: int, limit: int = 0) -> int:
    from app.core.llm_client import LLMClient
    from concurrent.futures import ThreadPoolExecutor, as_completed

    texts, by, _ = load_inputs()
    done: dict[str, dict] = {}
    if REWRITES.is_file():
        done = json.loads(REWRITES.read_text(encoding="utf-8")).get("results", {})
    print(f"已有改写结果 {len(done)} 条", flush=True)

    # 待改：含片段的，且（未改过 或 改后仍有片段）
    todo = []
    for idx, rec in by.items():
        orig = texts.get(idx, "")
        if not orig:
            continue
        cur = str(rec.get("instruct_text", ""))
        prev = done.get(str(idx), {}).get("rewritten", "")
        cand = prev or cur
        if max_fragment(cand, orig) >= MIN_FRAG:
            todo.append((idx, orig, cand))
    if limit:
        todo = todo[:limit]
    print(f"待改写 {len(todo)} 条（含正文 {MIN_FRAG}+ 字片段）\n", flush=True)
    if not todo:
        return 0

    client = LLMClient.for_flash_lite("tts_nofrag")

    def work(item):
        idx, orig, cur = item
        frag = max_fragment(cur, orig)
        user = (
            f"正文：\n{orig}\n\n控制词：\n{cur}\n\n"
            f"控制词中出现正文的连续 {frag} 字片段，请改写为不含正文任何连续 3 字的版本。"
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

    n_ok = n_frag = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(work, t) for t in todo]
        for k, f in enumerate(as_completed(futs), 1):
            idx, data = f.result()
            if data:
                orig = texts.get(idx, "")
                left = max_fragment(data["rewritten"], orig)
                done[str(idx)] = {**data, "fragment_after": left}
                n_ok += 1
                if left >= MIN_FRAG:
                    n_frag += 1
            if k % 50 == 0:
                REWRITES.write_text(json.dumps({"results": done}, ensure_ascii=False, indent=2),
                                    encoding="utf-8")
                print(f"  进度 {k}/{len(todo)}  成功 {n_ok}  仍有片段 {n_frag}", flush=True)

    REWRITES.write_text(json.dumps({"results": done}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n改写完成：成功 {n_ok}，其中仍有片段 {n_frag}（下轮继续）", flush=True)

    # 写回 checkpoint
    ckpt_path = PROJECT_ROOT / "output" / "streaming_tts.checkpoint.json"
    ckpt = json.loads(ckpt_path.read_text(encoding="utf-8"))
    segs = ckpt["segments"]
    for k, v in segs.items():
        i = str(int(v.get("index", k)))
        if i in done:
            v["instruct_text"] = done[i]["rewritten"]
    ckpt_path.write_text(json.dumps(ckpt, ensure_ascii=False, indent=2), encoding="utf-8")
    print("已写回 checkpoint", flush=True)
    return 0


def stage_gen(batch: int, workers: int) -> int:
    from run_full import load_config, resolve_cosyvoice_python, resolve_path

    texts, by, voice = load_inputs()
    rw = json.loads(REWRITES.read_text(encoding="utf-8"))["results"] if REWRITES.is_file() else {}
    if not rw:
        print("无改写结果，先跑 stage1", flush=True)
        return 1

    cfg = load_config(str(PROJECT_ROOT / "config" / "config.yaml"))
    cv = cfg.get("cosyvoice", {})
    py = str(resolve_cosyvoice_python(cfg))
    repo = str(resolve_path(cv["repo_path"]))
    model_dir = str(resolve_path(cv["model_path"]))

    done: set[int] = set()
    if GEN_RESULTS.is_file():
        done = {int(k) for k, v in json.loads(GEN_RESULTS.read_text(encoding="utf-8")).get("results", {}).items()
                if v.get("status") == "ok"}
    todo = sorted(int(k) for k in rw if int(k) not in done)
    print(f"待重生成 {len(todo)} 条（已完成 {len(done)}）\n", flush=True)

    for s in range(0, len(todo), batch):
        chunk = todo[s : s + batch]
        tasks = []
        for i in chunk:
            rec = by[i]
            p = Path(rec["audio_path"])
            p.unlink(missing_ok=True)
            ref = voice.get(rec.get("speaker", "")) or voice.get("旁白") or ""
            rp = Path(ref)
            tasks.append({
                "index": i, "text": texts.get(i, ""), "output_path": str(p),
                "fingerprint": rec.get("fingerprint", ""),
                "reference_audio": str(rp if rp.is_absolute() else PROJECT_ROOT / rp),
                "instruct_text": rw[str(i)]["rewritten"],
            })
        spec = {"tasks": tasks, "repo_path": repo, "model_path": model_dir,
                "results_path": f"output/_nofrag_b{chunk[0]}.json",
                "task_attempts": 3, "fp16": bool(cv.get("fp16", False))}
        sp = PROJECT_ROOT / "output" / f"_nofrag_spec{chunk[0]}.json"
        sp.write_text(json.dumps(spec, ensure_ascii=False), encoding="utf-8")
        print(f"--- 批次 {chunk[0]}-{chunk[-1]}（{len(chunk)} 条）---", flush=True)
        subprocess.call([py, str(PROJECT_ROOT / "backend" / "cosyvoice_worker.py"), str(sp)])
        rp2 = PROJECT_ROOT / "output" / f"_nofrag_b{chunk[0]}.json"
        if rp2.is_file():
            merged = json.loads(GEN_RESULTS.read_text(encoding="utf-8")) if GEN_RESULTS.is_file() else {"results": {}}
            merged["results"].update(json.loads(rp2.read_text(encoding="utf-8")).get("results", {}))
            GEN_RESULTS.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
        ok = sum(1 for v in json.loads(GEN_RESULTS.read_text(encoding="utf-8"))["results"].values()
                 if v.get("status") == "ok")
        print(f"  累计成功 {ok}", flush=True)
    print("重生成完成", flush=True)
    return 0


def stage_verify(workers: int) -> int:
    rw = json.loads(REWRITES.read_text(encoding="utf-8"))["results"] if REWRITES.is_file() else {}
    indices = ",".join(sorted((str(int(k)) for k in rw), key=int))
    print(f"复检 {len(rw)} 条", flush=True)
    rc = subprocess.call([
        sys.executable, "-u", str(PROJECT_ROOT / "scripts" / "tts_quality_loop.py"),
        "transcribe", "--transcripts", str(PROJECT_ROOT / "output" / "asr_transcripts.json"),
        "--indices", indices, "--force",
    ])
    print(f"  转录 exit={rc}", flush=True)
    rc = subprocess.call([
        sys.executable, "-u", str(PROJECT_ROOT / "scripts" / "tts_quality_loop.py"),
        "audit", "--report", str(PROJECT_ROOT / "output" / "tts_quality_report.json"),
        "--indices", indices, "--workers", str(workers),
    ])
    print(f"  判定 exit={rc}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["rewrite", "gen", "verify", "status"])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--batch", type=int, default=40)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    if args.stage == "status":
        texts, by, _ = load_inputs()
        rw = json.loads(REWRITES.read_text(encoding="utf-8"))["results"] if REWRITES.is_file() else {}
        gen = json.loads(GEN_RESULTS.read_text(encoding="utf-8"))["results"] if GEN_RESULTS.is_file() else {}
        n_frag = sum(1 for i, rec in by.items()
                     if max_fragment(str(rec.get("instruct_text", "")), texts.get(i, "")) >= MIN_FRAG)
        print(f"仍含片段的 instruct : {n_frag} / {len(by)}")
        print(f"已改写               : {len(rw)}")
        print(f"已重生成             : {sum(1 for v in gen.values() if v.get('status')=='ok')}")
        return 0

    if args.stage == "rewrite":
        return stage_rewrite(args.workers, args.limit)
    if args.stage == "gen":
        return stage_gen(args.batch, args.workers)
    return stage_verify(args.workers)


if __name__ == "__main__":
    sys.exit(main())
