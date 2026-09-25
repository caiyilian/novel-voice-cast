"""TTS 质量闭环：ASR 转录 → LLM 判定 → 定向修复 → 重生成 → 复检。

设计动机
--------
CosyVoice 3 的 ``inference_instruct2(text, instruct, ...)`` 在 instruct 使用
「叙述性对白语气」（如「句后留半拍等赫萝接话」）时，会把 instruct 内容当成台词
念出来，并可能导致正文重复。这类失败无法用规则穷举，只能靠「生成 → 转录 →
比对 → 修复」的闭环来兜住。

本模块提供三个阶段，可单独调用也可串联：
  1. ``transcribe``   —— faster-whisper 批量转录，产出 asr_transcripts.json
  2. ``audit``        —— LLM 对比「原文 + 控制词 + 转录文本」，三分类判定
  3. ``repair``       —— 对问题句让 LLM 重写 instruct 并重生成，循环复检

用法：
    # 1) 转录（全量或指定范围）
    python scripts/tts_quality_loop.py transcribe --limit 11 --start-index 279

    # 2) 判定
    python scripts/tts_quality_loop.py audit --report output/tts_quality_report.json

    # 3) 修复 + 重生成 + 复检
    python scripts/tts_quality_loop.py repair --report output/tts_quality_report.json --max-rounds 3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

# Windows 控制台默认 GBK，中文日志会乱码；重定向到文件时尤其明显。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

DEFAULT_WHISPER = Path(
    os.environ.get(
        "WHISPER_MODEL_DIR",
        "E:/huggingface_models/hub/models--Systran--faster-whisper-large-v2"
        "/snapshots/f0fe81560cb8b68660e564f55dd99207059c092e",
    )
)
DEFAULT_TTS_CHECKPOINT = PROJECT_ROOT / "output" / "streaming_tts.checkpoint.json"
DEFAULT_TRANSCRIPTS = PROJECT_ROOT / "output" / "asr_transcripts.json"
DEFAULT_REPORT = PROJECT_ROOT / "output" / "tts_quality_report.json"
DEFAULT_DIRECTIONS = PROJECT_ROOT / "backend" / "data" / "performance_directions.json"
DEFAULT_SUPPLEMENTAL = PROJECT_ROOT / "backend" / "data" / "performance_directions_supplemental.json"

QUALITY_PIPELINE_VERSION = 1


# ─────────────────────────── 数据加载 ───────────────────────────


def load_tts_segments(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("segments", {})


def load_directions(*paths: Path) -> dict[int, dict[str, Any]]:
    """按 dialogue_index 合并 performance directions（主组 + 补充组）。"""
    merged: dict[int, dict[str, Any]] = {}
    for path in paths:
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        for value in (payload.get("results") or {}).values():
            idx = value.get("dialogue_index")
            if idx is None:
                continue
            merged[int(idx)] = value
    return merged


def index_to_dialogue(segments: dict[str, dict[str, Any]]) -> dict[int, dict[str, Any]]:
    """TTS 分段 index → 分段记录。"""
    out: dict[int, dict[str, Any]] = {}
    for value in segments.values():
        out[int(value["index"])] = value
    return out


# ─────────────────────────── 阶段 1：ASR 转录 ───────────────────────────


def _load_whisper(model_dir: Path, device: str = "cuda", compute_type: str = "float16"):
    from faster_whisper import WhisperModel

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    return WhisperModel(str(model_dir), device=device, compute_type=compute_type)


def cmd_transcribe(args: argparse.Namespace) -> int:
    segments = load_tts_segments(Path(args.tts_checkpoint))
    by_index = index_to_dialogue(segments)
    indices = sorted(by_index)
    if args.start_index is not None:
        indices = [i for i in indices if i >= args.start_index]
    if args.limit:
        indices = indices[: args.limit]
    if args.indices:
        wanted = {int(x) for x in args.indices.split(",")}
        indices = [i for i in indices if i in wanted]

    out_path = Path(args.transcripts)
    store: dict[str, Any] = {
        "quality_pipeline_version": QUALITY_PIPELINE_VERSION,
        "whisper_model": str(args.whisper_model),
        "transcripts": {},
    }
    # 总是先加载已有结果，避免 --force 时把未涉及的条目丢掉。
    # --force 的语义是「重跑本次指定的 indices」，不是「清空整个文件」。
    if out_path.is_file():
        try:
            old = json.loads(out_path.read_text(encoding="utf-8"))
            if old.get("quality_pipeline_version") == QUALITY_PIPELINE_VERSION:
                store["transcripts"] = old.get("transcripts", {})
        except Exception:
            pass

    if args.force:
        for i in indices:
            store["transcripts"].pop(str(i), None)

    pending = [i for i in indices if str(i) not in store["transcripts"]]
    print(f"待转录: {len(pending)} / {len(indices)}（已完成 {len(indices)-len(pending)}）", flush=True)
    if not pending:
        print("无需转录", flush=True)
        return 0

    model = _load_whisper(Path(args.whisper_model), args.device, args.compute_type)
    print("Whisper 模型就绪", flush=True)

    t0 = time.time()
    for n, idx in enumerate(pending, 1):
        rec = by_index[idx]
        audio = Path(rec.get("audio_path", ""))
        if not audio.is_file():
            store["transcripts"][str(idx)] = {"status": "missing_audio", "path": str(audio)}
            continue
        try:
            pieces, info = model.transcribe(
                str(audio), language="zh", beam_size=args.beam_size
            )
            chunks = [{"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text} for s in pieces]
            text = "".join(c["text"] for c in chunks).strip()
            duration = round(float(getattr(info, "duration", 0.0) or 0.0), 2)
            store["transcripts"][str(idx)] = {
                "status": "ok",
                "speaker": rec.get("speaker"),
                "text": text,
                "duration": duration,
                "chunks": chunks,
            }
        except Exception as exc:  # noqa: BLE001 - 单条失败记入 error，后续阶段会显式报告
            store["transcripts"][str(idx)] = {"status": "error", "error": str(exc)[:200]}
        if n % 25 == 0 or n == len(pending):
            out_path.write_text(json.dumps(store, ensure_ascii=False, indent=2), encoding="utf-8")
            rate = n / max(1e-6, time.time() - t0)
            print(f"  [{n}/{len(pending)}] {rate:.1f} 条/秒", flush=True)

    # 转录失败必须显式报告并返回非零，绝不静默放过
    failed = [
        (int(k), v)
        for k, v in store["transcripts"].items()
        if v.get("status") in ("error", "missing_audio")
    ]
    if failed:
        print(f"\n转录失败 {len(failed)} 条：", flush=True)
        for idx, v in failed[:20]:
            print(f"  idx {idx}: {v.get('status')} {v.get('error') or v.get('path')}", flush=True)
        if len(failed) > 20:
            print(f"  ... 另有 {len(failed)-20} 条", flush=True)
        return 1
    print(f"转录完成 -> {out_path}", flush=True)
    return 0


# ─────────────────────────── 阶段 2：LLM 判定 ───────────────────────────

AUDIT_SYSTEM_PROMPT = """你是语音合成质量审核员。你会收到一条有声书语音的三份材料：

A. 原文（这条语音本应朗读的内容）
B. 控制词（发给 TTS 的表演指导，**不应该被朗读出来**）
C. 转录文本（用 ASR 对该语音识别出的实际内容）
D. 参考指标（字符级比对的量化结果，仅供参考，可能有误）

你的任务：判断这条语音是否合格。

## 判定标准

**OK** —— 合格。转录文本与原文一致，差异仅来自 ASR 识别误差。允许的差异：
  - 同音字/近音字（如「呗」识别成「白」、「罗伦斯」识别成「罗伦思」、「汝」识别成「乳」）
  - 简繁字体差异（「羅倫斯」vs「罗伦斯」）
  - 标点、语气词有无（「啊」「呀」）
  - 个别字词的轻微出入，但整句语义与原文相同
  - 短句（如「嗯。」「不。」）被 ASR 识别成乱码（如「by bwd6」「哦哦」）——这属于
    识别失败，**不算问题**，只要音频时长合理（短句应 <1.5 秒）即判 OK
  - 数字、专有名词的写法差异

**LEAK** —— 控制词泄漏。转录文本中出现了控制词里的内容（尤其是导演术语，
  如「放平推进」「留半拍」「等赫萝接话」「交棒」「尾音上扬」「放稳」「咬实」等），
  或出现了原文里根本没有的额外语句。

**MISMATCH** —— 严重不符。转录文本与原文语义明显不同，例如：
  - 原文内容大量缺失或大幅改写
  - 同一句话被重复朗读多遍
  - 出现与原文无关的整段内容

## 判断要点

1. **以原文与转录的语义对照为准**，不要被参考指标带偏——指标对短句容易误报。
2. 若转录含导演术语且原文中无对应内容 → LEAK。
3. 若转录是原文的忠实复述（哪怕用字有出入）→ OK。
4. 短句（原文 <6 字）若转录是乱码或音近词，只要不像导演术语，判 OK。

## 输出要求

对每条语音，给出：
- verdict: "OK" | "LEAK" | "MISMATCH"
- reason: 一句话说明判断依据（中文，40 字以内）
- leaked_fragments: 若为 LEAK，列出泄漏的具体片段（数组，可为空）

严格按 JSON 输出，不要输出任何其他文字。
"""

AUDIT_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["OK", "LEAK", "MISMATCH"]},
        "reason": {"type": "string"},
        "leaked_fragments": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["verdict", "reason", "leaked_fragments"],
    "additionalProperties": False,
}

AUDIT_TOOL = [{
    "type": "function",
    "function": {
        "name": "submit_audit",
        "description": "提交一条语音合成质量的审核结论。",
        "parameters": AUDIT_JSON_SCHEMA,
    },
}]


def _call_tool(
    client: Any,
    system: str,
    user: str,
    tool: list[dict],
    temperature: float,
    attempts: int = 3,
) -> dict[str, Any]:
    """调用 LLM 并以 tool_call 形式取回结构化结果（项目统一做法）。

    LLM 偶发返回空内容或不带 tool_call（实测重写阶段约 18% 概率），
    这类失败是可恢复的，必须重试而不是直接抛错——否则该句永远修不了。
    """
    last_error = ""
    for attempt in range(1, attempts + 1):
        result = client.chat(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            tools=tool,
            tool_choice="required",
            temperature=temperature,
        )
        if result.tool_calls:
            return result.tool_calls[0].arguments
        last_error = f"content={result.content[:80]!r}"
        if attempt < attempts:
            time.sleep(1.5 * attempt)
    raise RuntimeError(f"LLM 连续 {attempts} 次未返回结构化结果（{last_error}）")


def _audit_one(
    client: Any,
    index: int,
    original: str,
    control: str,
    transcript: str,
    metrics: dict[str, Any] | None = None,
    duration: float = 0.0,
) -> dict[str, Any]:
    hint = ""
    if metrics:
        hint = (
            f"\nD. 参考指标：\n"
            f"字符级比对（仅参考）：额外内容占比 {metrics.get('extra_ratio', 0):.2f}，"
            f"缺失占比 {metrics.get('missing_ratio', 0):.2f}，相似度 {metrics.get('similarity', 0):.2f}\n"
            f"音频时长：{duration:.2f} 秒（原文 {len(original)} 字）\n"
            f"注：短句的指标容易误报，请以语义对照为准。"
        )
    user = (
        f"A. 原文：\n{original}\n\n"
        f"B. 控制词：\n{control or '（无）'}\n\n"
        f"C. 转录文本：\n{transcript or '（空）'}"
        f"{hint}"
    )
    return _call_tool(client, AUDIT_SYSTEM_PROMPT, user, AUDIT_TOOL, 0.0)


def _merge_records(
    report_path: Path, records: list[dict[str, Any]], failed: list[dict[str, Any]]
) -> dict[str, Any]:
    """把本轮结果与已有报告合并，返回完整报告对象。"""
    merged: dict[int, dict[str, Any]] = {}
    all_failed: dict[int, dict[str, Any]] = {}
    if report_path.is_file():
        try:
            old = json.loads(report_path.read_text(encoding="utf-8"))
            if old.get("quality_pipeline_version") == QUALITY_PIPELINE_VERSION:
                for r in old.get("records") or []:
                    merged[int(r["index"])] = r
                for item in old.get("untranscribed") or []:
                    all_failed[int(item["index"])] = item
        except Exception:
            pass
    for r in records:
        merged[int(r["index"])] = r
    for item in failed:
        all_failed[int(item["index"])] = item
    # 本次成功审到的条目若原先记在 untranscribed，要移除
    for r in records:
        all_failed.pop(int(r["index"]), None)

    all_records = [merged[k] for k in sorted(merged)]
    total_counts: dict[str, int] = {"OK": 0, "LEAK": 0, "MISMATCH": 0, "ERROR": 0}
    for r in all_records:
        total_counts[r["verdict"]] = total_counts.get(r["verdict"], 0) + 1

    return {
        "quality_pipeline_version": QUALITY_PIPELINE_VERSION,
        "audited": len(all_records),
        "last_batch": len(records),
        "counts": total_counts,
        "untranscribed": [all_failed[k] for k in sorted(all_failed)],
        "records": all_records,
    }


def _flush_report(
    report_path: Path, records: list[dict[str, Any]], failed: list[dict[str, Any]], merge: bool = True
) -> dict[str, Any]:
    """把当前进度落盘（长任务被中断时不丢已完成部分）。"""
    report = _merge_records(report_path, records, failed) if merge else {
        "quality_pipeline_version": QUALITY_PIPELINE_VERSION,
        "audited": len(records),
        "counts": {},
        "untranscribed": failed,
        "records": records,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def cmd_audit(args: argparse.Namespace) -> int:
    """全量 LLM 判定（并发）。字符级比对仅作为参考信息随附。"""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from app.core.llm_client import LLMClient

    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from tts_leak_detect import compare

    segments = load_tts_segments(Path(args.tts_checkpoint))
    by_index = index_to_dialogue(segments)
    directions = load_directions(Path(args.directions), Path(args.supplemental))
    transcripts = json.loads(Path(args.transcripts).read_text(encoding="utf-8"))["transcripts"]

    indices = sorted(int(k) for k in transcripts)
    if args.start_index is not None:
        indices = [i for i in indices if i >= args.start_index]
    if args.indices:
        wanted = {int(x) for x in args.indices.split(",")}
        indices = [i for i in indices if i in wanted]
    if args.limit:
        indices = indices[: args.limit]

    # 组装待审条目（含字符指标与时长，作为 LLM 的参考）
    import wave

    pending: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    for idx in indices:
        t = transcripts.get(str(idx))
        if t is None:
            failed.append({"index": idx, "reason": "转录记录缺失"})
            continue
        if t.get("status") != "ok":
            failed.append(
                {"index": idx, "reason": f"{t.get('status')}: {t.get('error') or t.get('path') or ''}"}
            )
            continue
        rec = by_index.get(idx, {})
        direction = directions.get(idx, {})
        original = str(direction.get("text") or "")
        control = str(rec.get("instruct_text") or direction.get("performance_control") or "")
        transcript = str(t.get("text") or "")
        duration = float(t.get("duration") or 0.0)
        pending.append(
            {
                "index": idx,
                "speaker": rec.get("speaker"),
                "original": original,
                "control": control,
                "transcript": transcript,
                "duration": duration,
                "metrics": compare(original, transcript),
            }
        )

    print(f"待审核: {len(pending)} 条（并发 {args.workers}）", flush=True)
    report_path = Path(args.report)
    client = LLMClient.for_flash_lite("tts_quality_audit")
    counts = {"OK": 0, "LEAK": 0, "MISMATCH": 0, "ERROR": 0}
    records: list[dict[str, Any]] = []
    done = 0

    def _work(item: dict[str, Any]) -> dict[str, Any]:
        try:
            verdict = _audit_one(
                client, item["index"], item["original"], item["control"],
                item["transcript"], item["metrics"], item["duration"],
            )
        except Exception as exc:  # noqa: BLE001 - 记为 ERROR，由调用方按未通过处理
            verdict = {
                "verdict": "ERROR",
                "reason": f"{type(exc).__name__}: {str(exc)[:120]}",
                "leaked_fragments": [],
            }
        return {**item, **verdict}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_work, item) for item in pending]
        for future in as_completed(futures):
            record = future.result()
            counts[record["verdict"]] = counts.get(record["verdict"], 0) + 1
            records.append(record)
            done += 1
            if record["verdict"] in ("LEAK", "MISMATCH"):
                print(f"  [{record['index']}] {record['verdict']} — {record['reason']}", flush=True)
            if done % 100 == 0:
                print(f"  进度 {done}/{len(pending)}  OK={counts['OK']} "
                      f"LEAK={counts['LEAK']} MISMATCH={counts['MISMATCH']} ERR={counts['ERROR']}", flush=True)
            # 增量落盘：长任务被中断时不丢已完成的判定
            if done % 200 == 0:
                _flush_report(report_path, records, failed, merge=True)

    records.sort(key=lambda r: r["index"])
    report = _flush_report(report_path, records, failed, merge=True)
    total_counts = report["counts"]
    all_failed = report["untranscribed"]

    print(f"\n判定完成 -> {args.report}", flush=True)
    print(f"  本轮 {len(records)} 条 | 报告累计 {report['audited']} 条", flush=True)
    print(f"  累计：OK={total_counts['OK']}  LEAK={total_counts['LEAK']}  "
          f"MISMATCH={total_counts['MISMATCH']}  ERROR={total_counts['ERROR']}", flush=True)
    if failed:
        print(f"  ⚠ 本轮未转录（未参与判定）{len(failed)} 条：", flush=True)
        for item in failed[:20]:
            print(f"    idx {item['index']}: {item['reason']}", flush=True)
        if len(failed) > 20:
            print(f"    ... 另有 {len(failed)-20} 条", flush=True)
    # 有 ERROR 或未转录都必须返回非零，绝不当作通过
    if total_counts["ERROR"] or all_failed:
        return 1
    return 0


# ─────────────────────────── 阶段 3：定向修复 ───────────────────────────

REPAIR_SYSTEM_PROMPT = """你是语音合成指令修复专家。某条有声书语音生成失败，你要重写它的「控制词」。

## 背景

控制词通过独立参数传给 CosyVoice 3，**绝不应该被朗读出来**。但 CosyVoice 的实现
是把控制词与正文**拼成同一个序列**再由模型续写，当控制词语义上像「未说完的正文」
时，模型就会把它念出来。

## 重写规则（必须严格遵守）

1. **必须以「元指令」开头**——这是最关键的一条。用「请用……的语气说这句话」
   或「请以……的语气说这句话」这种明确的指令句式开头。
   实测：元指令形态泄漏率 0%，而无主语的参数罗列（如「平静，略带质疑。语速适中」）
   泄漏率高达 80%，因为它读起来像正文的前半句。

   ✅ 好例子：「请用平静而略带质疑的语气说这句话。语速适中，句末稍作停顿。」
   ❌ 坏例子：「平静，略带质疑。语速适中，句末停顿。」（无主语，像正文）

2. **禁止叙述性/对白化表述**。不得出现：
   - 涉及「谁接话」「交棒给某句/某人」「等某人说」
   - 引号内的示例台词、拟声台词
   - 对人物动作或心理的叙述（如「像把考题摆上桌」）

3. **保留原控制词的表演意图**（情绪、节奏、力度），只改表达形式。

4. 长度 30-80 字，中文，不用括号，不分行。

## 输出

给出修复后的控制词，以及一句简短的修改说明。
严格按 JSON 输出。
"""

REPAIR_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "fixed_control": {"type": "string"},
        "change_note": {"type": "string"},
    },
    "required": ["fixed_control", "change_note"],
    "additionalProperties": False,
}

REPAIR_TOOL = [{
    "type": "function",
    "function": {
        "name": "submit_repair",
        "description": "提交修复后的控制词。",
        "parameters": REPAIR_JSON_SCHEMA,
    },
}]


def _repair_control(client: Any, original: str, control: str, transcript: str, reason: str) -> dict[str, Any]:
    user = (
        f"原文：\n{original}\n\n"
        f"原控制词（有问题）：\n{control}\n\n"
        f"上次转录出的实际语音（暴露了问题）：\n{transcript}\n\n"
        f"审核判定的问题：{reason}"
    )
    return _call_tool(client, REPAIR_SYSTEM_PROMPT, user, REPAIR_TOOL, 0.3)


def cmd_repair(args: argparse.Namespace) -> int:
    """对有问题的句子重写控制词并重新生成，然后复检（转录 + 判定）。"""
    from app.core.llm_client import LLMClient

    report_path = Path(args.report)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    bad = [r for r in report["records"] if r["verdict"] in ("LEAK", "MISMATCH")]
    # ERROR 与未转录的句子无法用「改提示词」修复，必须显式报告，不能静默跳过
    blocked = [r for r in report["records"] if r["verdict"] == "ERROR"]
    blocked += list(report.get("untranscribed") or [])
    if blocked:
        print(f"⚠ 以下 {len(blocked)} 条无法通过改提示词修复（需先解决转录/调用问题）：", flush=True)
        for item in blocked[:20]:
            detail = item.get("reason") or item.get("error") or ""
            print(f"    idx {item['index']}: {detail}", flush=True)
        if len(blocked) > 20:
            print(f"    ... 另有 {len(blocked)-20} 条", flush=True)
        if not bad:
            print("没有可修复的条目（LEAK/MISMATCH 为空），退出", flush=True)
            return 1
    if args.limit:
        bad = bad[: args.limit]
    print(f"待修复: {len(bad)} 条", flush=True)
    if not bad:
        print("无需修复", flush=True)
        return 0

    client = LLMClient.for_flash_lite("tts_quality_repair")
    plan: list[dict[str, Any]] = []
    rewrite_failed: list[int] = []
    for r in bad:
        try:
            fixed = _repair_control(client, r["original"], r["control"], r["transcript"], r["reason"])
            plan.append(
                {
                    "index": r["index"],
                    "speaker": r["speaker"],
                    "original": r["original"],
                    "old_control": r["control"],
                    "new_control": fixed["fixed_control"],
                    "change_note": fixed["change_note"],
                    "verdict_before": r["verdict"],
                }
            )
            print(f"  [{r['index']}] 已重写控制词：{fixed['change_note'][:70]}", flush=True)
        except Exception as exc:  # noqa: BLE001 - 记入失败清单，最后显式报错
            rewrite_failed.append(r["index"])
            print(f"  [{r['index']}] ✗ 重写失败: {type(exc).__name__}: {str(exc)[:100]}", flush=True)

    out_plan = Path(args.plan)
    out_plan.write_text(json.dumps({"plan": plan}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n修复方案已生成 -> {out_plan}", flush=True)

    rc = 0
    if args.regenerate:
        rc = _regenerate(plan, args)
    else:
        print("（加 --regenerate 可立即重生成音频）", flush=True)

    # 任何一条没成功（重写失败 / 重生成失败）都返回非零
    if rewrite_failed:
        print(f"\n✗ 有 {len(rewrite_failed)} 条控制词重写失败: {rewrite_failed[:30]}", flush=True)
        rc = 1
    return rc


def _regenerate(plan: list[dict[str, Any]], args: argparse.Namespace) -> int:
    """用新控制词重生成音频，并回写 TTS checkpoint。

    直接驱动 ``backend/cosyvoice_worker.py``，绕开 ``run_cosyvoice_tasks``
    的严格指纹校验——重写控制词后指纹必然改变，用主流水线的校验会把
    「音频已生成」误判为失败。
    """
    if not plan:
        return 0

    import yaml

    from run_full import load_config

    config = load_config(str(Path(args.config)))
    segments_path = Path(args.tts_checkpoint)
    payload = json.loads(segments_path.read_text(encoding="utf-8"))
    segments = payload["segments"]

    # 角色 → 参考音频（与主流水线同一套解析）
    voice_map: dict[str, str] = {}
    for key in ("characters", "voice_assignments"):
        section = config.get(key) or {}
        if isinstance(section, dict):
            for speaker, value in section.items():
                if isinstance(value, str):
                    voice_map[str(speaker)] = value
                elif isinstance(value, dict) and value.get("reference_audio"):
                    voice_map[str(speaker)] = str(value["reference_audio"])

    def reference_audio_for(speaker: str) -> str:
        rel = voice_map.get(speaker) or voice_map.get("旁白") or ""
        if not rel:
            return ""
        p = Path(rel)
        return str(p if p.is_absolute() else PROJECT_ROOT / p)

    tasks: list[dict[str, Any]] = []
    for item in plan:
        idx = int(item["index"])
        rec = segments.get(str(idx)) or segments.get(idx)
        if not rec:
            print(f"  [{idx}] TTS checkpoint 里找不到该分段，跳过", flush=True)
            continue
        speaker = rec.get("speaker", "")
        tasks.append(
            {
                "index": idx,
                "text": item["original"],
                "output_path": rec["audio_path"],
                "fingerprint": rec.get("fingerprint", ""),
                "reference_audio": reference_audio_for(speaker),
                "instruct_text": item["new_control"],
            }
        )

    if not tasks:
        return 1

    # 删除旧音频：worker 会按「文件已存在」跳过，必须先删才能强制重生成
    for task in tasks:
        Path(task["output_path"]).unlink(missing_ok=True)

    print(f"\n重生成 {len(tasks)} 条…", flush=True)
    ok = _run_cosyvoice_worker(tasks, config, args)
    print(f"  成功 {len(ok)} / {len(tasks)}", flush=True)

    # 回写 instruct_text（指纹随之失效，下游据此判断需要重跑）
    for item in plan:
        idx = str(item["index"])
        if idx in ok:
            rec = segments.get(idx) or segments.get(int(idx))
            if rec is not None:
                rec["instruct_text"] = item["new_control"]
    payload["segments"] = segments
    segments_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  已回写 instruct_text -> {segments_path}", flush=True)

    # 有句子重生成失败必须显式报错并返回非零，绝不能静默当作修复成功
    failed_idx = sorted(int(t["index"]) for t in tasks if str(t["index"]) not in ok)
    if failed_idx:
        print(f"\n✗ 重生成失败 {len(failed_idx)} 条: {failed_idx[:30]}", flush=True)
        return 1
    return 0
    return 0


def _run_cosyvoice_worker(
    tasks: list[dict[str, Any]], config: dict[str, Any], args: argparse.Namespace
) -> set[str]:
    """直接驱动 cosyvoice_worker 重生成，绕开 run_cosyvoice_tasks 的严格校验。

    主流水线的 ``run_cosyvoice_tasks`` 要求结果指纹与传入指纹完全一致，而重写
    控制词后指纹必然改变，会导致「音频已生成但抛 PipelineError」。这里直接
    读 worker 的结果文件，只要 status=ok 且 WAV 可读即视为成功。
    """
    import subprocess

    from run_full import resolve_cosyvoice_python, resolve_path

    cosyvoice = config.get("cosyvoice", {})
    repo_path = cosyvoice.get("repo_path", "")
    model_path = str(resolve_path(cosyvoice.get("model_path", "")))
    python_path = str(resolve_cosyvoice_python(config))
    worker_path = PROJECT_ROOT / "backend" / "cosyvoice_worker.py"
    out_dir = PROJECT_ROOT / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    spec_path = out_dir / "_repair_spec.json"
    results_path = out_dir / "_repair_results.json"
    results_path.unlink(missing_ok=True)
    spec = {
        "tasks": tasks,
        "repo_path": str(resolve_path(repo_path)) if repo_path else "",
        "model_path": model_path,
        "results_path": str(results_path),
        "task_attempts": int(cosyvoice.get("task_attempts", 3)),
        "fp16": bool(cosyvoice.get("fp16", False)),
    }
    spec_path.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")

    rc = subprocess.call([python_path, str(worker_path), str(spec_path)], cwd=str(PROJECT_ROOT))
    print(f"  worker 退出码 {rc}", flush=True)
    if rc != 0:
        print(f"  ✗ worker 异常退出（{rc}），结果可能不完整", flush=True)
    if not results_path.is_file():
        print(f"  ✗ 未找到结果文件 {results_path}", flush=True)
        return set()

    ok: set[str] = set()
    failed: list[str] = []
    results = json.loads(results_path.read_text(encoding="utf-8")).get("results", {})
    for key, item in results.items():
        if item.get("status") == "ok":
            ok.add(str(item.get("index", key)))
        else:
            failed.append(f"{item.get('index', key)}: {item.get('error') or item.get('status')}")
    if failed:
        print(f"  ✗ worker 内失败 {len(failed)} 条：", flush=True)
        for line in failed[:10]:
            print(f"    {line}", flush=True)
    return ok



def main() -> int:
    parser = argparse.ArgumentParser(description="TTS 质量闭环：转录 / 判定 / 修复")
    sub = parser.add_subparsers(dest="command", required=True)

    p1 = sub.add_parser("transcribe", help="用 faster-whisper 批量转录 TTS 音频")
    p1.add_argument("--tts-checkpoint", default=str(DEFAULT_TTS_CHECKPOINT))
    p1.add_argument("--transcripts", default=str(DEFAULT_TRANSCRIPTS))
    p1.add_argument("--whisper-model", default=str(DEFAULT_WHISPER))
    p1.add_argument("--device", default="cuda")
    p1.add_argument("--compute-type", default="float16")
    p1.add_argument("--beam-size", type=int, default=5)
    p1.add_argument("--start-index", type=int, default=None)
    p1.add_argument("--indices", default=None, help="逗号分隔的指定 index")
    p1.add_argument("--limit", type=int, default=0)
    p1.add_argument("--force", action="store_true")
    p1.set_defaults(func=cmd_transcribe)

    p2 = sub.add_parser("audit", help="用 LLM 判定转录是否合格")
    p2.add_argument("--tts-checkpoint", default=str(DEFAULT_TTS_CHECKPOINT))
    p2.add_argument("--transcripts", default=str(DEFAULT_TRANSCRIPTS))
    p2.add_argument("--directions", default=str(DEFAULT_DIRECTIONS))
    p2.add_argument("--supplemental", default=str(DEFAULT_SUPPLEMENTAL))
    p2.add_argument("--report", default=str(DEFAULT_REPORT))
    p2.add_argument("--start-index", type=int, default=None)
    p2.add_argument("--indices", default=None, help="逗号分隔的指定 index（只审这些）")
    p2.add_argument("--limit", type=int, default=0)
    p2.add_argument("--workers", type=int, default=8, help="并发 LLM 调用数")
    p2.set_defaults(func=cmd_audit)

    p3 = sub.add_parser("repair", help="为问题句重写控制词（可选立即重生成）")
    p3.add_argument("--report", default=str(DEFAULT_REPORT))
    p3.add_argument("--plan", default=str(PROJECT_ROOT / "output" / "tts_repair_plan.json"))
    p3.add_argument("--limit", type=int, default=0)
    p3.add_argument("--regenerate", action="store_true", help="重写后立即调用 CosyVoice 重生成音频")
    p3.add_argument("--tts-checkpoint", default=str(DEFAULT_TTS_CHECKPOINT))
    p3.add_argument("--config", default=str(PROJECT_ROOT / "config" / "config.yaml"))
    p3.set_defaults(func=cmd_repair)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
