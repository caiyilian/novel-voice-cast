"""决定性实验：instruct 长度是否是泄漏的因果因素（而非相关）。

设计
----
取同一批句子，把 instruct 改写成三个长度版本：
  S 短版：约 30-40 字（两三句话，但比 VoxCPM 的极简更丰富）
  M 中版：约 55-65 字（当前多数句子的长度）
  L 长版：约 90-100 字（当前长条的长度）

关键：三个版本**保持相同的表演意图**（语速/停顿/重音/气息），
只调整描述的详细程度。若泄漏率随长度单调上升，则长度是因果因素。

样本：从当前 100+ 字的句子里挑（这些泄漏率最高，信号最强）。
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path("E:/projects/novel-voice-cast")
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

OUT = PROJECT_ROOT / "output" / "_len_test"
OUT.mkdir(parents=True, exist_ok=True)
REPEATS = 4


def norm(s: str) -> str:
    from opencc import OpenCC

    return re.sub(r"[^\u4e00-\u9fff0-9a-zA-Z]", "", OpenCC("t2s").convert(str(s or "")))


SYSTEM = """你要为同一句话写三个不同详细程度的「表演控制词」，用于对比实验。

## 背景

控制词会与正文拼接后交给 TTS。控制词越长，模型越容易把它也念出来（泄漏）。
本实验要验证这个规律，所以需要**同一表演意图的三个长度版本**。

## 三个版本

- **S（简短）**：约 30-40 字，2-3 个短句。只保留最关键的：语速 + 情绪基调 + 1 个关键停顿。
  例如「语速平缓，语气平静克制；句中稍作停顿；句末自然收住。」
- **M（中等）**：约 55-65 字，3-4 句。增加重音位置、气息、音量变化。
- **L（详细）**：约 90-100 字，5-6 句。完整保留原控制词的全部细节。

## 硬约束

1. 三个版本的**表演意图必须一致**（同一情绪、同一节奏、同一重点）
2. **都不含正文的任何连续 3 个字**，不使用「」引号
3. 用位置指代（句首/句尾/第N分句/动词短语等）而非引用原词
4. 只输出三个版本，不要解释

## 输出

用 submit 工具提交 s / m / l 三个字段。"""

TOOL = [
    {
        "type": "function",
        "function": {
            "name": "submit_versions",
            "description": "提交三个长度版本的控制词",
            "parameters": {
                "type": "object",
                "properties": {
                    "s": {"type": "string", "description": "简短版 30-40 字"},
                    "m": {"type": "string", "description": "中等版 55-65 字"},
                    "l": {"type": "string", "description": "详细版 90-100 字"},
                },
                "required": ["s", "m", "l"],
            },
        },
    }
]


def main() -> int:
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 4

    texts: dict[int, str] = {}
    for fn in ("performance_directions.json", "performance_directions_supplemental.json"):
        dd = json.loads((PROJECT_ROOT / "backend" / "data" / fn).read_text(encoding="utf-8"))["results"]
        for v in dd.values():
            texts[int(v["dialogue_index"])] = str(v.get("text", ""))

    ckpt = json.loads((PROJECT_ROOT / "output" / "streaming_tts.checkpoint.json").read_text(encoding="utf-8"))
    by = {int(v["index"]): v for v in ckpt["segments"].values()}

    # 挑 instruct 100+ 字的（泄漏率最高，信号最强）
    cands = []
    for i, rec in by.items():
        c = str(rec.get("instruct_text", ""))
        o = norm(texts.get(i, ""))
        if len(norm(c)) >= 95 and len(o) >= 15:
            cands.append(i)
    cands = cands[:limit]
    print(f"选取 {len(cands)} 条（instruct 100+ 字）: {cands}\n", flush=True)

    # 生成三个版本
    from app.core.llm_client import LLMClient

    client = LLMClient.for_flash_lite("tts_len_test")
    versions: dict[int, dict] = {}
    for i in cands:
        rec = by[i]
        user = (
            f"正文：\n{texts[i]}\n\n"
            f"当前控制词（约 {len(norm(str(rec.get('instruct_text',''))))} 字）：\n{rec['instruct_text']}\n\n"
            f"请给出 S/M/L 三个长度版本，表演意图保持一致。"
        )
        for _ in range(3):
            try:
                res = client.chat(
                    messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
                    tools=TOOL, tool_choice="required", temperature=0.3,
                )
                if res.tool_calls:
                    versions[i] = res.tool_calls[0].arguments
                    break
            except Exception as exc:  # noqa: BLE001
                print(f"  idx {i} 失败 {type(exc).__name__}", flush=True)
        if i in versions:
            v = versions[i]
            print(f"idx {i}: S={len(norm(v['s']))} M={len(norm(v['m']))} L={len(norm(v['l']))} 字", flush=True)

    (OUT / "versions.json").write_text(json.dumps(versions, ensure_ascii=False, indent=2), encoding="utf-8")

    # 生成音频
    from run_full import load_config, resolve_cosyvoice_python, resolve_path

    cfg = load_config(str(PROJECT_ROOT / "config" / "config.yaml"))
    cv = cfg.get("cosyvoice", {})
    voice: dict[str, str] = {}
    for key in ("characters", "voice_assignments"):
        sec = cfg.get(key) or {}
        if isinstance(sec, dict):
            for sp, val in sec.items():
                voice[str(sp)] = val if isinstance(val, str) else (val or {}).get("reference_audio", "")

    tasks = []
    for i, v in versions.items():
        rec = by[i]
        ref = voice.get(rec.get("speaker", "")) or voice.get("旁白") or ""
        rp = Path(ref)
        ref = str(rp if rp.is_absolute() else PROJECT_ROOT / rp)
        for tag in ("s", "m", "l"):
            for k in range(1, REPEATS + 1):
                tasks.append({
                    "task_key": f"{i}_{tag}_r{k}",   # worker 按 task_key 去重，必须唯一
                    "index": i, "text": texts.get(i, ""),
                    "output_path": str(OUT / f"{i}_{tag}_r{k}.wav"),
                    "fingerprint": f"lentest-{i}-{tag}-{k}",
                    "reference_audio": ref,
                    "instruct_text": v[tag],
                })
    spec = {"tasks": tasks, "repo_path": str(resolve_path(cv["repo_path"])),
            "model_path": str(resolve_path(cv["model_path"])),
            "results_path": "output/_len_test/results.json",
            "task_attempts": 3, "fp16": bool(cv.get("fp16", False))}
    (OUT / "spec.json").write_text(json.dumps(spec, ensure_ascii=False), encoding="utf-8")
    print(f"\n生成 {len(tasks)} 个音频…", flush=True)
    import subprocess
    rc = subprocess.call([str(resolve_cosyvoice_python(cfg)),
                          str(PROJECT_ROOT / "backend" / "cosyvoice_worker.py"),
                          str(OUT / "spec.json")])
    print(f"worker exit={rc}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
