"""按长度压缩 instruct：把 >65 字的压到 55-65 字（保留表演细节）。

实验依据（scripts/_len_experiment.py，6 句 × 3 版本 × 4 次）
----------------------------------------------------------
  版本                平均覆盖率   干净率
  S 简短(~30字)          1.00      24/24 = 100%
  M 中等(~60字)          1.00      24/24 = 100%   ← 采用
  L 详细(~100字)         1.21      20/24 = 83%

同一批句子、同一表演意图，只改长度 → 长度是因果因素。
且 60 字已达 100% 干净，**不需要压到 VoxCPM 那样的极简**。

压缩原则
--------
1. 保留：语速 + 情绪基调 + 关键停顿位置 + 重音位置 + 句末收法
2. 删减：重复的修饰、次要的过渡句、过度具体的比喻
3. 不引用正文（沿用无片段化规则）
4. 目标 55-65 字
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "backend"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

TARGET_MIN, TARGET_MAX = 38, 48
THRESHOLD = 50          # 超过此长度才压缩

SYSTEM = """你要压缩有声书的「表演控制词」，在保留关键表演细节的前提下缩短篇幅。

## 为什么要压缩

控制词与正文拼接后交给 TTS。实测（同一批句子、同一表演意图，只改长度）：

  控制词 0-50 字  → 泄漏率 0.0%
  控制词 50-60 字 → 泄漏率 1.3%
  控制词 60-70 字 → 泄漏率 2.8%
  控制词 70-80 字 → 泄漏率 3.3%
  控制词 80+ 字   → 泄漏率 6.1%

控制词越长，模型越容易把它也念出来。所以目标是把超过 50 字的压到 38-48 字（约两三句话）。

## 压缩原则

**必须保留**（这些是表演的灵魂）：
- 语速与整体情绪基调
- 关键停顿的位置（如「转折前顿半拍」）
- 重音落在哪（如「句尾四字短语加重」）
- 气息处理（如「轻吸一口气」「气声下沉」）
- 句末收法（如「放轻收尾留余味」）

**可以删减**：
- 重复或同义的修饰（「不抬声不念课文」与「克制」重复，留一个）
- 次要的过渡描述
- 过度具体的比喻（「如车轮碾过草地般」→ 删）
- 对正文内容的复述（本来就不该有）

## 硬约束

1. 目标长度 **38-48 字**（汉字数，不含标点，约两三句话）
2. **不含正文的任何连续 3 个字**，不使用「」引号
3. 用位置指代（句首/句尾/第N分句/动词短语/四字短语等）
4. 保持原控制词的表演意图，不新增、不改变情绪
5. 输出中文，用「；」分隔短句

## 输出

只输出压缩后的控制词。"""

TOOL = [
    {
        "type": "function",
        "function": {
            "name": "submit_compact",
            "description": "提交压缩后的控制词",
            "parameters": {
                "type": "object",
                "properties": {
                    "compacted": {"type": "string", "description": "压缩后的控制词（38-48 字）"},
                    "kept": {"type": "string", "description": "保留了哪些关键细节"},
                },
                "required": ["compacted", "kept"],
            },
        },
    }
]


def norm(s: str) -> str:
    from opencc import OpenCC

    return re.sub(r"[^\u4e00-\u9fff0-9a-zA-Z]", "", OpenCC("t2s").convert(str(s or "")))


def max_fragment(ctrl: str, orig: str, min_len: int = 3) -> int:
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
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="output/_compact.json")
    args = ap.parse_args()

    from app.core.llm_client import LLMClient
    from concurrent.futures import ThreadPoolExecutor, as_completed

    texts: dict[int, str] = {}
    for fn in ("performance_directions.json", "performance_directions_supplemental.json"):
        dd = json.loads((PROJECT_ROOT / "backend" / "data" / fn).read_text(encoding="utf-8"))["results"]
        for v in dd.values():
            texts[int(v["dialogue_index"])] = str(v.get("text", ""))

    ckpt = json.loads((PROJECT_ROOT / "output" / "streaming_tts.checkpoint.json").read_text(encoding="utf-8"))
    by = {int(v["index"]): v for v in ckpt["segments"].values()}

    out_path = PROJECT_ROOT / args.out
    done: dict[str, dict] = {}
    if out_path.is_file():
        done = json.loads(out_path.read_text(encoding="utf-8")).get("results", {})

    todo = []
    for i, rec in by.items():
        cur = str(rec.get("instruct_text", ""))
        prev = done.get(str(i), {}).get("compacted", "")
        cand = prev or cur
        if len(norm(cand)) > THRESHOLD:
            todo.append((i, texts.get(i, ""), cand))
    if args.limit:
        todo = todo[: args.limit]
    print(f"待压缩 {len(todo)} 条（>{THRESHOLD} 字），已有 {len(done)} 条\n", flush=True)
    if not todo:
        return 0

    client = LLMClient.for_flash_lite("tts_compact")

    def work(item):
        i, orig, cur = item
        user = (
            f"正文：\n{orig}\n\n"
            f"当前控制词（{len(norm(cur))} 字）：\n{cur}\n\n"
            f"请压缩到 38-48 字，保留语速/停顿/重音/气息/句末收法等关键细节。"
        )
        for _ in range(3):
            try:
                res = client.chat(
                    messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
                    tools=TOOL, tool_choice="required", temperature=0.2,
                )
                if res.tool_calls:
                    return i, res.tool_calls[0].arguments
            except Exception:
                pass
        return i, None

    ok = bad_len = bad_frag = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(work, t) for t in todo]
        for k, f in enumerate(as_completed(futs), 1):
            i, data = f.result()
            if not data:
                continue
            txt = data["compacted"]
            L = len(norm(txt))
            fg = max_fragment(txt, texts.get(i, ""))
            done[str(i)] = {**data, "len": L, "fragment": fg}
            ok += 1
            if not (TARGET_MIN - 10 <= L <= TARGET_MAX + 15):
                bad_len += 1
            if fg >= 3:
                bad_frag += 1
            if k % 100 == 0:
                out_path.write_text(json.dumps({"results": done}, ensure_ascii=False, indent=2), encoding="utf-8")
                print(f"  进度 {k}/{len(todo)}  成功 {ok}  超长 {bad_len}  含片段 {bad_frag}", flush=True)

    out_path.write_text(json.dumps({"results": done}, ensure_ascii=False, indent=2), encoding="utf-8")

    # 写回 checkpoint
    ckpt_path = PROJECT_ROOT / "output" / "streaming_tts.checkpoint.json"
    ckpt = json.loads(ckpt_path.read_text(encoding="utf-8"))
    for k, v in ckpt["segments"].items():
        i = str(int(v.get("index", k)))
        if i in done:
            v["instruct_text"] = done[i]["compacted"]
    ckpt_path.write_text(json.dumps(ckpt, ensure_ascii=False, indent=2), encoding="utf-8")

    lens = [v["len"] for v in done.values()]
    import statistics
    print(f"\n压缩完成：{ok} 条", flush=True)
    print(f"  平均长度 {statistics.mean(lens):.1f} 字（目标 55-65）", flush=True)
    print(f"  超出目标区间 {bad_len} 条，含片段 {bad_frag} 条", flush=True)
    print("已写回 checkpoint", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
