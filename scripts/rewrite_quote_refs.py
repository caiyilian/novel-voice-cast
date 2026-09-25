"""把 instruct 中「引用正文原词」改写成「位置描述」（不含正文原词）。

问题
----
实测三种变体的干净率（各 12 次）：
  A 原版（保留「」引用）      33%
  B 只去引号符号（保留词）    58%
  C 整体删除引用词           100%  ← 但破坏语义（「重音落与」）

C 有效但语义崩坏。本脚本采用 **D 方案**：把引用改写成位置/顺序描述，
既不出现正文原词，又保留可读性。

改写规则（用 LLM，因为需要理解语义）
------------------------------------
  重音落「迎风摇曳」与「狼」   →  重音落在句首词组与句尾单字
  「停止挥手」前略顿           →  该动作短语前略顿
  「啊」借呼气单点脱口         →  句首叹词借呼气单点脱口
  「麦子」音量放轻             →  句尾名词音量放轻

关键约束：
1. 输出中【绝不出现】正文的任何原词（含单字）
2. 用「句首/句尾/第一分句/动作短语/叹词/名词」等位置词代替
3. 长度尽量不增加，保留原有节奏感
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

SYSTEM = """你要改写有声书的「表演控制词」，消除其中的**正文词引用**。

## 背景

控制词通过独立参数传给 TTS 引擎，本不该被朗读。但实测发现：
控制词里用「」引用正文原词时，模型容易把控制词当成正文继续念出来。

实测数据（干净率）：
- 保留「」引用      33%
- 只去掉引号符号    58%
- 整体删除引用词    100%（但语义崩坏，如「重音落与」）

所以要你改写：**不出现正文任何原词**，但保持控制词的可读与可执行。

## 改写方法

把「引用具体词」换成「位置 / 类型描述」：

| 原写法 | 改写后 |
|---|---|
| 重音落「迎风摇曳」与「狼」 | 重音落在句首词组与句尾单字 |
| 「停止挥手」前略顿 | 该动作短语前略顿 |
| 「啊」借呼气单点脱口 | 句首叹词借呼气单点脱口 |
| 「麦子」音量放轻 | 句尾名词音量放轻 |
| 「司空见惯」加重、「这点小事」放轻 | 前半句成语加重、后半句短语放轻 |

## 硬约束

1. **输出绝不能出现正文中的任何词**（包括单字）——这是核心目的
2. 用「句首/句尾/第一分句/第二分句/动作短语/叹词/名词/成语/数量词」等位置词
3. 保留原有的语速、停顿、音量、气息描述
4. 长度尽量不增加（±10% 以内）
5. 语气与原文一致，不要新增表演意图

## 输出

只输出改写后的控制词文本，不要任何解释。"""

TOOL = [
    {
        "type": "function",
        "function": {
            "name": "submit_rewrite",
            "description": "提交改写后的控制词",
            "parameters": {
                "type": "object",
                "properties": {
                    "rewritten": {"type": "string", "description": "改写后的控制词，不含正文原词"},
                    "note": {"type": "string", "description": "一句话说明改了什么"},
                },
                "required": ["rewritten", "note"],
            },
        },
    }
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 条（0=全部）")
    ap.add_argument("--out", default="output/_quote_rewrite.json")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    from app.core.llm_client import LLMClient
    from concurrent.futures import ThreadPoolExecutor, as_completed

    report = json.loads((PROJECT_ROOT / "output" / "tts_quality_report.json").read_text(encoding="utf-8"))
    # 找含「引用正文词」的条目
    todo = []
    for r in report["records"]:
        qs = re.findall(r"「([^」]+)」", r["control"])
        if qs and any(x in r["original"] for x in qs):
            todo.append(r)
    if args.limit:
        todo = todo[: args.limit]
    print(f"待改写 {len(todo)} 条（含引用正文词的引号）\n", flush=True)

    client = LLMClient.for_flash_lite("tts_quote_rewrite")
    results: dict[int, dict] = {}
    failed: list[int] = []

    def work(r: dict) -> tuple[int, dict | None, str]:
        user = f"正文：\n{r['original']}\n\n原控制词：\n{r['control']}"
        try:
            res = client.chat(
                messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
                tools=TOOL, tool_choice="required", temperature=0.2,
            )
            if not res.tool_calls:
                return r["index"], None, "no tool_call"
            return r["index"], res.tool_calls[0].arguments, ""
        except Exception as exc:  # noqa: BLE001
            return r["index"], None, f"{type(exc).__name__}"

    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(work, r) for r in todo]
        for f in as_completed(futs):
            idx, data, err = f.result()
            done += 1
            if data:
                results[idx] = data
            else:
                failed.append(idx)
            if done % 20 == 0:
                print(f"  进度 {done}/{len(todo)}  成功 {len(results)} 失败 {len(failed)}", flush=True)

    # 校验：改写后是否仍含正文原词
    by_idx = {r["index"]: r for r in todo}
    violations = []
    for idx, data in results.items():
        r = by_idx[idx]
        t = r["original"]
        # 检查 2-gram 是否出现在正文
        txt = data["rewritten"]
        for i in range(len(txt) - 1):
            if txt[i : i + 2] in t:
                violations.append((idx, txt[i : i + 2], txt[:60]))
                break

    out = {"results": results, "failed": failed, "violations": violations}
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n完成 -> {args.out}", flush=True)
    print(f"  成功 {len(results)} / 失败 {len(failed)}", flush=True)
    print(f"  仍含正文 2-gram 的: {len(violations)}", flush=True)
    for idx, g, txt in violations[:8]:
        print(f"    idx {idx}: 「{g}」 {txt}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
