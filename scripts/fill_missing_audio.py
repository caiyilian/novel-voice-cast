"""补齐 checkpoint 中缺失或过期的音频。

依据：
- 以 checkpoint 的 instruct_text 为唯一真相源（sha256 指纹）
- 只生成「文件缺失」或「指纹不匹配」的条目
- 用 cosyvoice_worker.py 分批执行，支持断点续跑

用法：
    python scripts/fill_missing_audio.py            # 全部检查
    python scripts/fill_missing_audio.py --limit 20 # 仅前 20 条
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from run_full import load_config, resolve_cosyvoice_python, resolve_path  # noqa: E402

CHECKPOINT = PROJECT_ROOT / "output" / "streaming_tts.checkpoint.json"
MANIFEST = PROJECT_ROOT / "output" / "_audio_manifest.json"
# 基线指纹：gen9（nofrag_all.py gen）写出的记录，可直接复用，避免重复生成
BASELINE_MANIFEST = PROJECT_ROOT / "output" / "_nofrag_manifest.json"
DATA_FILES = (
    PROJECT_ROOT / "backend" / "data" / "performance_directions.json",
    PROJECT_ROOT / "backend" / "data" / "performance_directions_supplemental.json",
)


def fingerprint(text: str) -> str:
    return hashlib.sha256(str(text or "").encode("utf-8")).hexdigest()[:16]


def load_voice_map(cfg: dict) -> dict[str, str]:
    voice: dict[str, str] = {}
    for key in ("characters", "voice_assignments"):
        sec = cfg.get(key) or {}
        if isinstance(sec, dict):
            for sp, val in sec.items():
                voice[str(sp)] = val if isinstance(val, str) else (val or {}).get("reference_audio", "")
    return voice


def load_texts() -> dict[int, str]:
    texts: dict[int, str] = {}
    for fn in DATA_FILES:
        if not fn.is_file():
            continue
        data = json.loads(fn.read_text(encoding="utf-8")).get("results", {})
        for v in data.values():
            texts[int(v["dialogue_index"])] = str(v.get("text", ""))
    return texts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=20)
    ap.add_argument("--limit", type=int, default=0, help="最多处理多少条（0=全部）")
    ap.add_argument("--indices", type=str, default="", help="只处理指定索引，逗号分隔")
    ap.add_argument("--all", action="store_true", help="忽略指纹，全部重新生成")
    args = ap.parse_args()

    ckpt = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    by = {int(v["index"]): v for v in ckpt["segments"].values()}
    manifest: dict[str, str] = {}
    if MANIFEST.is_file():
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    # 继承基线（gen9 的产出），避免把已生成正确的 2454 条误判为过期
    if BASELINE_MANIFEST.is_file():
        base = json.loads(BASELINE_MANIFEST.read_text(encoding="utf-8"))
        for k, v in base.items():
            manifest.setdefault(k, v)

    cfg = load_config(str(PROJECT_ROOT / "config" / "config.yaml"))
    cv = cfg.get("cosyvoice", {})
    voice = load_voice_map(cfg)
    texts = load_texts()

    todo: list[int] = []
    for i, rec in sorted(by.items()):
        inst = str(rec.get("instruct_text", ""))
        p = Path(rec["audio_path"])
        fp = fingerprint(inst)
        if args.all:
            todo.append(i)
        elif not p.is_file() or p.stat().st_size == 0:
            todo.append(i)
        elif manifest.get(str(i)) != fp:
            todo.append(i)

    if args.limit:
        todo = todo[: args.limit]
    if args.indices:
        only = {int(x) for x in args.indices.split(",") if x.strip()}
        todo = [i for i in todo if i in only]
        # 指定索引时忽略指纹：强制重生成（供人工判定需要时使用）
        todo = sorted(set(todo) | (only & set(by)))

    missing = [i for i in todo if not Path(by[i]["audio_path"]).is_file()]
    stale = [i for i in todo if i not in missing]
    print(f"总条目 {len(by)} | 需生成 {len(todo)}（缺失 {len(missing)}、过期 {len(stale)}）", flush=True)
    if not todo:
        print("全部已是最新", flush=True)
        return 0

    py = str(resolve_cosyvoice_python(cfg))
    repo = str(resolve_path(cv["repo_path"]))
    model_dir = str(resolve_path(cv["model_path"]))

    for s in range(0, len(todo), args.batch):
        chunk = todo[s : s + args.batch]
        tasks = []
        for i in chunk:
            rec = by[i]
            ref = voice.get(rec.get("speaker", "")) or voice.get("旁白") or ""
            rp = Path(ref)
            tasks.append(
                {
                    "index": i,
                    "text": texts.get(i, ""),
                    "output_path": rec["audio_path"],
                    "fingerprint": fingerprint(str(rec.get("instruct_text", ""))),
                    "reference_audio": str(rp if rp.is_absolute() else PROJECT_ROOT / rp),
                    "instruct_text": str(rec.get("instruct_text", "")),
                }
            )
        spec_path = PROJECT_ROOT / "output" / f"_fill_spec{s}.json"
        res_path = PROJECT_ROOT / "output" / f"_fill_res{s}.json"
        spec_path.write_text(
            json.dumps(
                {
                    "tasks": tasks,
                    "repo_path": repo,
                    "model_path": model_dir,
                    "results_path": str(res_path),
                    "task_attempts": 3,
                    "fp16": bool(cv.get("fp16", False)),
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(f"--- 批次 {s//args.batch+1}: {chunk[0]}-{chunk[-1]}（{len(chunk)} 条）---", flush=True)
        subprocess.call([py, str(PROJECT_ROOT / "backend" / "cosyvoice_worker.py"), str(spec_path)])

        if res_path.is_file():
            got = json.loads(res_path.read_text(encoding="utf-8")).get("results", {})
            for t in tasks:
                k = str(t["index"])
                if got.get(k, {}).get("status") == "ok":
                    manifest[k] = t["fingerprint"]
        MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

        have = sum(1 for t in tasks if Path(t["output_path"]).is_file())
        print(f"  本批完成 {have}/{len(tasks)}", flush=True)

    left = [i for i in todo if not Path(by[i]["audio_path"]).is_file()]
    print(f"\n补全结束，仍缺失 {len(left)}" + (f": {left[:20]}" if left else ""), flush=True)
    return 0 if not left else 1


if __name__ == "__main__":
    raise SystemExit(main())
