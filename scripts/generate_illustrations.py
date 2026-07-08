"""Phase 2 — 批量生成插图

从 illustration_plan.json 读取所有插图 prompt，逐条调用 PuLID SDXL API 生成图片。
使用 animagine 方法（纯文生图，动漫画质最佳）。

用法:
    .venv\Scripts\python scripts/generate_illustrations.py
    .venv\Scripts\python scripts/generate_illustrations.py --resume
"""

import json
import time
from pathlib import Path

import requests

SERVER_HOST = "172.31.102.189"
SERVER_PORT = 8001
BASE_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"

PLAN_PATH = Path("output/illustration_plan.json")
OUTPUT_DIR = Path("output/illustrations")
CHECKPOINT_PATH = Path("output/illustrations_checkpoint.json")

MAX_RETRIES = 3
RETRY_DELAY = 5
REQUEST_INTERVAL = 1

NEG_PROMPT = (
    "flaws in the eyes, flaws in the face, flaws, lowres, non-HDRi, low quality, "
    "worst quality, artifacts noise, text, watermark, glitch, deformed, mutated, "
    "ugly, disfigured, hands, low resolution, partially rendered objects, "
    "deformed or partially rendered eyes, cross-eyed, blurry"
)


def load_plan() -> list[dict]:
    data = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    return data.get("illustrations", data) if isinstance(data, dict) else data


def load_checkpoint() -> set[int]:
    if CHECKPOINT_PATH.exists():
        return set(json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8")))
    return set()


def save_checkpoint(done: set[int]) -> None:
    CHECKPOINT_PATH.write_text(json.dumps(sorted(done)), encoding="utf-8")


def generate_one(prompt: str, idx: int, total: int) -> bytes:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(f"{BASE_URL}/generate", data={
                "prompt": prompt,
                "method": "animagine",
                "steps": 25,
                "cfg": 7.0,
                "width": 896,
                "height": 1152,
                "neg_prompt": NEG_PROMPT,
            }, timeout=300)
            if resp.status_code == 200:
                return resp.content
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
        except Exception as e:
            if attempt < MAX_RETRIES:
                print(f"    [{idx}/{total}] retry {attempt}/{MAX_RETRIES}: {e}")
                time.sleep(RETRY_DELAY * attempt)
            else:
                raise


def main():
    plan = load_plan()
    done = load_checkpoint()
    total = len(plan)
    resume = "--resume" in [a for a in __import__("sys").argv]

    print(f"plan: {total} images")
    print(f"done: {len(done)}")
    print(f"output: {OUTPUT_DIR}")
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    errors = []
    t0 = time.time()

    for i in range(total):
        if i in done:
            continue
        p = plan[i]
        prompt = p.get("prompt", "")
        title = p.get("title", f"img_{i:04d}")
        if not prompt.strip():
            done.add(i)
            save_checkpoint(done)
            continue

        elapsed = time.time() - t0
        done_cnt = len(done) + 1
        avg = elapsed / max(1, done_cnt)
        eta = f"{int((total-i-1)*avg//60):02d}:{int((total-i-1)*avg%60):02d}" if avg > 0 else "--:--"
        print(f"  [{i+1}/{total}] {title}  ETA {eta}")

        try:
            img = generate_one(prompt, i + 1, total)
            out = OUTPUT_DIR / f"{i+1:04d}_{title}.png"
            out.write_bytes(img)
            done.add(i)
            save_checkpoint(done)
            print(f"    -> {len(img)/1024:.0f} KB")
        except Exception as e:
            print(f"    [ERR] #{i+1} {title}: {e}")
            errors.append({"idx": i, "title": title, "err": str(e)})
            done.add(i)
            save_checkpoint(done)

        time.sleep(REQUEST_INTERVAL)

    elapsed = time.time() - t0
    ok = total - len(errors)
    print(f"\n{'='*50}")
    print(f"DONE: {ok}/{total} ok, {len(errors)} failed, {int(elapsed//60)}min")
    if errors:
        for e in errors[:5]:
            print(f"  #{e['idx']+1} {e['title']}: {e['err']}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()