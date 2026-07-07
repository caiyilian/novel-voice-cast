#!/usr/bin/env python
"""测试插图生成 API（连接 novel-illustration 服务）

用法:
    python scripts/test_illustration_api.py                          # 默认服务器
    python scripts/test_illustration_api.py --host 192.168.1.100     # 指定服务器
    python scripts/test_illustration_api.py --port 8000              # 指定端口
    python scripts/test_illustration_api.py --no-ref-only            # 只测无参考图
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import requests

DEFAULT_HOST = "172.31.102.189"
DEFAULT_PORT = 8001
REF_IMAGES_DIR = Path(__file__).resolve().parent.parent.parent / "novel-illustration" / "ref_images"
OUTPUT_DIR = Path("output/test_illustration")


def test_health(base: str) -> dict:
    r = requests.get(f"{base}/health", timeout=10)
    data = r.json()
    print(f"[健康检查] {data}")
    return data


def test_no_ref(base: str, out: Path) -> Path:
    prompt = (
        "anime style, a quiet medieval village street at sunset, "
        "cobblestone path, warm golden light, thatched roofs, "
        "flower boxes, masterpiece, high quality"
    )
    r = requests.post(
        f"{base}/generate",
        data={"prompt": prompt, "steps": 20, "cfg": 7.0},
        timeout=120,
    )
    path = out / "no_ref.png"
    path.write_bytes(r.content)
    print(f"[无参考图] {len(r.content) / 1024:.0f} KB -> {path}")
    return path


def test_with_ref(base: str, ref_path: Path, prompt: str, out_name: str, out: Path) -> Path:
    with open(ref_path, "rb") as f:
        r = requests.post(
            f"{base}/generate",
            data={
                "prompt": prompt,
                "id_scale": 0.8,
                "num_zero": 20,
                "steps": 25,
                "cfg": 7.0,
            },
            files={"ref_image": f},
            timeout=120,
        )
    path = out / out_name
    path.write_bytes(r.content)
    print(f"[有参考图] {len(r.content) / 1024:.0f} KB -> {path}")
    return path


def main() -> int:
    p = argparse.ArgumentParser(description="测试插图生成 API")
    p.add_argument("--host", default=DEFAULT_HOST, help=f"服务器地址（默认 {DEFAULT_HOST}）")
    p.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"端口（默认 {DEFAULT_PORT}）")
    p.add_argument("--no-ref-only", action="store_true", help="只测试无参考图模式")
    args = p.parse_args()

    base = f"http://{args.host}:{args.port}"
    out = Path(OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)

    print(f"服务器: {base}")
    print(f"输出目录: {out}")
    print()

    # 1. 健康检查
    try:
        test_health(base)
    except requests.ConnectionError as e:
        print(f"[错误] 无法连接服务器 {base}: {e}")
        return 1
    print()

    # 2. 无参考图
    test_no_ref(base, out)
    print()

    if args.no_ref_only:
        print("=== 完成（仅无参考图） ===")
        return 0

    # 3. 有参考图 - 需要 ref_images 目录
    for char_name, img_file, prompt in [
        (
            "holo",
            "holo.png",
            "portrait of a cute wolf girl with brown hair and wolf ears, "
            "medieval traveler outfit and hood, smiling, holding an apple, "
            "warm autumn forest, anime style, masterpiece",
        ),
        (
            "lls",
            "lls.png",
            "portrait of a young male traveling merchant with short brown hair, "
            "medieval cloak and hat, walking through a busy medieval market street, "
            "carrying a backpack, anime style, masterpiece",
        ),
    ]:
        img_path = REF_IMAGES_DIR / img_file
        if not img_path.is_file():
            print(f"[跳过] 参考图不存在: {img_path}")
            continue
        test_with_ref(base, img_path, prompt, f"ref_{char_name}.png", out)
        print()

    print("=== 全部完成 ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())