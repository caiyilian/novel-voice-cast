"""
音频去噪脚本
对 pipeline_full.py 输出的整卷音频进行去噪
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

from df.enhance import enhance, init_df, load_audio, save_audio

# ========== 配置 ==========
BASE_DIR = Path(__file__).parent.parent
INPUT_PATH = BASE_DIR / "output" / "full_volume.wav"
OUTPUT_PATH = BASE_DIR / "output" / "full_volume_denoised.wav"
MODEL_DIR = BASE_DIR / "backend" / "models" / "deepfilternet" / "DeepFilterNet3"


def format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}min"
    else:
        return f"{seconds/3600:.1f}h"


def main():
    total_start = time.time()

    print("=" * 60)
    print("音频去噪 - DeepFilterNet3")
    print("=" * 60)

    # 检查输入文件
    if not INPUT_PATH.exists():
        print(f"\n错误：输入文件不存在: {INPUT_PATH}")
        print("请先运行 pipeline_full.py 生成音频")
        return

    # 加载模型
    t0 = time.time()
    print(f"\n[1/3] 加载模型...")
    model, df_state, _ = init_df(model_base_dir=str(MODEL_DIR))
    print(f"  模型加载: {time.time()-t0:.2f}s")

    # 加载音频
    t0 = time.time()
    print(f"\n[2/3] 加载音频: {INPUT_PATH.name}")
    audio, _ = load_audio(str(INPUT_PATH), sr=df_state.sr())
    duration = len(audio) / df_state.sr()
    print(f"  时长: {duration:.1f}s ({duration/60:.1f}分钟)")
    print(f"  加载耗时: {time.time()-t0:.2f}s")

    # 去噪
    t0 = time.time()
    print(f"\n[3/3] 去噪处理...")
    enhanced = enhance(model, df_state, audio)
    print(f"  处理耗时: {time.time()-t0:.2f}s")
    print(f"  RTF: {(time.time()-t0)/duration:.3f}")

    # 保存
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    save_audio(str(OUTPUT_PATH), enhanced, df_state.sr())

    # 汇总
    total_time = time.time() - total_start
    print("\n" + "=" * 60)
    print("完成")
    print("=" * 60)
    print(f"  输入: {INPUT_PATH}")
    print(f"  输出: {OUTPUT_PATH}")
    print(f"  时长: {duration:.1f}s ({duration/60:.1f}分钟)")
    print(f"  总耗时: {format_time(total_time)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
