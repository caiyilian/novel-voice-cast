"""
测试 ACE-Step-1.5 BGM 生成
直接用 Python SDK，不需要启动 API server。
"""
import os
import sys
import time

# 切换到 ACE-Step-1.5 目录
ACE_DIR = r"E:\projects\novel-voice-cast\ACE-Step-1.5"
os.chdir(ACE_DIR)
sys.path.insert(0, ACE_DIR)

# 设置环境变量（不加载 LM，省 VRAM，RTX 3060 12GB）
os.environ["ACESTEP_CONFIG_PATH"] = "acestep-v15-turbo"
os.environ["ACESTEP_LM_MODEL_PATH"] = "acestep-5Hz-lm-1.7B"
os.environ["ACESTEP_DEVICE"] = "auto"
os.environ["ACESTEP_INIT_LLM"] = "false"  # 不加载 LM，先测试 DiT 单独跑

from acestep.handler import AceStepHandler
from acestep.inference import GenerationParams, GenerationConfig, generate_music

OUTPUT_DIR = r"E:\projects\novel-voice-cast\output\bgm_test"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    print("=" * 50)
    print("ACE-Step-1.5 BGM 生成测试")
    print("=" * 50)

    # 1. 初始化 DiT 模型（会自动下载或用本地缓存）
    print("\n[1/2] 加载 DiT 模型...")
    t0 = time.time()

    dit_handler = AceStepHandler()
    dit_handler.initialize_service(
        project_root=ACE_DIR,
        config_path="acestep-v15-turbo",
        device="cuda",
    )
    print(f"  模型加载完成 [{time.time() - t0:.1f}s]")

    # 不加载 LM，直接用 DiT text2music 模式

    # 2. 生成 BGM
    print("\n[2/2] 生成 BGM...")

    # 测试用例
    test_cases = [
        {
            "name": "紧张悬疑",
            "params": GenerationParams(
                caption="tense suspenseful dark ambient music, mysterious thriller soundtrack, low strings, eerie synth pads",
                duration=30,
                inference_steps=8,
            ),
        },
        {
            "name": "日常温馨",
            "params": GenerationParams(
                caption="light cheerful acoustic background music, warm and gentle, soft guitar and piano, everyday life mood",
                duration=30,
                inference_steps=8,
            ),
        },
    ]

    config = GenerationConfig(
        batch_size=1,
        audio_format="mp3",
    )

    for case in test_cases:
        name = case["name"]
        params = case["params"]
        save_dir = os.path.join(OUTPUT_DIR, name)
        os.makedirs(save_dir, exist_ok=True)

        print(f"\n  生成: {name}")
        t1 = time.time()

        result = generate_music(dit_handler, None, params, config, save_dir=save_dir)

        if result.success:
            for audio in result.audios:
                path = audio["path"]
                size_mb = os.path.getsize(path) / (1024 * 1024)
                print(f"    ✅ {path} ({size_mb:.1f} MB) [{time.time() - t1:.1f}s]")
        else:
            print(f"    ❌ 失败: {result.error}")

    print("\n" + "=" * 50)
    print("测试完成")
    print(f"输出目录: {OUTPUT_DIR}")
    print("=" * 50)


if __name__ == "__main__":
    main()
