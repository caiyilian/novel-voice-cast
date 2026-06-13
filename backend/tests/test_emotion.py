import sys
sys.path.insert(0, "..")

from app.core.ollama_client import OllamaClient
from app.core.emotion_labeler import label_emotion, EMOTIONS, TONES


def test_single_dialogue():
    """Test emotion identification for a single dialogue."""
    test_text = """第一章 初见

罗伦斯坐在马车上，望着远方的城镇。

「你好，我叫罗伦斯。」——罗伦斯微笑着说。

赫萝瞇起眼睛，露出不悦的表情。
「胆敢拿剑指着咱，真是不懂礼貌。」——赫萝冷冷地说。

「哈哈哈哈哈，咱是恶魔？」——赫萝大笑着说。

「呜……咱虽然不讨厌人类的外表，不过还是太冷了。」——赫萝颤抖着说。"""

    client = OllamaClient()

    # test happy dialogue
    print("Testing happy dialogue...")
    result = label_emotion(
        dialogue_text="你好，我叫罗伦斯。",
        dialogue_line=5,
        dialogue_index=0,
        text=test_text,
        client=client,
        max_tool_steps=8,
    )
    print(f"  Result: {result}")
    assert result["emotion"] in EMOTIONS, f"Invalid emotion: {result['emotion']}"
    assert result["tone"] in TONES, f"Invalid tone: {result['tone']}"
    print(f"  PASS: emotion={result['emotion']}, tone={result['tone']}")

    # test angry dialogue
    print("\nTesting angry dialogue...")
    result = label_emotion(
        dialogue_text="胆敢拿剑指着咱，真是不懂礼貌。",
        dialogue_line=10,
        dialogue_index=1,
        text=test_text,
        client=client,
        max_tool_steps=8,
    )
    print(f"  Result: {result}")
    assert result["emotion"] in EMOTIONS, f"Invalid emotion: {result['emotion']}"
    assert result["tone"] in TONES, f"Invalid tone: {result['tone']}"
    print(f"  PASS: emotion={result['emotion']}, tone={result['tone']}")

    # test laughing dialogue
    print("\nTesting laughing dialogue...")
    result = label_emotion(
        dialogue_text="哈哈哈哈哈，咱是恶魔？",
        dialogue_line=13,
        dialogue_index=2,
        text=test_text,
        client=client,
        max_tool_steps=8,
    )
    print(f"  Result: {result}")
    assert result["emotion"] in EMOTIONS, f"Invalid emotion: {result['emotion']}"
    assert result["tone"] in TONES, f"Invalid tone: {result['tone']}"
    print(f"  PASS: emotion={result['emotion']}, tone={result['tone']}")


def test_multiple_dialogues():
    """Test emotion labeling for multiple dialogues (one at a time)."""
    test_text = """第一章 初见

罗伦斯坐在马车上。

「你好，我叫罗伦斯。」——罗伦斯微笑着说。

赫萝瞇起眼睛。
「胆敢拿剑指着咱，真是不懂礼貌。」——赫萝冷冷地说。

「哈哈哈哈哈，咱是恶魔？」——赫萝大笑着说。"""

    client = OllamaClient()

    dialogues = [
        {"index": 0, "line": 3, "text": "你好，我叫罗伦斯。"},
        {"index": 1, "line": 6, "text": "胆敢拿剑指着咱，真是不懂礼貌。"},
        {"index": 2, "line": 8, "text": "哈哈哈哈哈，咱是恶魔？"},
    ]

    print("Testing multiple dialogues (one at a time)...")
    results = []
    for d in dialogues:
        result = label_emotion(
            dialogue_text=d["text"],
            dialogue_line=d["line"],
            dialogue_index=d["index"],
            text=test_text,
            client=client,
            max_tool_steps=8,
        )
        results.append(result)
        print(f"  Dialogue {result['dialogue_index']}: emotion={result['emotion']}, tone={result['tone']}")

    # verify all results have valid emotions and tones
    for r in results:
        assert r["emotion"] in EMOTIONS, f"Invalid emotion: {r['emotion']}"
        assert r["tone"] in TONES, f"Invalid tone: {r['tone']}"
    print("\nAll multiple dialogue tests passed!")


if __name__ == "__main__":
    test_single_dialogue()
    print()
    test_multiple_dialogues()
