import sys
sys.path.insert(0, "..")

from app.core.ollama_client import OllamaClient
from app.core.gender_identifier import identify_gender, identify_all_genders


def test_single_character():
    """Test gender identification for a single character."""
    test_text = """第一章 初见

「你好，我叫罗伦斯。」——一个年轻的旅行商人说道。
他是二十五岁的旅行商人，独自行商已有七个年头。

「汝是谁？」——一个女孩的声音从马车上传来。
她有着美丽的脸孔，头上长着像小狗的耳朵。
「咱的名字是赫萝。」——女孩微笑着说。

罗伦斯拔出短剑，指着赫萝。
「你到底是谁？」
赫萝瞇起带点红色的琥珀色眼睛。
「胆敢拿剑指着咱，真是不懂礼貌。」
"""

    client = OllamaClient()

    # test 罗伦斯
    print("Testing 罗伦斯...")
    result = identify_gender("罗伦斯", test_text, client, max_tool_steps=8)
    print(f"  Result: {result}")
    assert result["gender"] == "male", f"Expected male, got {result['gender']}"
    print(f"  PASS: {result['gender']} (confidence={result['confidence']})")

    # test 赫萝
    print("\nTesting 赫萝...")
    result = identify_gender("赫萝", test_text, client, max_tool_steps=8)
    print(f"  Result: {result}")
    assert result["gender"] == "female", f"Expected female, got {result['gender']}"
    print(f"  PASS: {result['gender']} (confidence={result['confidence']})")


def test_batch():
    """Test batch gender identification."""
    test_text = """第一章 初见

「你好，我叫罗伦斯。」——一个年轻的旅行商人说道。
他是二十五岁的旅行商人。

「汝是谁？」——一个女孩的声音从马车上传来。
「咱的名字是赫萝。」——女孩微笑着说。

骑士站在门口，他穿着银色的盔甲。
「你是什么人？」——骑士大声喊道。
"""

    client = OllamaClient()

    print("Testing batch identification...")
    results = identify_all_genders(["罗伦斯", "赫萝", "骑士"], test_text, client, max_tool_steps=8)
    for r in results:
        print(f"  {r['character_name']}: {r['gender']} (confidence={r['confidence']})")

    # verify results
    by_name = {r["character_name"]: r for r in results}
    assert by_name["罗伦斯"]["gender"] == "male"
    assert by_name["赫萝"]["gender"] == "female"
    assert by_name["骑士"]["gender"] == "male"
    print("\nAll batch tests passed!")


if __name__ == "__main__":
    test_single_character()
    print()
    test_batch()
