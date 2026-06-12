import sys
sys.path.insert(0, "..")

from app.core.parser import parse, parse_line


def test_jp_format():
    """「对话」——说话人"""
    dialogues, chars = parse('「你终于来了。」——绫小路清隆')
    assert len(dialogues) == 1
    assert dialogues[0]["text"] == "你终于来了。"
    assert dialogues[0]["speaker"] == "绫小路清隆"


def test_cn_format():
    """说话人：「对话」"""
    dialogues, chars = parse('绫小路清隆：「嗯，路上有点事。」')
    assert len(dialogues) == 1
    assert dialogues[0]["text"] == "嗯，路上有点事。"
    assert dialogues[0]["speaker"] == "绫小路清隆"


def test_mixed_formats():
    text = """「你终于来了。」——绫小路清隆
绫小路清隆：「嗯，路上有点事。」
「那就好。」——绫小路清隆"""
    dialogues, chars = parse(text)
    assert len(dialogues) == 3
    assert chars == ["绫小路清隆"]


def test_chapter_detection():
    text = """第1章 入学
「你好。」——绫小路清隆
第二章 冲突
「走开。」——堀北铃音"""
    dialogues, chars = parse(text)
    assert len(dialogues) == 2
    assert dialogues[0]["chapter"] == "第1章"
    assert dialogues[1]["chapter"] == "第二章"
    assert "堀北铃音" in chars


def test_chapter_variants():
    variants = ["第一章", "第1章", "第百章", "序章", "尾声", "前言", "后记", "Vol.1", "vol.2"]
    for v in variants:
        result = parse_line(v)
        assert result["type"] == "chapter", f"Failed to detect chapter: {v}"


def test_narrative():
    text = "教室里一片寂静。"
    result = parse_line(text)
    assert result["type"] == "narrative"


def test_blank_lines():
    dialogues, chars = parse("\n\n\n")
    assert len(dialogues) == 0


def test_narrative_skipped():
    text = """「你好。」——绫小路清隆
教室的门被推开了。
「你是谁。」——绫小路清隆"""
    dialogues, chars = parse(text)
    assert len(dialogues) == 2


def test_character_ordering():
    text = """「a」——A
「b」——B
「c」——A
「d」——C"""
    dialogues, chars = parse(text)
    assert chars == ["A", "B", "C"]


def test_character_ordering_ties_preserve_appearance():
    text = """「a」——B
「b」——A
「c」——C"""
    dialogues, chars = parse(text)
    assert chars == ["B", "A", "C"]


def test_dialogue_only_no_speaker():
    result = parse_line("「你好吗」")
    assert result["type"] == "dialogue"
    assert result["speaker"] == ""
    assert result["text"] == "你好吗"


def test_realistic_excerpt():
    text = """第3章 实力的证明

「终于来了呢。」——绫小路清隆
堀北铃音：「嗯，虽然迟了一点。」
「那么开始吧。」——绫小路清隆

旁白：教室里一片寂静，所有人都屏住了呼吸。

「哇，好厉害！」——轻井泽惠
「这怎么可能……」——平田洋介"""
    dialogues, chars = parse(text)
    assert len(dialogues) == 5
    assert chars[0] == "绫小路清隆"  # 2 dialogues, others 1 each
    assert len(chars) == 4
    assert dialogues[0]["chapter"] == "第3章"


def run():
    tests = [
        test_jp_format,
        test_cn_format,
        test_mixed_formats,
        test_chapter_detection,
        test_chapter_variants,
        test_narrative,
        test_blank_lines,
        test_narrative_skipped,
        test_character_ordering,
        test_character_ordering_ties_preserve_appearance,
        test_dialogue_only_no_speaker,
        test_realistic_excerpt,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
            print(f"  PASS {t.__name__}")
        except Exception as e:
            print(f"  FAIL {t.__name__}: {e}")
    print(f"\n{passed}/{len(tests)} passed")


if __name__ == "__main__":
    run()
