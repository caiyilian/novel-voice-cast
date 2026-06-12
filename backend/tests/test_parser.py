import sys
sys.path.insert(0, "..")

from app.core.parser import parse, parse_line, extract_chapters


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
    assert dialogues[0]["chapter"] == "第1章 入学"
    assert dialogues[1]["chapter"] == "第二章 冲突"
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
    assert dialogues[0]["chapter"] == "第3章 实力的证明"


# ─── Stage 4.5b: Chapter extraction tests ──────────────────────────

def test_chinese_chapter_with_digits():
    """第1章, 第12节, 第100部"""
    for v in ["第1章", "第12节", "第100部", "第5集", "第3卷", "第7篇"]:
        result = parse_line(v)
        assert result["type"] == "chapter", f"Failed: {v}"


def test_chinese_chapter_with_chinese_numbers():
    """第一章, 第十二节, 第一百部"""
    for v in ["第一章", "第十二节", "第一百部", "第三卷", "第七篇"]:
        result = parse_line(v)
        assert result["type"] == "chapter", f"Failed: {v}"


def test_chinese_chapter_with_subtitle():
    """第1章：觉醒, 第2章 - 风暴"""
    for v in ["第1章：觉醒", "第2章——风暴", "第三章.新的开始", "第5章 - 出发"]:
        result = parse_line(v)
        assert result["type"] == "chapter", f"Failed: {v}"


def test_english_chapter():
    """Chapter 1, Chapter 12, Chap. 3"""
    for v in ["Chapter 1", "Chapter 12", "Chap. 3", "Ch 5", "Part 1", "Volume 2", "Vol.3", "Book 1"]:
        result = parse_line(v)
        assert result["type"] == "chapter", f"Failed: {v}"


def test_special_chapters():
    """序章, 前言, 尾声, etc."""
    for v in ["序章", "前言", "后记", "尾声", "番外", "楔子", "引子", "序幕", "终章", "番外篇", "特别篇"]:
        result = parse_line(v)
        assert result["type"] == "chapter", f"Failed: {v}"


def test_chapter_with_trailing_text():
    """Chapter titles with trailing descriptive text"""
    text = "第1章 觉醒——那一天改变了世界"
    result = parse_line(text)
    assert result["type"] == "chapter"


def test_extract_chapters_basic():
    text = """这是开头
第1章 初见
一些内容
第二章 风暴
更多内容"""
    chapters = extract_chapters(text)
    assert len(chapters) == 2
    assert chapters[0]["title"] == "第1章 初见"
    assert chapters[0]["line_number"] == 2
    assert chapters[1]["title"] == "第二章 风暴"
    assert chapters[1]["line_number"] == 4


def test_extract_chapters_empty():
    chapters = extract_chapters("")
    assert len(chapters) == 0


def test_extract_chapters_no_chapters():
    chapters = extract_chapters("just some text\nno chapters here")
    assert len(chapters) == 0


def test_extract_chapters_realistic():
    text = """在这个村落，人们会把迎风摇曳的饱满麦穗形容成狼在奔跑。
第7章 初见
「你好。」——赫萝
赫萝：「嗯，刚刚好。」
序章 回忆
「那是很久以前的事了。」——罗伦斯"""
    chapters = extract_chapters(text)
    assert len(chapters) == 2
    assert chapters[0]["title"] == "第7章 初见"
    assert chapters[1]["title"] == "序章 回忆"


def test_extract_chapters_line_numbers():
    text = """line 1
line 2
第1章 First
line 4
line 5
Chapter 2 Second
line 7"""
    chapters = extract_chapters(text)
    assert len(chapters) == 2
    assert chapters[0]["line_number"] == 3
    assert chapters[1]["line_number"] == 6


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
        # Stage 4.5b chapter tests
        test_chinese_chapter_with_digits,
        test_chinese_chapter_with_chinese_numbers,
        test_chinese_chapter_with_subtitle,
        test_english_chapter,
        test_special_chapters,
        test_chapter_with_trailing_text,
        test_extract_chapters_basic,
        test_extract_chapters_empty,
        test_extract_chapters_no_chapters,
        test_extract_chapters_realistic,
        test_extract_chapters_line_numbers,
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
