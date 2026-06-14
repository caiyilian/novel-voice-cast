import re
from typing import List, Tuple, Dict


# ─── Chapter patterns ──────────────────────────────────────────────
# Chinese: 第X章/节/部/集/卷/篇 (X = digits or Chinese numbers)
_CH_NUM = r'[一二三四五六七八九十百千万零\d]+'
CHINESE_CH = rf'第{_CH_NUM}[章节部集卷篇]'

# English: Chapter X / Chap. X / Part X / Volume X
ENGLISH_CH = r'(?:Chapter|Chap\.?|Ch\.?|Part|Volume|Vol\.?|Book)\s*\d+'

# Special markers
SPECIAL_CH = r'(?:序章|前言|后记|尾声|番外|楔子|引子|序幕|终章|完结|后记|番外篇|特别篇)'

# Combined: must match at line start, capture the full title
CHAPTER_PATTERN = re.compile(
    rf'^(?:{CHINESE_CH}|{ENGLISH_CH}|{SPECIAL_CH})'
    rf'(?:[\s:：.．\-—]+.*)?$',
    re.IGNORECASE
)

# ─── Dialogue patterns ─────────────────────────────────────────────
DIALOGUE_JP = re.compile(r'「([^」]*)」[—－\-—\s]*(.+?)$')
DIALOGUE_CN = re.compile(r'「([^」]*)」[\s　]*$')
SPEAKER_PREFIX = re.compile(r'^(.+?)：[「『]')
# 匹配行中间的「对话」（不在行尾）
DIALOGUE_INLINE = re.compile(r'「([^」]*)」')


def parse_line(text: str, current_chapter: str = "") -> dict:
    text = text.strip()
    if not text:
        return {"type": "blank", "chapter": current_chapter}

    m = CHAPTER_PATTERN.match(text)
    if m:
        current_chapter = m.group(0).strip()
        return {"type": "chapter", "chapter": current_chapter, "title": text}

    # 「对话」——说话人
    m = DIALOGUE_JP.match(text)
    if m:
        dialogue_text = m.group(1).strip()
        speaker = m.group(2).strip()
        return {"type": "dialogue", "chapter": current_chapter, "text": dialogue_text, "speaker": speaker}

    # 说话人：「对话」
    m = SPEAKER_PREFIX.match(text)
    if m:
        speaker = m.group(1).strip()
        dialogue_text = text[m.end():]
        if dialogue_text.endswith('」'):
            dialogue_text = dialogue_text[:-1]
        return {"type": "dialogue", "chapter": current_chapter, "text": dialogue_text.strip(), "speaker": speaker}

    # 独立「对话」——说话人在上一行或本身不带标注
    m = DIALOGUE_CN.match(text)
    if m:
        dialogue_text = m.group(1).strip()
        return {"type": "dialogue", "chapter": current_chapter, "text": dialogue_text, "speaker": ""}

    # 行中间的「对话」
    m = DIALOGUE_INLINE.search(text)
    if m:
        dialogue_text = m.group(1).strip()
        return {"type": "dialogue", "chapter": current_chapter, "text": dialogue_text, "speaker": ""}

    return {"type": "narrative", "chapter": current_chapter, "text": text}


def parse(text: str, labels: List[str] = None) -> Tuple[List[dict], List[str]]:
    """Parse novel text and extract dialogues and characters.

    Args:
        text: Novel text content
        labels: Optional list of speaker labels from labels.txt (one per dialogue)

    Returns:
        Tuple of (dialogues, character_list)
    """
    chapters = []
    dialogues = []
    characters = []
    seen = set()
    current_chapter = ""
    dialogue_index = 0

    for line in text.splitlines():
        result = parse_line(line, current_chapter)
        current_chapter = result.get("chapter", current_chapter)

        if result["type"] == "chapter":
            chapters.append(result)
        elif result["type"] == "dialogue":
            # If labels provided, assign speaker from labels.txt
            if labels and dialogue_index < len(labels):
                label = labels[dialogue_index].strip()
                if label and label != "非人物发声":
                    result["speaker"] = label
                else:
                    result["speaker"] = ""
            dialogue_index += 1

            dialogues.append(result)
            speaker = result.get("speaker")
            if speaker and speaker not in seen:
                seen.add(speaker)
                characters.append(speaker)

        elif result["type"] == "narrative":
            # 叙述性文字也作为对话处理（旁白）
            text = result.get("text", "").strip()
            if text and len(text) > 5:  # 忽略太短的叙述
                dialogues.append({
                    "type": "dialogue",
                    "chapter": result.get("chapter", ""),
                    "text": text,
                    "speaker": "旁白",
                })
                if "旁白" not in seen:
                    seen.add("旁白")
                    characters.append("旁白")

    character_list = sorted(characters, key=lambda x: -sum(1 for d in dialogues if d.get("speaker") == x))
    return dialogues, character_list


def extract_chapters(text: str) -> List[Dict]:
    """Extract all chapter markers from text.

    Returns a list of dicts: [{"title": str, "line_number": int}]
    line_number is 1-based.
    """
    chapters = []
    for i, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        m = CHAPTER_PATTERN.match(stripped)
        if m:
            chapters.append({"title": stripped, "line_number": i})
    return chapters
