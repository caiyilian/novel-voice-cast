import re
from typing import List, Tuple

CHAPTER_PATTERN = re.compile(r'^(?:第[一二三四五六七八九十百千\d]+[章节部集]|[Vv][oO][lL]\.?\s*\d+|前言|序章|尾声|后记|番外)')
DIALOGUE_JP = re.compile(r'「([^」]*)」[—－\-—\s]*(.+?)$')
DIALOGUE_CN = re.compile(r'「([^」]*)」[\s　]*$')
SPEAKER_PREFIX = re.compile(r'^(.+?)：[「『]')


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

    return {"type": "narrative", "chapter": current_chapter, "text": text}


def parse(text: str) -> Tuple[List[dict], List[str]]:
    chapters = []
    dialogues = []
    characters = []
    seen = set()
    current_chapter = ""

    for line in text.splitlines():
        result = parse_line(line, current_chapter)
        current_chapter = result.get("chapter", current_chapter)

        if result["type"] == "chapter":
            chapters.append(result)
        elif result["type"] == "dialogue":
            dialogues.append(result)
            speaker = result.get("speaker")
            if speaker and speaker not in seen:
                seen.add(speaker)
                characters.append(speaker)

    character_list = sorted(characters, key=lambda x: -sum(1 for d in dialogues if d.get("speaker") == x))
    return dialogues, character_list
