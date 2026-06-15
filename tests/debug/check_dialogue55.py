import re
import sys

novel = open('novels/novel.txt', 'r', encoding='utf-8').read()
labels = open('novels/labels.txt', 'r', encoding='utf-8').read().splitlines()

# 找到第55条对话的原始文本
dialogues_simple = re.findall(r'「(.*?)」', novel)
target = dialogues_simple[54]

# 在novel.txt中找到这条对话
lines = novel.splitlines()
for i, line in enumerate(lines):
    if target in line:
        print(f"Line {i+1}:")
        print(f"  Content: {line}")
        print(f"  Label: {labels[54]}")
        
        # 检查各种pattern
        DIALOGUE_JP = re.compile(r'「([^」]*)」[—－\-—\s]*(.+?)$')
        SPEAKER_PREFIX = re.compile(r'^(.+?)：[「『]')
        DIALOGUE_CN = re.compile(r'「([^」]*)」[\s　]*$')
        
        m1 = DIALOGUE_JP.match(line)
        m2 = SPEAKER_PREFIX.match(line)
        m3 = DIALOGUE_CN.match(line)
        
        print(f"  DIALOGUE_JP: {'Match' if m1 else 'No match'}")
        print(f"  SPEAKER_PREFIX: {'Match' if m2 else 'No match'}")
        print(f"  DIALOGUE_CN: {'Match' if m3 else 'No match'}")
        
        if m1:
            print(f"    dialogue: {m1.group(1)}")
            print(f"    speaker: {m1.group(2)}")
        if m3:
            print(f"    dialogue: {m3.group(1)}")
        break
