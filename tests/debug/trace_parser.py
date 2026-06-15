import re
import sys
sys.path.insert(0, 'backend')
from app.core.parser import parse

novel = open('novels/novel.txt', 'r', encoding='utf-8').read()
labels = open('novels/labels.txt', 'r', encoding='utf-8').read().splitlines()

# 方法1：简单正则
dialogues_simple = re.findall(r'「(.*?)」', novel)
print(f"简单正则: {len(dialogues_simple)} 条对话")
print(f"第55条对话: {dialogues_simple[54]}")
print(f"第55条标签: {labels[54]}")

# 方法2：parser
f = open('novels/novel.txt', 'r', encoding='utf-8')
novel_text = f.read()
f.close()
f = open('novels/labels.txt', 'r', encoding='utf-8')
label_list = [l.strip() for l in f if l.strip()]
f.close()

dialogues_parser, _ = parse(novel_text, label_list)
print(f"\nparser: {len(dialogues_parser)} 条对话")

# 统计speaker
speaker_counts = {}
for d in dialogues_parser:
    speaker = d.get('speaker', '')
    speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1

print("\n说话人统计:")
for speaker, count in sorted(speaker_counts.items(), key=lambda x: -x[1])[:10]:
    print(f"  {speaker}: {count}")

# 找到第55条对话在parser中的位置
target = dialogues_simple[54]
print(f"\n目标对话: {target}")

# 在parser中搜索
for i, d in enumerate(dialogues_parser):
    if target in d.get('text', ''):
        print(f"在parser中找到: 第{i}条")
        print(f"  speaker: {d['speaker']}")
        print(f"  预期speaker: {labels[54]}")
        break
