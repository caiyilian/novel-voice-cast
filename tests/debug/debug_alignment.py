import re
import sys
sys.path.insert(0, 'backend')
from app.core.parser import parse

novel = open('novels/novel.txt', 'r', encoding='utf-8').read()
labels = open('novels/labels.txt', 'r', encoding='utf-8').read().splitlines()

# 方法1：简单正则
dialogues_simple = re.findall(r'「(.*?)」', novel)
print(f"简单正则找到: {len(dialogues_simple)} 条对话")
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
print(f"\nparser找到: {len(dialogues_parser)} 条对话")

# 找到第55条对话（简单正则的）在parser中的位置
target_text = dialogues_simple[54]
print(f"\n目标对话: {target_text}")

# 在parser结果中搜索
for i, d in enumerate(dialogues_parser):
    if target_text in d.get('text', ''):
        print(f"在parser中找到: 第{i}条, speaker={d['speaker']}")
        # 显示前后5条
        start = max(0, i-3)
        end = min(len(dialogues_parser), i+4)
        for j in range(start, end):
            marker = " ← 目标" if j == i else ""
            print(f"  [{j}] {dialogues_parser[j]['speaker']}: {dialogues_parser[j]['text'][:50]}{marker}")
        break
