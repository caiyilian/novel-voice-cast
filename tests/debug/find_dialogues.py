import sys
sys.path.insert(0, 'backend')
from app.core.parser import parse

f = open('novels/novel.txt', 'r', encoding='utf-8')
novel = f.read()
f.close()

f = open('novels/labels.txt', 'r', encoding='utf-8')
labels = [l.strip() for l in f if l.strip()]
f.close()

dialogues, _ = parse(novel, labels)

# 统计
print(f"总对话: {len(dialogues)}")
print(f"罗伦斯: {sum(1 for d in dialogues if d['speaker'] == '罗伦斯')} 条")
print(f"赫萝: {sum(1 for d in dialogues if d['speaker'] == '赫萝')} 条")
print(f"旁白: {sum(1 for d in dialogues if d['speaker'] == '旁白')} 条")

# 找第一个赫萝的对话
for i, d in enumerate(dialogues):
    if d['speaker'] == '赫萝':
        print(f"\n第一个赫萝对话: [{i}]")
        # 显示前后各5条
        start = max(0, i - 3)
        end = min(len(dialogues), i + 8)
        for j in range(start, end):
            marker = " ← 赫萝" if dialogues[j]['speaker'] == '赫萝' else ""
            marker += " ← 罗伦斯" if dialogues[j]['speaker'] == '罗伦斯' else ""
            print(f"  [{j}] {dialogues[j]['speaker']}: {dialogues[j]['text'][:50]}{marker}")
        break

# 找连续对话
print("\n\n寻找连续的罗伦斯-赫萝对话：")
for i in range(len(dialogues) - 10):
    window = dialogues[i:i+10]
    has_lolans = any(d['speaker'] == '罗伦斯' for d in window)
    has_holo = any(d['speaker'] == '赫萝' for d in window)
    if has_lolans and has_holo:
        print(f"\n从 [{i}] 开始的10条对话：")
        for j in range(i, min(i+10, len(dialogues))):
            d = dialogues[j]
            marker = ""
            if d['speaker'] in ['罗伦斯', '赫萝']:
                marker = f" ← {d['speaker']}"
            print(f"  [{j}] {d['speaker']}: {d['text'][:50]}{marker}")
        break
