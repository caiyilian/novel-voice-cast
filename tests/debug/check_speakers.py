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

# 检查是否有重复的speaker
from collections import Counter
speaker_counts = Counter(d.get('speaker', '') for d in dialogues)

print("说话人统计:")
for speaker, count in speaker_counts.most_common(10):
    print(f"  {speaker}: {count}")

# 检查前20条对话
print("\n前20条对话:")
for i, d in enumerate(dialogues[:20]):
    speaker = d.get('speaker', '?')
    text = d.get('text', '')[:30]
    print(f'[{i}] {speaker}: {text}')
