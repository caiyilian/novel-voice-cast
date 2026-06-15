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

# 检查前10条对话
for i, d in enumerate(dialogues[:10]):
    speaker = d.get('speaker', '?')
    text = d.get('text', '')[:30]
    print(f'[{i}] {speaker}: {text}')
