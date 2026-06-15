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

# 检查前25条对话的index和speaker
print("前25条对话:")
for i, d in enumerate(dialogues[:25]):
    speaker = d.get('speaker', '?')
    filename = f"{i:05d}_{speaker}.wav"
    print(f'[{i:5d}] {speaker:10s} -> {filename}')
