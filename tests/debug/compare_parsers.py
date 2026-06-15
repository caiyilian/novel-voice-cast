import re
import sys
sys.path.insert(0, 'backend')
from app.core.parser import parse_line

novel = open('novels/novel.txt', 'r', encoding='utf-8').read()

# 简单正则找到的对话
dialogues_simple = re.findall(r'「(.*?)」', novel)
print(f"简单正则: {len(dialogues_simple)} 条对话")

# parser检测的对话
lines = novel.splitlines()
parser_dialogues = []
for line in lines:
    result = parse_line(line)
    if result['type'] == 'dialogue':
        parser_dialogues.append(result['text'])

print(f"parser检测: {len(parser_dialogues)} 条对话")

# 找出简单正则有但parser没有的对话
missing = []
for text in dialogues_simple:
    if text not in parser_dialogues:
        missing.append(text)

print(f"\n简单正则有但parser没有: {len(missing)} 条")
if missing:
    print("前5条:")
    for text in missing[:5]:
        print(f"  {text[:50]}")
