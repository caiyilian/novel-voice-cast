import re
import sys
sys.path.insert(0, 'backend')
from app.core.parser import parse_line

novel = open('novels/novel.txt', 'r', encoding='utf-8').read()

# 找出missing的对话
dialogues_simple = re.findall(r'「(.*?)」', novel)

lines = novel.splitlines()
parser_dialogues = []
for line in lines:
    result = parse_line(line)
    if result['type'] == 'dialogue':
        parser_dialogues.append(result['text'])

missing = []
for text in dialogues_simple:
    if text not in parser_dialogues:
        missing.append(text)

print(f"Missing dialogues: {len(missing)}")
print("\n前10条missing对话及其上下文:")
for text in missing[:10]:
    # 在novel中找到这条对话
    pattern = f'「{re.escape(text)}」'
    match = re.search(pattern, novel)
    if match:
        # 找到这一行
        pos = match.start()
        line_start = novel.rfind('\n', 0, pos) + 1
        line_end = novel.find('\n', pos)
        if line_end == -1:
            line_end = len(novel)
        line = novel[line_start:line_end]
        print(f"\n对话: 「{text}」")
        print(f"所在行: {line[:100]}")
        
        # 测试parse_line
        result = parse_line(line)
        print(f"parse_line结果: type={result['type']}")
