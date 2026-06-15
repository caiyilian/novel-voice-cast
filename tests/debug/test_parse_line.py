import re
import sys
sys.path.insert(0, 'backend')
from app.core.parser import parse_line

# 测试第192行
line = "「……呼，真是好月色，有没有酒啊？」"
result = parse_line(line)
print(f"parse_line 结果: {result}")

# 测试其他行
test_lines = [
    "「你好」——罗伦斯",
    "罗伦斯：「你好」",
    "「你好」",
    "这是一段叙述性文字。",
]

for line in test_lines:
    result = parse_line(line)
    print(f"\n{line}")
    print(f"  type: {result['type']}")
    if result['type'] == 'dialogue':
        print(f"  speaker: {result.get('speaker', '')}")
        print(f"  text: {result.get('text', '')}")
