import re

novel = open('novels/novel.txt', 'r', encoding='utf-8').read()
lines = novel.splitlines()

# 检查第192行的格式
line_192 = lines[191] if len(lines) > 191 else ""
print(f"第192行: {line_192}")

# 检查DIALOGUE_JP pattern
DIALOGUE_JP = re.compile(r'「([^」]*)」[—－\-—\s]*(.+?)$')
m = DIALOGUE_JP.match(line_192)
if m:
    print(f"DIALOGUE_JP 匹配:")
    print(f"  对话: {m.group(1)}")
    print(f"  说话人: {m.group(2)}")
else:
    print("DIALOGUE_JP 未匹配")

# 检查所有包含「……呼，真是好月色，有没有酒啊？」的行
target = "……呼，真是好月色，有没有酒啊？"
for i, line in enumerate(lines):
    if target in line:
        print(f"\n找到目标对话在第{i+1}行:")
        print(f"  内容: {line}")
        m = DIALOGUE_JP.match(line)
        if m:
            print(f"  DIALOGUE_JP 匹配: speaker={m.group(2)}")
        else:
            print(f"  DIALOGUE_JP 未匹配")
