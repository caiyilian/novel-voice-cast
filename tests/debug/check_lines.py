import re
import sys

novel = open('novels/novel.txt', 'r', encoding='utf-8').read()
labels = open('novels/labels.txt', 'r', encoding='utf-8').read().splitlines()

# 显示novel.txt第185-195行
lines = novel.splitlines()
print("novel.txt 第185-195行:")
for i in range(184, 195):
    if i < len(lines):
        print(f"  [{i+1}] {lines[i][:80]}")

print("\n\nlabels.txt 第50-60行:")
for i in range(49, 60):
    if i < len(labels):
        print(f"  [{i+1}] {labels[i]}")
