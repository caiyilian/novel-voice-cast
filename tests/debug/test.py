import os
import re


novel = open('novels/novel.txt', 'r', encoding='utf-8').read()
labels = open('novels/labels.txt', 'r', encoding='utf-8').read().splitlines()

# 获取所有对话，就是被「」包裹的

dialogues = re.findall(r'「(.*?)」', novel)
print(dialogues[1347])  # 打印第55个对话
# 查看对应的标签
print(labels[1347])  # 打印第55个标签