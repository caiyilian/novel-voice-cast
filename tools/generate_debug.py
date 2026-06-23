"""
生成 debug.txt - 显示所有对话的说话人分配情况
用于检查对齐是否正确
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))

from app.core.parser import parse

# 读取文件
novel_path = os.path.join(os.path.dirname(__file__), "novels", "novel.txt")
labels_path = os.path.join(os.path.dirname(__file__), "novels", "labels.txt")

with open(novel_path, "r", encoding="utf-8") as f:
    novel_text = f.read()
with open(labels_path, "r", encoding="utf-8") as f:
    labels = [line.strip() for line in f if line.strip()]

# 解析
dialogues, characters = parse(novel_text, labels)

# 生成 debug.txt
output_path = os.path.join(os.path.dirname(__file__), "output", "debug.txt")
os.makedirs(os.path.dirname(output_path), exist_ok=True)

with open(output_path, "w", encoding="utf-8") as f:
    f.write(f"小说: {novel_path}\n")
    f.write(f"标注: {labels_path}\n")
    f.write(f"对话总数: {len(dialogues)}\n")
    f.write(f"标签总数: {len(labels)}\n")
    f.write("=" * 60 + "\n\n")

    # 只显示对话（不显示旁白）
    label_index = 0
    for i, d in enumerate(dialogues):
        speaker = d.get("speaker", "")
        text = d.get("text", "")
        
        # 旁白跳过
        if speaker == "旁白" or not speaker:
            continue
        
        # 非旁白对话，用 label_index 匹配
        label = labels[label_index] if label_index < len(labels) else "无标签"
        match = "✓" if speaker == label else "✗"
        f.write(f"[{i:4d}] {match} parser={speaker:10s} label={label:10s} | {text[:50]}\n")
        label_index += 1

print(f"已生成: {output_path}")
print(f"对话总数: {len(dialogues)}")
print(f"标签总数: {len(labels)}")
