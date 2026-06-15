import os

import re

def extract_speakers(txt_path):
    """
    从小说txt文件中提取对话说话人列表
    
    Args:
        txt_path: txt文件路径
    
    Returns:
        列表，包含按对话顺序出现的说话人
    """
    speakers = []
    
    with open(txt_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 匹配模式：【说话人】「对话内容」
    pattern = r'【(.*?)】「.*?」'
    matches = re.findall(pattern, content)
    
    speakers = matches
    
    return speakers
speaks = extract_speakers(r"novels\第1卷.txt")
new_list = []
for man in speaks:
    if "|" in man:
        new_list.append(man.split("|")[0]+"\n")
    else:
        new_list.append(man+"\n")
with open("labels.txt", 'w', encoding='utf-8') as f:
    f.writelines(new_list)