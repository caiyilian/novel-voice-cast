"""
Debug script to trace parser behavior and find alignment issues
"""
import re
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))

from app.core.parser import parse, parse_line

# Load data
novel = open("novels/novel.txt", "r", encoding="utf-8").read()
labels = [l.strip() for l in open("novels/labels.txt", "r", encoding="utf-8").readlines() if l.strip()]

print(f"Labels count: {len(labels)}")

# Test parse_line on specific lines
novel_lines = open("novels/novel.txt", "r", encoding="utf-8").read().splitlines()

# Check lines around 1347
print("Checking lines around 1347:")
for i in range(1340, 1355):
    line = novel_lines[i]
    result = parse_line(line.strip())
    print(f"Line {i}: type={result['type']}, speaker={result.get('speaker','')}, text={line[:50]}")