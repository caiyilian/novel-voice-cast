#!/usr/bin/env python
"""Scan all Python files for hardcoded character names or novel-specific content.

This project is a *general* novel processing tool.  No character name,
work title, or genre-specific assumption may appear in the code —
all character info comes from user-provided ``docs/角色卡.md``.

Usage:
    python scripts/check_hardcode.py
    python scripts/check_hardcode.py --fix       # not implemented yet
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent

# ── Rules ──────────────────────────────────────────────────────────────

# Character / work names that should NEVER appear in code.
FORBIDDEN_SUBSTRINGS: list[str] = [
    # Novel-specific characters (狼与香辛料)
    "赫萝", "Holo", "罗伦斯", "Lawrence",
    "克拉福", "Kraft",
    # Novel-specific works
    "狼与香辛料", "Spice and Wolf", "狼と香辛料",
]

# Whole-word patterns (to avoid matching "lls" inside "calls", "fills", etc.)
FORBIDDEN_WORDS: list[str] = [
    "lls",  # Lawrence filename (whole word only)
]

# Patterns that suggest genre-specific assumptions in prompts or logic.
SUSPICIOUS_PATTERNS: list[str] = [
    r"\bwolf.?girl\b",
    r"\bwolf.?ear\b",
    r"\bvillage\b",       # assumes rural setting
    r"\bmerchant\b",      # assumes economic theme
]

# Files/dirs to skip
SKIP_DIRS: set[str] = {".venv", "__pycache__", ".git", "ACE-Step-1.5",
                       "whl_files", "_archive", "output", "node_modules",
                       "tests", "tools", "backend/models"}

SKIP_FILES: set[str] = {".gitignore", "README.md"}

# Files that are allowed to contain specific content (docs, test data)
ALLOWED_CHARACTER_NAMES_FILES: set[str] = {
    "角色卡.md",
    "插图生成环境配置指南.md",
    "插图生成API接入文档.md",
    "插图生成策略讨论.md",
    "开源动画视频生成项目调研.md",
    "方案.md",
    "hardware_resources.md",
}

# Self: the scanner itself defines the forbidden list, so it's exempt
SELF = "check_hardcode.py"


# ── Scanner ────────────────────────────────────────────────────────────

def _walk_py() -> list[Path]:
    py_files = []
    for path in PROJECT.rglob("*.py"):
        rel = path.relative_to(PROJECT)
        if any(part.startswith(".") for part in path.parts):
            continue
        if any(parent.name in SKIP_DIRS for parent in path.parents):
            continue
        if path.name in SKIP_FILES:
            continue
        py_files.append(path)
    return sorted(py_files)


def _check_file(path: Path) -> list[str]:
    rel = path.relative_to(PROJECT)

    # Skip the scanner itself
    if path.name == SELF:
        return []

    issues: list[str] = []

    # Check plain text content
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return []

    lines = text.splitlines()

    for lineno, line in enumerate(lines, start=1):
        stripped = line.strip()

        # Skip comments and docstrings (less strict about those)
        is_comment = stripped.startswith("#") or stripped.startswith('"""') or stripped.startswith("'''")

        for forbidden in FORBIDDEN_SUBSTRINGS:
            if forbidden.lower() in stripped.lower():
                if is_comment:
                    issues.append(f"  {rel}:{lineno}  Comment contains forbidden name '{forbidden}': {stripped[:80]}")
                else:
                    issues.append(f"  {rel}:{lineno}  Contains forbidden name '{forbidden}': {stripped[:80]}")

        # Whole-word checks
        for word in FORBIDDEN_WORDS:
            if re.search(rf"\b{re.escape(word)}\b", stripped, re.IGNORECASE):
                if is_comment:
                    issues.append(f"  {rel}:{lineno}  Comment contains forbidden word '{word}': {stripped[:80]}")
                else:
                    issues.append(f"  {rel}:{lineno}  Contains forbidden word '{word}': {stripped[:80]}")

    # Check AST for suspicious string literals in prompts
    try:
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                val = node.value.lower()
                for pattern in SUSPICIOUS_PATTERNS:
                    if re.search(pattern, val):
                        # Only flag if it looks like a prompt (long strings)
                        if len(val) > 50:
                            col = getattr(node, "col_offset", 0)
                            issues.append(f"  {rel}:{node.lineno}:{col}  Prompt-style string matches '{pattern}': {val[:80]}...")
    except SyntaxError:
        pass

    return issues


def _in_string_literal(line: str, keyword: str) -> bool:
    """Heuristic: check if keyword appears inside a string literal."""
    # Very rough — looks for keyword inside quotes
    return keyword.lower() in line.lower()


def _check_prompt_files() -> list[str]:
    """Check non-Python files for hardcoded character names in system prompts."""
    issues: list[str] = []
    for path in PROJECT.rglob("*.py"):
        rel = path.relative_to(PROJECT)
        if path.name == SELF:
            continue
        if any(part.startswith(".") for part in path.parts):
            continue
        if any(parent.name in SKIP_DIRS for parent in path.parents):
            continue
        text = path.read_text(encoding="utf-8")
        for keyword in FORBIDDEN_SUBSTRINGS:
            if keyword in text:
                # Check if it's inside a string that looks like a system prompt
                if "SYSTEM_PROMPT" in text or "system" in text.lower():
                    issues.append(f"  {rel}  Contains '{keyword}' in a file with system prompts")
    return issues


def main() -> int:
    py_files = _walk_py()
    print(f"Scanning {len(py_files)} Python files...")
    print()

    all_issues: list[str] = []
    for pf in py_files:
        issues = _check_file(pf)
        all_issues.extend(issues)

    # Prompt file check
    prompt_issues = _check_prompt_files()
    all_issues.extend(prompt_issues)

    if all_issues:
        print(f"Found {len(all_issues)} potential issue(s):")
        print()
        for issue in all_issues:
            print(issue)
        print()
        print("Review these and remove any hardcoded character/work references.")
        return 1
    else:
        print("No hardcoded character names or novel-specific content found.")
        print("Clean!")
        return 0


if __name__ == "__main__":
    sys.exit(main())