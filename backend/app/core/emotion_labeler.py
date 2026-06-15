"""
Emotion labeler agent for novel dialogues.
Uses Ollama with tool calling to identify emotion and tone for each dialogue.
Pattern adapted from opencode-novel-loop's AgentRunner.
"""
import json
import re
from typing import List, Dict, Any, Optional

from app.core.ollama_client import OllamaClient, OllamaConfig, OllamaError, ToolCall


# ─── Emotion taxonomy ──────────────────────────────────────────────

EMOTIONS = ["happy", "sad", "angry", "surprised", "calm", "nervous", "cold"]
TONES = ["loud", "soft", "stutter", "sarcastic", "gentle", "serious", "whisper"]


# ─── Novel Index (simplified) ──────────────────────────────────────

class NovelIndex:
    """Indexes novel text for search and read operations."""

    def __init__(self, text: str, dialogues: List[Dict] = None):
        self.text = text
        self.lines = text.splitlines()
        # 构建行号到说话人的映射
        self.line_to_speaker = {}
        if dialogues:
            for d in dialogues:
                line_num = d.get("line", 0)
                speaker = d.get("speaker", "")
                if line_num > 0 and speaker and speaker != "旁白":
                    self.line_to_speaker[line_num] = speaker

    def _format_line(self, line_num: int, line: str) -> str:
        """给对话行加上说话人标记"""
        speaker = self.line_to_speaker.get(line_num)
        if speaker and "「" in line:
            # 给对话行加上【说话人】标记
            return f"{line_num}: 【{speaker}】{line.strip()}"
        return f"{line_num}: {line.strip()}"

    def search(self, keyword: str, limit: int = 20) -> Dict[str, Any]:
        """Search for keyword in novel text."""
        matches = []
        for i, line in enumerate(self.lines, start=1):
            if keyword in line:
                formatted = self._format_line(i, line)
                matches.append({"line_number": i, "line": formatted[:120]})
                if len(matches) >= limit:
                    break
        total = sum(1 for line in self.lines if keyword in line)
        return {
            "total_matches": total,
            "truncated": total > len(matches),
            "matches": matches,
        }

    def read_lines(self, start: int, end: int, limit: int = 300) -> Dict[str, Any]:
        """Read line range from novel (1-based, inclusive)."""
        start = max(1, start)
        end = min(len(self.lines), end)
        if start > end:
            return {"text": "", "truncated": False}
        lines = self.lines[start - 1 : end]
        if len(lines) > limit:
            lines = lines[:limit]
            truncated = True
        else:
            truncated = False
        text = "\n".join(self._format_line(start + i, line) for i, line in enumerate(lines))
        return {"text": text, "truncated": truncated}

    def get_dialogue_context(self, dialogue_index: int, context_lines: int = 5) -> Dict[str, Any]:
        """Get context around a specific dialogue line."""
        if dialogue_index < 1 or dialogue_index > len(self.lines):
            return {"text": "", "truncated": False}
        start = max(1, dialogue_index - context_lines)
        end = min(len(self.lines), dialogue_index + context_lines)
        lines = self.lines[start - 1 : end]
        text = "\n".join(f"{start + i}: {line.strip()}" for i, line in enumerate(lines))
        return {"text": text, "truncated": False}


# ─── Tool definitions ──────────────────────────────────────────────

TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "read_lines",
            "description": "Read specific lines from the novel. 1-based inclusive.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_line": {"type": "integer", "description": "Start line (1-based)"},
                    "end_line": {"type": "integer", "description": "End line (1-based, inclusive)"},
                },
                "required": ["start_line", "end_line"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_novel",
            "description": "Search the novel for a keyword. Returns matching lines with line numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {"type": "string", "description": "Keyword to search for"},
                    "limit": {"type": "integer", "description": "Max results (default 20)"},
                },
                "required": ["keyword"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_emotion",
            "description": "Submit emotion and tone judgment for a dialogue.",
            "parameters": {
                "type": "object",
                "properties": {
                    "dialogue_index": {"type": "integer", "description": "Index of the dialogue (1-based)"},
                    "emotion": {"type": "string", "enum": EMOTIONS, "description": "Emotion label"},
                    "tone": {"type": "string", "enum": TONES, "description": "Tone label"},
                    "confidence": {"type": "number", "description": "Confidence 0.0-1.0"},
                    "evidence": {"type": "string", "description": "Evidence from the text"},
                },
                "required": ["dialogue_index", "emotion", "tone", "confidence", "evidence"],
            },
        },
    },
]


# ─── System prompt ─────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a novel dialogue emotion analyst. Your task is to determine the emotion and tone of each dialogue.

## Emotion Labels
- happy: 高兴/快乐
- sad: 悲伤/难过
- angry: 愤怒/生气
- surprised: 惊讶/震惊
- calm: 平静/冷静
- nervous: 紧张/焦虑
- cold: 冷漠/淡漠

## Tone Labels
- loud: 大声/喊叫
- soft: 小声/轻声
- stutter: 结巴/犹豫
- sarcastic: 讽刺/嘲讽
- gentle: 温柔/轻柔
- serious: 严肃/认真
- whisper: 低语/耳语

## Analysis Rules

1. **First read context**: Call `read_lines` to read 30+ lines around the dialogue
2. **Search if needed**: Call `search_novel` to find related descriptions or character emotions
3. **Analyze multiple factors**:
   - Dialogue text content (words, punctuation, exclamations)
   - Context from surrounding lines (what happened before/after)
   - Narrator descriptions (e.g., "冷冷地说", "微笑着说", "颤抖着说")
   - Character's emotional state from previous dialogues
   - Punctuation (!, ..., ?, ！！！)
4. **Consider character relationships**: How characters feel about each other affects emotion
5. **Default when uncertain**: Use "calm" emotion + "serious" tone with confidence < 0.5

## Evidence Requirements

Always provide specific evidence from the text:
- Quote the narrator description that indicates emotion
- Note any punctuation that suggests tone
- Reference context from surrounding lines

## Output

Call `submit_emotion` with:
- dialogue_index: the line number of the dialogue
- emotion: one of the emotion labels above
- tone: one of the tone labels above
- confidence: 0.0 to 1.0 (higher = more certain)
- evidence: specific lines or descriptions that support your judgment
"""


# ─── Tool execution ────────────────────────────────────────────────

def _execute_tool(tool: ToolCall, index: NovelIndex) -> str:
    """Execute a tool call and return the result."""
    if tool.name == "read_lines":
        start = tool.arguments.get("start_line", 1)
        end = tool.arguments.get("end_line", start + 100)
        result = index.read_lines(start, end)
        output = result["text"]
        if result["truncated"]:
            output += "\n... (truncated)"
        return output if output else "No lines in range"

    elif tool.name == "search_novel":
        keyword = tool.arguments.get("keyword", "")
        limit = tool.arguments.get("limit", 20)
        result = index.search(keyword, limit)
        lines = [f"Line {m['line_number']}: {m['line']}" for m in result["matches"]]
        output = f"Found {result['total_matches']} matches"
        if result["truncated"]:
            output += f" (showing {len(result['matches'])} of {result['total_matches']})"
        output += ":\n" + "\n".join(lines) if lines else "\nNo matches found"
        return output

    return f"Unknown tool: {tool.name}"


# ─── Main agent loop ───────────────────────────────────────────────

def label_emotion(
    dialogue_text: str,
    dialogue_line: int,
    dialogue_index: int,
    text: str,
    client: OllamaClient,
    max_tool_steps: int = 10,
    speaker: str = "",
    dialogues: List[Dict] = None,
) -> Dict[str, Any]:
    """Identify emotion for a single dialogue.

    Args:
        dialogue_text: The dialogue text
        dialogue_line: Line number of the dialogue (1-based)
        dialogue_index: Index in the dialogues list (0-based)
        text: Full novel text
        client: Ollama client
        max_tool_steps: Maximum tool-calling iterations
        speaker: Speaker name (optional)
        dialogues: List of all dialogues with speaker info (optional)

    Returns:
        {"dialogue_index": int, "emotion": str, "tone": str, "confidence": float, "evidence": str}
    """
    index = NovelIndex(text, dialogues)

    # 只提供对话本身，让 LLM 自己探索上下文
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"""Analyze the emotion and tone of this dialogue:

Dialogue (line {dialogue_line}): 【{speaker}】「{dialogue_text}」

Use read_lines to read the context around this dialogue, and search_novel to find related descriptions or character emotions. Then call submit_emotion."""},
    ]

    for step in range(1, max_tool_steps + 1):
        result = client.chat(messages, tools=TOOL_SPECS)

        if result.tool_calls:
            for tc in result.tool_calls:
                tool_result = _execute_tool(tc, index)
                messages.append({
                    "role": "assistant",
                    "content": result.content or "",
                    "tool_calls": [{
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                        },
                    }],
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": tool_result,
                })

                # check if submit_emotion was called
                if tc.name == "submit_emotion":
                    return {
                        "dialogue_index": dialogue_index,
                        "emotion": tc.arguments.get("emotion", "calm"),
                        "tone": tc.arguments.get("tone", "serious"),
                        "confidence": tc.arguments.get("confidence", 0.0),
                        "evidence": tc.arguments.get("evidence", ""),
                    }

        elif result.content:
            # try to parse JSON fallback
            match = re.search(r'\{.*?\}', result.content, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group(0))
                    if "emotion" in data and "tone" in data:
                        return {
                            "dialogue_index": dialogue_index,
                            "emotion": data.get("emotion", "calm"),
                            "tone": data.get("tone", "serious"),
                            "confidence": data.get("confidence", 0.0),
                            "evidence": data.get("evidence", ""),
                        }
                except json.JSONDecodeError:
                    pass

    return {
        "dialogue_index": dialogue_index,
        "emotion": "calm",
        "tone": "serious",
        "confidence": 0.0,
        "evidence": "Failed to determine within max_tool_steps",
    }
