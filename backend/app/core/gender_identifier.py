"""
Gender identifier agent for novel characters.
Uses Ollama with tool calling to identify character genders.
Pattern adapted from opencode-novel-loop's AgentRunner.
"""
import json
import re
from typing import List, Dict, Any, Optional

from app.core.ollama_client import OllamaClient, OllamaConfig, OllamaError, ToolCall


# ─── Novel Index (simplified DialogueIndex) ────────────────────────

class NovelIndex:
    """Indexes novel text for search and read operations."""

    def __init__(self, text: str):
        self.text = text
        self.lines = text.splitlines()

    def search(self, keyword: str, limit: int = 20) -> Dict[str, Any]:
        """Search for keyword in novel text."""
        matches = []
        for i, line in enumerate(self.lines, start=1):
            if keyword in line:
                matches.append({"line_number": i, "line": line.strip()[:120]})
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
        text = "\n".join(f"{start + i}: {line.strip()}" for i, line in enumerate(lines))
        return {"text": text, "truncated": truncated}

    def get_dialogues(self, character_name: str, limit: int = 50) -> List[Dict]:
        """Get dialogues for a character."""
        dialogues = []
        for i, line in enumerate(self.lines, start=1):
            stripped = line.strip()
            if character_name in stripped and "「" in stripped:
                dialogues.append({"line_number": i, "text": stripped[:120]})
                if len(dialogues) >= limit:
                    break
        return dialogues


# ─── Tool definitions ──────────────────────────────────────────────

TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "search_novel",
            "description": "Search the novel for a keyword. Returns matching lines with line numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {"type": "string", "description": "Keyword to search for (e.g. character name)"},
                    "limit": {"type": "integer", "description": "Max results (default 20)"},
                },
                "required": ["keyword"],
            },
        },
    },
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
            "name": "get_dialogues",
            "description": "Get dialogues for a specific character.",
            "parameters": {
                "type": "object",
                "properties": {
                    "character_name": {"type": "string", "description": "Character name to search for"},
                    "limit": {"type": "integer", "description": "Max dialogues to return (default 20)"},
                },
                "required": ["character_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_gender",
            "description": "Submit the gender judgment for a character.",
            "parameters": {
                "type": "object",
                "properties": {
                    "character_name": {"type": "string", "description": "Character name"},
                    "gender": {"type": "string", "enum": ["male", "female", "unknown"], "description": "Gender"},
                    "confidence": {"type": "number", "description": "Confidence 0.0-1.0"},
                    "evidence": {"type": "string", "description": "Evidence from the novel"},
                },
                "required": ["character_name", "gender", "confidence", "evidence"],
            },
        },
    },
]


# ─── System prompt ─────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a novel character gender analyst. Your task is to determine the gender of each character.

## Rules

1. First call `search_novel` with the character's name to find where they appear in the novel
2. Then call `read_lines` to read context around those appearances (at least 100 lines)
3. Also call `get_dialogues` to see what the character says
4. Analyze based on:
   - The character's name (Chinese names often indicate gender)
   - Pronouns used: 他 (he/him) = male, 她 (she/her) = female
   - Physical descriptions (hair, body, etc.)
   - How others address the character (先生/小姐/大人/etc.)
5. If unsure, mark as "unknown" with confidence < 0.5
6. Non-human characters: use pronouns in the narration to determine gender

## Output

After analyzing, call `submit_gender` with:
- character_name: the character's name
- gender: "male", "female", or "unknown"
- confidence: 0.0 to 1.0
- evidence: specific lines or descriptions that support your judgment
"""


# ─── Tool execution ────────────────────────────────────────────────

def _execute_tool(tool: ToolCall, index: NovelIndex) -> str:
    """Execute a tool call and return the result."""
    if tool.name == "search_novel":
        keyword = tool.arguments.get("keyword", "")
        limit = tool.arguments.get("limit", 20)
        result = index.search(keyword, limit)
        lines = [f"Line {m['line_number']}: {m['line']}" for m in result["matches"]]
        output = f"Found {result['total_matches']} matches"
        if result["truncated"]:
            output += f" (showing {len(result['matches'])} of {result['total_matches']})"
        output += ":\n" + "\n".join(lines) if lines else "\nNo matches found"
        return output

    elif tool.name == "read_lines":
        start = tool.arguments.get("start_line", 1)
        end = tool.arguments.get("end_line", start + 100)
        result = index.read_lines(start, end)
        output = result["text"]
        if result["truncated"]:
            output += "\n... (truncated)"
        return output if output else "No lines in range"

    elif tool.name == "get_dialogues":
        name = tool.arguments.get("character_name", "")
        limit = tool.arguments.get("limit", 20)
        dialogues = index.get_dialogues(name, limit)
        if not dialogues:
            return f"No dialogues found for '{name}'"
        lines = [f"Line {d['line_number']}: {d['text']}" for d in dialogues]
        return f"Found {len(dialogues)} dialogues:\n" + "\n".join(lines)

    return f"Unknown tool: {tool.name}"


# ─── Main agent loop ───────────────────────────────────────────────

def _normalize_gender(result: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize gender: unknown → male."""
    if result.get("gender") == "unknown":
        result["gender"] = "male"
        result["confidence"] = max(result.get("confidence", 0.0), 0.3)
    return result


def identify_gender(
    character_name: str,
    text: str,
    client: OllamaClient,
    max_tool_steps: int = 10,
) -> Dict[str, Any]:
    """Identify gender for a single character.

    Args:
        character_name: Name of the character to analyze
        text: Full novel text
        client: Ollama client
        max_tool_steps: Maximum tool-calling iterations

    Returns:
        {"character_name": str, "gender": str, "confidence": float, "evidence": str}
    """
    index = NovelIndex(text)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Please identify the gender of the character '{character_name}'.\n\nThe novel has {len(index.lines)} lines. Search for the character's name and read context to determine gender."},
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

                # check if submit_gender was called
                if tc.name == "submit_gender":
                    return _normalize_gender({
                        "character_name": tc.arguments.get("character_name", character_name),
                        "gender": tc.arguments.get("gender", "unknown"),
                        "confidence": tc.arguments.get("confidence", 0.0),
                        "evidence": tc.arguments.get("evidence", ""),
                    })

        elif result.content:
            # try to parse JSON fallback
            match = re.search(r'\{.*?\}', result.content, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group(0))
                    if "character_name" in data and "gender" in data:
                        return _normalize_gender({
                            "character_name": data.get("character_name", character_name),
                            "gender": data.get("gender", "unknown"),
                            "confidence": data.get("confidence", 0.0),
                            "evidence": data.get("evidence", ""),
                        })
                except json.JSONDecodeError:
                    pass

    return _normalize_gender({
        "character_name": character_name,
        "gender": "unknown",
        "confidence": 0.0,
        "evidence": "Failed to determine within max_tool_steps",
    })


def identify_all_genders(
    character_names: List[str],
    text: str,
    client: Optional[OllamaClient] = None,
    max_tool_steps: int = 10,
) -> List[Dict[str, Any]]:
    """Identify gender for multiple characters.

    Args:
        character_names: List of character names
        text: Full novel text
        client: Ollama client (creates one if None)
        max_tool_steps: Maximum tool-calling iterations per character

    Returns:
        List of gender identification results
    """
    if client is None:
        client = OllamaClient()

    results = []
    for name in character_names:
        try:
            result = identify_gender(name, text, client, max_tool_steps)
            results.append(result)
        except Exception as e:
            results.append(_normalize_gender({
                "character_name": name,
                "gender": "unknown",
                "confidence": 0.0,
                "evidence": f"Error: {str(e)}",
            }))

    return results
