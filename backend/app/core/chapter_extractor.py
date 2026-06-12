import json
import re
from typing import List, Dict, Optional

from app.core.ollama_client import OllamaClient, OllamaError, ToolCall


# ─── Tool definitions ──────────────────────────────────────────────

TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "search_novel",
            "description": "Search the novel text for a keyword. Returns matching line numbers with context.",
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {
                        "type": "string",
                        "description": "The keyword to search for (e.g. '第', '章', 'Chapter', '卷')"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max number of results to return (default 20)"
                    }
                },
                "required": ["keyword"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_lines",
            "description": "Read specific lines from the novel. Line numbers are 1-based and inclusive.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_line": {
                        "type": "integer",
                        "description": "Start line number (1-based)"
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "End line number (1-based, inclusive)"
                    }
                },
                "required": ["start_line", "end_line"]
            }
        }
    }
]


# ─── System prompt ─────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a novel chapter structure analyst. Your task is to identify all chapter markers and their line numbers in the novel text.

## Rules

1. First call `search_novel` with keyword "第" and limit=200 to find ALL potential chapter markers across the ENTIRE file
2. Also search for "幕" and "章" to catch different chapter formats
3. Then call `read_lines` to confirm each candidate - the chapter marker must be a STANDALONE line, not part of a table of contents
4. A valid chapter marker must:
   - Appear at the beginning of a line, ALONE (not in a list or paragraph)
   - Follow a consistent numbering pattern (第1幕, 第2幕, 第1章, 第2章, etc.)
   - Be followed by narrative content on the next non-empty line
5. Ignore false positives:
   - Lines in a table of contents (目錄/目录) section
   - Lines that are just "第" or "一" split across multiple lines
   - Lines that contain chapter names but are part of a sentence
6. After finding all candidates, verify by reading lines around each one to confirm it's a real chapter start
7. Output a JSON list of chapters: [{"title": "full chapter title", "line_number": N}]
8. Sort by line_number ascending

## Important

- Search the ENTIRE file using limit=200 or higher
- Chapter markers are typically on their own line, not embedded in text
- If you see "第一幕" appearing multiple times close together, only count the FIRST occurrence as the chapter marker
- Make multiple search calls with different keywords to be thorough

## Output format

After exploring the text, output ONLY a JSON list like:
[{"title": "第1章 觉醒", "line_number": 123}, {"title": "第2章 风暴", "line_number": 456}]
"""


# ─── Tool execution ────────────────────────────────────────────────

def _execute_tool(tool: ToolCall, lines: List[str]) -> str:
    """Execute a tool call and return the result as a string."""
    if tool.name == "search_novel":
        keyword = tool.arguments.get("keyword", "")
        limit = tool.arguments.get("limit", 20)
        results = []
        for i, line in enumerate(lines, start=1):
            if keyword in line:
                results.append(f"Line {i}: {line.strip()[:100]}")
                if len(results) >= limit:
                    break
        if not results:
            return f"No matches found for '{keyword}'"
        return "\n".join(results)

    elif tool.name == "read_lines":
        start = tool.arguments.get("start_line", 1)
        end = tool.arguments.get("end_line", start + 20)
        start = max(1, start)
        end = min(len(lines), end)
        result_lines = []
        for i in range(start - 1, end):
            result_lines.append(f"Line {i + 1}: {lines[i].strip()[:120]}")
        return "\n".join(result_lines) if result_lines else "No lines in range"

    return f"Unknown tool: {tool.name}"


# ─── Main function ─────────────────────────────────────────────────

def llm_extract_chapters(
    text: str,
    client: OllamaClient,
    max_tool_steps: int = 10,
    regex_candidates: Optional[List[Dict]] = None,
    max_retries: int = 2,
) -> List[Dict]:
    """Extract chapters using LLM with tool calling.

    Args:
        text: The full novel text
        client: OllamaClient instance
        max_tool_steps: Maximum number of tool-calling iterations
        regex_candidates: Optional pre-extracted chapter list from regex
        max_retries: Number of retries if LLM finds fewer chapters than regex

    Returns:
        List of {"title": str, "line_number": int}
    """
    best_result = []
    regex_count = len(regex_candidates) if regex_candidates else 0

    for attempt in range(max_retries + 1):
        result = _run_llm_extraction(text, client, max_tool_steps, regex_candidates)

        # if LLM found >= regex candidates, use it
        if len(result) >= regex_count and regex_count > 0:
            return result

        # if LLM found something and regex found nothing, use it
        if len(result) > 0 and regex_count == 0:
            return result

        # keep track of best result
        if len(result) > len(best_result):
            best_result = result

        # if this was the last attempt, return best
        if attempt == max_retries:
            return best_result

    return best_result


def _run_llm_extraction(
    text: str,
    client: OllamaClient,
    max_tool_steps: int,
    regex_candidates: Optional[List[Dict]],
) -> List[Dict]:
    """Single LLM extraction attempt."""
    lines = text.splitlines()

    # build initial prompt with regex candidates
    user_msg = f"The novel has {len(lines)} lines total.\n\n"
    if regex_candidates:
        user_msg += "I already found these potential chapter markers via regex:\n"
        for ch in regex_candidates:
            user_msg += f"  Line {ch['line_number']}: {ch['title']}\n"
        user_msg += "\nPlease verify these and find any I missed. Use search_novel and read_lines to explore."
    else:
        user_msg += "No regex candidates available. Please search for chapter markers using the tools."

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    for step in range(1, max_tool_steps + 1):
        result = client.chat(messages, tools=TOOL_SPECS)

        if result.tool_calls:
            # execute each tool call
            for tc in result.tool_calls:
                tool_result = _execute_tool(tc, lines)
                messages.append({
                    "role": "assistant",
                    "content": result.content or "",
                    "tool_calls": [{
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments, ensure_ascii=False)
                        }
                    }]
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": tool_result
                })

        elif result.content:
            # model returned text instead of tool calls — try to parse JSON
            chapters = _parse_chapter_json(result.content)
            if chapters is not None:
                return chapters

            # if no JSON found, ask model to continue
            messages.append({"role": "assistant", "content": result.content})
            messages.append({
                "role": "user",
                "content": "Please continue exploring and output the final chapter list as JSON."
            })
        else:
            break

    # fallback: return regex candidates if available
    return regex_candidates or []


def _parse_chapter_json(text: str) -> Optional[List[Dict]]:
    """Try to extract a JSON chapter list from model output."""
    # find JSON array in text
    match = re.search(r'\[.*?\]', text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            if isinstance(data, list):
                # validate structure
                valid = []
                for item in data:
                    if isinstance(item, dict) and "title" in item and "line_number" in item:
                        valid.append({
                            "title": str(item["title"]),
                            "line_number": int(item["line_number"])
                        })
                if valid:
                    valid.sort(key=lambda x: x["line_number"])
                    return valid
        except (json.JSONDecodeError, ValueError):
            pass
    return None
