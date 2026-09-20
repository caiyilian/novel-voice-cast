"""Shared numbered novel index with dialogue-speaker metadata and evidence lookup.

The gender / emotion / performance stages each used to define their own copy
of ``NovelIndex``.  They are consolidated here: ``read_lines`` / ``search``
annotate each line with its speaker when ``dialogues`` are supplied, and the
gender/performance-specific lookups (``get_dialogues``, ``evidence_packet``,
``character_samples``) are folded into the same class.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence


def evenly_spaced_positions(total: int, count: int) -> list[int]:
    if total <= 0 or count <= 0:
        return []
    if count == 1:
        return [0]
    return sorted({round(position * (total - 1) / (count - 1)) for position in range(count)})


class NovelIndex:
    """Numbered novel access plus dialogue-speaker metadata and evidence lookups."""

    def __init__(self, text: str, dialogues: Optional[Sequence[dict[str, Any]]] = None):
        self.lines = text.splitlines()
        self.dialogues = list(dialogues) if dialogues else []
        self.line_to_speakers: dict[int, list[str]] = {}
        self.speaker_to_dialogues: dict[str, list[tuple[int, dict[str, Any]]]] = {}
        self._search_cache: dict[tuple[str, int], dict[str, Any]] = {}
        self._evidence_cache: dict[str, str] = {}
        for dialogue_index, dialogue in enumerate(self.dialogues):
            line = int(dialogue.get("line", 0) or 0)
            speaker = str(dialogue.get("speaker", "")).strip()
            if line > 0 and speaker:
                speakers = self.line_to_speakers.setdefault(line, [])
                if speaker not in speakers:
                    speakers.append(speaker)
                self.speaker_to_dialogues.setdefault(speaker, []).append((dialogue_index, dialogue))

    def _format_line(self, number: int) -> str:
        speakers = self.line_to_speakers.get(number, [])
        label = f" [speaker: {', '.join(speakers)}]" if speakers else ""
        return f"{number}{label}: {self.lines[number - 1].strip()}"

    def read_lines(self, start: int, end: int, limit: int = 240) -> dict[str, Any]:
        start = max(1, int(start))
        end = min(len(self.lines), int(end))
        if start > end:
            return {"text": "", "truncated": False}
        numbers = list(range(start, end + 1))
        truncated = len(numbers) > limit
        numbers = numbers[:limit]
        return {
            "text": "\n".join(self._format_line(number) for number in numbers),
            "truncated": truncated,
        }

    def search(self, keyword: str, limit: int = 20) -> dict[str, Any]:
        key = (keyword, limit)
        if key in self._search_cache:
            return self._search_cache[key]
        matches = [
            {"line_number": number, "line": self._format_line(number)[:500]}
            for number, line in enumerate(self.lines, 1)
            if keyword and keyword in line
        ]
        result = {
            "total_matches": len(matches),
            "truncated": len(matches) > limit,
            "matches": matches[:limit],
        }
        self._search_cache[key] = result
        return result

    def context(self, target_line: int, radius: int = 100) -> str:
        text = self.read_lines(target_line - radius, target_line + radius, limit=radius * 2 + 1)["text"]
        target_prefixes = (f"{target_line}:", f"{target_line} ")
        return "\n".join(
            f">>> TARGET SOURCE LINE {line}" if line.startswith(target_prefixes) else line
            for line in text.splitlines()
        )

    def get_dialogues(self, character_name: str, limit: int = 50) -> list[dict]:
        matched: list[dict] = []
        for index, dialogue in enumerate(self.dialogues):
            if dialogue.get("speaker") == character_name:
                matched.append(
                    {
                        "dialogue_index": index,
                        "line_number": int(dialogue.get("line", 0)),
                        "text": str(dialogue.get("text", ""))[:240],
                    }
                )
                if len(matched) >= limit:
                    break
        if matched:
            return matched
        for number, line in enumerate(self.lines, 1):
            if character_name in line and ("\u300c" in line or "\u300d" in line):
                matched.append({"dialogue_index": -1, "line_number": number, "text": line.strip()[:240]})
                if len(matched) >= limit:
                    break
        return matched

    def evidence_packet(self, character_name: str, max_occurrences: int = 12, radius: int = 5) -> str:
        if character_name in self._evidence_cache:
            return self._evidence_cache[character_name]
        matches = self.search(character_name, limit=80)["matches"]
        if not matches:
            packet = "No literal name occurrence was found. Use dialogue metadata and return unknown if evidence remains absent."
            self._evidence_cache[character_name] = packet
            return packet

        positions = [item["line_number"] for item in matches]
        if len(positions) > max_occurrences:
            last = len(positions) - 1
            chosen = sorted({positions[round(i * last / (max_occurrences - 1))] for i in range(max_occurrences)})
        else:
            chosen = positions
        blocks = []
        for position in chosen:
            block = self.read_lines(position - radius, position + radius, limit=radius * 2 + 1)["text"]
            blocks.append(block)
        dialogues = self.get_dialogues(character_name, limit=12)
        dialogue_text = "\n".join(
            f"line {item['line_number']}: {item['text']}" for item in dialogues
        ) or "No speaker-labelled dialogue found."
        packet = (
            f"Representative name contexts ({len(chosen)} of {len(positions)} occurrences):\n"
            + "\n---\n".join(blocks)
            + "\n\nSpeaker-labelled dialogue samples:\n"
            + dialogue_text
        )
        self._evidence_cache[character_name] = packet
        return packet

    def character_samples(self, speaker: str, max_samples: int = 18, radius: int = 2) -> str:
        occurrences = self.speaker_to_dialogues.get(speaker, [])
        if not occurrences:
            return "(no labeled occurrences)"
        positions = evenly_spaced_positions(len(occurrences), min(max_samples, len(occurrences)))
        windows: list[str] = []
        used_lines: set[int] = set()
        for position in positions:
            _, dialogue = occurrences[position]
            line = int(dialogue.get("line", 0) or 0)
            if line <= 0 or line in used_lines:
                continue
            used_lines.add(line)
            windows.append(self.read_lines(line - radius, line + radius, limit=radius * 2 + 1)["text"])
        return "\n\n--- REPRESENTATIVE OCCURRENCE ---\n".join(windows)
