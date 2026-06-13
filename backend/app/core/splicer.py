"""
音频拼接引擎 — 按原文顺序拼接音频片段。

功能：
1. 按原文顺序拼接音频
2. 对话间间隔 0.3s，段落间隔 1s，章节间隔 2s
3. 淡入淡出效果（每段 50ms）
4. 逐章节拆分输出
"""
import os
from pathlib import Path
from typing import Dict, List, Optional

from pydub import AudioSegment


# ─── Configuration ─────────────────────────────────────────────────

# 间隔配置（毫秒）
GAP_DIALOGUE = 300    # 对话间间隔
GAP_PARAGRAPH = 1000  # 段落间隔
GAP_CHAPTER = 2000    # 章节间隔

# 淡入淡出配置（毫秒）
FADE_DURATION = 50


# ─── Audio Splicer ─────────────────────────────────────────────────

class AudioSplicer:
    """音频拼接引擎"""

    def __init__(
        self,
        gap_dialogue: int = GAP_DIALOGUE,
        gap_paragraph: int = GAP_PARAGRAPH,
        gap_chapter: int = GAP_CHAPTER,
        fade_duration: int = FADE_DURATION,
    ):
        self.gap_dialogue = gap_dialogue
        self.gap_paragraph = gap_paragraph
        self.gap_chapter = gap_chapter
        self.fade_duration = fade_duration

    def _create_silence(self, duration_ms: int) -> AudioSegment:
        """创建静音片段"""
        return AudioSegment.silent(duration=duration_ms)

    def _apply_fade(self, audio: AudioSegment) -> AudioSegment:
        """应用淡入淡出效果"""
        if len(audio) < self.fade_duration * 2:
            return audio
        return audio.fade_in(self.fade_duration).fade_out(self.fade_duration)

    def splice(
        self,
        segments: List[Dict],
        output_path: Optional[str] = None,
    ) -> AudioSegment:
        """拼接音频片段。

        Args:
            segments: [{"audio_path": str, "chapter": str, "order": int}]
            output_path: 输出文件路径（可选）

        Returns:
            拼接后的音频
        """
        if not segments:
            return AudioSegment.empty()

        # 按 order 排序
        sorted_segments = sorted(segments, key=lambda x: x.get("order", 0))

        # 加载所有音频
        audio_segments = []
        for seg in sorted_segments:
            try:
                audio = AudioSegment.from_file(seg["audio_path"])
                audio_segments.append(audio)
            except Exception as e:
                print(f"Warning: Failed to load {seg['audio_path']}: {e}")
                continue

        if not audio_segments:
            return AudioSegment.empty()

        # 拼接音频
        result = audio_segments[0]
        for i in range(1, len(audio_segments)):
            # 应用淡出到前一段
            result = self._apply_fade(result)

            # 添加间隔
            prev_chapter = sorted_segments[i - 1].get("chapter", "")
            curr_chapter = sorted_segments[i].get("chapter", "")

            if prev_chapter != curr_chapter:
                # 章节间隔
                result += self._create_silence(self.gap_chapter)
            elif i > 1 and sorted_segments[i].get("chapter") != sorted_segments[i - 1].get("chapter"):
                # 段落间隔
                result += self._create_silence(self.gap_paragraph)
            else:
                # 对话间隔
                result += self._create_silence(self.gap_dialogue)

            # 应用淡入到后一段
            result += self._apply_fade(audio_segments[i])

        # 保存文件
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            result.export(output_path, format="wav")

        return result

    def splice_by_chapter(
        self,
        segments: List[Dict],
        output_dir: str,
        filename_pattern: str = "chapter_{index}.wav",
    ) -> List[str]:
        """按章节拆分输出。

        Args:
            segments: [{"audio_path": str, "chapter": str, "order": int}]
            output_dir: 输出目录
            filename_pattern: 文件名模式

        Returns:
            输出文件路径列表
        """
        # 按章节分组
        chapters: Dict[str, List[Dict]] = {}
        for seg in segments:
            chapter = seg.get("chapter", "")
            if chapter not in chapters:
                chapters[chapter] = []
            chapters[chapter].append(seg)

        # 按章节顺序拼接
        output_files = []
        os.makedirs(output_dir, exist_ok=True)

        for i, (chapter_name, chapter_segments) in enumerate(chapters.items()):
            if not chapter_name:
                chapter_name = f"chapter_{i + 1}"

            audio = self.splice(chapter_segments)
            filename = filename_pattern.format(index=i + 1, chapter=chapter_name)
            output_path = os.path.join(output_dir, filename)
            audio.export(output_path, format="wav")
            output_files.append(output_path)

        return output_files

    def get_duration(self, audio: AudioSegment) -> float:
        """获取音频时长（秒）"""
        return len(audio) / 1000.0
