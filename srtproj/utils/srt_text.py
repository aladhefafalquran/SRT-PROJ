"""SRT text helpers: timestamp formatting, RTL embedding, splitting, parsing."""
from __future__ import annotations

import re
from typing import List

# Comprehensive Unicode ranges for Arabic script.
_ARABIC_RE = re.compile(
    r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]"
)


def format_timestamp(seconds: float) -> str:
    """Convert float seconds to SRT ``HH:MM:SS,mmm`` format."""
    if seconds is None or seconds < 0:
        seconds = 0
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    whole_secs = int(secs)
    millis = int(round((secs - whole_secs) * 1000))
    # Carry an extra second if rounding pushed millis to 1000.
    if millis == 1000:
        millis = 0
        whole_secs += 1
        if whole_secs == 60:
            whole_secs = 0
            minutes += 1
            if minutes == 60:
                minutes = 0
                hours += 1
    return f"{hours:02}:{minutes:02}:{whole_secs:02},{millis:03}"


def has_arabic_text(srt_path: str) -> bool:
    """Return True if the SRT file at ``srt_path`` contains Arabic script."""
    try:
        with open(srt_path, "r", encoding="utf-8") as handle:
            content = handle.read()
    except Exception:
        return False
    return bool(_ARABIC_RE.search(content))


def apply_rtl_formatting(text: str) -> str:
    """Wrap Arabic text in U+202B / U+202C so players render it RTL.

    For non-Arabic text the input is returned unchanged (whitespace
    stripped) so callers don't have to branch.
    """
    if text is None:
        return ""
    stripped = text.strip()
    if not stripped:
        return ""
    if _ARABIC_RE.search(stripped):
        return "\u202B" + stripped + "\u202C"
    return stripped


def create_rtl_srt(input_srt: str, output_srt: str) -> bool:
    """Create an RTL-aware SRT using U+202E / U+202C per text line.

    This is the legacy variant used by the burn-subs pipeline; the newer
    ``apply_rtl_formatting`` is preferred for block-level translation.
    """
    try:
        with open(input_srt, "r", encoding="utf-8") as handle:
            lines = handle.readlines()
        with open(output_srt, "w", encoding="utf-8") as handle:
            for line in lines:
                if line.strip() and not line.strip().isdigit() and "-->" not in line:
                    if _ARABIC_RE.search(line):
                        line = "\u202E" + line.strip() + "\u202C\n"
                handle.write(line)
        return True
    except Exception:
        return False


def split_long_text(text: str, max_chars: int = 80, max_words: int = 15) -> List[str]:
    """Split ``text`` into smaller chunks at natural break points.

    The strategy mirrors the original implementation:
      1. Try sentence boundaries.
      2. Fall back to clause boundaries.
      3. Fall back to hard word-count chunks.
    """
    if not text:
        return []
    text = text.strip()
    if len(text) <= max_chars and len(text.split()) <= max_words:
        return [text]

    # Sentences first
    sentences = re.split(r"(?<=[.!?])\s+", text)
    if len(sentences) > 1:
        return [s.strip() for s in sentences if s.strip()]

    # Clauses next
    clauses = re.split(r"(?<=[,;])\s+", text)
    if len(clauses) > 1:
        chunks: List[str] = []
        current = ""
        for clause in clauses:
            candidate = (current + " " + clause).strip() if current else clause
            if len(candidate) <= max_chars:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                current = clause.strip()
        if current:
            chunks.append(current)
        return [c for c in chunks if c]

    # Hard word-count chunks
    words = text.split()
    if len(words) > max_words:
        return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]
    return [text]


def parse_srt_content(content: str) -> list:
    """Parse an SRT string into a list of structured subtitle dicts."""
    subtitles: list = []
    if not content:
        return subtitles
    blocks = content.strip().split("\n\n")
    for i, block in enumerate(blocks):
        lines = block.strip().split("\n")
        if len(lines) < 3:
            continue
        try:
            sequence = int(lines[0].strip())
        except ValueError:
            continue
        timing_line = lines[1].strip()
        if "-->" not in timing_line:
            continue
        start_time, end_time = [s.strip() for s in timing_line.split("-->", 1)]
        text = "\n".join(lines[2:]).strip()
        subtitles.append({
            "id": i + 1,
            "sequence": sequence,
            "start_time": start_time,
            "end_time": end_time,
            "text": text,
        })
    return subtitles


def generate_srt_content(subtitles: list) -> str:
    """Serialise a list of subtitle dicts back to SRT text."""
    srt_lines: List[str] = []
    for i, subtitle in enumerate(subtitles):
        srt_lines.append(str(i + 1))
        srt_lines.append(f"{subtitle['start_time']} --> {subtitle['end_time']}")
        srt_lines.append(subtitle.get("text", ""))
        if i < len(subtitles) - 1:
            srt_lines.append("")
    return "\n".join(srt_lines)
