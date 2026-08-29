"""Tests for the pure SRT-text helpers."""
from __future__ import annotations

import os
import tempfile

import pytest

from srtproj.utils.srt_text import (
    apply_rtl_formatting,
    format_timestamp,
    generate_srt_content,
    has_arabic_text,
    parse_srt_content,
    split_long_text,
)


def test_format_timestamp_zero():
    assert format_timestamp(0) == "00:00:00,000"


def test_format_timestamp_3725_5():
    assert format_timestamp(3725.5) == "01:02:05,500"


def test_apply_rtl_formatting_no_arabic_returns_text():
    assert apply_rtl_formatting("hello") == "hello"


def test_apply_rtl_formatting_arabic_wrapped():
    result = apply_rtl_formatting("مرحبا")
    assert result.startswith("\u202B")
    assert result.endswith("\u202C")
    assert "مرحبا" in result


def test_has_arabic_text_arabic_file_true(tmp_path):
    p = tmp_path / "ar.srt"
    p.write_text("1\n00:00:00,000 --> 00:00:01,000\nمرحبا بالعالم\n", encoding="utf-8")
    assert has_arabic_text(str(p)) is True


def test_has_arabic_text_english_file_false(tmp_path):
    p = tmp_path / "en.srt"
    p.write_text("1\n00:00:00,000 --> 00:00:01,000\nHello world\n", encoding="utf-8")
    assert has_arabic_text(str(p)) is False


def test_split_long_text_creates_four_chunks():
    text = "a b c d e f g h i j k l m n o p"
    chunks = split_long_text(text, max_words=5)
    assert chunks == [
        "a b c d e",
        "f g h i j",
        "k l m n o",
        "p",
    ]


def test_split_long_text_short_returns_single_chunk():
    chunks = split_long_text("a b c", max_words=10)
    assert chunks == ["a b c"]


SAMPLE_SRT = (
    "1\n"
    "00:00:01,000 --> 00:00:03,500\n"
    "First subtitle line\n"
    "second line\n"
    "\n"
    "2\n"
    "00:00:04,000 --> 00:00:06,000\n"
    "Another entry\n"
)


def test_parse_generate_round_trip():
    parsed = parse_srt_content(SAMPLE_SRT)
    assert [s["sequence"] for s in parsed] == [1, 2]
    assert parsed[0]["start_time"] == "00:00:01,000"
    assert parsed[0]["end_time"] == "00:00:03,500"
    assert parsed[0]["text"] == "First subtitle line\nsecond line"
    # Renumbering happens during generation: sequential 1, 2, ...
    rebuilt = generate_srt_content(parsed)
    rebuilt_parsed = parse_srt_content(rebuilt)
    assert [s["text"] for s in rebuilt_parsed] == [s["text"] for s in parsed]
    assert [s["start_time"] for s in rebuilt_parsed] == [s["start_time"] for s in parsed]
    assert [s["end_time"] for s in rebuilt_parsed] == [s["end_time"] for s in parsed]
