"""Service: thin wrappers around the SRT editor utilities.

The actual parsing/serialisation lives in :mod:`srtproj.utils.srt_text`.
This module just re-exports the helpers with a stable import path so the
editor route can depend on a service-layer symbol.
"""
from __future__ import annotations

from ..utils.srt_text import generate_srt_content, parse_srt_content

__all__ = ["parse_srt_content", "generate_srt_content"]
