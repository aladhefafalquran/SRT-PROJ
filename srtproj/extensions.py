"""Shared module-level singletons used across blueprints."""
from __future__ import annotations

# These dicts hold per-job state for the four long-running features.
# They are deliberately module-level so background threads can mutate
# the entries and SSE endpoints can read them safely.
processing_status: dict = {}            # /merge + /upload (burn-subs)
video_processing_status: dict = {}      # /video_to_srt
translate_status: dict = {}             # /translate_srt
download_status: dict = {}              # /download_online_video

# Whisper import is optional. Other modules import this flag rather than
# re-trying the import themselves.
try:  # pragma: no cover - optional dep
    import whisper  # type: ignore
    WHISPER_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dep
    whisper = None  # type: ignore
    WHISPER_AVAILABLE = False
