"""Startup helpers: dependency checks, shared logger setup."""
from __future__ import annotations

import logging
import subprocess
from typing import List

from .extensions import WHISPER_AVAILABLE

logger = logging.getLogger(__name__)


def _check_tool(name: str) -> bool:
    try:
        subprocess.run([name, "--version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return False


def check_dependencies() -> List[str]:
    """Return a list of human-readable warnings for missing dependencies."""
    issues: List[str] = []

    if _check_tool("ffmpeg"):
        logger.info("FFmpeg found")
    else:
        issues.append("FFmpeg not found. Please install FFmpeg.")
        logger.warning("FFmpeg not found")

    if _check_tool("ffprobe"):
        logger.info("ffprobe found")
    else:
        issues.append("ffprobe not found. Please install FFmpeg.")
        logger.warning("ffprobe not found")

    if _check_tool("yt-dlp"):
        logger.info("yt-dlp found")
    else:
        issues.append("yt-dlp not found. Please install yt-dlp.")
        logger.warning("yt-dlp not found")

    if WHISPER_AVAILABLE:
        logger.info("Whisper available")
    else:
        issues.append("Whisper not found. Install with: pip install openai-whisper")
        logger.warning("Whisper not available")

    return issues


def configure_logging(level: int = logging.INFO) -> None:
    """Configure root logging once at app startup."""
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )
