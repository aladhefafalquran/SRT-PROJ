"""Application configuration loaded from environment variables."""
import os
from pathlib import Path


def _bool(name: str, default: bool = False) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


class Config:
    """Centralised configuration for the SRT-Proj Flask app.

    All values are sourced from environment variables. Sensible fallbacks
    are used for filesystem paths and feature flags, but secrets (DeepL
    API key) must be supplied explicitly.
    """

    # --- Filesystem paths (absolute, created on demand) ----------------
    BASE_DIR = Path(__file__).resolve().parent.parent
    UPLOAD_FOLDER = os.environ.get(
        "SRTPROJ_UPLOAD_FOLDER",
        str((BASE_DIR / "uploads").resolve()),
    )
    OUTPUT_FOLDER = os.environ.get(
        "SRTPROJ_OUTPUT_FOLDER",
        str((BASE_DIR / "outputs").resolve()),
    )

    # --- Upload constraints -------------------------------------------
    MAX_CONTENT_LENGTH = int(
        os.environ.get("SRTPROJ_MAX_CONTENT_LENGTH", 2048 * 1024 * 1024)
    )  # 2 GiB default

    # --- Allowed extensions -------------------------------------------
    ALLOWED_EXTENSIONS_VIDEO = {"mp4", "mov", "avi", "mkv"}
    ALLOWED_EXTENSIONS_SUB = {"srt", "vtt", "ass"}

    # --- Server runtime -----------------------------------------------
    HOST = os.environ.get("SRTPROJ_HOST", "0.0.0.0")
    PORT = int(os.environ.get("SRTPROJ_PORT", "5000"))
    DEBUG = _bool("SRTPROJ_DEBUG", False)

    # --- DeepL --------------------------------------------------------
    # Required: must be supplied via environment, no hardcoded fallback.
    DEEPL_API_KEY = os.environ.get("DEEPL_API_KEY", "").strip()
    DEEPL_API_URL = os.environ.get(
        "DEEPL_API_URL", "https://api-free.deepl.com/v2/translate"
    )

    @classmethod
    def ensure_dirs(cls) -> None:
        """Create upload and output directories if they do not exist."""
        os.makedirs(cls.UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(cls.OUTPUT_FOLDER, exist_ok=True)
