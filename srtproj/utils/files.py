"""Filesystem helpers: filename validation, secure path joining, cleanup."""
from __future__ import annotations

import os
import shutil
from typing import Iterable, Optional

from werkzeug.utils import secure_filename as _secure_filename


def allowed_file(filename: str, allowed_extensions: Iterable[str]) -> bool:
    """Return True if ``filename`` has an allowed extension (case-insensitive).

    The check is purely suffix-based and does not look at the filesystem.
    """
    if not filename or "." not in filename:
        return False
    ext = filename.rsplit(".", 1)[1].lower()
    return ext in {e.lower() for e in allowed_extensions}


def secure_filename(filename: str) -> str:
    """Thin wrapper around ``werkzeug.utils.secure_filename`` so the rest
    of the codebase only depends on this module."""
    return _secure_filename(filename) or ""


def secure_join(base: str, filename: str) -> Optional[str]:
    """Safely join ``base`` and ``filename`` and verify the result stays
    inside ``base``.

    The input filename is first rejected if it contains a path separator
    (``/`` or ``\\``), a parent reference (``..``), or an absolute path.
    The joined path is then resolved via :func:`os.path.realpath` so
    symbolic links cannot be used to escape. If the resolved path escapes
    the base directory, or if the filename is empty, ``None`` is returned
    so callers can reject the request with a 400.
    """
    if not filename:
        return None

    # Reject anything that looks like a path traversal / absolute path
    # before letting ``secure_filename`` silently rewrite it. The
    # original app used this same check ('..' in filename or any path
    # separator) and we want to surface bad input rather than quietly
    # rewrite it.
    if (
        ".." in filename
        or "/" in filename
        or "\\" in filename
        or filename.startswith("~")
    ):
        return None

    safe_name = secure_filename(filename)
    if not safe_name or safe_name in {".", ".."}:
        return None

    base_real = os.path.realpath(base)
    candidate = os.path.realpath(os.path.join(base_real, safe_name))

    # Ensure the resolved candidate is the base or sits beneath it.
    if candidate != base_real and not candidate.startswith(base_real + os.sep):
        return None

    return candidate


def cleanup_temp_files(path: str) -> None:
    """Best-effort cleanup of a temp file or directory."""
    if not path:
        return
    try:
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
        elif os.path.isfile(path):
            try:
                os.remove(path)
            except OSError:
                pass
    except Exception:
        # Cleanup must never raise; the caller is almost always in a
        # ``finally`` block.
        pass
