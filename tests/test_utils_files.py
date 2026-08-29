"""Tests for filesystem helpers in srtproj.utils.files."""
from __future__ import annotations

import os

import pytest

from srtproj.utils.files import allowed_file, secure_join


def test_allowed_file_case_insensitive():
    assert allowed_file("foo.SRT", {"srt"}) is True
    assert allowed_file("foo.srt", {"srt"}) is True


def test_allowed_file_rejects_other_extension():
    assert allowed_file("foo.exe", {"srt"}) is False
    assert allowed_file("foo", {"srt"}) is False
    assert allowed_file("", {"srt"}) is False


def test_secure_join_traversal_returns_none(tmp_path):
    base = str(tmp_path)
    # `..` is stripped by secure_filename, then we double-check the
    # candidate stays inside the base.
    assert secure_join(base, "../etc/passwd") is None
    assert secure_join(base, "..") is None
    assert secure_join(base, "") is None


def test_secure_join_legit_filename_resolves_under_base(tmp_path):
    base = str(tmp_path)
    target = tmp_path / "legit.txt"
    target.write_text("ok", encoding="utf-8")
    resolved = secure_join(base, "legit.txt")
    assert resolved == str(target.resolve())
    assert resolved.startswith(os.path.realpath(base))


def test_secure_join_absolute_path_rejected(tmp_path):
    base = str(tmp_path)
    # Werkzeug strips leading slashes; the result must still resolve under base.
    outside = "/etc/passwd"
    assert secure_join(base, outside) is None
