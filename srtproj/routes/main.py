"""Top-level routes: index + merge landing page."""
from __future__ import annotations

from flask import Blueprint, redirect, render_template, request, url_for

from ..bootstrap import check_dependencies

main_bp = Blueprint("main", __name__)


@main_bp.route("/", endpoint="index")
def index():
    """Redirect to the video-to-SRT extractor (legacy default)."""
    return redirect(url_for("transcribe.video_to_srt_page"))


@main_bp.route("/merge", endpoint="merge_page")
def merge_page():
    """Render the burn-subs landing page."""
    message = request.args.get("message")
    issues = check_dependencies()
    if issues:
        message = "⚠️ Missing dependencies: " + "; ".join(issues)
    return render_template("index.html", message=message)
