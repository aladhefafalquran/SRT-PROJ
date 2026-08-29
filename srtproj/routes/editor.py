"""Routes for the SRT editor."""
from __future__ import annotations

import logging
import os
import uuid

from flask import Blueprint, jsonify, render_template, request

from ..config import Config
from ..services.editor import generate_srt_content, parse_srt_content
from ..utils.files import allowed_file

logger = logging.getLogger(__name__)

editor_bp = Blueprint("editor", __name__)


@editor_bp.route("/edit_srt_page", endpoint="edit_srt_page")
def edit_srt_page():
    """Render the SRT editor page."""
    return render_template("edit_srt.html")


@editor_bp.route("/parse_srt", methods=["POST"], endpoint="parse_srt")
def parse_srt_route():
    """Parse an uploaded SRT and return structured JSON."""
    if "srtFile" not in request.files:
        return jsonify({"error": "No SRT file uploaded"}), 400
    srt_file = request.files["srtFile"]
    if srt_file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    if not allowed_file(srt_file.filename, {"srt"}):
        return jsonify({"error": "Only .srt files allowed"}), 400
    try:
        content = srt_file.read().decode("utf-8")
        subtitles = parse_srt_content(content)
        return jsonify({
            "success": True,
            "filename": srt_file.filename,
            "subtitles": subtitles,
            "total_count": len(subtitles),
        })
    except Exception as exc:
        logger.exception("parse_srt failed: %s", exc)
        return jsonify({"error": f"Failed to parse SRT: {exc}"}), 400


@editor_bp.route("/save_srt", methods=["POST"], endpoint="save_srt")
def save_srt_route():
    """Serialise edited subtitle JSON back to an SRT file."""
    try:
        data = request.get_json() or {}
        if "subtitles" not in data:
            return jsonify({"error": "No subtitle data provided"}), 400
        subtitles = data["subtitles"]
        unique_id = str(uuid.uuid4())
        filename = f"edited_{unique_id}.srt"
        output_path = os.path.join(Config.OUTPUT_FOLDER, filename)
        srt_content = generate_srt_content(subtitles)
        os.makedirs(Config.OUTPUT_FOLDER, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            handle.write(srt_content)
        return jsonify({
            "success": True,
            "filename": filename,
            "message": "SRT file saved successfully!",
        })
    except Exception as exc:
        logger.exception("save_srt failed: %s", exc)
        return jsonify({"error": f"Failed to save SRT: {exc}"}), 500
