"""Routes for the SRT translation feature."""
from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid

from flask import Blueprint, Response, jsonify, render_template, request

from ..config import Config
from ..extensions import translate_status
from ..services.translator import enhanced_translate_srt_task
from ..utils.files import allowed_file

logger = logging.getLogger(__name__)

translate_bp = Blueprint("translate", __name__)


def _require_api_key():
    """Return a 500 response if DEEPL_API_KEY is missing, else None."""
    if not Config.DEEPL_API_KEY:
        logger.error("DEEPL_API_KEY missing; rejecting translate request")
        return jsonify({
            "error": "DEEPL_API_KEY is not configured. Set it in the environment before using this endpoint."
        }), 500
    return None


@translate_bp.route("/translate_srt_page", endpoint="translate_srt_page")
def translate_srt_page():
    """Render the SRT translation page."""
    return render_template("translate_srt.html")


@translate_bp.route("/translate_srt", methods=["POST"], endpoint="translate_srt_enhanced")
def translate_srt_enhanced():
    """Accept an SRT file and run block-by-block DeepL translation."""
    missing = _require_api_key()
    if missing is not None:
        return missing

    if "srtFile" not in request.files:
        return jsonify({"error": "No SRT file uploaded"}), 400

    srt_file = request.files["srtFile"]
    if srt_file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    if not allowed_file(srt_file.filename, {"srt"}):
        return jsonify({"error": "Only .srt files allowed"}), 400

    unique_id = str(uuid.uuid4())
    input_path = os.path.join(Config.UPLOAD_FOLDER, f"{unique_id}_input.srt")
    output_filename = f"translated_arabic_rtl_{unique_id}.srt"
    output_path = os.path.join(Config.OUTPUT_FOLDER, output_filename)

    srt_file.save(input_path)
    logger.info("Saved translate input: %s", input_path)

    translate_status[unique_id] = {
        "status": "translating",
        "progress": 0,
        "message": "Starting Arabic translation with automatic RTL formatting...",
    }

    thread = threading.Thread(
        target=enhanced_translate_srt_task,
        args=(input_path, output_path, unique_id),
        daemon=True,
    )
    thread.start()

    return jsonify({
        "success": True,
        "translation_id": unique_id,
        "message": "Translation started - RTL formatting will be applied automatically!",
    })


@translate_bp.route("/translation_status/<translation_id>", endpoint="translation_status")
def translation_status(translation_id):
    """SSE feed for translation progress."""

    def generate():
        while True:
            status = translate_status.get(translation_id)
            if not status:
                yield f"data: {json.dumps({'status': 'error', 'message': 'Not found'})}\n\n"
                break
            if status["status"] == "completed":
                yield f"data: {json.dumps({'status': 'completed', 'progress': 100, 'message': 'Translation completed with automatic RTL formatting!', 'output_filename': status['output_filename']})}\n\n"
                del translate_status[translation_id]
                break
            if status["status"] == "error":
                yield f"data: {json.dumps({'status': 'error', 'message': status['message']})}\n\n"
                del translate_status[translation_id]
                break
            yield f"data: {json.dumps({'status': 'in_progress', 'progress': status['progress'], 'message': status['message']})}\n\n"
            time.sleep(0.5)

    return Response(generate(), mimetype="text/event-stream")
