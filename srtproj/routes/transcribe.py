"""Routes for the video-to-SRT (Whisper) feature."""
from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid

from flask import Blueprint, Response, jsonify, render_template, request

from ..config import Config
from ..extensions import WHISPER_AVAILABLE, video_processing_status
from ..services.transcribe import run_extraction
from ..utils.files import secure_filename

logger = logging.getLogger(__name__)

transcribe_bp = Blueprint("transcribe", __name__)


@transcribe_bp.route("/video_to_srt_page", endpoint="video_to_srt_page")
def video_to_srt_page():
    """Render the video-to-SRT extractor page."""
    return render_template("video_to_srt.html")


@transcribe_bp.route("/video_to_srt", methods=["POST"], endpoint="start_video_to_srt")
def start_video_to_srt():
    if not WHISPER_AVAILABLE:
        return jsonify({
            "error": "Whisper is not installed. Please install with: pip install openai-whisper"
        }), 500

    if "video" not in request.files:
        return jsonify({"error": "No video uploaded"}), 400

    video = request.files["video"]
    model = request.form.get("model", "small")
    translate = request.form.get("translate") == "true"

    if video.filename == "":
        return jsonify({"error": "No file selected"}), 400

    unique_id = str(uuid.uuid4())
    temp_dir = os.path.join(Config.UPLOAD_FOLDER, unique_id)
    os.makedirs(temp_dir, exist_ok=True)

    safe_name = secure_filename(video.filename) or f"{unique_id}.mp4"
    video_path = os.path.join(temp_dir, safe_name)
    video.save(video_path)

    srt_filename = f"extracted_{unique_id}.srt"
    srt_path = os.path.join(Config.OUTPUT_FOLDER, srt_filename)

    video_processing_status[unique_id] = {
        "status": "running",
        "progress": 0,
        "message": "Loading Whisper model...",
        "srt_file": srt_filename,
    }

    thread = threading.Thread(
        target=run_extraction,
        kwargs=dict(
            video_path=video_path,
            srt_path=srt_path,
            unique_id=unique_id,
            model_name=model,
            translate=translate,
            output_folder=Config.OUTPUT_FOLDER,
            temp_dir=temp_dir,
        ),
        daemon=True,
    )
    thread.start()

    return jsonify({"success": True, "conversion_id": unique_id})


@transcribe_bp.route("/video_status/<conversion_id>", endpoint="video_status")
def video_status(conversion_id):
    def generate():
        try:
            while True:
                status = video_processing_status.get(conversion_id)
                if not status:
                    yield f"data: {json.dumps({'status': 'error', 'message': 'Job not found'})}\n\n"
                    break
                if status["status"] == "completed":
                    yield f"data: {json.dumps({'status': 'completed', 'progress': 100, 'message': 'Enhanced SRT completed!', 'data': {'srt_file': status['srt_file']}})}\n\n"
                    del video_processing_status[conversion_id]
                    break
                if status["status"] == "error":
                    yield f"data: {json.dumps({'status': 'error', 'message': status['message']})}\n\n"
                    del video_processing_status[conversion_id]
                    break
                yield f"data: {json.dumps({'status': 'in_progress', 'progress': status['progress'], 'message': status['message']})}\n\n"
                time.sleep(0.5)
        except Exception as exc:
            logger.exception("video_status SSE error: %s", exc)
            yield f"data: {json.dumps({'status': 'error', 'message': f'Server error: {exc}'})}\n\n"

    return Response(generate(), mimetype="text/event-stream")


@transcribe_bp.route("/video_status_json/<conversion_id>", endpoint="video_status_json")
def video_status_json(conversion_id):
    """JSON polling fallback for the video-to-SRT job."""
    try:
        status = video_processing_status.get(conversion_id)
        if not status:
            return jsonify({"status": "error", "message": "Job not found"}), 404

        if status["status"] == "completed":
            completed_status = status.copy()
            del video_processing_status[conversion_id]
            return jsonify({
                "status": "completed",
                "progress": 100,
                "message": "Enhanced SRT completed!",
                "data": {"srt_file": completed_status["srt_file"]},
            })
        if status["status"] == "error":
            error_message = status["message"]
            del video_processing_status[conversion_id]
            return jsonify({"status": "error", "message": error_message})
        return jsonify({
            "status": "in_progress",
            "progress": status["progress"],
            "message": status["message"],
        })
    except Exception as exc:
        logger.exception("video_status_json error: %s", exc)
        return jsonify({"status": "error", "message": f"Server error: {exc}"}), 500
