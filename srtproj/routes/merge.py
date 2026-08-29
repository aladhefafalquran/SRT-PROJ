"""Routes for the burn-subtitles feature."""
from __future__ import annotations

import json
import logging
import os
import subprocess
import threading
import time
import uuid

from flask import Blueprint, Response, current_app, jsonify, render_template, request

from ..config import Config
from ..extensions import processing_status
from ..services.merge import process_video_job
from ..utils.files import allowed_file, secure_filename

logger = logging.getLogger(__name__)

merge_bp = Blueprint("merge", __name__)


@merge_bp.route("/upload", methods=["POST"], endpoint="upload_files")
def upload_files():
    """Accept a video + subtitle pair, kick off the burn-subs job."""
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return render_template("index.html", message="FFmpeg is not installed or not found in PATH."), 500

    if "videoFile" not in request.files or "subtitleFile" not in request.files:
        return render_template("index.html", message="Please select both files."), 400

    video = request.files["videoFile"]
    sub = request.files["subtitleFile"]
    if video.filename == "" or sub.filename == "":
        return render_template("index.html", message="Please select both files."), 400
    if not allowed_file(video.filename, Config.ALLOWED_EXTENSIONS_VIDEO):
        return render_template("index.html", message="Invalid video file."), 400
    if not allowed_file(sub.filename, Config.ALLOWED_EXTENSIONS_SUB):
        return render_template("index.html", message="Invalid subtitle file."), 400

    unique_id = str(uuid.uuid4())
    temp_dir = os.path.join(Config.UPLOAD_FOLDER, unique_id)
    os.makedirs(temp_dir, exist_ok=True)

    vpath = os.path.join(temp_dir, secure_filename(video.filename))
    spath = os.path.join(temp_dir, secure_filename(sub.filename))
    opath = os.path.join(Config.OUTPUT_FOLDER, f"merged_{unique_id}.mp4")

    video.save(vpath)
    sub.save(spath)

    processing_status[unique_id] = {
        "status": "running",
        "progress": 0,
        "status_text": "Starting...",
        "logs": [],
        "output_filename": f"merged_{unique_id}.mp4",
    }

    thread = threading.Thread(
        target=process_video_job,
        args=(vpath, spath, opath, unique_id, temp_dir),
        daemon=True,
    )
    thread.start()

    return render_template("processing.html", unique_id=unique_id, output_filename=f"merged_{unique_id}.mp4")


@merge_bp.route("/stream/<unique_id>", endpoint="stream_progress")
def stream_progress(unique_id):
    """SSE feed for the burn-subs frontend."""

    def generate():
        last_log = 0
        last_progress = -1
        while True:
            if unique_id not in processing_status:
                yield f"data: {json.dumps({'type': 'complete', 'success': False})}\n\n"
                break
            status = processing_status[unique_id]
            progress = status.get("progress", 0)
            if progress != last_progress:
                yield f"data: {json.dumps({'type': 'progress', 'percentage': progress, 'status': status.get('status_text', '')})}\n\n"
                last_progress = progress
            logs = status.get("logs", [])
            if len(logs) > last_log:
                for log in logs[last_log:]:
                    yield f"data: {json.dumps({'type': 'log', 'message': log})}\n\n"
                last_log = len(logs)
            if status["status"] == "success":
                yield f"data: {json.dumps({'type': 'complete', 'success': True, 'output_filename': status['output_filename']})}\n\n"
                del processing_status[unique_id]
                break
            if status["status"] == "failed":
                yield f"data: {json.dumps({'type': 'complete', 'success': False, 'message': status.get('message', 'Unknown error')})}\n\n"
                del processing_status[unique_id]
                break
            time.sleep(1)

    return Response(generate(), mimetype="text/event-stream")


@merge_bp.route("/status/<unique_id>", endpoint="check_status")
def check_status(unique_id):
    """JSON snapshot of a burn-subs job."""
    status = processing_status.get(unique_id, {"status": "not_found"})
    if status["status"] in {"success", "failed"}:
        copy = status.copy()
        if unique_id in processing_status:
            del processing_status[unique_id]
        return jsonify(copy)
    return jsonify(status)
