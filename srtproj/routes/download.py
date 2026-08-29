"""Routes for the online video downloader."""
from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from urllib.parse import urlparse

from flask import Blueprint, Response, jsonify, render_template, request

from ..config import Config
from ..extensions import download_status
from ..services.downloader import run_download

logger = logging.getLogger(__name__)

download_bp = Blueprint("download", __name__)


@download_bp.route("/download_video_page", endpoint="download_video_page")
def download_video_page():
    """Render the online video downloader page."""
    return render_template("download_video.html")


@download_bp.route("/download_online_video", methods=["POST"], endpoint="download_online_video")
def download_online_video():
    """Kick off a yt-dlp download in the background."""
    try:
        data = request.get_json() or {}
        url = (data.get("url") or "").strip()
        quality = data.get("quality", "best")
        audio_only = bool(data.get("audio_only", False))

        if not url:
            return jsonify({"error": "Please provide a valid URL"}), 400
        try:
            parsed = urlparse(url)
            if not all([parsed.scheme, parsed.netloc]):
                raise ValueError("invalid")
        except Exception:
            return jsonify({"error": "Invalid URL format"}), 400

        unique_id = str(uuid.uuid4())
        download_status[unique_id] = {
            "status": "starting",
            "progress": 0,
            "message": "Initializing download...",
            "url": url,
            "filename": None,
            "file_size": 0,
        }

        thread = threading.Thread(
            target=run_download,
            args=(url, quality, audio_only, unique_id, Config.OUTPUT_FOLDER),
            daemon=True,
        )
        thread.start()

        return jsonify({
            "success": True,
            "download_id": unique_id,
            "message": "Download started",
        })
    except Exception as exc:
        logger.exception("Failed to start download: %s", exc)
        return jsonify({"error": f"Failed to start download: {exc}"}), 500


@download_bp.route("/download_status/<download_id>", endpoint="get_download_status")
def get_download_status(download_id):
    """SSE feed for download progress."""

    def generate():
        while True:
            status = download_status.get(download_id)
            if not status:
                yield f"data: {json.dumps({'status': 'error', 'message': 'Download not found'})}\n\n"
                break
            if status["status"] == "completed":
                yield f"data: {json.dumps({'status': 'completed', 'progress': 100, 'message': status['message'], 'filename': status['filename'], 'original_name': status.get('original_name', ''), 'file_size': status['file_size']})}\n\n"
                break
            if status["status"] == "error":
                yield f"data: {json.dumps({'status': 'error', 'message': status['message']})}\n\n"
                if download_id in download_status:
                    del download_status[download_id]
                break
            yield f"data: {json.dumps({'status': 'downloading', 'progress': status['progress'], 'message': status['message']})}\n\n"
            time.sleep(1)

    return Response(generate(), mimetype="text/event-stream")
