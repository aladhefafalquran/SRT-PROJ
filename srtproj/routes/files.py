"""Routes for serving generated output files to the user.

The legacy ``/download/<filename>`` route had a zero-width character
between ``<`` and ``filename``; this version uses the explicit
``<string:filename>`` converter so the parameter can never contain a
slash, and runs the candidate path through :func:`secure_join` to
defeat any remaining traversal attempts.
"""
from __future__ import annotations

import logging
import os

from flask import Blueprint, current_app, jsonify, send_file

from ..config import Config
from ..utils.files import secure_join

logger = logging.getLogger(__name__)

files_bp = Blueprint("files", __name__)


@files_bp.route("/download/<string:filename>", endpoint="download_file")
def download_file(filename):
    """Serve a file from the output folder.

    The filename is first filtered through ``secure_filename`` and the
    resolved path is verified to live inside ``OUTPUT_FOLDER``; any
    attempt to escape (e.g. ``../etc/passwd``) is rejected with a 400.
    """
    logger.info("Download requested: %s", filename)
    abs_path = secure_join(Config.OUTPUT_FOLDER, filename)
    if abs_path is None:
        logger.warning("Rejected unsafe filename: %s", filename)
        return jsonify({"error": "Invalid filename"}), 400

    if not os.path.exists(abs_path):
        logger.info("File not found: %s", abs_path)
        if os.path.isdir(Config.OUTPUT_FOLDER):
            logger.debug("Files in output folder: %s", os.listdir(Config.OUTPUT_FOLDER))
        return jsonify({"error": "File not found"}), 404

    try:
        return send_file(abs_path, as_attachment=True, download_name=filename)
    except Exception as exc:
        logger.exception("send_file failed: %s", exc)
        return jsonify({"error": f"Error sending file: {exc}"}), 500
