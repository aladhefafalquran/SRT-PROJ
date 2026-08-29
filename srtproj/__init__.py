"""SRT-Proj application factory.

This module wires the blueprints together and exposes a single
``create_app()`` function used by ``app.py`` at the repo root.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

from flask import Flask

from .bootstrap import check_dependencies, configure_logging
from .config import Config
from .routes.download import download_bp
from .routes.editor import editor_bp
from .routes.files import files_bp
from .routes.main import main_bp
from .routes.merge import merge_bp
from .routes.transcribe import transcribe_bp
from .routes.translate import translate_bp

logger = logging.getLogger(__name__)


def create_app(config: Optional[Config] = None) -> Flask:
    """Build and return a configured Flask app instance."""
    configure_logging()
    app = Flask(
        __name__,
        template_folder=os.path.join(Config.BASE_DIR, "templates"),
        static_folder=os.path.join(Config.BASE_DIR, "static"),
    )
    cfg = config or Config
    app.config["UPLOAD_FOLDER"] = cfg.UPLOAD_FOLDER
    app.config["OUTPUT_FOLDER"] = cfg.OUTPUT_FOLDER
    app.config["MAX_CONTENT_LENGTH"] = cfg.MAX_CONTENT_LENGTH
    cfg.ensure_dirs()

    app.register_blueprint(main_bp)
    app.register_blueprint(merge_bp)
    app.register_blueprint(transcribe_bp)
    app.register_blueprint(download_bp)
    app.register_blueprint(translate_bp)
    app.register_blueprint(editor_bp)
    app.register_blueprint(files_bp)

    issues = check_dependencies()
    if issues:
        for issue in issues:
            logger.warning("Dependency issue: %s", issue)
    else:
        logger.info("All dependencies found")

    return app
