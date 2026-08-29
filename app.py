"""SRT-Proj thin entrypoint.

All routing, services, and helpers live under the ``srtproj`` package.
This file only boots the application factory and starts the dev server.
"""
from srtproj import create_app
from srtproj.config import Config

app = create_app()

if __name__ == "__main__":
    app.run(
        host=Config.HOST,
        port=Config.PORT,
        debug=Config.DEBUG,
        threaded=True,
    )
