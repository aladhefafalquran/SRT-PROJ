"""Service: yt-dlp-backed online video downloader."""
from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess

from ..extensions import download_status

logger = logging.getLogger(__name__)


_PERCENT_RE = re.compile(r"(\d+(?:\.\d+)?)%")
_SPEED_RE = re.search  # placeholder to keep the symbol reserved


def _build_ytdlp_cmd(url: str, quality: str, audio_only: bool, output_dir: str) -> list:
    cmd = ["yt-dlp"]
    if audio_only:
        cmd.extend([
            "--extract-audio",
            "--audio-format", "mp3",
            "--audio-quality", "0",
            "--output", os.path.join(output_dir, "%(title)s.%(ext)s"),
        ])
    else:
        if quality == "best":
            cmd.extend(["--format", "best[height<=2160]"])
        elif quality == "1080p":
            cmd.extend(["--format", "best[height<=1080]"])
        elif quality == "720p":
            cmd.extend(["--format", "best[height<=720]"])
        elif quality == "480p":
            cmd.extend(["--format", "best[height<=480]"])
        else:
            cmd.extend(["--format", "best"])
        cmd.extend([
            "--output", os.path.join(output_dir, "%(title)s.%(ext)s"),
            "--merge-output-format", "mp4",
        ])
    cmd.extend([
        "--no-playlist",
        "--write-info-json",
        url,
    ])
    return cmd


def run_download(
    url: str,
    quality: str,
    audio_only: bool,
    unique_id: str,
    output_folder: str,
) -> None:
    """Background task that runs yt-dlp and moves the result into the
    shared output folder."""
    status = download_status[unique_id]
    output_dir = ""
    try:
        status["message"] = "Checking URL and fetching info..."
        status["progress"] = 5

        output_dir = os.path.join(output_folder, f"downloads_{unique_id}")
        os.makedirs(output_dir, exist_ok=True)

        cmd = _build_ytdlp_cmd(url, quality, audio_only, output_dir)

        status["message"] = "Starting download..."
        status["progress"] = 10
        logger.info("Starting yt-dlp: %s", " ".join(cmd[:6]))

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        speed_re = re.compile(r"at\s+([\d.]+\w+/s)")
        for line in iter(proc.stdout.readline, ""):
            if not line.strip():
                continue
            logger.debug("yt-dlp: %s", line.strip())
            if "[download]" in line and "%" in line:
                match = _PERCENT_RE.search(line)
                if match:
                    status["progress"] = min(float(match.group(1)), 95.0)
                speed_match = speed_re.search(line)
                if speed_match:
                    status["message"] = f"Downloading... {speed_match.group(1)}"
            if "has already been downloaded" in line or "Destination:" in line:
                status["progress"] = 95
                status["message"] = "Finalizing download..."

        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError(f"yt-dlp failed with return code {proc.returncode}")

        downloaded_files = [
            f for f in os.listdir(output_dir)
            if not f.endswith(".json") and not f.endswith(".part")
        ]
        if not downloaded_files:
            raise RuntimeError("No output file found after download")

        filename = downloaded_files[0]
        original_path = os.path.join(output_dir, filename)
        final_filename = f"downloaded_{unique_id}_{filename}"
        final_path = os.path.join(output_folder, final_filename)
        shutil.move(original_path, final_path)

        file_size = os.path.getsize(final_path)
        status.update({
            "status": "completed",
            "progress": 100,
            "message": "Download completed!",
            "filename": final_filename,
            "file_size": file_size,
            "original_name": filename,
        })
        logger.info("Download completed: %s", final_filename)
    except Exception as exc:
        logger.exception("Download failed: %s", exc)
        status.update({
            "status": "error",
            "message": str(exc),
            "progress": 0,
        })
    finally:
        if output_dir and os.path.exists(output_dir):
            try:
                shutil.rmtree(output_dir)
            except OSError:
                pass
