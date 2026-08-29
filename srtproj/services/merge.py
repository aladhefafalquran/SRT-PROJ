"""Service: burn subtitles into a video using FFmpeg.

Mirrors the original ``process_video_job`` exactly, just split out for
clarity. The MAXIMUM-quality FFmpeg flag set lives in
:mod:`srtproj.utils.ffmpeg`.
"""
from __future__ import annotations

import logging
import os
import subprocess

from ..extensions import processing_status
from ..utils.ffmpeg import (
    build_burn_subs_cmd,
    get_detailed_video_info,
    get_video_duration,
    parse_ffmpeg_progress,
)
from ..utils.files import cleanup_temp_files
from ..utils.srt_text import create_rtl_srt, has_arabic_text

logger = logging.getLogger(__name__)


def process_video_job(
    video_path: str,
    subtitle_path: str,
    output_path: str,
    unique_id: str,
    temp_dir: str,
) -> None:
    """Run the heavy burn-subs pipeline and update ``processing_status``."""
    status = processing_status[unique_id]
    try:
        duration = get_video_duration(video_path)
        video_info = get_detailed_video_info(video_path)
        status["status_text"] = "Analyzing video for MAXIMUM quality preservation..."

        if video_info:
            status["logs"].append(
                f'🎬 Original resolution: {video_info["width"]}x{video_info["height"]}'
            )
            status["logs"].append(f'🎬 Original codec: {video_info["codec"]}')
            status["logs"].append(f'🎬 Original bitrate: {video_info["bitrate"]} bps')
            status["logs"].append(f'🎬 Original FPS: {video_info["fps"]}')
            status["logs"].append(f'🎬 Original pixel format: {video_info["pix_fmt"]}')

        is_arabic = has_arabic_text(subtitle_path)
        if is_arabic:
            status["status_text"] = "Processing Arabic RTL subtitles..."
            rtl_subtitle_path = os.path.join(temp_dir, "rtl_subtitles.srt")
            if create_rtl_srt(subtitle_path, rtl_subtitle_path):
                subtitle_path = rtl_subtitle_path
                status["logs"].append(
                    "✅ Created RTL-compatible subtitle file for Arabic text"
                )

        cmd = build_burn_subs_cmd(
            video_path=video_path,
            subtitle_path=subtitle_path,
            output_path=output_path,
            video_info=video_info,
            is_arabic=is_arabic,
        )

        if is_arabic:
            status["status_text"] = "Starting MAXIMUM quality encoding with Arabic RTL..."
            status["logs"].append("🎬 Arabic RTL + Maximum Quality Mode Activated")
        else:
            status["status_text"] = "Starting MAXIMUM quality encoding..."
            status["logs"].append("🎬 Maximum Quality Mode Activated")

        status["logs"].append(f'🔧 FFmpeg command preview: {" ".join(cmd[:15])}...')

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        for line in iter(proc.stdout.readline, ""):
            if not line.strip():
                continue
            status["logs"].append(line.strip())
            if duration:
                progress = parse_ffmpeg_progress(line, duration)
                if progress is not None:
                    status["progress"] = progress
                    status["status_text"] = (
                        f"Maximum quality encoding... {progress:.1f}%"
                    )

        proc.wait()

        if proc.returncode == 0:
            output_info = get_detailed_video_info(output_path)
            if output_info and video_info:
                status["logs"].append(
                    f'✅ Output resolution: {output_info["width"]}x{output_info["height"]}'
                )
                status["logs"].append(
                    f'✅ Input resolution: {video_info["width"]}x{video_info["height"]}'
                )
                if (
                    output_info["width"] == video_info["width"]
                    and output_info["height"] == video_info["height"]
                ):
                    status["logs"].append("🎯 ✅ PERFECT RESOLUTION MATCH!")
                else:
                    status["logs"].append("⚠️ Resolution mismatch detected")
                if os.path.exists(video_path) and os.path.exists(output_path):
                    input_size = os.path.getsize(video_path)
                    output_size = os.path.getsize(output_path)
                    size_ratio = output_size / input_size if input_size else 0
                    status["logs"].append(
                        f"📊 Size ratio: {size_ratio:.2f}x (output/input)"
                    )

            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                status.update({
                    "status": "success",
                    "progress": 100,
                    "status_text": "MAXIMUM quality processing completed!",
                    "output_filename": os.path.basename(output_path),
                })
                if is_arabic:
                    status["logs"].append(
                        "🎉 ✅ Arabic RTL subtitles processed with ORIGINAL QUALITY preserved!"
                    )
                else:
                    status["logs"].append(
                        "🎉 ✅ Subtitles burned with ORIGINAL QUALITY preserved!"
                    )
            else:
                raise Exception("Output file was not created or is empty")
        else:
            raise Exception(f"FFmpeg failed with return code {proc.returncode}")

    except Exception as exc:
        error_msg = str(exc)
        logger.exception("Burn-subs job %s failed: %s", unique_id, error_msg)
        status.update({
            "status": "failed",
            "message": error_msg,
            "status_text": f"Failed: {error_msg}",
        })
    finally:
        cleanup_temp_files(temp_dir)
