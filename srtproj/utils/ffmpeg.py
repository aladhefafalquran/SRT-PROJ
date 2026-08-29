"""FFmpeg / ffprobe helpers: duration, progress parsing, video metadata,
and the high-quality burn-subs command builder."""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_TIME_RE = re.compile(r"time=(\d+):(\d+):(\d+\.\d+)")


def get_video_duration(video_path: str) -> Optional[float]:
    """Return the duration of a media file in seconds, or ``None`` on error."""
    try:
        cmd = [
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_format", video_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        return float(data["format"]["duration"])
    except Exception as exc:  # pragma: no cover - depends on ffprobe
        logger.debug("get_video_duration failed for %s: %s", video_path, exc)
        return None


def parse_ffmpeg_progress(line: str, total_duration: Optional[float]) -> Optional[float]:
    """Extract a 0-100 progress percentage from a single FFmpeg stderr line."""
    if "time=" not in line or not total_duration:
        return None
    match = _TIME_RE.search(line)
    if not match:
        return None
    h, m, s = match.groups()
    current = int(h) * 3600 + int(m) * 60 + float(s)
    return min((current / total_duration) * 100, 100)


def get_detailed_video_info(video_path: str) -> Optional[Dict[str, Any]]:
    """Return a dict with the most important stream/format properties."""
    try:
        cmd = [
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_streams", "-show_format",
            video_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)

        video_stream = None
        for stream in data.get("streams", []):
            if stream.get("codec_type") == "video":
                video_stream = stream
                break
        if not video_stream:
            return None

        fps_str = video_stream.get("r_frame_rate", "30/1")
        try:
            fps = eval(fps_str) if "/" in fps_str else float(fps_str)  # noqa: S307
        except Exception:
            fps = 30.0

        bitrate = 0
        if video_stream.get("bit_rate"):
            bitrate = int(video_stream["bit_rate"])
        elif data.get("format", {}).get("bit_rate"):
            # Fallback: estimate video bitrate as 85% of container bitrate.
            bitrate = int(int(data["format"]["bit_rate"]) * 0.85)

        return {
            "width": int(video_stream.get("width", 0)),
            "height": int(video_stream.get("height", 0)),
            "fps": fps,
            "pix_fmt": video_stream.get("pix_fmt", "yuv420p"),
            "bitrate": bitrate,
            "codec": video_stream.get("codec_name", "unknown"),
            "profile": video_stream.get("profile", "unknown"),
            "level": video_stream.get("level", "unknown"),
            "color_space": video_stream.get("color_space", "unknown"),
            "sample_aspect_ratio": video_stream.get("sample_aspect_ratio", "1:1"),
            "display_aspect_ratio": video_stream.get("display_aspect_ratio", "unknown"),
            "duration": float(data.get("format", {}).get("duration", 0)),
        }
    except Exception as exc:  # pragma: no cover
        logger.debug("get_detailed_video_info failed for %s: %s", video_path, exc)
        return None


def _escape_subtitle_path(path: str) -> str:
    if os.name == "nt":  # Windows
        return path.replace("\\", "/").replace(":", "\\:")
    return path.replace(":", "\\:")


def build_burn_subs_cmd(
    video_path: str,
    subtitle_path: str,
    output_path: str,
    video_info: Optional[Dict[str, Any]],
    is_arabic: bool,
) -> list:
    """Return the FFmpeg command list for "MAXIMUM quality" burn-subs.

    The exact flag set is preserved from the original app.py: veryslow
    preset, 150% bitrate or CRF 10-12 fallback, advanced x264-params,
    +faststart, audio stream-copy, and Arabic RTL styling.
    """
    escaped_sub = _escape_subtitle_path(subtitle_path)
    cmd: list = ["ffmpeg", "-y", "-i", video_path]

    if video_info and video_info.get("width", 0) > 0 and video_info.get("height", 0) > 0:
        if is_arabic:
            subtitle_filter = (
                f"subtitles='{escaped_sub}'"
                f":force_style='Fontsize=20,PrimaryColour=&Hffffff,OutlineColour=&H000000,"
                f"BackColour=&H80000000,Outline=3,Shadow=1,Alignment=2,MarginV=40,"
                f"Fontname=Arial,Bold=1'"
            )
        else:
            subtitle_filter = (
                f"subtitles='{escaped_sub}'"
                f":force_style='Fontsize=18,PrimaryColour=&Hffffff,OutlineColour=&H000000,"
                f"BackColour=&H80000000,Outline=2,Shadow=1'"
            )

        cmd.extend(["-vf", subtitle_filter])
        cmd.extend(["-c:v", "libx264"])
        cmd.extend(["-preset", "veryslow"])

        if video_info.get("bitrate", 0) > 0:
            target_bitrate = max(int(video_info["bitrate"]), 2_000_000)
            enhanced_bitrate = int(target_bitrate * 1.5)
            cmd.extend(["-b:v", str(enhanced_bitrate)])
            cmd.extend(["-maxrate", str(int(enhanced_bitrate * 1.3))])
            cmd.extend(["-bufsize", str(int(enhanced_bitrate * 2))])
        else:
            # Near-lossless fallback when original bitrate is unknown.
            cmd.extend(["-crf", "12"])

        cmd.extend(["-s", f"{video_info['width']}x{video_info['height']}"])
        cmd.extend(["-pix_fmt", video_info["pix_fmt"]])
        cmd.extend(["-r", str(video_info["fps"])])
        if video_info.get("sample_aspect_ratio") != "1:1":
            cmd.extend(["-aspect", video_info["display_aspect_ratio"]])
        cmd.extend([
            "-x264-params",
            "ref=16:bframes=16:b-adapt=2:direct=auto:me=umh:subme=11:trellis=2:rc-lookahead=60:keyint=300:min-keyint=30",
        ])
    else:
        # Fallback path when ffprobe could not parse the input.
        if is_arabic:
            subtitle_filter = (
                f"subtitles='{escaped_sub}'"
                f":force_style='Fontsize=20,PrimaryColour=&Hffffff,OutlineColour=&H000000,"
                f"BackColour=&H80000000,Outline=3,Shadow=1,Alignment=2,MarginV=40,"
                f"Fontname=Arial,Bold=1'"
            )
        else:
            subtitle_filter = f"subtitles='{escaped_sub}'"
        cmd.extend(["-vf", subtitle_filter])
        cmd.extend(["-c:v", "libx264", "-preset", "veryslow", "-crf", "10"])

    # Audio copy + output stream flags
    cmd.extend(["-c:a", "copy"])
    cmd.extend(["-movflags", "+faststart"])
    cmd.extend(["-avoid_negative_ts", "make_zero"])
    cmd.extend(["-fflags", "+genpts+igndts"])
    cmd.extend(["-max_muxing_queue_size", "2048"])
    cmd.extend(["-muxdelay", "0"])
    cmd.extend(["-muxpreload", "0"])
    cmd.extend(["-progress", "pipe:1", "-nostats"])
    cmd.append(output_path)
    return cmd
