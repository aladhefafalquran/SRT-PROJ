"""Service: Whisper-backed video-to-SRT extraction with silence-aware
segmentation. Behaviour matches the original ``enhanced_background_task``
inside ``app.py``."""
from __future__ import annotations

import logging
import os
import re
import subprocess

from ..extensions import WHISPER_AVAILABLE, whisper, video_processing_status
from ..utils.srt_text import format_timestamp, split_long_text

logger = logging.getLogger(__name__)


def _detect_silence(video_path: str) -> list:
    """Return a list of (start, end) silence tuples using FFmpeg's
    silencedetect filter. Empty list on failure."""
    cmd = [
        "ffmpeg", "-i", video_path,
        "-af", "silencedetect=noise=-30dB:d=1.0",
        "-f", "null", "-",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
    except Exception as exc:
        logger.debug("silencedetect failed: %s", exc)
        return []

    output = result.stderr or ""
    periods: list = []
    silence_start = None
    for line in output.split("\n"):
        if "silence_start:" in line:
            match = re.search(r"silence_start: ([\d.]+)", line)
            if match:
                silence_start = float(match.group(1))
        elif "silence_end:" in line and silence_start is not None:
            match = re.search(r"silence_end: ([\d.]+)", line)
            if match:
                periods.append((silence_start, float(match.group(1))))
                silence_start = None
    return periods


def _find_speech_start(segments: list) -> float:
    if not segments:
        return 0.0
    for segment in segments:
        text = (segment.get("text") or "").strip()
        if len(text) > 3 and any(c.isalpha() for c in text):
            return float(segment.get("start", 0) or 0)
    return float(segments[0].get("start", 0) or 0)


def _filter_silence_segments(segments: list, silence_periods: list, buffer: float = 0.2) -> list:
    filtered: list = []
    for segment in segments:
        start = float(segment.get("start", 0) or 0)
        end = float(segment.get("end", 0) or 0)
        text = (segment.get("text") or "").strip()
        if start < 0:
            start = 0
        if end <= start:
            continue
        skip = False
        for silence_start, silence_end in silence_periods:
            buffered_start = silence_start + buffer
            buffered_end = silence_end - buffer
            overlap_start = max(start, buffered_start)
            overlap_end = min(end, buffered_end)
            if overlap_end > overlap_start:
                overlap_duration = overlap_end - overlap_start
                segment_duration = end - start
                if segment_duration > 0 and overlap_duration / segment_duration > 0.7:
                    skip = True
                    break
        if not skip and text:
            filtered.append({"start": start, "end": end, "text": text})
    return filtered


def _build_enhanced_segments(segments: list, silence_periods: list) -> list:
    if not segments:
        return []

    speech_start = _find_speech_start(segments)
    enhanced: list = []
    for segment in segments:
        start = float(segment.get("start", 0) or 0)
        end = float(segment.get("end", 0) or 0)
        text = (segment.get("text") or "").strip()
        if start < speech_start - 0.5:
            continue
        if len(text) < 2 or (end - start) < 0.3:
            continue

        candidates = [{"start": start, "end": end, "text": text}]
        candidates = _filter_silence_segments(candidates, silence_periods)
        if not candidates:
            continue

        text = re.sub(r"\s+", " ", text).strip()
        chunks = split_long_text(text)
        if len(chunks) == 1:
            enhanced.append({"start": start, "end": end, "text": text})
        else:
            chunk_duration = (end - start) / len(chunks)
            for i, chunk in enumerate(chunks):
                enhanced.append({
                    "start": start + i * chunk_duration,
                    "end": start + (i + 1) * chunk_duration,
                    "text": chunk,
                })
    return enhanced


def _write_enhanced_srt(enhanced_segments: list, srt_path: str) -> None:
    os.makedirs(os.path.dirname(srt_path), exist_ok=True)
    with open(srt_path, "w", encoding="utf-8") as handle:
        if not enhanced_segments:
            handle.write(
                "1\n00:00:00,000 --> 00:00:05,000\nNo clear speech detected\n\n"
            )
            return
        for i, segment in enumerate(enhanced_segments):
            start_time = format_timestamp(segment["start"])
            end_time = format_timestamp(segment["end"])
            text = segment["text"]
            handle.write(f"{i + 1}\n{start_time} --> {end_time}\n{text}\n\n")


def run_extraction(
    video_path: str,
    srt_path: str,
    unique_id: str,
    model_name: str,
    translate: bool,
    output_folder: str,
    temp_dir: str,
) -> None:
    """Background entry point. Mirrors the original background task."""
    from ..utils.files import cleanup_temp_files  # local import to avoid cycle

    status = video_processing_status[unique_id]
    try:
        if not WHISPER_AVAILABLE:
            raise RuntimeError("Whisper is not installed")

        status["message"] = f"Loading {model_name} model..."
        status["progress"] = 10
        logger.info("Loading Whisper model: %s", model_name)
        model = whisper.load_model(model_name)

        status["message"] = "Analyzing audio with enhanced processing..."
        status["progress"] = 20

        silence_periods = _detect_silence(video_path)
        logger.info("Detected %d silence periods", len(silence_periods))

        status["message"] = "Transcribing with smart segmentation..."
        status["progress"] = 30

        result = model.transcribe(
            video_path,
            task="translate" if translate else "transcribe",
            word_timestamps=True,
            condition_on_previous_text=False,
            compression_ratio_threshold=2.4,
            logprob_threshold=-1.0,
            no_speech_threshold=0.6,
            temperature=0.0,
        )

        status["message"] = "Processing segments with smart timing..."
        status["progress"] = 70

        enhanced_segments = _build_enhanced_segments(
            result.get("segments", []), silence_periods
        )

        status["message"] = "Writing enhanced SRT file..."
        status["progress"] = 90

        os.makedirs(output_folder, exist_ok=True)
        _write_enhanced_srt(enhanced_segments, srt_path)

        if os.path.exists(srt_path):
            status["status"] = "completed"
            status["progress"] = 100
            status["message"] = "Enhanced SRT completed with smart timing!"
            logger.info("Enhanced SRT written: %s", srt_path)
        else:
            raise RuntimeError("Enhanced SRT file was not created")
    except Exception as exc:
        logger.exception("Transcription failed: %s", exc)
        status["status"] = "error"
        status["message"] = f"Enhanced processing failed: {exc}"
    finally:
        cleanup_temp_files(temp_dir)
