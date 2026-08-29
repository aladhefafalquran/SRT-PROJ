"""Service: DeepL-powered SRT translation with automatic RTL embedding.

The DeepL API key is mandatory — it is loaded from the Flask config which
reads ``os.environ['DEEPL_API_KEY']``. If the key is missing the caller
(the translate blueprint) must reject the request before this service is
invoked; we still raise defensively here.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import requests

from ..config import Config
from ..extensions import translate_status
from ..utils.srt_text import apply_rtl_formatting

logger = logging.getLogger(__name__)


class DeepLKeyMissing(RuntimeError):
    """Raised when the DeepL API key is not configured."""


def _deepl_request(text: str, target_lang: str, api_key: Optional[str] = None) -> str:
    key = api_key if api_key is not None else Config.DEEPL_API_KEY
    if not key:
        raise DeepLKeyMissing(
            "DEEPL_API_KEY is not configured. Set it in the environment "
            "before using the translate endpoint."
        )
    response = requests.post(
        Config.DEEPL_API_URL,
        data={"text": text, "target_lang": target_lang},
        headers={"Authorization": f"DeepL-Auth-Key {key}"},
        timeout=60,
    )
    if response.status_code != 200:
        raise RuntimeError(f"DeepL API error: {response.text}")
    payload = response.json()
    return payload["translations"][0]["text"]


def enhanced_translate_srt_task(input_path: str, output_path: str, unique_id: str) -> None:
    """Block-by-block translation with RTL embedding. Behaviour matches
    the original implementation."""
    status = translate_status[unique_id]
    try:
        status["message"] = "Reading SRT file..."
        status["progress"] = 5

        with open(input_path, "r", encoding="utf-8") as handle:
            content = handle.read().strip()

        blocks = content.split("\n\n")
        translated_blocks = []
        total_blocks = len([b for b in blocks if b.strip()])
        current_block = 0

        status["message"] = f"Found {total_blocks} subtitle blocks to translate..."
        status["progress"] = 10

        for block in blocks:
            if not block.strip():
                continue

            lines = block.strip().split("\n")
            if len(lines) < 3:
                translated_blocks.append(block)
                current_block += 1
                continue

            sequence_line = lines[0]
            timing_line = lines[1]
            text_lines = lines[2:]
            full_text = " ".join(text_lines)

            if full_text.strip():
                current_block += 1
                status["message"] = (
                    f"Translating subtitle {current_block}/{total_blocks}..."
                )
                try:
                    translated_text = _deepl_request(full_text, "AR")
                    formatted_text = apply_rtl_formatting(translated_text)
                    translated_block = f"{sequence_line}\n{timing_line}\n{formatted_text}"
                    translated_blocks.append(translated_block)
                    logger.info(
                        "Translated block %d: '%s...' -> '%s...'",
                        current_block,
                        full_text[:30],
                        formatted_text[:30],
                    )
                except DeepLKeyMissing:
                    raise
                except Exception as exc:
                    logger.exception("DeepL block %d failed: %s", current_block, exc)
                    translated_blocks.append(block)
            else:
                translated_blocks.append(block)
                current_block += 1

            progress = 10 + int((current_block / max(total_blocks, 1)) * 80)
            status["progress"] = min(progress, 90)

        status["message"] = "Finalizing Arabic RTL formatting..."
        status["progress"] = 95

        final_content = "\n\n".join(translated_blocks)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            handle.write(final_content)

        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            status.update({
                "status": "completed",
                "progress": 100,
                "message": "Translation completed with automatic RTL formatting!",
                "output_filename": os.path.basename(output_path),
            })
        else:
            raise RuntimeError("Output file was not created or is empty")
    except DeepLKeyMissing as exc:
        status["status"] = "error"
        status["message"] = f"Translation failed: {exc}"
    except Exception as exc:
        logger.exception("Translation failed: %s", exc)
        status["status"] = "error"
        status["message"] = f"Translation failed: {exc}"
    finally:
        try:
            if os.path.exists(input_path):
                os.remove(input_path)
        except OSError:
            pass
