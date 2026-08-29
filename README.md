# SRT-Proj

A small Flask web app that bundles four subtitle / video utilities into
one UI plus an in-browser SRT editor.

## Features

1. **Burn subtitles into a video** — `/merge` and `/upload`. Uses FFmpeg
   with a "MAXIMUM quality" x264 preset (veryslow, CRF 10-12, 150%
   bitrate, advanced x264-params, +faststart, audio stream-copy).
   Detects Arabic text in the SRT and applies RTL styling automatically.
2. **Extract SRT from a video** — `/video_to_srt_page` and
   `/video_to_srt`. Runs OpenAI Whisper locally, applies silence-aware
   segmentation, splits long utterances, and writes a clean SRT.
3. **Download online video** — `/download_video_page` and
   `/download_online_video`. yt-dlp wrapper, up to 4K or audio-only MP3
   at quality 0.
4. **Translate an SRT to Arabic** — `/translate_srt_page` and
   `/translate_srt`. Block-by-block DeepL translation with automatic
   U+202B / U+202C RTL embedding per line.
5. **SRT editor** — `/edit_srt_page`, `/parse_srt`, `/save_srt`. Round-
   trips an SRT through JSON, edits in the browser, writes a new file.

## Prerequisites

- **Python 3.10+** (Whisper and the type hints in this codebase assume
  modern Python).
- **FFmpeg** and **ffprobe** on `PATH` (required by every feature).
- **yt-dlp** on `PATH` (used by the online downloader).
- **OpenAI Whisper** Python package (the extractor). The model weights
  are downloaded on first use.

## Install

```bash
# 1. System deps
sudo apt install ffmpeg          # Debian/Ubuntu; macOS: `brew install ffmpeg`
python -m pip install -U yt-dlp

# 2. Python deps (Whisper pulls in torch and friends; expect a few GB)
python -m pip install -r requirements.txt

# 3. Configure secrets
cp .env.example .env             # then edit and set DEEPL_API_KEY
```

## Configuration

All configuration is read from environment variables (see
[`.env.example`](.env.example)). The most important one is
`DEEPL_API_KEY`: the translate endpoint refuses to start without it and
returns HTTP 500 with a descriptive error.

## Running

```bash
python app.py
```

The server binds to `0.0.0.0:5000` by default. Override with
`SRTPROJ_HOST` and `SRTPROJ_PORT`.

## Routes

| Method | Path                              | Blueprint       | Description                          |
| ------ | --------------------------------- | --------------- | ------------------------------------ |
| GET    | `/`                               | main            | Redirect to the extractor            |
| GET    | `/merge`                          | main            | Burn-subtitles landing page          |
| POST   | `/upload`                         | merge           | Accept video + SRT, start burn job   |
| GET    | `/stream/<id>`                    | merge           | SSE progress feed                    |
| GET    | `/status/<id>`                    | merge           | JSON snapshot                        |
| GET    | `/video_to_srt_page`              | transcribe      | Extractor landing page               |
| POST   | `/video_to_srt`                   | transcribe      | Accept video, start Whisper          |
| GET    | `/video_status/<id>`              | transcribe      | SSE progress feed                    |
| GET    | `/video_status_json/<id>`         | transcribe      | JSON polling fallback                |
| GET    | `/download_video_page`            | download        | Downloader landing page              |
| POST   | `/download_online_video`          | download        | Start a yt-dlp job                   |
| GET    | `/download_status/<id>`           | download        | SSE progress feed                    |
| GET    | `/translate_srt_page`             | translate       | Translator landing page              |
| POST   | `/translate_srt`                  | translate       | Accept SRT, start DeepL              |
| GET    | `/translation_status/<id>`        | translate       | SSE progress feed                    |
| GET    | `/edit_srt_page`                  | editor          | SRT editor landing page              |
| POST   | `/parse_srt`                      | editor          | Parse uploaded SRT to JSON           |
| POST   | `/save_srt`                       | editor          | Write edited JSON back to SRT        |
| GET    | `/download/<string:filename>`     | files           | Serve a generated output file        |

## Project layout

```
app.py                 # thin entrypoint: create_app() + app.run()
srtproj/
  __init__.py          # application factory
  config.py            # env-driven Config class
  extensions.py        # shared status dicts + Whisper import
  bootstrap.py         # configure_logging(), check_dependencies()
  utils/
    files.py           # allowed_file(), secure_filename(), secure_join()
    ffmpeg.py          # duration, progress, burn-subs command builder
    srt_text.py        # timestamp, RTL, split, parse, generate
  services/
    merge.py           # burn-subs background job
    transcribe.py      # Whisper + silence-aware segmentation
    downloader.py      # yt-dlp wrapper
    translator.py      # DeepL block translation with RTL embedding
    editor.py          # thin re-export of parse/generate
  routes/
    main.py
    merge.py
    transcribe.py
    download.py
    translate.py
    editor.py
    files.py           # /download/<string:filename> (zero-width char FIXED)
templates/             # base + 7 feature pages
static/                # CSS/JS
uploads/, outputs/     # runtime folders (gitignored)
tests/                 # pytest suite for the pure helpers
requirements.txt       # re-encoded UTF-8, trimmed to what's imported
.env.example
.gitignore
```

## Security note

The previous source tree contained a hardcoded DeepL API key
(`6e05e993-b62b-43c5-aaa1-24b25aa8c3ae:fx`). That key has been rotated
and is no longer shipped; the translate endpoint now requires
`DEEPL_API_KEY` to be supplied through the environment. If you ever find
a committed secret in this repo, treat it as compromised and rotate it
immediately.

## Tests

```bash
python -m pip install pytest
pytest tests/
```

The test suite covers the pure helpers in `srtproj/utils/` and the
`secure_join` path-traversal protection. Background jobs (FFmpeg,
Whisper, yt-dlp, DeepL) are out of scope for the unit tests.
