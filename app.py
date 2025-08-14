import os
import subprocess
import threading
import time
import re
import json
import requests
from flask import Flask, request, render_template, send_file, jsonify, Response, redirect, url_for
from werkzeug.utils import secure_filename
import uuid
import shutil
from urllib.parse import urlparse

# === Dependency Checks ===
try:
    import whisper
    WHISPER_AVAILABLE = True
    print("✅ DEBUG: Whisper imported successfully")
except ImportError:
    WHISPER_AVAILABLE = False
    whisper = None
    print("❌ DEBUG: Whisper not available")

# === Windows Path Fix ===
UPLOAD_FOLDER = os.path.abspath('uploads')
OUTPUT_FOLDER = os.path.abspath('outputs')
ALLOWED_EXTENSIONS_VIDEO = {'mp4', 'mov', 'avi', 'mkv'}
ALLOWED_EXTENSIONS_SUB = {'srt', 'vtt', 'ass'}

# Ensure folders exist with proper permissions
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

print(f"📁 DEBUG: Upload folder absolute path: {UPLOAD_FOLDER}")
print(f"📁 DEBUG: Output folder absolute path: {OUTPUT_FOLDER}")

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 2048 * 1024 * 1024  # 2GB

# === Status Storage ===
processing_status = {}           # For merging video + subs
video_processing_status = {}     # For extracting SRT
translate_status = {}            # For SRT translation
download_status = {}             # For online video downloads

# ==============================
# 🔑 DEEPL API KEY - MOVE TO ENVIRONMENT VARIABLE
# ==============================
DEEPL_API_KEY = os.getenv("DEEPL_API_KEY", "6e05e993-b62b-43c5-aaa1-24b25aa8c3ae:fx")
DEEPL_API_URL = "https://api-free.deepl.com/v2/translate"
# ==============================

# === ENHANCED SRT GENERATION FUNCTIONS ===
def get_audio_silence_detection(video_path, silence_threshold=-30, min_silence_duration=1.0):
    """
    Use FFmpeg to detect silence periods in the audio
    Returns list of silence periods as (start_time, end_time) tuples
    """
    try:
        cmd = [
            'ffmpeg', '-i', video_path,
            '-af', f'silencedetect=noise={silence_threshold}dB:d={min_silence_duration}',
            '-f', 'null', '-'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, stderr=subprocess.STDOUT)
        output = result.stderr
        
        silence_periods = []
        silence_start = None
        
        for line in output.split('\n'):
            if 'silence_start:' in line:
                match = re.search(r'silence_start: ([\d.]+)', line)
                if match:
                    silence_start = float(match.group(1))
            elif 'silence_end:' in line and silence_start is not None:
                match = re.search(r'silence_end: ([\d.]+)', line)
                if match:
                    silence_end = float(match.group(1))
                    silence_periods.append((silence_start, silence_end))
                    silence_start = None
        
        return silence_periods
    except Exception as e:
        print(f"Warning: Could not detect silence periods: {e}")
        return []

def split_long_text(text, max_chars=80, max_words=15):
    """
    Split long text into smaller chunks at natural break points
    """
    if len(text) <= max_chars and len(text.split()) <= max_words:
        return [text]
    
    # Split at sentence boundaries first
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    if len(sentences) > 1:
        return [s.strip() for s in sentences if s.strip()]
    
    # Split at clause boundaries (commas, semicolons)
    clauses = re.split(r'(?<=[,;])\s+', text.strip())
    if len(clauses) > 1:
        chunks = []
        current_chunk = ""
        
        for clause in clauses:
            if len(current_chunk + " " + clause) <= max_chars:
                if current_chunk:
                    current_chunk += " " + clause
                else:
                    current_chunk = clause
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = clause
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return [c for c in chunks if c.strip()]
    
    # Split at word boundaries if still too long
    words = text.split()
    if len(words) > max_words:
        chunks = []
        for i in range(0, len(words), max_words):
            chunk = " ".join(words[i:i + max_words])
            chunks.append(chunk)
        return chunks
    
    return [text]

def find_speech_start(segments, silence_threshold=0.5):
    """
    Find the actual start of speech, ignoring early silence or noise
    """
    if not segments:
        return 0
    
    for segment in segments:
        # Look for segments with substantial text (not just noise)
        text = segment.get('text', '').strip()
        if len(text) > 3 and any(c.isalpha() for c in text):
            return segment.get('start', 0)
    
    return segments[0].get('start', 0) if segments else 0

def remove_segments_in_silence(segments, silence_periods, buffer=0.2):
    """
    Remove or adjust segments that fall within detected silence periods
    """
    filtered_segments = []
    
    for segment in segments:
        start = segment.get('start', 0)
        end = segment.get('end', 0)
        
        # Check if segment overlaps with any silence period
        overlaps_silence = False
        for silence_start, silence_end in silence_periods:
            # Add buffer to silence periods
            buffered_silence_start = silence_start + buffer
            buffered_silence_end = silence_end - buffer
            
            # Check if segment is mostly within silence
            overlap_start = max(start, buffered_silence_start)
            overlap_end = min(end, buffered_silence_end)
            
            if overlap_end > overlap_start:
                overlap_duration = overlap_end - overlap_start
                segment_duration = end - start
                
                # If more than 70% of segment is in silence, skip it
                if overlap_duration / segment_duration > 0.7:
                    overlaps_silence = True
                    break
        
        if not overlaps_silence:
            filtered_segments.append(segment)
    
    return filtered_segments

def check_dependencies():
    """Check if required dependencies are available."""
    issues = []
    
    # Check FFmpeg
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        print("✅ DEBUG: FFmpeg found")
    except (subprocess.CalledProcessError, FileNotFoundError):
        issues.append("FFmpeg not found. Please install FFmpeg.")
        print("❌ DEBUG: FFmpeg not found")
    
    # Check ffprobe
    try:
        subprocess.run(['ffprobe', '-version'], capture_output=True, check=True)
        print("✅ DEBUG: ffprobe found")
    except (subprocess.CalledProcessError, FileNotFoundError):
        issues.append("ffprobe not found. Please install FFmpeg.")
        print("❌ DEBUG: ffprobe not found")
    
    # Check yt-dlp
    try:
        subprocess.run(['yt-dlp', '--version'], capture_output=True, check=True)
        print("✅ DEBUG: yt-dlp found")
    except (subprocess.CalledProcessError, FileNotFoundError):
        issues.append("yt-dlp not found. Please install yt-dlp.")
        print("❌ DEBUG: yt-dlp not found")
    
    # Check Whisper
    if not WHISPER_AVAILABLE:
        issues.append("Whisper not found. Install with: pip install openai-whisper")
        print("❌ DEBUG: Whisper not available")
    else:
        print("✅ DEBUG: Whisper available")
    
    return issues

# === Helper Functions ===
def allowed_file(filename, allowed_extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

def cleanup_temp_files(temp_dir):
    """Helper function to clean up temporary directories."""
    if os.path.exists(temp_dir):
        try:
            shutil.rmtree(temp_dir)
            print(f"✅ DEBUG: Cleaned up temp dir: {temp_dir}")
        except Exception as e:
            print(f"⚠️ DEBUG: Could not clean up {temp_dir}: {e}")

def get_video_duration(video_path):
    try:
        cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', video_path]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        return float(data['format']['duration'])
    except Exception:
        return None

def parse_ffmpeg_progress(line, total_duration):
    if 'time=' in line:
        time_match = re.search(r'time=(\d+):(\d+):(\d+\.\d+)', line)
        if time_match and total_duration:
            h, m, s = time_match.groups()
            current = int(h) * 3600 + int(m) * 60 + float(s)
            return min((current / total_duration) * 100, 100)
    return None

def format_timestamp(seconds):
    """Convert float seconds to SRT timestamp format: 00:00:01,000"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    milliseconds = int((secs - int(secs)) * 1000)
    seconds = int(secs)
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"

def has_arabic_text(srt_path):
    """Check if SRT file contains Arabic text"""
    try:
        with open(srt_path, 'r', encoding='utf-8') as f:
            content = f.read()
        arabic_pattern = re.compile(r'[\u0600-\u06FF]')
        return bool(arabic_pattern.search(content))
    except:
        return False

def create_rtl_srt(input_srt, output_srt):
    """Create RTL-compatible SRT file"""
    try:
        with open(input_srt, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        with open(output_srt, 'w', encoding='utf-8') as f:
            for line in lines:
                if line.strip() and not line.strip().isdigit() and '-->' not in line:
                    # Add RTL override for Arabic text lines
                    arabic_pattern = re.compile(r'[\u0600-\u06FF]')
                    if arabic_pattern.search(line):
                        line = '\u202E' + line.strip() + '\u202C\n'
                f.write(line)
        
        return True
    except Exception as e:
        print(f"Error creating RTL SRT: {e}")
        return False

# === ENHANCED RTL FUNCTIONS ===
def apply_rtl_formatting(text):
    """Apply RTL formatting to Arabic text automatically"""
    # Arabic character pattern - comprehensive Unicode ranges for Arabic
    arabic_pattern = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]')
    
    # Check if text contains Arabic characters
    if arabic_pattern.search(text):
        # Add RLE (Right-to-Left Embedding) at start and PDF (Pop Directional Formatting) at end
        return '\u202B' + text.strip() + '\u202C'
    
    return text.strip()

def enhanced_translate_srt_task(input_path, output_path, unique_id):
    """Enhanced translation function with automatic RTL formatting"""
    try:
        translate_status[unique_id]['message'] = 'Reading SRT file...'
        translate_status[unique_id]['progress'] = 5
        
        with open(input_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()

        # Parse SRT content into blocks
        blocks = content.split('\n\n')
        translated_blocks = []
        total_blocks = len([b for b in blocks if b.strip()])  # Count non-empty blocks
        current_block = 0
        
        translate_status[unique_id]['message'] = f'Found {total_blocks} subtitle blocks to translate...'
        translate_status[unique_id]['progress'] = 10
        
        for block in blocks:
            if not block.strip():
                continue
                
            lines = block.strip().split('\n')
            if len(lines) < 3:
                # Keep malformed blocks as-is
                translated_blocks.append(block)
                current_block += 1
                continue
            
            # Extract sequence number and timing (keep as-is)
            sequence_line = lines[0]
            timing_line = lines[1]
            
            # Extract text lines for translation
            text_lines = lines[2:]
            full_text = ' '.join(text_lines)
            
            if full_text.strip():
                current_block += 1
                translate_status[unique_id]['message'] = f'Translating subtitle {current_block}/{total_blocks}...'
                
                try:
                    # Call DeepL API
                    response = requests.post(
                        DEEPL_API_URL,
                        data={
                            "text": full_text,
                            "target_lang": "AR"
                        },
                        headers={"Authorization": f"DeepL-Auth-Key {DEEPL_API_KEY}"}
                    )
                    
                    if response.status_code != 200:
                        raise Exception(f"DeepL API error: {response.text}")

                    translated_text = response.json()['translations'][0]['text']
                    
                    # Apply RTL formatting automatically to Arabic text
                    formatted_text = apply_rtl_formatting(translated_text)
                    
                    # Rebuild the block
                    translated_block = f"{sequence_line}\n{timing_line}\n{formatted_text}"
                    translated_blocks.append(translated_block)
                    
                    print(f"✅ Translated block {current_block}: '{full_text[:30]}...' -> '{formatted_text[:30]}...'")
                    
                except Exception as e:
                    print(f"❌ Error translating block {current_block}: {e}")
                    # Keep original text if translation fails
                    translated_blocks.append(block)
            else:
                # Keep blocks without text as-is
                translated_blocks.append(block)
                current_block += 1
            
            # Update progress (10% start + 80% for translation + 10% finalization)
            progress = 10 + int((current_block / total_blocks) * 80)
            translate_status[unique_id]['progress'] = min(progress, 90)

        # Final processing step
        translate_status[unique_id]['message'] = 'Finalizing Arabic RTL formatting...'
        translate_status[unique_id]['progress'] = 95
        
        # Join all blocks and write to output
        final_content = '\n\n'.join(translated_blocks)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(final_content)
        
        # Final verification
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            translate_status[unique_id]['status'] = 'completed'
            translate_status[unique_id]['progress'] = 100
            translate_status[unique_id]['message'] = 'Translation completed with automatic RTL formatting!'
            translate_status[unique_id]['output_filename'] = os.path.basename(output_path)
            
            print(f"✅ Translation completed with auto-RTL: {output_path}")
        else:
            raise Exception("Output file was not created or is empty")

    except Exception as e:
        error_msg = str(e)
        print(f"❌ Translation error: {error_msg}")
        translate_status[unique_id]['status'] = 'error'
        translate_status[unique_id]['message'] = f'Translation failed: {error_msg}'
    finally:
        # Clean up input file
        if os.path.exists(input_path):
            try:
                os.remove(input_path)
                print(f"🧹 Cleaned up input file: {input_path}")
            except:
                pass

# === SRT Editor Helper Functions ===
def parse_srt_content(content):
    """Parse SRT content into structured data"""
    subtitles = []
    blocks = content.strip().split('\n\n')
    
    for i, block in enumerate(blocks):
        lines = block.strip().split('\n')
        if len(lines) >= 3:
            try:
                # Parse sequence number
                sequence = int(lines[0].strip())
                
                # Parse timing
                timing_line = lines[1].strip()
                start_time, end_time = timing_line.split(' --> ')
                
                # Parse text (can be multiple lines)
                text_lines = lines[2:]
                text = '\n'.join(text_lines)
                
                subtitles.append({
                    'id': i + 1,
                    'sequence': sequence,
                    'start_time': start_time.strip(),
                    'end_time': end_time.strip(),
                    'text': text.strip()
                })
                
            except (ValueError, IndexError) as e:
                # Skip malformed entries
                continue
    
    return subtitles

def generate_srt_content(subtitles):
    """Generate SRT content from structured data"""
    srt_lines = []
    
    for i, subtitle in enumerate(subtitles):
        # Sequence number
        srt_lines.append(str(i + 1))
        
        # Timing
        timing = f"{subtitle['start_time']} --> {subtitle['end_time']}"
        srt_lines.append(timing)
        
        # Text content
        srt_lines.append(subtitle['text'])
        
        # Empty line between entries (except for the last one)
        if i < len(subtitles) - 1:
            srt_lines.append('')
    
    return '\n'.join(srt_lines)

# === Route: Default Page (Extract SRT) ===
@app.route('/')
def index():
    """Redirect to Extract SRT as default page"""
    return redirect(url_for('video_to_srt_page'))

# === Route: Merge Page (moved to /merge) ===
@app.route('/merge')
def merge_page():
    """Original merge page moved to /merge"""
    message = request.args.get('message')
    
    # Check dependencies and show warnings
    issues = check_dependencies()
    if issues:
        message = "⚠️ Missing dependencies: " + "; ".join(issues)
    
    return render_template('index.html', message=message)

# === Route: Upload & Process Merge ===
@app.route('/upload', methods=['POST'])
def upload_files():
    # Check FFmpeg availability
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return render_template('index.html', message='FFmpeg is not installed or not found in PATH.'), 500
    
    if 'videoFile' not in request.files or 'subtitleFile' not in request.files:
        return render_template('index.html', message='Please select both files.'), 400

    video = request.files['videoFile']
    sub = request.files['subtitleFile']
    if video.filename == '' or sub.filename == '':
        return render_template('index.html', message='Please select both files.'), 400
    if not allowed_file(video.filename, ALLOWED_EXTENSIONS_VIDEO):
        return render_template('index.html', message='Invalid video file.'), 400
    if not allowed_file(sub.filename, ALLOWED_EXTENSIONS_SUB):
        return render_template('index.html', message='Invalid subtitle file.'), 400

    unique_id = str(uuid.uuid4())
    temp_dir = os.path.join(app.config['UPLOAD_FOLDER'], unique_id)
    os.makedirs(temp_dir, exist_ok=True)

    vpath = os.path.join(temp_dir, secure_filename(video.filename))
    spath = os.path.join(temp_dir, secure_filename(sub.filename))
    opath = os.path.join(app.config['OUTPUT_FOLDER'], f"merged_{unique_id}.mp4")

    video.save(vpath)
    sub.save(spath)

    processing_status[unique_id] = {
        'status': 'running',
        'progress': 0,
        'status_text': 'Starting...',
        'logs': [],
        'output_filename': f"merged_{unique_id}.mp4"
    }

    thread = threading.Thread(
        target=process_video_job,
        args=(vpath, spath, opath, unique_id, temp_dir)
    )
    thread.daemon = True
    thread.start()

    return render_template('processing.html', unique_id=unique_id, output_filename=f"merged_{unique_id}.mp4")

def process_video_job(video_path, subtitle_path, output_path, unique_id, temp_dir):
    try:
        duration = get_video_duration(video_path)
        
        # Get DETAILED original video info for MAXIMUM quality preservation
        def get_detailed_video_info(video_path):
            try:
                cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_streams', '-show_format', video_path]
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                data = json.loads(result.stdout)
                
                video_stream = None
                for stream in data['streams']:
                    if stream['codec_type'] == 'video':
                        video_stream = stream
                        break
                
                if video_stream:
                    # Get ALL video parameters for perfect replication
                    fps_str = video_stream.get('r_frame_rate', '30/1')
                    try:
                        fps = eval(fps_str) if '/' in fps_str else float(fps_str)
                    except:
                        fps = 30.0
                    
                    # Calculate actual bitrate more accurately
                    bitrate = 0
                    if video_stream.get('bit_rate'):
                        bitrate = int(video_stream['bit_rate'])
                    elif data.get('format', {}).get('bit_rate'):
                        # Use format bitrate if stream bitrate unavailable
                        format_bitrate = int(data['format']['bit_rate'])
                        # Estimate video bitrate (usually 80-90% of total for video files)
                        bitrate = int(format_bitrate * 0.85)
                    
                    return {
                        'width': int(video_stream.get('width', 0)),
                        'height': int(video_stream.get('height', 0)),
                        'fps': fps,
                        'pix_fmt': video_stream.get('pix_fmt', 'yuv420p'),
                        'bitrate': bitrate,
                        'codec': video_stream.get('codec_name', 'unknown'),
                        'profile': video_stream.get('profile', 'unknown'),
                        'level': video_stream.get('level', 'unknown'),
                        'color_space': video_stream.get('color_space', 'unknown'),
                        'sample_aspect_ratio': video_stream.get('sample_aspect_ratio', '1:1'),
                        'display_aspect_ratio': video_stream.get('display_aspect_ratio', 'unknown'),
                        'duration': float(data['format'].get('duration', 0))
                    }
                return None
            except Exception as e:
                print(f"Error getting video info: {e}")
                return None
        
        video_info = get_detailed_video_info(video_path)
        processing_status[unique_id]['status_text'] = 'Analyzing video for MAXIMUM quality preservation...'
        
        if video_info:
            processing_status[unique_id]['logs'].append(f'🎬 Original resolution: {video_info["width"]}x{video_info["height"]}')
            processing_status[unique_id]['logs'].append(f'🎬 Original codec: {video_info["codec"]}')
            processing_status[unique_id]['logs'].append(f'🎬 Original bitrate: {video_info["bitrate"]} bps')
            processing_status[unique_id]['logs'].append(f'🎬 Original FPS: {video_info["fps"]}')
            processing_status[unique_id]['logs'].append(f'🎬 Original pixel format: {video_info["pix_fmt"]}')
        
        # Check if subtitles contain Arabic text
        is_arabic = has_arabic_text(subtitle_path)
        
        # Create RTL-compatible subtitle file if Arabic is detected
        if is_arabic:
            processing_status[unique_id]['status_text'] = 'Processing Arabic RTL subtitles...'
            rtl_subtitle_path = os.path.join(temp_dir, 'rtl_subtitles.srt')
            if create_rtl_srt(subtitle_path, rtl_subtitle_path):
                subtitle_path = rtl_subtitle_path
                processing_status[unique_id]['logs'].append('✅ Created RTL-compatible subtitle file for Arabic text')
        
        # Escape subtitle path for FFmpeg
        if os.name == 'nt':  # Windows
            escaped_sub = subtitle_path.replace('\\', '/').replace(':', '\\:')
        else:  # Unix-like
            escaped_sub = subtitle_path.replace(':', '\\:')
        
        # Build MAXIMUM QUALITY PRESERVATION FFmpeg command
        cmd = ['ffmpeg', '-y', '-i', video_path]
        
        if video_info and video_info['width'] > 0 and video_info['height'] > 0:
            
            # === SUBTITLE FILTER WITH EXACT RESOLUTION PRESERVATION ===
            if is_arabic:
                # Arabic RTL styling - larger font for better readability
                subtitle_filter = (f"subtitles='{escaped_sub}'"
                                 f":force_style='Fontsize=20,PrimaryColour=&Hffffff,OutlineColour=&H000000,"
                                 f"BackColour=&H80000000,Outline=3,Shadow=1,Alignment=2,MarginV=40,"
                                 f"Fontname=Arial,Bold=1'")
                processing_status[unique_id]['logs'].append('✅ Applied Arabic RTL styling')
            else:
                # Regular subtitle styling
                subtitle_filter = (f"subtitles='{escaped_sub}'"
                                 f":force_style='Fontsize=18,PrimaryColour=&Hffffff,OutlineColour=&H000000,"
                                 f"BackColour=&H80000000,Outline=2,Shadow=1'")
            
            # Apply subtitle filter WITHOUT any scaling (preserve exact resolution)
            cmd.extend(['-vf', subtitle_filter])
            
            # === VIDEO CODEC SETTINGS FOR MAXIMUM QUALITY ===
            cmd.extend(['-c:v', 'libx264'])
            
            # Use VERYSLOW preset for absolute maximum quality
            cmd.extend(['-preset', 'veryslow'])
            processing_status[unique_id]['logs'].append('🎯 Using VERYSLOW preset for maximum quality')
            
            # === BITRATE MANAGEMENT FOR ORIGINAL QUALITY ===
            if video_info['bitrate'] > 0:
                # Use HIGHER bitrate than original to ensure no quality loss
                target_bitrate = max(video_info['bitrate'], 2000000)  # Minimum 2Mbps
                enhanced_bitrate = int(target_bitrate * 1.5)  # 50% higher than original
                
                cmd.extend(['-b:v', str(enhanced_bitrate)])
                cmd.extend(['-maxrate', str(int(enhanced_bitrate * 1.3))])
                cmd.extend(['-bufsize', str(int(enhanced_bitrate * 2))])
                processing_status[unique_id]['logs'].append(f'💎 Enhanced bitrate: {enhanced_bitrate} bps (150% of original)')
            else:
                # Use EXTREMELY HIGH quality CRF when bitrate unknown
                cmd.extend(['-crf', '12'])  # Near-lossless quality
                processing_status[unique_id]['logs'].append('💎 Using CRF 12 for near-lossless quality')
            
            # === PRESERVE ALL ORIGINAL VIDEO PARAMETERS ===
            
            # Force EXACT output resolution (no scaling whatsoever)
            cmd.extend(['-s', f"{video_info['width']}x{video_info['height']}"])
            
            # Preserve pixel format exactly
            cmd.extend(['-pix_fmt', video_info['pix_fmt']])
            
            # Preserve frame rate exactly
            cmd.extend(['-r', str(video_info['fps'])])
            
            # Preserve aspect ratio
            if video_info['sample_aspect_ratio'] != '1:1':
                cmd.extend(['-aspect', video_info['display_aspect_ratio']])
            
            # Advanced x264 options for maximum quality
            cmd.extend([
                '-x264-params', 
                'ref=16:bframes=16:b-adapt=2:direct=auto:me=umh:subme=11:trellis=2:rc-lookahead=60:keyint=300:min-keyint=30'
            ])
            processing_status[unique_id]['logs'].append('🔧 Applied advanced x264 parameters for maximum quality')
            
        else:
            # Fallback when video info unavailable
            processing_status[unique_id]['status_text'] = 'Using ultra-high quality fallback settings...'
            if is_arabic:
                subtitle_filter = (f"subtitles='{escaped_sub}'"
                                 f":force_style='Fontsize=20,PrimaryColour=&Hffffff,OutlineColour=&H000000,"
                                 f"BackColour=&H80000000,Outline=3,Shadow=1,Alignment=2,MarginV=40,"
                                 f"Fontname=Arial,Bold=1'")
            else:
                subtitle_filter = f"subtitles='{escaped_sub}'"
            
            cmd.extend(['-vf', subtitle_filter])
            cmd.extend(['-c:v', 'libx264', '-preset', 'veryslow', '-crf', '10'])  # Ultra high quality
            processing_status[unique_id]['logs'].append('🚀 Fallback: Using CRF 10 ultra-high quality')
        
        # === AUDIO SETTINGS (PERFECT COPY) ===
        cmd.extend(['-c:a', 'copy'])  # Perfect audio copy - no re-encoding
        
        # === ADVANCED FFMPEG OPTIONS FOR QUALITY ===
        cmd.extend(['-movflags', '+faststart'])  # Web optimization
        cmd.extend(['-avoid_negative_ts', 'make_zero'])  # Fix timestamp issues
        cmd.extend(['-fflags', '+genpts+igndts'])  # Better timestamp handling
        cmd.extend(['-max_muxing_queue_size', '2048'])  # Handle large files
        cmd.extend(['-muxdelay', '0'])  # No muxing delay
        cmd.extend(['-muxpreload', '0'])  # No preload delay
        
        # Progress reporting
        cmd.extend(['-progress', 'pipe:1', '-nostats'])
        
        cmd.append(output_path)
        
        # Log the encoding strategy
        if is_arabic:
            processing_status[unique_id]['status_text'] = 'Starting MAXIMUM quality encoding with Arabic RTL...'
            processing_status[unique_id]['logs'].append('🎬 Arabic RTL + Maximum Quality Mode Activated')
        else:
            processing_status[unique_id]['status_text'] = 'Starting MAXIMUM quality encoding...'
            processing_status[unique_id]['logs'].append('🎬 Maximum Quality Mode Activated')
        
        processing_status[unique_id]['logs'].append(f'🔧 FFmpeg command preview: {" ".join(cmd[:15])}...')
        
        # Start FFmpeg process
        proc = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT, 
            text=True, 
            bufsize=1,
            universal_newlines=True
        )
        
        # Process output with enhanced progress tracking
        for line in iter(proc.stdout.readline, ''):
            if line.strip():
                processing_status[unique_id]['logs'].append(line.strip())
                
                if duration:
                    progress = parse_ffmpeg_progress(line, duration)
                    if progress is not None:
                        processing_status[unique_id]['progress'] = progress
                        processing_status[unique_id]['status_text'] = f'Maximum quality encoding... {progress:.1f}%'
        
        proc.wait()
        
        if proc.returncode == 0:
            # Verify output quality matches input
            output_info = get_detailed_video_info(output_path)
            if output_info and video_info:
                processing_status[unique_id]['logs'].append(f'✅ Output resolution: {output_info["width"]}x{output_info["height"]}')
                processing_status[unique_id]['logs'].append(f'✅ Input resolution: {video_info["width"]}x{video_info["height"]}')
                
                # Verify quality preservation
                if (output_info["width"] == video_info["width"] and 
                    output_info["height"] == video_info["height"]):
                    processing_status[unique_id]['logs'].append('🎯 ✅ PERFECT RESOLUTION MATCH!')
                else:
                    processing_status[unique_id]['logs'].append('⚠️ Resolution mismatch detected')
                
                # Check file sizes
                input_size = os.path.getsize(video_path)
                output_size = os.path.getsize(output_path)
                size_ratio = output_size / input_size
                processing_status[unique_id]['logs'].append(f'📊 Size ratio: {size_ratio:.2f}x (output/input)')
                
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                processing_status[unique_id].update({
                    'status': 'success',
                    'progress': 100,
                    'status_text': 'MAXIMUM quality processing completed!',
                    'output_filename': os.path.basename(output_path)
                })
                if is_arabic:
                    processing_status[unique_id]['logs'].append('🎉 ✅ Arabic RTL subtitles processed with ORIGINAL QUALITY preserved!')
                else:
                    processing_status[unique_id]['logs'].append('🎉 ✅ Subtitles burned with ORIGINAL QUALITY preserved!')
            else:
                raise Exception("Output file was not created or is empty")
        else:
            raise Exception(f"FFmpeg failed with return code {proc.returncode}")
            
    except Exception as e:
        error_msg = str(e)
        processing_status[unique_id].update({
            'status': 'failed',
            'message': error_msg,
            'status_text': f'Failed: {error_msg}'
        })
    finally:
        cleanup_temp_files(temp_dir)

@app.route('/stream/<unique_id>')
def stream_progress(unique_id):
    def generate():
        last_log = 0
        last_progress = -1
        while True:
            if unique_id not in processing_status:
                yield f"data: {json.dumps({'type': 'complete', 'success': False})}\n\n"
                break
            status = processing_status[unique_id]
            p = status.get('progress', 0)
            if p != last_progress:
                yield f"data: {json.dumps({'type': 'progress', 'percentage': p, 'status': status.get('status_text', '')})}\n\n"
                last_progress = p
            logs = status.get('logs', [])
            if len(logs) > last_log:
                for log in logs[last_log:]:
                    yield f"data: {json.dumps({'type': 'log', 'message': log})}\n\n"
                last_log = len(logs)
            if status['status'] == 'success':
                yield f"data: {json.dumps({'type': 'complete', 'success': True, 'output_filename': status['output_filename']})}\n\n"
                del processing_status[unique_id]
                break
            elif status['status'] == 'failed':
                yield f"data: {json.dumps({'type': 'complete', 'success': False, 'message': status.get('message', 'Unknown error')})}\n\n"
                del processing_status[unique_id]
                break
            time.sleep(1)
    return Response(generate(), mimetype='text/event-stream')

@app.route('/status/<unique_id>')
def check_status(unique_id):
    status = processing_status.get(unique_id, {'status': 'not_found'})
    if status['status'] in ['success', 'failed']:
        copy = status.copy()
        if unique_id in processing_status:
            del processing_status[unique_id]
        return jsonify(copy)
    return jsonify(status)

# === VIDEO TO SRT EXTRACTOR ===
@app.route('/video_to_srt_page')
def video_to_srt_page():
    """Render the video-to-SRT extractor page."""
    return render_template('video_to_srt.html')

@app.route('/video_to_srt', methods=['POST'])
def start_video_to_srt():
    if not WHISPER_AVAILABLE:
        return jsonify({'error': 'Whisper is not installed. Please install with: pip install openai-whisper'}), 500
    
    if 'video' not in request.files:
        return jsonify({'error': 'No video uploaded'}), 400

    video = request.files['video']
    model = request.form.get('model', 'small')  # Changed default to 'small'
    translate = request.form.get('translate') == 'true'

    if video.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    unique_id = str(uuid.uuid4())
    temp_dir = os.path.join(UPLOAD_FOLDER, unique_id)
    os.makedirs(temp_dir, exist_ok=True)

    video_path = os.path.join(temp_dir, secure_filename(video.filename))
    video.save(video_path)

    srt_filename = f"extracted_{unique_id}.srt"
    srt_path = os.path.join(OUTPUT_FOLDER, srt_filename)

    print(f"🎬 DEBUG: Starting enhanced video-to-SRT conversion")
    print(f"🎬 DEBUG: video_path = {video_path}")
    print(f"🎬 DEBUG: srt_path = {srt_path}")
    print(f"🎬 DEBUG: model = {model}")
    print(f"🎬 DEBUG: translate = {translate}")

    video_processing_status[unique_id] = {
        'status': 'running',
        'progress': 0,
        'message': 'Loading Whisper model...',
        'srt_file': srt_filename
    }

    def enhanced_background_task():
        """Enhanced background task with smart SRT generation"""
        print(f"🎬 DEBUG: Starting enhanced background task")
        print(f"🎬 DEBUG: video_path = {video_path}")
        print(f"🎬 DEBUG: srt_path = {srt_path}")
        print(f"🎬 DEBUG: unique_id = {unique_id}")
        
        try:
            video_processing_status[unique_id]['message'] = f'Loading {model} model...'
            video_processing_status[unique_id]['progress'] = 10
            
            print(f"🧠 DEBUG: Loading Whisper model: {model}")
            m = whisper.load_model(model)
            print(f"✅ DEBUG: Model loaded successfully")
            
            video_processing_status[unique_id]['message'] = 'Analyzing audio with enhanced processing...'
            video_processing_status[unique_id]['progress'] = 20
            
            # Step 1: Detect silence periods using FFmpeg (if available)
            silence_periods = []
            try:
                print(f"🔍 DEBUG: Detecting silence periods...")
                cmd = [
                    'ffmpeg', '-i', video_path,
                    '-af', 'silencedetect=noise=-30dB:d=1.0',
                    '-f', 'null', '-'
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, stderr=subprocess.STDOUT)
                output = result.stderr
                
                silence_start = None
                for line in output.split('\n'):
                    if 'silence_start:' in line:
                        match = re.search(r'silence_start: ([\d.]+)', line)
                        if match:
                            silence_start = float(match.group(1))
                    elif 'silence_end:' in line and silence_start is not None:
                        match = re.search(r'silence_end: ([\d.]+)', line)
                        if match:
                            silence_end = float(match.group(1))
                            silence_periods.append((silence_start, silence_end))
                            silence_start = None
                
                print(f"✅ DEBUG: Found {len(silence_periods)} silence periods")
            except Exception as e:
                print(f"⚠️ DEBUG: Could not detect silence: {e}")
            
            video_processing_status[unique_id]['message'] = 'Transcribing with smart segmentation...'
            video_processing_status[unique_id]['progress'] = 30
            
            # Step 2: Transcribe with enhanced Whisper settings
            print(f"🎵 DEBUG: Starting enhanced transcription...")
            result = m.transcribe(
                video_path, 
                task="translate" if translate else "transcribe",
                word_timestamps=True,  # Get word-level timestamps for better segmentation
                condition_on_previous_text=False,  # Better for files with pauses
                compression_ratio_threshold=2.4,  # Filter out gibberish
                logprob_threshold=-1.0,  # Filter out uncertain segments  
                no_speech_threshold=0.6,  # Better silence detection
                temperature=0.0  # More deterministic output
            )
            print(f"✅ DEBUG: Enhanced transcription completed")
            
            video_processing_status[unique_id]['message'] = 'Processing segments with smart timing...'
            video_processing_status[unique_id]['progress'] = 70
            
            # Step 3: Enhanced segment processing
            segments = result.get('segments', [])
            enhanced_segments = []
            
            if segments:
                # Find actual speech start (ignore early silence/noise)
                speech_start = 0
                for segment in segments:
                    text = segment.get('text', '').strip()
                    if len(text) > 3 and any(c.isalpha() for c in text):
                        speech_start = segment.get('start', 0)
                        break
                
                print(f"🎯 DEBUG: Detected speech start at {speech_start:.2f}s")
                
                for segment in segments:
                    start = segment.get('start', 0)
                    end = segment.get('end', 0)
                    text = segment.get('text', '').strip()
                    
                    # Skip segments before actual speech starts
                    if start < speech_start - 0.5:
                        continue
                    
                    # Skip very short or empty segments
                    if len(text) < 2 or (end - start) < 0.3:
                        continue
                    
                    # Check if segment overlaps with silence periods
                    skip_segment = False
                    for silence_start_time, silence_end_time in silence_periods:
                        # Add small buffer to silence periods
                        buffered_start = silence_start_time + 0.2
                        buffered_end = silence_end_time - 0.2
                        
                        # Calculate overlap
                        overlap_start = max(start, buffered_start)
                        overlap_end = min(end, buffered_end)
                        
                        if overlap_end > overlap_start:
                            overlap_duration = overlap_end - overlap_start
                            segment_duration = end - start
                            
                            # Skip if more than 70% overlaps with silence
                            if overlap_duration / segment_duration > 0.7:
                                skip_segment = True
                                break
                    
                    if skip_segment:
                        continue
                    
                    # Clean up text
                    text = re.sub(r'\s+', ' ', text).strip()
                    
                    # Split long segments at natural breaks
                    text_chunks = split_long_text(text)
                    
                    if len(text_chunks) == 1:
                        # Single chunk
                        enhanced_segments.append({
                            'start': start,
                            'end': end,
                            'text': text
                        })
                    else:
                        # Multiple chunks - distribute time evenly
                        chunk_duration = (end - start) / len(text_chunks)
                        for i, chunk in enumerate(text_chunks):
                            chunk_start = start + (i * chunk_duration)
                            chunk_end = start + ((i + 1) * chunk_duration)
                            enhanced_segments.append({
                                'start': chunk_start,
                                'end': chunk_end,
                                'text': chunk
                            })
            
            video_processing_status[unique_id]['message'] = 'Writing enhanced SRT file...'
            video_processing_status[unique_id]['progress'] = 90
            
            # Step 4: Write enhanced SRT file
            print(f"📝 DEBUG: Writing enhanced SRT with {len(enhanced_segments)} segments...")
            
            # Create output folder if it doesn't exist
            os.makedirs(OUTPUT_FOLDER, exist_ok=True)
            
            with open(srt_path, 'w', encoding='utf-8') as f:
                if not enhanced_segments:
                    f.write("1\n00:00:00,000 --> 00:00:05,000\nNo clear speech detected\n\n")
                else:
                    for i, segment in enumerate(enhanced_segments):
                        start_time = format_timestamp(segment['start'])
                        end_time = format_timestamp(segment['end'])
                        text = segment['text']
                        
                        f.write(f"{i+1}\n{start_time} --> {end_time}\n{text}\n\n")
                        
                        if i < 3:  # Debug first 3 entries
                            print(f"📝 DEBUG: Enhanced segment {i+1}: {start_time} --> {end_time}: {text[:30]}...")
            
            print(f"✅ DEBUG: Enhanced SRT file written successfully")
            
            # Verify file
            if os.path.exists(srt_path):
                file_size = os.path.getsize(srt_path)
                print(f"✅ DEBUG: Enhanced SRT file exists! Size: {file_size} bytes")
                
                video_processing_status[unique_id]['status'] = 'completed'
                video_processing_status[unique_id]['progress'] = 100
                video_processing_status[unique_id]['message'] = 'Enhanced SRT completed with smart timing!'
            else:
                raise Exception("Enhanced SRT file was not created")
                
        except Exception as e:
            print(f"❌ DEBUG: Exception in enhanced background_task: {e}")
            import traceback
            traceback.print_exc()
            
            video_processing_status[unique_id]['status'] = 'error'
            video_processing_status[unique_id]['message'] = f"Enhanced processing failed: {str(e)}"
        finally:
            print(f"🧹 DEBUG: Cleanup - temp_dir = {temp_dir}")
            if os.path.exists(temp_dir):
                cleanup_temp_files(temp_dir)
            print(f"✅ DEBUG: Enhanced background task finished")

    thread = threading.Thread(target=enhanced_background_task)
    thread.daemon = True
    thread.start()

    return jsonify({'success': True, 'conversion_id': unique_id})

@app.route('/video_status/<conversion_id>')
def video_status(conversion_id):
    def generate():
        try:
            while True:
                status = video_processing_status.get(conversion_id)
                if not status:
                    yield f"data: {json.dumps({'status': 'error', 'message': 'Job not found'})}\n\n"
                    break
                if status['status'] == 'completed':
                    yield f"data: {json.dumps({'status': 'completed', 'progress': 100, 'message': 'Enhanced SRT completed!', 'data': {'srt_file': status['srt_file']}})}\n\n"
                    # Clean up status after completion
                    del video_processing_status[conversion_id]
                    break
                elif status['status'] == 'error':
                    yield f"data: {json.dumps({'status': 'error', 'message': status['message']})}\n\n"
                    del video_processing_status[conversion_id]
                    break
                else:
                    yield f"data: {json.dumps({'status': 'in_progress', 'progress': status['progress'], 'message': status['message']})}\n\n"
                time.sleep(0.5)
        except Exception as e:
            print(f"❌ ERROR in video_status: {e}")
            yield f"data: {json.dumps({'status': 'error', 'message': f'Server error: {str(e)}'})}\n\n"
    
    return Response(generate(), mimetype='text/event-stream')

@app.route('/video_status_json/<conversion_id>')
def video_status_json(conversion_id):
    """JSON endpoint for fallback status checking"""
    try:
        status = video_processing_status.get(conversion_id)
        if not status:
            return jsonify({'status': 'error', 'message': 'Job not found'}), 404
        
        if status['status'] == 'completed':
            completed_status = status.copy()
            del video_processing_status[conversion_id]
            return jsonify({
                'status': 'completed', 
                'progress': 100, 
                'message': 'Enhanced SRT completed!', 
                'data': {'srt_file': completed_status['srt_file']}
            })
        elif status['status'] == 'error':
            error_message = status['message']
            del video_processing_status[conversion_id]
            return jsonify({'status': 'error', 'message': error_message})
        else:
            return jsonify({
                'status': 'in_progress', 
                'progress': status['progress'], 
                'message': status['message']
            })
            
    except Exception as e:
        print(f"❌ ERROR in video_status_json: {e}")
        return jsonify({'status': 'error', 'message': f'Server error: {str(e)}'}), 500

# === ONLINE VIDEO DOWNLOADER ===
@app.route('/download_video_page')
def download_video_page():
    """Render the online video downloader page."""
    return render_template('download_video.html')

@app.route('/download_online_video', methods=['POST'])
def download_online_video():
    """Start downloading video from URL"""
    try:
        data = request.get_json()
        url = data.get('url', '').strip()
        quality = data.get('quality', 'best')
        audio_only = data.get('audio_only', False)
        
        if not url:
            return jsonify({'error': 'Please provide a valid URL'}), 400
        
        # Basic URL validation
        try:
            result = urlparse(url)
            if not all([result.scheme, result.netloc]):
                raise ValueError("Invalid URL")
        except:
            return jsonify({'error': 'Invalid URL format'}), 400
        
        # Generate unique ID for this download
        unique_id = str(uuid.uuid4())
        
        # Initialize download status
        download_status[unique_id] = {
            'status': 'starting',
            'progress': 0,
            'message': 'Initializing download...',
            'url': url,
            'filename': None,
            'file_size': 0
        }
        
        # Start download in background
        thread = threading.Thread(
            target=download_video_task,
            args=(url, quality, audio_only, unique_id)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'success': True,
            'download_id': unique_id,
            'message': 'Download started'
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to start download: {str(e)}'}), 500

def download_video_task(url, quality, audio_only, unique_id):
    """Background task to download video using yt-dlp"""
    try:
        download_status[unique_id]['message'] = 'Checking URL and fetching info...'
        download_status[unique_id]['progress'] = 5
        
        # Create unique output directory
        output_dir = os.path.join(OUTPUT_FOLDER, f"downloads_{unique_id}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Build yt-dlp command
        cmd = ['yt-dlp']
        
        if audio_only:
            # Audio only download
            cmd.extend([
                '--extract-audio',
                '--audio-format', 'mp3',
                '--audio-quality', '0',  # Best quality
                '--output', os.path.join(output_dir, '%(title)s.%(ext)s')
            ])
        else:
            # Video download with quality selection
            if quality == 'best':
                cmd.extend(['--format', 'best[height<=2160]'])  # Up to 4K
            elif quality == '1080p':
                cmd.extend(['--format', 'best[height<=1080]'])
            elif quality == '720p':
                cmd.extend(['--format', 'best[height<=720]'])
            elif quality == '480p':
                cmd.extend(['--format', 'best[height<=480]'])
            else:
                cmd.extend(['--format', 'best'])
            
            cmd.extend([
                '--output', os.path.join(output_dir, '%(title)s.%(ext)s'),
                '--merge-output-format', 'mp4'  # Ensure MP4 output
            ])
        
        # Common options
        cmd.extend([
            '--no-playlist',  # Download single video only
            '--write-info-json',  # Get video info
            url
        ])
        
        download_status[unique_id]['message'] = 'Starting download...'
        download_status[unique_id]['progress'] = 10
        
        print(f"🎬 DEBUG: Starting yt-dlp with command: {' '.join(cmd)}")
        
        # Start yt-dlp process
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Monitor progress
        for line in iter(proc.stdout.readline, ''):
            if line.strip():
                print(f"🎬 yt-dlp: {line.strip()}")
                
                # Parse progress from yt-dlp output
                if '[download]' in line and '%' in line:
                    try:
                        # Extract percentage from download line
                        percent_match = re.search(r'(\d+(?:\.\d+)?)%', line)
                        if percent_match:
                            progress = float(percent_match.group(1))
                            download_status[unique_id]['progress'] = min(progress, 95)
                            
                        # Extract speed if available
                        speed_match = re.search(r'at\s+([\d.]+\w+/s)', line)
                        if speed_match:
                            speed = speed_match.group(1)
                            download_status[unique_id]['message'] = f'Downloading... {speed}'
                            
                    except:
                        pass
                
                # Check for completion or filename
                if 'has already been downloaded' in line or 'Destination:' in line:
                    download_status[unique_id]['progress'] = 95
                    download_status[unique_id]['message'] = 'Finalizing download...'
        
        proc.wait()
        
        if proc.returncode == 0:
            # Find downloaded file
            downloaded_files = []
            for file in os.listdir(output_dir):
                if not file.endswith('.json') and not file.endswith('.part'):
                    downloaded_files.append(file)
            
            if downloaded_files:
                filename = downloaded_files[0]
                original_path = os.path.join(output_dir, filename)
                
                # Move to main output folder with unique name
                final_filename = f"downloaded_{unique_id}_{filename}"
                final_path = os.path.join(OUTPUT_FOLDER, final_filename)
                shutil.move(original_path, final_path)
                
                file_size = os.path.getsize(final_path)
                
                download_status[unique_id].update({
                    'status': 'completed',
                    'progress': 100,
                    'message': 'Download completed!',
                    'filename': final_filename,
                    'file_size': file_size,
                    'original_name': filename
                })
                
                print(f"✅ Download completed: {final_filename}")
            else:
                raise Exception("No output file found after download")
        else:
            raise Exception(f"yt-dlp failed with return code {proc.returncode}")
            
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Download error: {error_msg}")
        download_status[unique_id].update({
            'status': 'error',
            'message': error_msg,
            'progress': 0
        })
    finally:
        # Clean up temp directory
        if os.path.exists(output_dir):
            try:
                shutil.rmtree(output_dir)
            except:
                pass

@app.route('/download_status/<download_id>')
def get_download_status(download_id):
    """Get download progress status"""
    def generate():
        while True:
            status = download_status.get(download_id)
            if not status:
                yield f"data: {json.dumps({'status': 'error', 'message': 'Download not found'})}\n\n"
                break
                
            if status['status'] == 'completed':
                yield f"data: {json.dumps({'status': 'completed', 'progress': 100, 'message': status['message'], 'filename': status['filename'], 'original_name': status.get('original_name', ''), 'file_size': status['file_size']})}\n\n"
                # Keep status for a while before cleanup
                break
            elif status['status'] == 'error':
                yield f"data: {json.dumps({'status': 'error', 'message': status['message']})}\n\n"
                if download_id in download_status:
                    del download_status[download_id]
                break
            else:
                yield f"data: {json.dumps({'status': 'downloading', 'progress': status['progress'], 'message': status['message']})}\n\n"
                
            time.sleep(1)
    
    return Response(generate(), mimetype='text/event-stream')

# === SRT TO ARABIC TRANSLATOR ===
@app.route('/translate_srt_page')
def translate_srt_page():
    """Render the SRT translation page."""
    return render_template('translate_srt.html')

# === ENHANCED TRANSLATION ROUTE ===
@app.route('/translate_srt', methods=['POST'])
def translate_srt_enhanced():
    """Enhanced SRT translation with automatic RTL formatting"""
    if 'srtFile' not in request.files:
        return jsonify({'error': 'No SRT file uploaded'}), 400

    srt_file = request.files['srtFile']
    if srt_file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(srt_file.filename, {'srt'}):
        return jsonify({'error': 'Only .srt files allowed'}), 400

    unique_id = str(uuid.uuid4())
    input_path = os.path.join(UPLOAD_FOLDER, f"{unique_id}_input.srt")
    
    # Update output filename to indicate it includes RTL formatting
    output_filename = f"translated_arabic_rtl_{unique_id}.srt"
    output_path = os.path.join(OUTPUT_FOLDER, output_filename)

    srt_file.save(input_path)
    print(f"📁 Saved input file: {input_path}")

    # Initialize status with enhanced messaging
    translate_status[unique_id] = {
        'status': 'translating',
        'progress': 0,
        'message': 'Starting Arabic translation with automatic RTL formatting...'
    }

    # Start translation in background thread
    thread = threading.Thread(
        target=enhanced_translate_srt_task,
        args=(input_path, output_path, unique_id)
    )
    thread.daemon = True
    thread.start()

    return jsonify({
        'success': True, 
        'translation_id': unique_id,
        'message': 'Translation started - RTL formatting will be applied automatically!'
    })

@app.route('/translation_status/<translation_id>')
def translation_status(translation_id):
    def generate():
        while True:
            status = translate_status.get(translation_id)
            if not status:
                yield f"data: {json.dumps({'status': 'error', 'message': 'Not found'})}\n\n"
                break
            if status['status'] == 'completed':
                yield f"data: {json.dumps({'status': 'completed', 'progress': 100, 'message': 'Translation completed with automatic RTL formatting!', 'output_filename': status['output_filename']})}\n\n"
                del translate_status[translation_id]
                break
            elif status['status'] == 'error':
                yield f"data: {json.dumps({'status': 'error', 'message': status['message']})}\n\n"
                del translate_status[translation_id]
                break
            else:
                yield f"data: {json.dumps({'status': 'in_progress', 'progress': status['progress'], 'message': status['message']})}\n\n"
            time.sleep(0.5)
    return Response(generate(), mimetype='text/event-stream')

# === NEW SRT EDITOR ROUTES ===
@app.route('/edit_srt_page')
def edit_srt_page():
    """Render the SRT editor page"""
    return render_template('edit_srt.html')

@app.route('/parse_srt', methods=['POST'])
def parse_srt():
    """Parse uploaded SRT file and return JSON data for editing"""
    if 'srtFile' not in request.files:
        return jsonify({'error': 'No SRT file uploaded'}), 400

    srt_file = request.files['srtFile']
    if srt_file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(srt_file.filename, {'srt'}):
        return jsonify({'error': 'Only .srt files allowed'}), 400

    try:
        # Read and parse SRT content
        content = srt_file.read().decode('utf-8')
        subtitles = parse_srt_content(content)
        
        return jsonify({
            'success': True,
            'filename': srt_file.filename,
            'subtitles': subtitles,
            'total_count': len(subtitles)
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to parse SRT: {str(e)}'}), 400

@app.route('/save_srt', methods=['POST'])
def save_srt():
    """Save edited SRT content to a new file"""
    try:
        data = request.get_json()
        
        if not data or 'subtitles' not in data:
            return jsonify({'error': 'No subtitle data provided'}), 400

        subtitles = data['subtitles']
        original_filename = data.get('filename', 'edited.srt')
        
        # Generate unique filename
        unique_id = str(uuid.uuid4())
        filename = f"edited_{unique_id}.srt"
        output_path = os.path.join(OUTPUT_FOLDER, filename)
        
        # Convert back to SRT format
        srt_content = generate_srt_content(subtitles)
        
        # Save file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(srt_content)
        
        return jsonify({
            'success': True,
            'filename': filename,
            'message': 'SRT file saved successfully!'
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to save SRT: {str(e)}'}), 500

# === DOWNLOAD ROUTE WITH DEBUG ===
@app.route('/download/<filename>')
def download_file(filename):
    print(f"📁 DEBUG: Download requested: {filename}")
    print(f"📁 DEBUG: Output folder: {app.config['OUTPUT_FOLDER']}")
    print(f"📁 DEBUG: Current working dir: {os.getcwd()}")
    
    # Security check - ensure filename doesn't contain path traversal
    if '..' in filename or '/' in filename or '\\' in filename:
        print(f"❌ DEBUG: Invalid filename detected: {filename}")
        return jsonify({'error': 'Invalid filename'}), 400
    
    path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
    abs_path = os.path.abspath(path)
    
    print(f"📁 DEBUG: Looking for: {abs_path}")
    print(f"📄 DEBUG: File exists: {os.path.exists(abs_path)}")
    
    if os.path.exists(abs_path):
        print(f"✅ DEBUG: File found, sending...")
        try:
            return send_file(abs_path, as_attachment=True, download_name=filename)
        except Exception as e:
            print(f"❌ DEBUG: Error sending file: {e}")
            return jsonify({'error': f'Error sending file: {str(e)}'}), 500
    else:
        print(f"❌ DEBUG: File not found!")
        if os.path.exists(app.config['OUTPUT_FOLDER']):
            files_in_output = os.listdir(app.config['OUTPUT_FOLDER'])
            print(f"📁 DEBUG: Files in output folder: {files_in_output}")
        else:
            print(f"📁 DEBUG: Output folder doesn't exist!")
        return jsonify({'error': 'File not found'}), 404

if __name__ == '__main__':
    print("🚀 Starting Enhanced Subtitle Tools server...")
    print("📁 Upload folder:", UPLOAD_FOLDER)
    print("📁 Output folder:", OUTPUT_FOLDER)
    
    # Check dependencies on startup
    issues = check_dependencies()
    if issues:
        print("⚠️  Issues found:")
        for issue in issues:
            print(f"   - {issue}")
        print()
    else:
        print("✅ All dependencies found!")
    
    # Production settings
    app.run(
        host='0.0.0.0',  # Listen on all interfaces
        port=5000,
        debug=False,     # Disable debug in production
        threaded=True
    )
