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

# ==============================
# 🔑 DEEPL API KEY - MOVE TO ENVIRONMENT VARIABLE
# ==============================
DEEPL_API_KEY = os.getenv("DEEPL_API_KEY", "6e05e993-b62b-43c5-aaa1-24b25aa8c3ae:fx")
DEEPL_API_URL = "https://api-free.deepl.com/v2/translate"
# ==============================


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
        
        # Get original video info to preserve quality
        def get_video_info(video_path):
            try:
                cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_streams', video_path]
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                data = json.loads(result.stdout)
                
                video_stream = None
                for stream in data['streams']:
                    if stream['codec_type'] == 'video':
                        video_stream = stream
                        break
                
                if video_stream:
                    return {
                        'width': int(video_stream.get('width', 0)),
                        'height': int(video_stream.get('height', 0)),
                        'fps': eval(video_stream.get('r_frame_rate', '30/1')),
                        'pix_fmt': video_stream.get('pix_fmt', 'yuv420p'),
                        'bitrate': int(video_stream.get('bit_rate', 0))
                    }
                return None
            except Exception:
                return None
        
        video_info = get_video_info(video_path)
        processing_status[unique_id]['status_text'] = 'Analyzing video and subtitles...'
        
        # Check if subtitles contain Arabic text
        is_arabic = has_arabic_text(subtitle_path)
        
        # Create RTL-compatible subtitle file if Arabic is detected
        if is_arabic:
            processing_status[unique_id]['status_text'] = 'Processing Arabic RTL subtitles...'
            rtl_subtitle_path = os.path.join(temp_dir, 'rtl_subtitles.srt')
            if create_rtl_srt(subtitle_path, rtl_subtitle_path):
                subtitle_path = rtl_subtitle_path
                processing_status[unique_id]['logs'].append('Created RTL-compatible subtitle file for Arabic text')
        
        # Escape subtitle path for FFmpeg
        if os.name == 'nt':  # Windows
            escaped_sub = subtitle_path.replace('\\', '/').replace(':', '\\:')
        else:  # Unix-like
            escaped_sub = subtitle_path.replace(':', '\\:')
        
        # Build improved FFmpeg command
        cmd = ['ffmpeg', '-y', '-i', video_path]
        
        # Video filter with RTL support for Arabic
        if video_info and video_info['width'] > 0 and video_info['height'] > 0:
            if is_arabic:
                # Special styling for Arabic RTL text
                vf = (f"subtitles='{escaped_sub}'"
                     f":force_style='Fontsize=18,PrimaryColour=&Hffffff,OutlineColour=&H000000,"
                     f"BackColour=&H80000000,Outline=2,Shadow=1,Alignment=2,MarginV=30,"
                     f"Fontname=Arial,Bold=1'")
                processing_status[unique_id]['logs'].append('Applied Arabic RTL text styling')
            else:
                # Regular subtitle styling for non-Arabic text
                vf = (f"subtitles='{escaped_sub}'"
                     f":force_style='Fontsize=16,PrimaryColour=&Hffffff,OutlineColour=&H000000,"
                     f"BackColour=&H80000000,Outline=2,Shadow=1'")
            
            cmd.extend(['-vf', vf])
            
            # Preserve original resolution explicitly
            cmd.extend(['-s', f"{video_info['width']}x{video_info['height']}"])
            
            # Video codec settings for quality preservation
            cmd.extend(['-c:v', 'libx264'])
            cmd.extend(['-preset', 'medium'])  # Better quality than 'fast'
            
            # Use original bitrate if available, otherwise use CRF
            if video_info['bitrate'] > 0:
                target_bitrate = min(video_info['bitrate'], 8000000)  # Cap at 8Mbps
                cmd.extend(['-b:v', str(target_bitrate)])
                cmd.extend(['-maxrate', str(int(target_bitrate * 1.2))])
                cmd.extend(['-bufsize', str(int(target_bitrate * 2))])
            else:
                cmd.extend(['-crf', '18'])  # Higher quality than 23
            
            # Preserve pixel format
            cmd.extend(['-pix_fmt', video_info['pix_fmt']])
            
            # Frame rate preservation
            if video_info['fps'] > 0:
                cmd.extend(['-r', str(video_info['fps'])])
        else:
            # Fallback for when we can't get video info
            processing_status[unique_id]['status_text'] = 'Using fallback encoding...'
            if is_arabic:
                vf = (f"subtitles='{escaped_sub}'"
                     f":force_style='Fontsize=18,PrimaryColour=&Hffffff,OutlineColour=&H000000,"
                     f"BackColour=&H80000000,Outline=2,Shadow=1,Alignment=2,MarginV=30,"
                     f"Fontname=Arial,Bold=1'")
            else:
                vf = f"subtitles='{escaped_sub}'"
            cmd.extend(['-vf', vf])
            cmd.extend(['-c:v', 'libx264', '-preset', 'medium', '-crf', '18'])
        
        # Audio settings (copy to avoid re-encoding)
        cmd.extend(['-c:a', 'copy'])
        
        # Additional quality settings
        cmd.extend(['-movflags', '+faststart'])  # Web optimization
        cmd.extend(['-avoid_negative_ts', 'make_zero'])  # Fix timestamp issues
        cmd.extend(['-fflags', '+genpts'])  # Generate presentation timestamps
        
        # Progress reporting
        cmd.extend(['-progress', 'pipe:1', '-nostats'])
        
        cmd.append(output_path)
        
        if is_arabic:
            processing_status[unique_id]['status_text'] = 'Starting encoding with Arabic RTL support...'
            processing_status[unique_id]['logs'].append('Arabic text detected - using RTL processing')
        else:
            processing_status[unique_id]['status_text'] = 'Starting encoding...'
        
        # Start FFmpeg process
        proc = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT, 
            text=True, 
            bufsize=1,
            universal_newlines=True
        )
        
        # Process output
        for line in iter(proc.stdout.readline, ''):
            if line.strip():
                processing_status[unique_id]['logs'].append(line.strip())
                
                if duration:
                    progress = parse_ffmpeg_progress(line, duration)
                    if progress is not None:
                        processing_status[unique_id]['progress'] = progress
                        processing_status[unique_id]['status_text'] = f'Encoding... {progress:.1f}%'
        
        proc.wait()
        
        if proc.returncode == 0:
            # Verify output file
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                processing_status[unique_id].update({
                    'status': 'success',
                    'progress': 100,
                    'status_text': 'Completed successfully!',
                    'output_filename': os.path.basename(output_path)
                })
                if is_arabic:
                    processing_status[unique_id]['logs'].append('✅ Arabic RTL subtitles processed successfully!')
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
    model = request.form.get('model', 'base')
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

    print(f"🎬 DEBUG: Starting video-to-SRT conversion")
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

    def background_task():
        print(f"🎬 DEBUG: Starting background task")
        print(f"🎬 DEBUG: video_path = {video_path}")
        print(f"🎬 DEBUG: srt_path = {srt_path}")
        print(f"🎬 DEBUG: unique_id = {unique_id}")
        
        try:
            video_processing_status[unique_id]['message'] = f'Loading {model} model...'
            video_processing_status[unique_id]['progress'] = 10
            
            print(f"🧠 DEBUG: Loading Whisper model: {model}")
            m = whisper.load_model(model)
            print(f"✅ DEBUG: Model loaded successfully")
            
            video_processing_status[unique_id]['message'] = 'Transcribing audio...'
            video_processing_status[unique_id]['progress'] = 30
            
            print(f"🎵 DEBUG: Starting transcription...")
            result = m.transcribe(video_path, task="translate" if translate else "transcribe")
            print(f"✅ DEBUG: Transcription completed. Result keys: {list(result.keys())}")
            
            if 'segments' in result:
                print(f"✅ DEBUG: Found {len(result['segments'])} segments")
                # Show first few segments for debugging
                for i, seg in enumerate(result['segments'][:3]):
                    print(f"📄 DEBUG: Segment {i}: start={seg.get('start')}, end={seg.get('end')}, text='{seg.get('text', 'NO TEXT')[:50]}...'")
            else:
                print(f"❌ DEBUG: No 'segments' key in result. Available keys: {list(result.keys())}")
            
            video_processing_status[unique_id]['message'] = 'Writing SRT file...'
            video_processing_status[unique_id]['progress'] = 90
            
            print(f"📝 DEBUG: About to write SRT file...")
            print(f"📁 DEBUG: Output folder exists: {os.path.exists(OUTPUT_FOLDER)}")
            print(f"📁 DEBUG: Output folder path: {os.path.abspath(OUTPUT_FOLDER)}")
            print(f"📄 DEBUG: SRT file path: {os.path.abspath(srt_path)}")
            
            # Create output folder if it doesn't exist
            os.makedirs(OUTPUT_FOLDER, exist_ok=True)
            print(f"✅ DEBUG: Output folder created/confirmed")
            
            with open(srt_path, 'w', encoding='utf-8') as f:
                if 'segments' not in result or len(result['segments']) == 0:
                    print(f"❌ DEBUG: No segments found in transcription result!")
                    f.write("1\n00:00:00,000 --> 00:00:05,000\nNo speech detected\n\n")
                else:
                    for i, seg in enumerate(result['segments']):
                        start = format_timestamp(seg['start'])
                        end = format_timestamp(seg['end'])
                        text = seg['text'].strip()
                        
                        srt_entry = f"{i+1}\n{start} --> {end}\n{text}\n\n"
                        f.write(srt_entry)
                        
                        if i < 3:  # Debug first 3 entries
                            print(f"📝 DEBUG: Wrote segment {i+1}: {text[:30]}...")
            
            print(f"✅ DEBUG: SRT file writing completed")
            
            # Verify file was created
            if os.path.exists(srt_path):
                file_size = os.path.getsize(srt_path)
                print(f"✅ DEBUG: SRT file exists! Size: {file_size} bytes")
                
                # Read back first few lines to verify
                with open(srt_path, 'r', encoding='utf-8') as f:
                    first_lines = f.read(200)
                    print(f"📄 DEBUG: File content preview:\n{first_lines}")
            else:
                print(f"❌ DEBUG: SRT file was NOT created!")
            
            video_processing_status[unique_id]['status'] = 'completed'
            video_processing_status[unique_id]['progress'] = 100
            video_processing_status[unique_id]['message'] = 'Completed!'
            print(f"✅ DEBUG: Background task completed successfully")
            
        except Exception as e:
            print(f"❌ DEBUG: Exception in background_task: {e}")
            print(f"❌ DEBUG: Exception type: {type(e)}")
            import traceback
            traceback.print_exc()
            
            video_processing_status[unique_id]['status'] = 'error'
            video_processing_status[unique_id]['message'] = str(e)
        finally:
            print(f"🧹 DEBUG: Cleanup - temp_dir = {temp_dir}")
            if os.path.exists(temp_dir):
                print(f"🧹 DEBUG: Temp dir exists, cleaning up...")
                cleanup_temp_files(temp_dir)
            else:
                print(f"🧹 DEBUG: Temp dir doesn't exist, nothing to clean")
            print(f"✅ DEBUG: Background task finished")

    thread = threading.Thread(target=background_task)
    thread.daemon = True
    thread.start()

    return jsonify({'success': True, 'conversion_id': unique_id})


@app.route('/video_status/<conversion_id>')
def video_status(conversion_id):
    def generate():
        while True:
            status = video_processing_status.get(conversion_id)
            if not status:
                yield f"data: {json.dumps({'status': 'error', 'message': 'Job not found'})}\n\n"
                break
            if status['status'] == 'completed':
                yield f"data: {json.dumps({'status': 'completed', 'progress': 100, 'message': 'Completed!', 'data': {'srt_file': status['srt_file']}})}\n\n"
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
    return Response(generate(), mimetype='text/event-stream')


# === SRT TO ARABIC TRANSLATOR ===
@app.route('/translate_srt_page')
def translate_srt_page():
    """Render the SRT translation page."""
    return render_template('translate_srt.html')


@app.route('/translate_srt', methods=['POST'])
def translate_srt():
    if 'srtFile' not in request.files:
        return jsonify({'error': 'No SRT file uploaded'}), 400

    srt_file = request.files['srtFile']
    if srt_file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(srt_file.filename, {'srt'}):
        return jsonify({'error': 'Only .srt files allowed'}), 400

    unique_id = str(uuid.uuid4())
    input_path = os.path.join(UPLOAD_FOLDER, f"{unique_id}_input.srt")
    output_path = os.path.join(OUTPUT_FOLDER, f"translated_{unique_id}.srt")

    srt_file.save(input_path)

    translate_status[unique_id] = {
        'status': 'translating',
        'progress': 0,
        'message': 'Starting translation...'
    }

    def do_translation():
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                lines = f.read().strip().split('\n')

            translated_lines = []
            total_subs = len([l for l in lines if re.match(r'\d+', l)])
            current = 0

            i = 0
            while i < len(lines):
                line = lines[i]
                translated_lines.append(line)
                i += 1

                if i < len(lines) and '-->' in lines[i]:
                    translated_lines.append(lines[i])
                    i += 1
                    text_lines = []

                    while i < len(lines) and lines[i].strip() != '':
                        text_lines.append(lines[i].strip())
                        i += 1

                    full_text = ' '.join(text_lines)
                    if full_text:
                        response = requests.post(
                            DEEPL_API_URL,
                            data={
                                "text": full_text,
                                "target_lang": "AR"
                            },
                            headers={"Authorization": f"DeepL-Auth-Key {DEEPL_API_KEY}"}
                        )
                        if response.status_code != 200:
                            raise Exception(f"DeepL error: {response.text}")

                        translated = response.json()['translations'][0]['text']
                        translated_lines.append(translated)

                    current += 1
                    if total_subs > 0:
                        translate_status[unique_id]['progress'] = int((current / total_subs) * 100)
                    translate_status[unique_id]['message'] = f"Translating... {current}/{total_subs}"

                if i < len(lines) and lines[i].strip() == '':
                    translated_lines.append('')
                    i += 1

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(translated_lines))

            translate_status[unique_id]['status'] = 'completed'
            translate_status[unique_id]['output_filename'] = f"translated_{unique_id}.srt"

        except Exception as e:
            translate_status[unique_id]['status'] = 'error'
            translate_status[unique_id]['message'] = str(e)
        finally:
            # Clean up input file
            if os.path.exists(input_path):
                try:
                    os.remove(input_path)
                except:
                    pass

    thread = threading.Thread(target=do_translation)
    thread.daemon = True
    thread.start()

    return jsonify({'success': True, 'translation_id': unique_id})


@app.route('/translation_status/<translation_id>')
def translation_status(translation_id):
    def generate():
        while True:
            status = translate_status.get(translation_id)
            if not status:
                yield f"data: {json.dumps({'status': 'error', 'message': 'Not found'})}\n\n"
                break
            if status['status'] == 'completed':
                yield f"data: {json.dumps({'status': 'completed', 'progress': 100, 'message': 'Done!', 'output_filename': status['output_filename']})}\n\n"
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
    print(f"🔍 DEBUG: Download requested: {filename}")
    print(f"📁 DEBUG: Output folder: {app.config['OUTPUT_FOLDER']}")
    print(f"📁 DEBUG: Current working dir: {os.getcwd()}")
    
    # Security check - ensure filename doesn't contain path traversal
    if '..' in filename or '/' in filename or '\\' in filename:
        print(f"❌ DEBUG: Invalid filename detected: {filename}")
        return jsonify({'error': 'Invalid filename'}), 400
    
    path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
    abs_path = os.path.abspath(path)
    
    print(f"🔍 DEBUG: Looking for: {abs_path}")
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
    print("🚀 Starting Subtitle Tools server...")
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