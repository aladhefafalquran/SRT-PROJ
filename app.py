import os
import subprocess
import threading
import time
import re
import json
import requests
from flask import Flask, request, render_template, send_file, jsonify, Response
from werkzeug.utils import secure_filename
import uuid
import shutil

# === Dependency Checks ===
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    whisper = None

# === Configuration ===
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
ALLOWED_EXTENSIONS_VIDEO = {'mp4', 'mov', 'avi', 'mkv'}
ALLOWED_EXTENSIONS_SUB = {'srt', 'vtt', 'ass'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

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
    except (subprocess.CalledProcessError, FileNotFoundError):
        issues.append("FFmpeg not found. Please install FFmpeg.")
    
    # Check ffprobe
    try:
        subprocess.run(['ffprobe', '-version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        issues.append("ffprobe not found. Please install FFmpeg.")
    
    # Check Whisper
    if not WHISPER_AVAILABLE:
        issues.append("Whisper not found. Install with: pip install openai-whisper")
    
    return issues


# === Helper Functions ===
def allowed_file(filename, allowed_extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions


def cleanup_temp_files(temp_dir):
    """Helper function to clean up temporary directories."""
    if os.path.exists(temp_dir):
        try:
            shutil.rmtree(temp_dir)
        except Exception as e:
            print(f"Warning: Could not clean up {temp_dir}: {e}")


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


# === Route: Main Page (Merge Tool) ===
@app.route('/')
def index():
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
                        'fps': eval(video_stream.get('r_frame_rate', '30/1')),  # Convert fraction to float
                        'pix_fmt': video_stream.get('pix_fmt', 'yuv420p'),
                        'bitrate': int(video_stream.get('bit_rate', 0))
                    }
                return None
            except Exception:
                return None
        
        video_info = get_video_info(video_path)
        processing_status[unique_id]['status_text'] = 'Analyzing video...'
        
        # Escape subtitle path for FFmpeg
        if os.name == 'nt':  # Windows
            escaped_sub = subtitle_path.replace('\\', '/').replace(':', '\\:')
        else:  # Unix-like
            escaped_sub = subtitle_path.replace(':', '\\:')
        
        # Build improved FFmpeg command
        cmd = ['ffmpeg', '-y', '-i', video_path]
        
        # Video filter with resolution preservation
        if video_info and video_info['width'] > 0 and video_info['height'] > 0:
            # Preserve original dimensions and add subtitles
            vf = f"subtitles='{escaped_sub}':force_style='Fontsize=16,PrimaryColour=&Hffffff,OutlineColour=&H000000,BackColour=&H80000000,Outline=2,Shadow=1'"
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

    video_processing_status[unique_id] = {
        'status': 'running',
        'progress': 0,
        'message': 'Loading Whisper model...',
        'srt_file': srt_filename
    }

    def background_task():
        try:
            video_processing_status[unique_id]['message'] = f'Loading {model} model...'
            video_processing_status[unique_id]['progress'] = 10
            
            m = whisper.load_model(model)
            
            video_processing_status[unique_id]['message'] = 'Transcribing audio...'
            video_processing_status[unique_id]['progress'] = 30
            
            result = m.transcribe(video_path, task="translate" if translate else "transcribe")
            
            video_processing_status[unique_id]['message'] = 'Writing SRT file...'
            video_processing_status[unique_id]['progress'] = 90
            
            with open(srt_path, 'w', encoding='utf-8') as f:
                for i, seg in enumerate(result['segments']):
                    start = format_timestamp(seg['start'])
                    end = format_timestamp(seg['end'])
                    f.write(f"{i+1}\n{start} --> {end}\n{seg['text'].strip()}\n\n")
            
            video_processing_status[unique_id]['status'] = 'completed'
            video_processing_status[unique_id]['progress'] = 100
            video_processing_status[unique_id]['message'] = 'Completed!'
            
        except Exception as e:
            video_processing_status[unique_id]['status'] = 'error'
            video_processing_status[unique_id]['message'] = str(e)
        finally:
            cleanup_temp_files(temp_dir)

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


@app.route('/download/<filename>')
def download_file(filename):
    path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
    if os.path.exists(path):
        return send_file(path, as_attachment=True, download_name=filename)
    return render_template('index.html', message="File not found."), 404


if __name__ == '__main__':
    print("🔍 Checking dependencies...")
    issues = check_dependencies()
    if issues:
        print("⚠️  Issues found:")
        for issue in issues:
            print(f"   - {issue}")
        print()
    else:
        print("✅ All dependencies found!")
    
    app.run(debug=True, threaded=True)