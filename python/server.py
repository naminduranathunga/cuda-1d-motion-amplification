from flask import Flask, request, render_template, send_from_directory, jsonify
from flask_cors import CORS
import os
import uuid
import subprocess
from main import process_video
import cv2
import numpy as np

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

# Use absolute paths to avoid issues with different startup directories
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
PROCESSED_FOLDER = os.path.join(BASE_DIR, 'processed')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    file = request.files['video']
    alpha = float(request.form.get('alpha', 50.0))
    sigma = float(request.form.get('sigma', 1.0))
    low_freq = float(request.form.get('low_freq', 0.5))
    high_freq = float(request.form.get('high_freq', 2.0))
    fps = request.form.get('fps')
    if fps:
        fps = float(fps)
    else:
        fps = None
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    file_id = str(uuid.uuid4())
    input_path = os.path.join(UPLOAD_FOLDER, f"{file_id}_{file.filename}")
    output_filename = f"amp_{file_id}_{file.filename}"
    output_path = os.path.join(PROCESSED_FOLDER, output_filename)
    
    file.save(input_path)
    
    # Process the video
    try:
        metrics = process_video(input_path, output_path, alpha, sigma, low_freq, high_freq, fps)
        
        # Transcode BOTH to H.264 for web compatibility
        import imageio_ffmpeg
        ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
        
        web_input_filename = f"web_in_{file_id}_{file.filename}"
        web_input_path = os.path.join(UPLOAD_FOLDER, web_input_filename)
        
        web_output_filename = f"web_out_{file_id}_{file.filename}"
        web_output_path = os.path.join(PROCESSED_FOLDER, web_output_filename)
        
        # Transcode input
        subprocess.run([ffmpeg_bin, '-y', '-i', input_path, '-c:v', 'libx264', '-pix_fmt', 'yuv420p', web_input_path], check=True)
        # Transcode output
        subprocess.run([ffmpeg_bin, '-y', '-i', output_path, '-c:v', 'libx264', '-pix_fmt', 'yuv420p', web_output_path], check=True)
        
        return jsonify({
            'success': True,
            'input_url': f'/uploads/{web_input_filename}',
            'output_url': f'/processed/{web_output_filename}',
            'metrics': {
                'host_to_device': float(metrics[0]),
                'gaussian_blur': float(metrics[1]),
                'sobel_x': float(metrics[2]),
                'temporal_filter': float(metrics[3]),
                'amplification': float(metrics[4]),
                'device_to_host': float(metrics[5]),
                'total': float(sum(metrics))
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/uploads/<filename>')
def serve_upload(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/processed/<filename>')
def serve_processed(filename):
    return send_from_directory(PROCESSED_FOLDER, filename)

@app.route('/histogram', methods=['POST'])
def get_roi_histogram():
    data = request.json
    video_path = data.get('video_path')
    frame_time = float(data.get('time', 0))
    roi = data.get('roi') # {x1, y1, x2, y2}
    
    # Correct path if it's a relative URL
    if video_path.startswith('/uploads/'):
        video_path = os.path.join(UPLOAD_FOLDER, video_path.replace('/uploads/', ''))
    elif video_path.startswith('/processed/'):
        video_path = os.path.join(PROCESSED_FOLDER, video_path.replace('/processed/', ''))

    if not os.path.exists(video_path):
        return jsonify({'error': 'Video not found'}), 404

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_MSEC, frame_time * 1000)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return jsonify({'error': 'Could not read frame'}), 500

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h_input = gray.astype(np.float32) / 255.0
    
    from main import compute_histogram
    hist = compute_histogram(h_input, roi['x1'], roi['y1'], roi['x2'], roi['y2'])
    
    return jsonify({'histogram': hist.tolist()})

@app.route('/frequency_spectrum', methods=['POST'])
def get_frequency_spectrum():
    data = request.json
    video_path = data.get('video_path')
    roi = data.get('roi') # {x1, y1, x2, y2}
    
    if video_path.startswith('/uploads/'):
        video_path = os.path.join(UPLOAD_FOLDER, video_path.replace('/uploads/', ''))
    elif video_path.startswith('/processed/'):
        video_path = os.path.join(PROCESSED_FOLDER, video_path.replace('/processed/', ''))

    if not os.path.exists(video_path):
        return jsonify({'error': 'Video not found'}), 404

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    intensities = []
    
    # Sample every frame for the first 10 seconds or total duration to get a spectrum
    # For performance, maybe limit to 300 frames (~10s at 30fps)
    frame_count = 0
    max_frames = 300
    
    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Crop to ROI
        x1, y1, x2, y2 = int(roi['x1']), int(roi['y1']), int(roi['x2']), int(roi['y2'])
        # Ensure ROI is within bounds
        h, w = gray.shape
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)
        
        if x2 > x1 and y2 > y1:
            roi_slice = gray[y1:y2, x1:x2]
            avg_intensity = np.mean(roi_slice)
            intensities.append(avg_intensity)
        
        frame_count += 1
    
    cap.release()

    if not intensities:
        return jsonify({'error': 'No data for ROI'}), 400

    # Compute FFT
    n = len(intensities)
    # Detrend
    intensities = np.array(intensities) - np.mean(intensities)
    fft_vals = np.fft.rfft(intensities)
    fft_freqs = np.fft.rfftfreq(n, d=1.0/fps)
    
    magnitudes = np.abs(fft_vals)
    
    return jsonify({
        'frequencies': fft_freqs.tolist(),
        'magnitudes': magnitudes.tolist()
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

