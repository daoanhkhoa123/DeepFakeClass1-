import torch
import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, render_template, request, jsonify
import youtube_dl
import tempfile
import os
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.9)

import torch

# Set the device based on GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Use the device in your code
model = torch.load('/content/DeepFakeClass1-/models/deepfake_cnn_optimized.pth', 
                   map_location=device)
model = model.to(device)  # Move the model to the selected device
model.eval()  # Set the model to evaluation mode

# Function to download video from YouTube
def download_youtube_video(url):
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': tempfile.mktemp() + '.mp4',
    }
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(url, download=True)
        video_file = ydl.prepare_filename(info_dict)
        return video_file

# Process Video and Detect Faces
def process_video(video_path):
    # Initialize OpenCV Video Capture
    cap = cv2.VideoCapture(video_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert BGR frame to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_frame)
        
        if results.detections:
            for detection in results.detections:
                # Draw bounding box on the face
                mp_drawing.draw_detection(frame, detection)
                
                # Crop the face based on the detection coordinates
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                face = frame[y:y+h, x:x+w]
                
                # Predict if the face is real or fake using PyTorch model
                face_resized = cv2.resize(face, (224, 224))  # Assuming model input size is 224x224
                face_normalized = np.expand_dims(face_resized, axis=0) / 255.0  # Normalize the image
                face_tensor = torch.tensor(face_normalized).permute(0, 3, 1, 2).float()  # Convert to tensor

                # Perform inference
                with torch.no_grad():  # Disable gradient calculation for inference
                    prediction = model(face_tensor)

                # Assuming the model outputs a binary classification (real or fake)
                label = 'Real' if prediction.item() > 0.5 else 'Fake'
                cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        # Convert frame to JPEG for web display
        _, jpeg_frame = cv2.imencode('.jpg', frame)
        frame_bytes = jpeg_frame.tobytes()
        
        # Return as Response to web
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

# Route for index page
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle video processing
@app.route('/process_video', methods=['POST'])
def process_video_route():
    url = request.form.get('url')
    if not url:
        return jsonify({'error': 'No URL provided'}), 400

    try:
        video_path = download_youtube_video(url)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    return jsonify({'status': 'Video processing started', 'video_path': video_path}), 200

# Route to stream the video and processed face detection
@app.route('/video_feed')
def video_feed():
    url = request.args.get('url')
    if not url:
        return jsonify({'error': 'No video URL provided'}), 400

    video_path = download_youtube_video(url)
    return Response(process_video(video_path),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
