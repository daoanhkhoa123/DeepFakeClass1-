from flask import Flask, render_template, Response, request
from vidgear.gears import CamGear
import cv2
import mediapipe as mp

# Initialize Flask app
app = Flask(__name__)

# Initialize MediaPipe face detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.9)

# YouTube URL (initial stream)
youtube_url = "https://www.youtube.com/watch?v=VdAiKJTf5NY"

# Initialize video stream
video_stream = CamGear(
    source=youtube_url,
    stream_mode=True,
    logging=True
).start()

# Route for the index page
@app.route('/', methods=['GET', 'POST'])
def index():
    global video_stream
    if request.method == 'POST':
        new_url = request.form.get('youtube_url')
        if new_url:
            # Stop the current video stream and update it with the new URL
            video_stream.stop()
            video_stream = CamGear(
                source=new_url,
                stream_mode=True,
                logging=True
            ).start()
    
    return render_template('index.html')


# Function to process video frames with face detection and cropping
def generate_frames():
    while True:
        frame = video_stream.read()
        if frame is None:
            continue
        
        # Convert the frame to RGB (MediaPipe works with RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(frame_rgb)

        # If faces are detected
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape

                # Calculate bounding box coordinates
                x1, y1 = int(bboxC.xmin * w), int(bboxC.ymin * h)
                x2, y2 = int((bboxC.xmin + bboxC.width) * w), int((bboxC.ymin + bboxC.height) * h)

                # Ensure coordinates are within frame boundaries
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                # Draw bounding box around the face
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Crop the face from the frame
                face_crop = frame[y1:y2, x1:x2]

        # Encode the frame for streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield the frame for streaming
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# Route for video feed
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


# Run the Flask app
if __name__ == '__main__':
    app.run(port=8000, debug=True)
