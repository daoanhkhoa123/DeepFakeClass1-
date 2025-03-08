from flask import Flask, render_template, Response
from vidgear.gears import CamGear
import cv2

app = Flask(__name__)

from vidgear.gears import CamGear
import cv2
youtube_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  


# Video stream source (YouTube video URL)
video_stream = CamGear(
    source=youtube_url, 
    stream_mode=True,
    logging=True
).start()


# Route for index page
@app.route('/')
def index():
    return render_template('index.html')

# Generate video frames and stream
def generate_frames():
    while True:
        frame = video_stream.read()
        if frame is None:
            continue

        # Encode the frame for streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Route for video feed
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(port=8000,debug=True)
