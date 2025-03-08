from flask import Flask, render_template, Response, request
from vidgear.gears import CamGear
import cv2
import mediapipe as mp
import torch
import torch.nn as nn
from torchvision import models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize Flask app
app = Flask(__name__)

# Define the model architecture and load weights
def load_model(pth_file_path):
    model = models.resnet50(pretrained=False)  # Set pretrained to False when loading custom weights

    # Modify the fully connected (fc) layer for binary classification
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 1),
        nn.Sigmoid()
    )

    # Load the state dictionary
    state_dict = torch.load(pth_file_path, map_location=device)
    model.load_state_dict(state_dict)

    # Set the model to evaluation mode
    model.eval()
    return model

# Load the model
pth_file_path = '/content/deepfake_resnest50.pth'
model = load_model(pth_file_path).to(device)

# Define the prediction function
def predict(input_tensor):
    """
    Predict whether input corresponds to class 0 (Real) or 1 (Fake).

    Args:
        input_tensor (torch.Tensor): The input image tensor, shape (1, 3, 224, 224).

    Returns:
        int: Predicted class (0 for Real, 1 for Fake).
        float: Confidence score (probability of the predicted class).
    """
    with torch.no_grad():
        # Pass the input through the model
        outputs = model(input_tensor)

        # Apply sigmoid activation to get probability
        probability = outputs.squeeze().item()  # Squeeze removes redundant dimensions

        # Classify based on threshold: real (0) if probability < 0.5, fake (1) otherwise
        predicted_class = int(probability >= 0.5)

        # Calculate confidence score
        confidence_score = probability if predicted_class == 1 else 1 - probability

    return predicted_class, confidence_score

# Initialize MediaPipe face detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.9)

# Default YouTube URL for initial stream
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

# Function to process video frames with face detection, cropping, and prediction
def generate_frames():
    while True:
        frame = video_stream.read()
        if frame is None:
            continue
        
        # Resize the frame for better fit
        frame = cv2.resize(frame, (640, 360))

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

                # Crop the face from the frame
                face_crop = frame[y1:y2, x1:x2]
                
                # Resize the cropped face to match the model input size
                face_resized = cv2.resize(face_crop, (224, 224))
                
                # Convert the image to a tensor and normalize
                face_tensor = torch.tensor(face_resized).permute(2, 0, 1).unsqueeze(0) / 255.0
                face_tensor = face_tensor.to(device)
                
                # Predict whether the face is real or fake
                predicted_class, confidence_score = predict(face_tensor)
                
                # Set rectangle color based on prediction
                color = (0, 255, 0) if predicted_class == 0 else (0, 0, 255)  # Green for real, Red for fake
                
                # Draw bounding box around the face
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Display confidence score as a label
                label = f"{'Real' if predicted_class == 0 else 'Fake'}: {confidence_score:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

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
