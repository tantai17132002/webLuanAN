import os
import cv2
from flask import Flask, render_template, Response, request, redirect, url_for, send_from_directory
from ultralytics import YOLO

app = Flask(__name__)

# Load the YOLOv8 model
model = YOLO("best.pt")
names = model.model.names

# Define the base and upload directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOADS_DIR = os.path.join(BASE_DIR, 'uploads')
RESULTS_DIR = os.path.join(BASE_DIR, 'static', 'results')

# Create directories if they do not exist
def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

ensure_directory_exists(UPLOADS_DIR)
ensure_directory_exists(RESULTS_DIR)

# Function to generate a unique filename
def get_unique_filename(directory, filename):
    base, ext = os.path.splitext(filename)
    counter = 1
    unique_filename = filename

    while os.path.exists(os.path.join(directory, unique_filename)):
        unique_filename = f"{base}_{counter}{ext}"
        counter += 1

    return unique_filename

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/display_image/<filename>')
def display_image(filename):
    return render_template('display_image.html', filename=filename)

@app.route('/uploads/<filename>')
def play_video(filename):
    return render_template('play_video.html', filename=filename)

@app.route('/video/<path:filename>')
def send_video(filename):
    return send_from_directory('uploads', filename)

def resize_frame(frame, size=None):
    """ Resize the frame if size is provided """
    if size is not None:
        return cv2.resize(frame, size)
    return frame

@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    file_path = os.path.join(UPLOADS_DIR, get_unique_filename(UPLOADS_DIR, file.filename))
    file.save(file_path)

    img = cv2.imread(file_path)
    # Perform object detection on the image
    results = model.predict(img)

    if results[0].boxes is not None:
        boxes = results[0].boxes.xyxy.int().cpu().tolist()
        class_ids = results[0].boxes.cls.int().cpu().tolist()

        for box, class_id in zip(boxes, class_ids):
            c = names[class_id]
            x1, y1, x2, y2 = box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, c, (x1, max(10, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

    output_path = os.path.join(RESULTS_DIR, get_unique_filename(RESULTS_DIR, file.filename))
    cv2.imwrite(output_path, img)

    return redirect(url_for('display_image', filename=os.path.basename(output_path)))

@app.route('/upload_video', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    file_path = os.path.join(UPLOADS_DIR, get_unique_filename(UPLOADS_DIR, file.filename))
    file.save(file_path)

    return redirect(url_for('play_video', filename=os.path.basename(file_path)))

def detect_objects_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform object detection on the frame
        results = model.track(frame)  # Use track for tracking instead of predict

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.int().cpu().tolist()
            class_ids = results[0].boxes.cls.int().cpu().tolist()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            for box, class_id, track_id in zip(boxes, class_ids, track_ids):
                class_name = names[class_id]
                x1, y1, x2, y2 = box
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(frame.shape[1], x2), min(frame.shape[0], y2)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{track_id} - {class_name}"
                cv2.putText(frame, label, (x1, max(10, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/video_feed/<filename>')
def video_feed(filename):
    video_path = os.path.join(UPLOADS_DIR, filename)
    return Response(detect_objects_from_video(video_path),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run('0.0.0.0', debug=False, port=8080)
