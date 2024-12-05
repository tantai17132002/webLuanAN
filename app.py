import os
import cv2
from flask import Flask, render_template, Response, request, redirect, url_for, send_from_directory
from ultralytics import YOLO

app = Flask(__name__)

# Load the YOLOv8 model
model = YOLO("best.pt")
names = model.model.names

# Định nghĩa thư mục gốc
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOADS_DIR = os.path.join(BASE_DIR, 'uploads')
RESULTS_DIR = os.path.join(BASE_DIR, 'static', 'results')

# Tạo thư mục nếu chưa tồn tại
def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

ensure_directory_exists(UPLOADS_DIR)
ensure_directory_exists(RESULTS_DIR)

# Hàm tạo tên tệp duy nhất
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

@app.route('/start_webcam')
def start_webcam():
    return render_template('webcam.html')

@app.route('/display_image/<filename>')
def display_image(filename):
    return render_template('display_image.html', filename=filename)

@app.route('/uploads/<filename>')
def play_video(filename):
    return render_template('play_video.html', filename=filename)

@app.route('/video/<path:filename>')
def send_video(filename):
    return send_from_directory(UPLOADS_DIR, filename)

def resize_frame(frame, size=(640, 640)):
    return cv2.resize(frame, size)

def detect_objects_from_webcam():
    count = 0
    cap = cv2.VideoCapture(0)  # 0 for the default webcam
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        count += 1
        if count % 2 != 0:
            continue

        # Resize frame to 640x640
        frame = resize_frame(frame)

        results = model.track(frame, persist=True)

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.int().cpu().tolist()
            class_ids = results[0].boxes.cls.int().cpu().tolist()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            font_scale = max(0.5, min(frame.shape[1] / 1000, 1.5))
            thickness = max(1, int(frame.shape[1] / 500))

            for box, class_id, track_id in zip(boxes, class_ids, track_ids):
                c = names[class_id]
                x1, y1, x2, y2 = box
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(frame.shape[1], x2), min(frame.shape[0], y2)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), thickness)

                label_size = cv2.getTextSize(f'{track_id} - {c}', cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
                label_x1, label_y1 = x1, max(0, y1 - label_size[1] - 10)
                label_x2, label_y2 = x1 + label_size[0], y1

                cv2.rectangle(frame, (label_x1, label_y1), (label_x2, label_y2), (0, 0, 0), -1)
                cv2.putText(frame, f'{track_id} - {c}', (x1, max(10, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/webcam_feed')
def webcam_feed():
    return Response(detect_objects_from_webcam(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

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
    img = resize_frame(img)

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

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    # Lưu video gốc vào thư mục `uploads`
    file_path = os.path.join(UPLOADS_DIR, get_unique_filename(UPLOADS_DIR, file.filename))
    file.save(file_path)

    # Xử lý video và lưu kết quả vào `static/results`
    result_filename = detect_objects_and_save_video(file_path)

    # Chuyển hướng tới trang phát video kết quả
    return redirect(url_for('play_video', filename=result_filename))

def detect_objects_and_save_video(video_path):
    cap = cv2.VideoCapture(video_path)

    # Lấy thông tin video gốc
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    if fps == 0:  # Xử lý trường hợp không lấy được FPS
        fps = 30

    # Đường dẫn lưu video kết quả
    result_filename = get_unique_filename(RESULTS_DIR, os.path.basename(video_path))
    result_path = os.path.join(RESULTS_DIR, result_filename)

    # Khởi tạo bộ ghi video đầu ra
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(result_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame (nếu cần đảm bảo kích thước phù hợp với video)
        if frame.shape[1] != width or frame.shape[0] != height:
            frame = cv2.resize(frame, (width, height))

        # Dự đoán đối tượng
        results = model.track(frame, persist=True)

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.int().cpu().tolist()
            class_ids = results[0].boxes.cls.int().cpu().tolist()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            # Vẽ bounding box và nhãn lên frame
            for box, class_id, track_id in zip(boxes, class_ids, track_ids):
                class_name = names[class_id]
                x1, y1, x2, y2 = box
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(frame.shape[1], x2), min(frame.shape[0], y2)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{track_id} - {class_name}"
                cv2.putText(frame, label, (x1, max(10, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Ghi khung hình vào video kết quả
        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Trả về tên file kết quả
    return os.path.basename(result_path)

@app.route('/video_feed/<filename>')
def video_feed(filename):
    video_path = os.path.join(UPLOADS_DIR, filename)
    return Response(detect_objects_from_video(video_path), # type: ignore
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run('0.0.0.0', debug=False, port=8080)
