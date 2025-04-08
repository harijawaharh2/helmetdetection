from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
import cv2
import os
from ultralytics import YOLO

app = Flask(__name__, static_folder="static")
CORS(app)

# Load YOLOv8 Model
yolo_model = YOLO("trained_yolov8_model_intruder_v10_run1 (1).pt")

# Create 'output' directory if it doesn't exist
os.makedirs("output", exist_ok=True)

def is_image(file_name):
    return file_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))

def is_video(file_name):
    return file_name.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))

@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/process", methods=["POST"])
def process_media():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    filename = file.filename or "live_feed_frame.jpg"
    file_path = os.path.join("output", filename)
    file.save(file_path)

    if is_image(filename):
        return process_image(file_path)
    elif is_video(filename):
        return process_video(file_path)
    else:
        return jsonify({"error": "Unsupported file format! Upload an image or video."}), 400

def process_image(file_path):
    image = cv2.imread(file_path)
    results = yolo_model(image)
    helmet_count = 0
    bike_count = 0

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = f"{yolo_model.names[cls]}: {conf:.2f}"
            color = (0, 255, 0) if cls == 0 else (0, 0, 255)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, label, (x2, y2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            if cls == 0:
                helmet_count += 1
            elif cls == 1:
                bike_count += 1

    output_image_path = os.path.join("output", "processed_image.jpg")
    cv2.imwrite(output_image_path, image)

    return jsonify({
        "output_file": f"/output/processed_image.jpg?t={int(os.path.getmtime(output_image_path))}",
        "total_helmets_yolo": helmet_count,
        "bikes_detected": bike_count,
        "warning": "⚠ Warning: More than 2 helmets detected!" if helmet_count > 2 else "Safe"
    })

def process_video(file_path):
    cap = cv2.VideoCapture(file_path)
    frame_width, frame_height = int(cap.get(3)), int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    output_video_path = os.path.join("output", "processed_video.mp4")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    total_helmets_yolo = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = yolo_model(frame)
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = f"{yolo_model.names[cls]}: {conf:.2f}"
                color = (0, 255, 0) if cls == 0 else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                if cls == 0:
                    total_helmets_yolo += 1

        out.write(frame)

    cap.release()
    out.release()

    if not os.path.exists(output_video_path):
        return jsonify({"error": "Video processing failed!"}), 500

    return jsonify({
        "output_file": f"/output/processed_video.mp4?t={int(os.path.getmtime(output_video_path))}",
        "total_helmets_yolo": total_helmets_yolo,
        "warning": "⚠ Warning: More than 2 helmets detected!" if total_helmets_yolo > 2 else "Safe"
    })

@app.route("/output/<path:filename>")
def serve_output(filename):
    file_path = os.path.join("output", filename)
    if not os.path.exists(file_path):
        return jsonify({"error": "File not found"}), 404

    return send_file(file_path, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5050)