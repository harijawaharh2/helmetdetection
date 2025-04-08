'''from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
import cv2
import os
from ultralytics import YOLO
from transformers import DetrForObjectDetection, DetrFeatureExtractor
from PIL import Image
import torch

# Initialize Flask app
app = Flask(__name__, static_folder="static")
CORS(app)

# Create 'output' directory if it doesn't exist
os.makedirs("output", exist_ok=True)

# Load DETR model and feature extractor
detr_model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
detr_model.load_state_dict(torch.load("detr50_helmet_final_30EPOCH.pth", map_location=torch.device('cpu')))
detr_model.eval()
feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50")

# Load YOLOv8 model
yolo_model = YOLO("v12march3.pt")

# Helper functions
def is_image(file_name):
    return file_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))

def detect_with_detr(image_path, model, feature_extractor, class_names):
    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    inputs = feature_extractor(images=image, return_tensors="pt")

    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)

    # Process predictions
    probas = outputs.logits.softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > 0.9
    bboxes_scaled = outputs.pred_boxes[0, keep].detach().numpy()

    # Convert bounding boxes to pixel coordinates
    width, height = image.size
    bboxes_scaled[:, 0] = bboxes_scaled[:, 0] * width  # x_center
    bboxes_scaled[:, 1] = bboxes_scaled[:, 1] * height  # y_center
    bboxes_scaled[:, 2] = bboxes_scaled[:, 2] * width  # width
    bboxes_scaled[:, 3] = bboxes_scaled[:, 3] * height  # height

    # Extract labels
    labels = []
    for i in range(len(bboxes_scaled)):
        predicted_class_index = probas[keep][i].argmax().item()
        if predicted_class_index < len(class_names):
            labels.append(class_names[predicted_class_index])
        else:
            labels.append("Unknown")

    return bboxes_scaled, labels

def detect_with_yolov8(image_path, model):
    image = cv2.imread(image_path)
    results = model(image)

    bboxes = []
    labels = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = f"{model.names[cls]}: {conf:.2f}"
            bboxes.append([x1, y1, x2, y2])
            labels.append(label)

    return bboxes, labels

def combine_detections(detr_bboxes, detr_labels, yolov8_bboxes, yolov8_labels):
    combined_bboxes = []
    combined_labels = []

    # Add DETR detections
    for bbox, label in zip(detr_bboxes, detr_labels):
        x_center, y_center, bbox_w, bbox_h = bbox
        x_min = int(x_center - bbox_w / 2)
        y_min = int(y_center - bbox_h / 2)
        x_max = int(x_center + bbox_w / 2)
        y_max = int(y_center + bbox_h / 2)
        combined_bboxes.append([x_min, y_min, x_max, y_max])
        combined_labels.append(f"DETR: {label}")

    # Add YOLOv8 detections
    for bbox, label in zip(yolov8_bboxes, yolov8_labels):
        combined_bboxes.append(bbox)
        combined_labels.append(f"YOLOv8: {label}")

    return combined_bboxes, combined_labels

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/process", methods=["POST"])
def process_media():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    filename = file.filename or "uploaded_media.mp4"
    file_path = os.path.join("output", filename)
    file.save(file_path)

    # Check if it's an image or video
    if is_image(filename):
        return process_image(file_path)
    elif filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        return process_video(file_path)
    else:
        return jsonify({"error": "Unsupported file format! Upload an image or video."}), 400

def process_image(file_path):
    """Handles image processing"""
    COCO_CLASSES = ['Helmet', 'NOHelmet']
    detr_bboxes, detr_labels = detect_with_detr(file_path, detr_model, feature_extractor, COCO_CLASSES)
    yolov8_bboxes, yolov8_labels = detect_with_yolov8(file_path, yolo_model)

    # Combine results
    combined_bboxes, combined_labels = combine_detections(detr_bboxes, detr_labels, yolov8_bboxes, yolov8_labels)

    # Draw bounding boxes on the image
    image = cv2.imread(file_path)
    for i, bbox in enumerate(combined_bboxes):
        x_min, y_min, x_max, y_max = bbox
        label = combined_labels[i]
        color = (0, 255, 0) if "Helmet" in label else (0, 0, 255)
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
        cv2.putText(image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    output_image_path = os.path.join("output", "processed_image_combined.jpg")
    cv2.imwrite(output_image_path, image)

    return jsonify({
        "output_file": f"/output/processed_image_combined.jpg?t={int(os.path.getmtime(output_image_path))}",
        "detected_objects": combined_labels
    })

def process_video(video_path):
    """Handles video processing frame-by-frame"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return jsonify({"error": "Could not open video"}), 500

    output_video_path = os.path.join("output", "processed_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Stop processing when video ends

        # Save frame as an image and process it
        temp_frame_path = "output/temp_frame.jpg"
        cv2.imwrite(temp_frame_path, frame)

        # Run DETR and YOLO
        COCO_CLASSES = ['Helmet', 'NOHelmet']
        detr_bboxes, detr_labels = detect_with_detr(temp_frame_path, detr_model, feature_extractor, COCO_CLASSES)
        yolov8_bboxes, yolov8_labels = detect_with_yolov8(temp_frame_path, yolo_model)

        # Combine detections
        combined_bboxes, combined_labels = combine_detections(detr_bboxes, detr_labels, yolov8_bboxes, yolov8_labels)

        # Draw bounding boxes
        for i, bbox in enumerate(combined_bboxes):
            x_min, y_min, x_max, y_max = bbox
            label = combined_labels[i]
            color = (0, 255, 0) if "Helmet" in label else (0, 0, 255)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
            cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Write processed frame to output video
        out.write(frame)

    cap.release()
    out.release()

    return jsonify({
        "output_file": f"/output/processed_video.mp4?t={int(os.path.getmtime(output_video_path))}",
        "message": "Video processed successfully!"
    })


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)

'''
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import cv2
import os
from ultralytics import YOLO
from transformers import DetrForObjectDetection, DetrFeatureExtractor
from PIL import Image
import torch
import time  # Import for timestamping output files

# Initialize Flask app
app = Flask(__name__, static_folder="output")
CORS(app)

# Create 'output' directory if it doesn't exist
os.makedirs("output", exist_ok=True)

# Load DETR model and feature extractor
detr_model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
detr_model.load_state_dict(torch.load("detr50_helmet_final_30EPOCH.pth", map_location=torch.device('cpu')))
detr_model.eval()
feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50")

# Load YOLOv8 model
yolo_model = YOLO("v12march3.pt")

# Helper functions
def is_image(file_name):
    return file_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))

def convert_detr_bboxes(detr_bboxes, width, height):
    """Convert DETR bounding boxes from [center_x, center_y, width, height] to [x_min, y_min, x_max, y_max]"""
    converted_bboxes = []
    for bbox in detr_bboxes:
        x_center, y_center, bbox_w, bbox_h = bbox
        x_min = max(0, int(x_center - bbox_w / 2))
        y_min = max(0, int(y_center - bbox_h / 2))
        x_max = min(width, int(x_center + bbox_w / 2))
        y_max = min(height, int(y_center + bbox_h / 2))
        converted_bboxes.append([x_min, y_min, x_max, y_max])
    return converted_bboxes

def detect_with_detr(image_path, model, feature_extractor, class_names):
    """Perform object detection using DETR"""
    image = Image.open(image_path).convert('RGB')
    inputs = feature_extractor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    probas = outputs.logits.softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > 0.9
    bboxes_scaled = outputs.pred_boxes[0, keep].detach().numpy()

    width, height = image.size
    converted_bboxes = convert_detr_bboxes(bboxes_scaled, width, height)

    labels = []
    for i in range(len(converted_bboxes)):
        predicted_class_index = probas[keep][i].argmax().item()
        labels.append(class_names[predicted_class_index] if predicted_class_index < len(class_names) else "Unknown")

    return converted_bboxes, labels

def detect_with_yolov8(image_path, model):
    """Perform object detection using YOLOv8"""
    image = cv2.imread(image_path)
    results = model(image)

    bboxes, labels = [], []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = f"{model.names[cls]}: {conf:.2f}"
            bboxes.append([x1, y1, x2, y2])
            labels.append(label)

    return bboxes, labels

def combine_detections(detr_bboxes, detr_labels, yolov8_bboxes, yolov8_labels):
    """Combine DETR and YOLO detections"""
    combined_bboxes = detr_bboxes + yolov8_bboxes
    combined_labels = [f"DETR: {label}" for label in detr_labels] + [f"YOLOv8: {label}" for label in yolov8_labels]
    return combined_bboxes, combined_labels

@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/process", methods=["POST"])
def process_media():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    filename = file.filename or "uploaded_media.mp4"
    file_path = os.path.join("output", filename)
    file.save(file_path)

    if is_image(filename):
        return process_image(file_path)
    elif filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        return process_video(file_path)
    else:
        return jsonify({"error": "Unsupported file format! Upload an image or video."}), 400

def process_image(file_path):
    """Handles image processing"""
    COCO_CLASSES = ['Helmet', 'NOHelmet']
    detr_bboxes, detr_labels = detect_with_detr(file_path, detr_model, feature_extractor, COCO_CLASSES)
    yolov8_bboxes, yolov8_labels = detect_with_yolov8(file_path, yolo_model)

    combined_bboxes, combined_labels = combine_detections(detr_bboxes, detr_labels, yolov8_bboxes, yolov8_labels)

    image = cv2.imread(file_path)
    for i, bbox in enumerate(combined_bboxes):
        x_min, y_min, x_max, y_max = bbox
        color = (0, 255, 0) if "Helmet" in combined_labels[i] else (0, 0, 255)
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
        cv2.putText(image, combined_labels[i], (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    output_image_path = os.path.join("output", "processed_image_combined.jpg")
    cv2.imwrite(output_image_path, image)

    return jsonify({
        "output_file": f"/output/processed_image_combined.jpg?t={int(time.time())}",
        "detected_objects": combined_labels
    })

def process_video(video_path):
    """Handles video processing frame-by-frame"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return jsonify({"error": "Could not open video"}), 500  

    output_video_path = os.path.join("output", "processed_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    temp_frame_path = "output/temp_frame.jpg"

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  

        cv2.imwrite(temp_frame_path, frame)

        COCO_CLASSES = ['Helmet', 'NOHelmet']
        detr_bboxes, detr_labels = detect_with_detr(temp_frame_path, detr_model, feature_extractor, COCO_CLASSES)
        yolov8_bboxes, yolov8_labels = detect_with_yolov8(temp_frame_path, yolo_model)

        combined_bboxes, combined_labels = combine_detections(detr_bboxes, detr_labels, yolov8_bboxes, yolov8_labels)

        for i, bbox in enumerate(combined_bboxes):
            x_min, y_min, x_max, y_max = bbox
            color = (0, 255, 0) if "Helmet" in combined_labels[i] else (0, 0, 255)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
            cv2.putText(frame, combined_labels[i], (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        out.write(frame)
        os.remove(temp_frame_path)  

    cap.release()
    out.release()

    return jsonify({
        "output_file": f"/output/processed_video.mp4?t={int(time.time())}",
        "message": "Video processed successfully!"
    })

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)


