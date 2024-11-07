import eventlet
eventlet.monkey_patch()

from flask import Flask, render_template, url_for
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import random
import os
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

app = Flask(__name__, static_folder='static')
socketio = SocketIO(app, cors_allowed_origins="*")


# Sensor Data Simulation Function
def get_sensor_data():
    light = random.uniform(0, 100)
    temperature = random.uniform(-10, 40)
    humidity = random.uniform(0, 100)
    return {"light": light, "temperature": temperature, "humidity": humidity}


# COCO Class Names for YOLOv4
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'N/A', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
    'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'handbag', 'tie',
    'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
    'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
    'bed', 'N/A', 'dining table', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]


# Load YOLO Model
def load_yolo_model():
    weights_path = r"C:\Users\shast\Downloads\yolov4.weights"  # Update with the actual path
    config_path = r"C:\Users\shast\Downloads\yolov4.cfg"  # Update with the actual path
    net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net


# Image Restoration (Placeholder Function)
def restore_image(image):
    # Denoising function (for now)
    restored_image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    return restored_image


# Highlighting Objects in the Restored Image
def highlight_objects(image, net):
    # Prepare the image for object detection
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    layer_names = net.getLayerNames()  # Get the layer names
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]  # Corrected this line
    detections = net.forward(output_layers)

    height, width, channels = image.shape
    class_ids = []
    confidences = []
    boxes = []

    for out in detections:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # Minimum confidence threshold
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Non-maximum Suppression to remove overlapping boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw bounding boxes and add class names
    for i in range(len(boxes)):
        if i in indices:
            x, y, w, h = boxes[i]
            label = COCO_CLASSES[class_ids[i]]
            color = (0, 255, 0)  # Green color for bounding boxes
            thickness = 4  # Increase the thickness of the bounding box
            cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)

            # Add the class name label above the bounding box
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(image, label, (x, y - 10), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    return image


# Calculate PSNR
def calculate_psnr(original_image, restored_image):
    return psnr(original_image, restored_image)


# Calculate SSIM
def calculate_ssim(original_image, restored_image):
    original_gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    restored_gray = cv2.cvtColor(restored_image, cv2.COLOR_BGR2GRAY)
    return ssim(original_gray, restored_gray)


# Process Image and Detect Objects
def process_image(image_path, net):
    original_image = cv2.imread(image_path)
    if original_image is None:
        return None

    restored_image = restore_image(original_image)

    # Highlight objects in the restored image
    highlighted_restored_image = highlight_objects(restored_image.copy(), net)

    psnr_score = calculate_psnr(original_image, restored_image)
    ssim_score = calculate_ssim(original_image, restored_image)
    sensor_data = get_sensor_data()

    return {
        "original_image": original_image,
        "restored_image": restored_image,
        "highlighted_restored_image": highlighted_restored_image,
        "psnr": psnr_score,
        "ssim": ssim_score,
        "sensor_data": sensor_data
    }


# SocketIO Event for Image Processing Loop
@socketio.on('start_image_processing')
def start_image_processing():
    image_dir = r"C:\Users\shast\Downloads\archive\test"  # Directory where test images are stored
    net = load_yolo_model()

    processed_dir = os.path.join(app.static_folder, 'processed')
    os.makedirs(processed_dir, exist_ok=True)

    for filename in os.listdir(image_dir):
        image_path = os.path.join(image_dir, filename)
        metrics = process_image(image_path, net)
        if metrics:
            original_image_path = os.path.join(processed_dir, f"original_{filename}")
            restored_image_path = os.path.join(processed_dir, f"restored_{filename}")
            highlighted_image_path = os.path.join(processed_dir, f"highlighted_{filename}")

            # Save original, restored, and highlighted images
            cv2.imwrite(original_image_path, metrics["original_image"])
            cv2.imwrite(restored_image_path, metrics["restored_image"])
            cv2.imwrite(highlighted_image_path, metrics["highlighted_restored_image"])

            # Emit real-time data for each processed image
            original_image_url = url_for('static', filename=f'processed/original_{filename}')
            restored_image_url = url_for('static', filename=f'processed/restored_{filename}')
            highlighted_image_url = url_for('static', filename=f'processed/highlighted_{filename}')
            emit('image_data', {
                "psnr": metrics['psnr'],
                "ssim": metrics['ssim'],
                "sensor_data": metrics['sensor_data'],
                "original_image_path": original_image_url,
                "restored_image_path": restored_image_url,
                "highlighted_image_path": highlighted_image_url
            })
            socketio.sleep(3)  # Wait for 3 seconds before processing the next image


# Web Page to Display Images and Metrics
@app.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    socketio.run(app, debug=True)

