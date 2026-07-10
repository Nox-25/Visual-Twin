import os
import cv2
import base64
import numpy as np
from flask import Flask, request, jsonify, render_template

from clip_model import ClipModelWrapper
from heatmap_generator import HeatmapGenerator
from prompts import NORMAL_PROMPTS, DEFECT_PROMPTS

app = Flask(__name__)

# Initialize ML components on startup
print("Initializing ML models...")
clip_wrapper = ClipModelWrapper()
generator = HeatmapGenerator(clip_wrapper, NORMAL_PROMPTS, DEFECT_PROMPTS)
THRESHOLD = 0.05 # Configurable threshold for PASS/FAIL (relative similarity score)

def encode_image_base64(img_array):
    _, buffer = cv2.imencode('.jpg', img_array)
    return base64.b64encode(buffer).decode('utf-8')

def process_image(img_bgr):
    """Runs the heatmap generator and formats the response."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Generate Heatmap
    heatmap_norm, binary_mask, score = generator.generate(img_rgb)

    # Create an overlay for visualization
    heatmap_colored = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_bgr, 0.6, heatmap_colored, 0.4, 0)

    verdict = "FAIL" if score > THRESHOLD else "PASS"

    return {
        "original": encode_image_base64(img_bgr),
        "heatmap_overlay": encode_image_base64(overlay),
        "score": float(score),
        "verdict": verdict
    }

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/capture", methods=["GET", "POST"])
def capture():
    # Attempt to open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return jsonify({"error": "Webcam not found or inaccessible."}), 500

    # Read a few frames to let auto-exposure settle
    for _ in range(5):
        cap.read()

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        return jsonify({"error": "Failed to capture image from webcam."}), 500

    # Process the frame
    try:
        result = process_image(frame)
        return jsonify(result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/upload", methods=["POST"])
def upload():
    if 'image' not in request.files:
        return jsonify({"error": "No image part in the request"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        # Read the file as numpy array
        npimg = np.frombuffer(file.read(), np.uint8)
        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        if frame is None:
             return jsonify({"error": "Invalid image file."}), 400

        result = process_image(frame)
        return jsonify(result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
