from flask import Flask, jsonify, request, Response, render_template
import threading
import cv2
import time
import os
from core.camera import get_camera
from core.inference import VisualTwinInference
from core.digital_twin import DigitalTwin

app = Flask(__name__)

# Ensure data directories exist
os.makedirs("data/images", exist_ok=True)
os.makedirs("data/snapshots", exist_ok=True)

# Global instances
camera = get_camera(fallback_folder="data/images")
inference = VisualTwinInference()
digital_twin = DigitalTwin()

# Global state for sharing between background thread and API
current_frame = None
current_heatmap_frame = None
current_predictions = []
current_prediction_idx = 0
inference_running = True
processing_fps = 0

def inference_loop():
    global current_frame, current_heatmap_frame, current_predictions, current_prediction_idx, processing_fps

    frame_count = 0
    start_time = time.time()

    while inference_running:
        frame = camera.get_frame()
        if frame is None:
            time.sleep(0.1)
            continue

        digital_twin.set_state("inspecting")

        # BGR to RGB for PIL/CLIP
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        try:
            # Run inference
            ontology = digital_twin.active_ontology
            probs, heatmap = inference.get_class_specific_heatmap(rgb_frame, ontology, target_idx=current_prediction_idx)

            # Find max probability class
            max_idx = probs.argmax()
            max_prob = probs[max_idx]
            detected_class = ontology[max_idx]

            # Format predictions for API
            preds = [{"class": ontology[i], "prob": float(probs[i])} for i in range(len(ontology))]
            preds.sort(key=lambda x: x["prob"], reverse=True)
            current_predictions = preds

            # Determine if defect (assume the class containing "normal" or "no defect" is the healthy state,
            # otherwise fallback to index 0 if not found)
            normal_idx = 0
            for i, desc in enumerate(ontology):
                if "normal" in desc.lower() or "no defect" in desc.lower():
                    normal_idx = i
                    break

            is_defect = (max_idx != normal_idx and max_prob > 0.5)

            if is_defect:
                # Save frame temporarily for reference
                frame_ref = f"frame_{int(time.time()*1000)}.jpg"
                cv2.imwrite(os.path.join("data/images", frame_ref), frame)
                digital_twin.log_anomaly(detected_class, max_prob, frame_ref)
            else:
                if digital_twin.current_state == "defect_detected":
                    # For simplicity, if we detect a normal frame after a defect, we can auto-resolve,
                    # or stay in defect_detected until manually resolved. Let's auto-resolve for visual demo.
                    pass # digital_twin.set_state("resolved")

            # Generate overlay (using heatmap from the highest predicted defect class if we wanted,
            # but currently targeting `current_prediction_idx` which we can update)
            # Default to tracking the max probability class if it's a defect, or keep the requested one
            target_idx = max_idx if is_defect else 1 # Default heatmap to the first defect class if normal
            if target_idx >= len(ontology): target_idx = len(ontology) - 1

            # re-compute heatmap for the highest defect class if different
            if target_idx != current_prediction_idx:
                _, heatmap = inference.get_class_specific_heatmap(rgb_frame, ontology, target_idx=target_idx)

            overlay = inference.overlay_heatmap(frame, heatmap, alpha=0.5)

            # Update globals for streaming
            current_frame = frame.copy()
            current_heatmap_frame = overlay.copy()

            # Calculate FPS
            frame_count += 1
            if frame_count % 10 == 0:
                elapsed = time.time() - start_time
                processing_fps = frame_count / elapsed
                frame_count = 0
                start_time = time.time()

            time.sleep(0.05) # Prevent maxing out CPU

        except Exception as e:
            print(f"Inference error: {e}")
            time.sleep(1)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/state', methods=['GET'])
def get_state():
    state = {
        "digital_twin": {
            "station_id": digital_twin.station_id,
            "current_state": digital_twin.current_state,
            "active_ontology": digital_twin.active_ontology,
            "recent_anomalies": digital_twin.anomaly_log[-5:] # Last 5 anomalies
        },
        "inference": {
            "predictions": current_predictions,
            "fps": round(processing_fps, 1)
        }
    }
    return jsonify(state)


@app.route('/api/ontology', methods=['POST'])
def update_ontology():
    data = request.json
    action = data.get('action')

    if action == 'add':
        desc = data.get('description')
        if desc:
            digital_twin.add_ontology_class(desc)
    elif action == 'remove':
        idx = data.get('index')
        if idx is not None:
            digital_twin.remove_ontology_class(idx)
    elif action == 'update':
        descriptions = data.get('descriptions')
        if descriptions and isinstance(descriptions, list):
            digital_twin.update_ontology(descriptions)

    return jsonify({"status": "success", "active_ontology": digital_twin.active_ontology})


@app.route('/api/snapshot', methods=['POST'])
def trigger_snapshot():
    filepath = digital_twin.save()
    return jsonify({"status": "success", "filepath": filepath})


def gen_frames(stream_type="raw"):
    global current_frame, current_heatmap_frame
    while True:
        frame = current_heatmap_frame if stream_type == "heatmap" else current_frame

        if frame is None:
            time.sleep(0.1)
            continue

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        time.sleep(0.1) # Limit streaming frame rate


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames("raw"), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/heatmap_feed')
def heatmap_feed():
    return Response(gen_frames("heatmap"), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    # Start inference thread
    thread = threading.Thread(target=inference_loop, daemon=True)
    thread.start()

    app.run(host='0.0.0.0', port=5000, threaded=True)
