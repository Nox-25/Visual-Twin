import json
import time
import os
import uuid
import datetime

class DigitalTwin:
    def __init__(self, station_id="VisualTwin_Station_1", snapshot_dir="data/snapshots"):
        self.station_id = station_id
        self.snapshot_dir = snapshot_dir
        os.makedirs(self.snapshot_dir, exist_ok=True)

        # State machine: "idle", "inspecting", "defect_detected", "resolved"
        self.current_state = "idle"
        self.state_transition_history = []
        self._record_transition("idle")

        # Living Ontology
        self.active_ontology = [
            "a photo of a normal, defect-free surface",
            "a photo of a surface with a scratch",
            "a photo of a surface with a crack",
            "a photo of a surface with corrosion or rust"
        ]

        self.anomaly_log = []

        self.session_metadata = {
            "model_version": "openai/clip-vit-base-patch32",
            "explainability_method": "Class-Specific Attention Rollout",
            "session_start_time": self._get_timestamp()
        }

    def _get_timestamp(self):
        return datetime.datetime.now().isoformat()

    def _record_transition(self, new_state):
        if self.current_state != new_state or not self.state_transition_history:
            self.current_state = new_state
            self.state_transition_history.append({
                "timestamp": self._get_timestamp(),
                "state": new_state
            })

    def set_state(self, new_state):
        if new_state in ["idle", "inspecting", "defect_detected", "resolved"]:
            self._record_transition(new_state)

    def update_ontology(self, descriptions):
        """Replaces the active ontology with a new list of descriptions."""
        self.active_ontology = descriptions

    def add_ontology_class(self, description):
        """Adds a single description if it doesn't exist."""
        if description not in self.active_ontology:
            self.active_ontology.append(description)

    def remove_ontology_class(self, idx):
        """Removes a description by index."""
        if 0 <= idx < len(self.active_ontology):
            self.active_ontology.pop(idx)

    def log_anomaly(self, detected_class, confidence_score, frame_ref):
        """Logs an anomaly and updates the state machine."""
        log_entry = {
            "timestamp": self._get_timestamp(),
            "detected_class": detected_class,
            "confidence_score": float(confidence_score),
            "frame_ref": frame_ref,
            "explainability_method_used": self.session_metadata["explainability_method"]
        }
        self.anomaly_log.append(log_entry)
        self.set_state("defect_detected")
        return log_entry

    def save(self):
        """Saves a JSON snapshot of the twin's complete state."""
        snapshot = {
            "station_id": self.station_id,
            "timestamp": self._get_timestamp(),
            "current_state": self.current_state,
            "active_ontology": self.active_ontology,
            "anomaly_log": self.anomaly_log,
            "state_transition_history": self.state_transition_history,
            "session_metadata": self.session_metadata
        }

        filename = f"snapshot_{int(time.time())}_{uuid.uuid4().hex[:6]}.json"
        filepath = os.path.join(self.snapshot_dir, filename)

        with open(filepath, 'w') as f:
            json.dump(snapshot, f, indent=4)

        return filepath
