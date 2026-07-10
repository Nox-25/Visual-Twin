# VisualTwin: Zero-Shot PCB Defect Detection

VisualTwin is a working prototype for a conference paper demonstrating a zero-shot anomaly detection system for Printed Circuit Boards (PCBs). It uses a pretrained CLIP model (ViT-B-16) to evaluate image patches against normal and defect text prompts without any fine-tuning.

## Features
- **Zero-shot Anomaly Detection**: No training or fine-tuning required. Uses open-vocabulary text embeddings via `open_clip`.
- **Live Camera Pipeline**: Captures from a standard webcam (Logitech C270 recommended).
- **Web Dashboard**: A Flask-powered UI to capture images, generate heatmaps, and see pass/fail verdicts.
- **Evaluation Script**: Benchmarks the algorithm against standard industrial datasets like MVTec AD to produce AUROC metrics.

## Hardware Setup
- **Camera**: Logitech C270 (fixed focus, 720p).
- **Placement**: Mount the camera at a fixed distance (~30-40cm) from the PCB base for consistent focus.
- **Lighting**: Ensure adequate, even lighting to reduce shadows and reflections on metallic traces.

## Installation

1. Clone the repository and navigate to the `visualtwin` folder.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Web App
1. Start the Flask server:
   ```bash
   python app.py
   ```
2. Open your browser and go to `http://localhost:5000`.
3. Click "Capture from Webcam & Inspect" to run the pipeline.

## Evaluation (MVTec AD)

To generate AUROC metrics for your paper, use the `evaluate.py` script against your local copy of the MVTec AD dataset.

```bash
python evaluate.py --data_path /path/to/mvtec/transistor
```

### Dataset Structure Requirement
The script expects the standard MVTec AD directory structure:
```
transistor/
  ├── train/
  │   └── good/
  ├── test/
  │   ├── good/
  │   ├── bent_lead/
  │   └── damaged_case/
  └── ground_truth/
      ├── bent_lead/
      └── damaged_case/
```

Results are saved to a CSV file (e.g. `results_transistor.csv`), containing the Image AUROC and a simplified Pixel AUROC. *Note: A strict implementation of AUPRO requires region-based overlap thresholding and is left as a TODO for future implementations.*
