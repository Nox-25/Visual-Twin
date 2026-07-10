import argparse
import os
import cv2
import numpy as np
import csv
from sklearn.metrics import roc_auc_score
from dataset_loader import load_mvtec_category
from clip_model import ClipModelWrapper
from heatmap_generator import HeatmapGenerator
from prompts import NORMAL_PROMPTS, DEFECT_PROMPTS

def evaluate_dataset(data_path):
    category = os.path.basename(os.path.normpath(data_path))
    print(f"Loading MVTec AD category: {category} from {data_path}")

    try:
        dataset = load_mvtec_category(data_path)
    except ValueError as e:
        print(f"Error: {e}")
        return

    if not dataset:
        print("No test images found.")
        return

    print(f"Found {len(dataset)} test images.")

    print("Initializing CLIP Model and Heatmap Generator...")
    clip_wrapper = ClipModelWrapper()
    generator = HeatmapGenerator(clip_wrapper, NORMAL_PROMPTS, DEFECT_PROMPTS)

    image_labels = []
    image_scores = []

    pixel_labels = []
    pixel_scores = []

    for i, item in enumerate(dataset):
        img_path = item["image_path"]
        mask_path = item["mask_path"]
        label = item["label"]

        print(f"[{i+1}/{len(dataset)}] Processing {os.path.basename(img_path)} (Class: {item['type']})...")

        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to read image: {img_path}")
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Generate heatmap and scores
        heatmap_norm, binary_mask, img_score = generator.generate(img_rgb)

        image_labels.append(label)
        image_scores.append(img_score)

        # For pixel level metrics (only if we have ground truth, or if it's a good image we treat all pixels as 0)
        # Note: True AUPRO is quite complex (requires region-wise overlap thresholds).
        # TODO: Implement proper AUPRO metric here in the future (e.g. referencing the anomalib library's approach).
        # For now, we compute simplified pixel-level AUROC.

        if label == 0:
            gt_mask = np.zeros(img.shape[:2], dtype=np.uint8)
        elif mask_path and os.path.exists(mask_path):
            gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if gt_mask is None:
                 gt_mask = np.zeros(img.shape[:2], dtype=np.uint8)
                 print(f"Failed to load mask: {mask_path}")
        else:
            # Mask missing for a defect, skip pixel evaluation for this image
            continue

        # Ensure sizes match (sometimes masks are strictly binary, sometimes 0-255)
        if gt_mask.shape != heatmap_norm.shape:
             gt_mask = cv2.resize(gt_mask, (heatmap_norm.shape[1], heatmap_norm.shape[0]), interpolation=cv2.INTER_NEAREST)

        gt_mask_binary = (gt_mask > 127).astype(np.int32).flatten()
        heatmap_flat = (heatmap_norm / 255.0).flatten()

        # To avoid exploding memory, we could subsample pixels, but for typical PCB images (e.g., 1024x1024)
        # flattening and accumulating might blow up RAM over 100+ images.
        # Instead, we'll accumulate a fixed number of pixels per image or evaluate per image and average.
        # Let's subsample 10,000 pixels per image to keep memory bounds safe while getting an estimate.
        sub_idx = np.random.choice(len(gt_mask_binary), min(10000, len(gt_mask_binary)), replace=False)
        pixel_labels.extend(gt_mask_binary[sub_idx])
        pixel_scores.extend(heatmap_flat[sub_idx])

    # Calculate metrics
    image_auroc = roc_auc_score(image_labels, image_scores) if len(set(image_labels)) > 1 else 0.0

    pixel_auroc = 0.0
    if len(pixel_labels) > 0 and len(set(pixel_labels)) > 1:
        pixel_auroc = roc_auc_score(pixel_labels, pixel_scores)

    # Output to CSV
    csv_file = f"results_{category}.csv"
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["category", "image_auroc", "pixel_auroc", "aupro"])
        writer.writerow([category, f"{image_auroc:.4f}", f"{pixel_auroc:.4f}", "TODO"])

    print("\n--- Evaluation Complete ---")
    print(f"Category: {category}")
    print(f"Image AUROC: {image_auroc:.4f}")
    print(f"Pixel AUROC (Simplified): {pixel_auroc:.4f}")
    print(f"AUPRO: TODO (Implementation pending)")
    print(f"Results saved to {csv_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate VisualTwin zero-shot CLIP pipeline.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to MVTec AD category folder (e.g. /path/to/transistor)")
    args = parser.parse_args()

    evaluate_dataset(args.data_path)
