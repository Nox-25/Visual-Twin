import os
import glob

def load_mvtec_category(base_path):
    """
    Parses MVTec AD format directory structure.
    Expected structure:
      base_path/train/good/*.png
      base_path/test/good/*.png
      base_path/test/<defect_type>/*.png
      base_path/ground_truth/<defect_type>/*_mask.png

    Returns a list of dicts:
    [
        {
            "image_path": str,
            "mask_path": str or None,
            "label": int (0 for good, 1 for defect),
            "type": str ("good" or defect_type)
        }, ...
    ]
    """
    if not os.path.isdir(base_path):
        raise ValueError(f"Directory not found: {base_path}. Please provide a valid MVTec AD category folder.")

    test_dir = os.path.join(base_path, "test")
    gt_dir = os.path.join(base_path, "ground_truth")

    if not os.path.isdir(test_dir):
        raise ValueError(f"Expected 'test' subdirectory not found in {base_path}")

    dataset = []

    # Iterate over all subdirectories in 'test' (e.g., 'good', 'scratch', 'bent_lead', etc.)
    for defect_type in os.listdir(test_dir):
        defect_path = os.path.join(test_dir, defect_type)
        if not os.path.isdir(defect_path):
            continue

        label = 0 if defect_type == "good" else 1

        # Get all images
        for img_path in glob.glob(os.path.join(defect_path, "*.*")):
            if not (img_path.lower().endswith('.png') or img_path.lower().endswith('.jpg') or img_path.lower().endswith('.jpeg')):
                continue

            mask_path = None
            if label == 1:
                # Deduce mask path. E.g., '000.png' -> '000_mask.png'
                img_filename = os.path.basename(img_path)
                name, ext = os.path.splitext(img_filename)
                mask_filename = f"{name}_mask{ext}"
                potential_mask_path = os.path.join(gt_dir, defect_type, mask_filename)

                if os.path.exists(potential_mask_path):
                    mask_path = potential_mask_path
                else:
                    print(f"Warning: Ground truth mask not found for {img_path}")

            dataset.append({
                "image_path": img_path,
                "mask_path": mask_path,
                "label": label,
                "type": defect_type
            })

    return dataset
