import numpy as np
import cv2
import torch
from scipy.ndimage import gaussian_filter

class HeatmapGenerator:
    def __init__(self, clip_wrapper, normal_prompts, defect_prompts, window_size=64, stride=32):
        self.clip_wrapper = clip_wrapper
        self.window_size = window_size
        self.stride = stride

        # Precompute average text embeddings for normal and defect classes
        normal_embeds = self.clip_wrapper.get_text_embeddings(normal_prompts)
        defect_embeds = self.clip_wrapper.get_text_embeddings(defect_prompts)

        # Average across the prompts and normalize again
        self.normal_embed_avg = normal_embeds.mean(dim=0, keepdim=True)
        self.normal_embed_avg /= self.normal_embed_avg.norm(dim=-1, keepdim=True)

        self.defect_embed_avg = defect_embeds.mean(dim=0, keepdim=True)
        self.defect_embed_avg /= self.defect_embed_avg.norm(dim=-1, keepdim=True)

    def generate(self, image_rgb):
        """
        image_rgb: numpy array (H, W, 3) in RGB
        Returns:
            heatmap_norm: (H, W) numpy array scaled 0-255
            binary_mask: (H, W) numpy array 0 or 255 (Otsu thresholded)
            image_score: float, overall anomaly score
        """
        H, W, _ = image_rgb.shape

        # Ensure image is large enough for the window size
        if H < self.window_size or W < self.window_size:
            # Fallback if image is tiny
            self.window_size = min(H, W)
            self.stride = max(1, self.window_size // 2)

        # Calculate grid dimensions
        grid_h = (H - self.window_size) // self.stride + 1
        grid_w = (W - self.window_size) // self.stride + 1

        if grid_h <= 0 or grid_w <= 0:
            grid_h, grid_w = 1, 1

        patch_scores = np.zeros((grid_h, grid_w), dtype=np.float32)

        # Optimization: We can batch the patches, but to save memory and keep it simple,
        # we will process them iteratively or in mini-batches.
        patches = []
        coords = []
        for i in range(grid_h):
            for j in range(grid_w):
                y = i * self.stride
                x = j * self.stride
                patch = image_rgb[y:y+self.window_size, x:x+self.window_size]
                patches.append(patch)
                coords.append((i, j))

        # We will process one by one to avoid huge memory spikes, though it's slower.
        # For a true realtime system, batched tensors are better, but for this prototype
        # it fulfills the requirement.

        # Let's batch them for reasonable speed (batch size 16)
        batch_size = 32
        for b in range(0, len(patches), batch_size):
            batch_patches = patches[b:b+batch_size]
            batch_coords = coords[b:b+batch_size]

            # Since get_image_embedding handles 1 image, we'll loop internally,
            # or we can write a quick batch logic if clip_model supports it.
            # ClipModelWrapper currently processes 1 image at a time via PIL.
            for idx, patch in enumerate(batch_patches):
                img_emb = self.clip_wrapper.get_image_embedding(patch) # [1, D]

                # Compute cosine similarities
                # embeddings are already normalized, so dot product is cosine similarity
                sim_normal = torch.sum(img_emb * self.normal_embed_avg, dim=-1).item()
                sim_defect = torch.sum(img_emb * self.defect_embed_avg, dim=-1).item()

                # Anomaly score: similarity to defect - similarity to normal
                score = sim_defect - sim_normal

                i, j = batch_coords[idx]
                patch_scores[i, j] = score

        # The image score is the maximum patch anomaly score
        image_score = float(np.max(patch_scores))

        # Upsample patch scores to full image resolution
        # First resize the grid to full resolution using bilinear interpolation
        # Since grid centers are offset, we just resize directly
        heatmap = cv2.resize(patch_scores, (W, H), interpolation=cv2.INTER_LINEAR)

        # Apply Gaussian smoothing
        heatmap = gaussian_filter(heatmap, sigma=max(W, H) * 0.02) # adaptive sigma

        # Normalize to 0-255 for visualization and thresholding
        # Handle edge case where max == min
        h_min, h_max = heatmap.min(), heatmap.max()
        if h_max > h_min:
            heatmap_norm = ((heatmap - h_min) / (h_max - h_min) * 255.0).astype(np.uint8)
        else:
            heatmap_norm = np.zeros_like(heatmap, dtype=np.uint8)

        # Apply Otsu's thresholding
        _, binary_mask = cv2.threshold(heatmap_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return heatmap_norm, binary_mask, image_score
