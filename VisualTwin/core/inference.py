import torch
import cv2
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch.nn.functional as F

class VisualTwinInference:
    def __init__(self, model_name="openai/clip-vit-base-patch32", device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading {model_name} on {self.device}...")

        # Load model and processor. Ensure output_attentions=True for rollout
        self.model = CLIPModel.from_pretrained(model_name, output_attentions=True).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)

        self.model.eval()
        self.patch_size = self.model.config.vision_config.patch_size
        self.image_size = self.model.config.vision_config.image_size
        self.grid_size = self.image_size // self.patch_size

    def compute_attention_rollout(self, attentions):
        """
        Computes standard class-agnostic attention rollout.
        attentions: tuple of attention matrices from each layer.
                    Each shape: (batch_size, num_heads, seq_len, seq_len)
        """
        # Start with an identity matrix
        result = torch.eye(attentions[0].size(-1)).to(self.device)

        for attention in attentions:
            # Average across heads
            attention_heads_fused = attention.mean(axis=1)

            # Add identity matrix to account for residual connections (and to avoid vanishing values)
            attention_heads_fused += torch.eye(attention_heads_fused.size(-1)).to(self.device)

            # Normalize the rows
            attention_heads_fused = attention_heads_fused / attention_heads_fused.sum(dim=-1, keepdim=True)

            # Matrix multiply to compute rollout
            result = torch.matmul(attention_heads_fused, result)

        return result

    def get_class_specific_heatmap(self, image_input, text_inputs, target_idx=0):
        """
        Computes a class-specific heatmap for the ViT backbone using Attention Rollout
        weighted by patch-level text similarity.
        """
        # Process inputs
        inputs = self.processor(
            text=text_inputs,
            images=image_input,
            return_tensors="pt",
            padding=True
        ).to(self.device)

        with torch.no_grad():
            # Get text embeddings
            text_outputs = self.model.text_model(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask
            )
            text_embeds = text_outputs[1] # pooled output
            text_embeds = self.model.text_projection(text_embeds)
            text_embeds = F.normalize(text_embeds, p=2, dim=-1)

            # Get image embeddings and attentions
            vision_outputs = self.model.vision_model(
                pixel_values=inputs.pixel_values,
                output_attentions=True
            )

            # Image global embedding
            image_embeds = vision_outputs[1] # pooled output
            image_embeds = self.model.visual_projection(image_embeds)
            image_embeds = F.normalize(image_embeds, p=2, dim=-1)

            # Compute classification scores (cosine similarity)
            logits_per_image = torch.matmul(image_embeds, text_embeds.t()) * self.model.logit_scale.exp()
            probs = logits_per_image.softmax(dim=1)[0].cpu().numpy()

            # -- Heatmap Generation --
            # 1. Compute standard rollout
            attentions = vision_outputs.attentions
            rollout = self.compute_attention_rollout(attentions)

            # 2. Get attention from CLS token (index 0) to all patch tokens (index 1:)
            # rollout shape: (batch_size, seq_len, seq_len). We want the first batch.
            mask = rollout[0, 0, 1:]

            # 3. Get patch-level embeddings for class conditioning
            # last_hidden_state shape: (batch_size, seq_len, hidden_size)
            patch_embeds = vision_outputs.last_hidden_state[0, 1:, :] # shape: (num_patches, hidden_size)
            patch_embeds = self.model.visual_projection(patch_embeds)
            patch_embeds = F.normalize(patch_embeds, p=2, dim=-1)

            # Compute similarity of each patch to the target text embedding
            target_text_embed = text_embeds[target_idx]
            patch_similarities = torch.matmul(patch_embeds, target_text_embed)

            # Relu to keep only positive similarities (excite)
            patch_similarities = F.relu(patch_similarities)

            # Weight the rollout mask by patch similarity
            class_specific_mask = mask * patch_similarities

            # Reshape to grid
            class_specific_mask = class_specific_mask.reshape(self.grid_size, self.grid_size)

            # Normalize to [0, 1] for visualization
            class_specific_mask = class_specific_mask - class_specific_mask.min()
            max_val = class_specific_mask.max()
            if max_val > 0:
                class_specific_mask = class_specific_mask / max_val

            heatmap = class_specific_mask.cpu().numpy()

        return probs, heatmap

    def overlay_heatmap(self, original_image, heatmap, alpha=0.5):
        """
        Upsamples heatmap to original image size and overlays it.
        original_image: numpy array (H, W, 3) in BGR (from OpenCV)
        heatmap: numpy array (grid_size, grid_size)
        """
        h, w = original_image.shape[:2]

        # Resize heatmap
        heatmap_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_CUBIC)

        # Convert to colormap (JET)
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)

        # Overlay
        overlayed_img = cv2.addWeighted(heatmap_colored, alpha, original_image, 1 - alpha, 0)
        return overlayed_img
