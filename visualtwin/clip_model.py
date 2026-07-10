import torch
import open_clip
from PIL import Image

class ClipModelWrapper:
    def __init__(self, model_name="ViT-B-16", pretrained="laion2b_s34b_b88k"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading CLIP model {model_name} ({pretrained}) on {self.device}...")

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            device=self.device
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()

    @torch.no_grad()
    def get_image_embedding(self, image_patch):
        """
        image_patch: PIL Image or numpy array (RGB)
        Returns: normalized tensor embedding (shape: [1, hidden_dim])
        """
        if not isinstance(image_patch, Image.Image):
            image_patch = Image.fromarray(image_patch)

        # Preprocess and add batch dimension
        image_input = self.preprocess(image_patch).unsqueeze(0).to(self.device)

        # Get embeddings and normalize
        image_features = self.model.encode_image(image_input)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features.cpu()

    @torch.no_grad()
    def get_text_embeddings(self, prompt_list):
        """
        prompt_list: list of strings
        Returns: normalized tensor embedding (shape: [len(prompt_list), hidden_dim])
        """
        text_tokens = self.tokenizer(prompt_list).to(self.device)
        text_features = self.model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features.cpu()
