from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image


class VisionEmbeddingModel:

    def __init__(self):
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def embed_image(self, image: Image.Image):
        inputs = self.processor(images=image, return_tensors="pt")

        with torch.no_grad():
            features = self.model.get_image_features(**inputs)

        return features[0].cpu().numpy()

    def embed_text(self, text: str):
        inputs = self.processor(text=[text], return_tensors="pt", padding=True)

        with torch.no_grad():
            features = self.model.get_text_features(**inputs)

        return features[0].cpu().numpy()