import cv2
import numpy as np
import torch
from torchvision.models import resnet18, ResNet18_Weights

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

MIN_CROP_SIZE = 40

class EmbeddingExtractor:

    def __init__(self):

        self.device = "cuda" if torch.cuda.is_available() else "cpu" # gpu or cpu

        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.model.fc = torch.nn.Identity()
        self.model.eval()
        self.model.to(self.device)

    def extract(self, crop_bgr: np.ndarray | None) -> np.ndarray | None:
        """Return a normalized 512-d embedding for a person crop, or None if unusable."""
        if crop_bgr is None or crop_bgr.size == 0:
            return None
        h, w = crop_bgr.shape[:2]
        if h < MIN_CROP_SIZE or w < MIN_CROP_SIZE:
            return None

        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (224, 224))
        rgb = rgb.astype(np.float32) / 255.0
        rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD

        tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self.model(tensor)[0].cpu().numpy()

        norm = np.linalg.norm(features)
        if norm == 0:
            return None
        return features / norm
