import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from config import DISPOSITIVO

class PerdaTextura(nn.Module):
    def __init__(self):
        super(PerdaTextura, self).__init__()
        try:
            resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            self.features = nn.Sequential(*list(resnet.children())[:7])  # até layer3
            for param in self.features.parameters():
                param.requires_grad = False
            self.features.to(DISPOSITIVO)
        except Exception as e:
            raise RuntimeError(f"Falha ao carregar ResNet-50: {e}")

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        try:
            f1 = self.features(img1)
            f2 = self.features(img2)
            perda = F.l1_loss(f1, f2)
            return perda
        except Exception as e:
            raise RuntimeError(f"Falha ao calcular PerdaTextura: {e}")