import torch
import torch.nn as nn
from torchvision.models import vgg16
from torch import Tensor
from config import REGISTRADOR, DISPOSITIVO
import psutil

class PerdaPerceptual(nn.Module):
    def __init__(self):
        try:
            super(PerdaPerceptual, self).__init__()
            with torch.no_grad():
                vgg = vgg16(weights='IMAGENET1K_V1').features[:9].eval().to(DISPOSITIVO)
                for param in vgg.parameters():
                    param.requires_grad = False
            self.vgg = vgg
            self.criterio = nn.MSELoss()
        except Exception as e:
            raise RuntimeError(f"Falha na inicialização de PerdaPerceptual: {str(e)}")

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        try:
            memoria = psutil.virtual_memory()
            if memoria.percent > 85:
                REGISTRADOR.warning(f"Memória alta ({memoria.percent}%). Liberando recursos antes de calcular perda.")

            x_vgg = self.vgg(x)

            with torch.no_grad():
                y_vgg = self.vgg(y)

            perda = self.criterio(x_vgg, y_vgg)

            return perda
        except Exception as e:
            raise RuntimeError(f"Falha ao calcular perda perceptual: {str(e)}")