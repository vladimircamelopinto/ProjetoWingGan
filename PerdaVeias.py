import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from config import REGISTRADOR

class PerdaVeias(nn.Module):
    def __init__(self):
        try:
            super().__init__()
            self.criterio = nn.L1Loss()
            sobel_x_k = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
            sobel_y_k = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
            self.register_buffer('sobel_x', sobel_x_k)
            self.register_buffer('sobel_y', sobel_y_k)
        except Exception as e:
            raise RuntimeError(f"Falha na inicialização de PerdaVeias: {str(e)}")

    def forward(self, img_gerada: Tensor, img_real: Tensor) -> Tensor:
        try:
            if img_gerada.dtype != torch.float32 or img_real.dtype != torch.float32:
                REGISTRADOR.error("Tensores não são float32. Convertendo...")
                img_gerada = img_gerada.float()
                img_real = img_real.float()
            if img_gerada.shape != img_real.shape:
                raise ValueError(f"Dimensões incompatíveis: gerada={img_gerada.shape}, real={img_real.shape}")

            def calcular_bordas(imagem: Tensor) -> Tensor:
                bordas_x = F.conv2d(imagem, self.sobel_x, padding=1)
                bordas_y = F.conv2d(imagem, self.sobel_y, padding=1)
                return torch.sqrt(bordas_x ** 2 + bordas_y ** 2 + 1e-10)

            cinza_gerada = img_gerada.mean(dim=1, keepdim=True)
            with torch.no_grad():
                cinza_real = img_real.mean(dim=1, keepdim=True)
                bordas_reais = calcular_bordas(cinza_real)

            bordas_geradas = calcular_bordas(cinza_gerada)
            perda = self.criterio(bordas_geradas, bordas_reais)

            return perda
        except Exception as e:
            raise RuntimeError(f"Falha ao calcular perda de veias: {str(e)}")