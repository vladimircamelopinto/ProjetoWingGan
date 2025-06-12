import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from config import REGISTRADOR  # Assume que REGISTRADOR está em .config

class PerdaAltaFrequencia(nn.Module):
    def __init__(self):
        try:
            super(PerdaAltaFrequencia, self).__init__()
            self.criterio = nn.L1Loss()
            sobel_x_k = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
            sobel_y_k = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
            self.register_buffer('sobel_x', sobel_x_k)
            self.register_buffer('sobel_y', sobel_y_k)
        except Exception as e:
            raise  # Re-lança a exceção para indicar falha na inicialização

    def forward(self, img_gerada: Tensor, img_real: Tensor) -> Tensor:
        try:
            perda_total_hf = 0.0
            num_canais = img_gerada.size(1)

            if num_canais == 0:  # Evitar divisão por zero
                device_fallback = img_real.device if isinstance(img_real, Tensor) and img_real.numel() > 0 else torch.device("cpu")
                REGISTRADOR.warning(
                    f"Imagem gerada com 0 canais em PerdaAltaFrequencia. Retornando perda zero no dispositivo {device_fallback}."
                )
                return torch.tensor(0.0, device=device_fallback, requires_grad=True)  # Adicionado requires_grad=True

            for i in range(num_canais):
                canal_gerado = img_gerada[:, i:i + 1, :, :]
                canal_real = img_real[:, i:i + 1, :, :]

                grad_gerada_x = F.conv2d(canal_gerado, self.sobel_x, padding=1)
                grad_gerada_y = F.conv2d(canal_gerado, self.sobel_y, padding=1)
                grad_real_x = F.conv2d(canal_real, self.sobel_x, padding=1)
                grad_real_y = F.conv2d(canal_real, self.sobel_y, padding=1)

                perda_total_hf += self.criterio(grad_gerada_x, grad_real_x) + self.criterio(grad_gerada_y, grad_real_y)

            return perda_total_hf / num_canais

        except Exception as e:
            raise RuntimeError(f"Falha ao calcular perda de alta frequência (detalhes no log): {str(e)}")