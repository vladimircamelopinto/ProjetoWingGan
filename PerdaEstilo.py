import torch
import torch.nn as nn
from torchvision.models import vgg16
from torch import Tensor
from config import DISPOSITIVO

class PerdaEstilo(nn.Module):
    def __init__(self):
        try:
            super(PerdaEstilo, self).__init__()
            with torch.no_grad():
                vgg = vgg16(weights='IMAGENET1K_V1').features[:9].eval().to(DISPOSITIVO)
                for param in vgg.parameters():
                    param.requires_grad = False
            self.vgg = vgg
            self.criterio = nn.MSELoss()
        except Exception as e:
            raise RuntimeError(f"Falha na inicialização de PerdaEstilo: {str(e)}")

    def matriz_gram(self, x: Tensor) -> Tensor:
        # A matriz de Gram precisa ser diferenciável para o backpropagation
        try:
            b, c, h, w = x.size()
            features = x.view(b, c, h * w)
            features_t = features.transpose(1, 2)
            gram = torch.bmm(features, features_t) / (c * h * w)
            return gram
        except Exception as e:
            raise RuntimeError(f"Erro ao calcular a matriz de Gram: {str(e)}")

    def forward(self, img_gerada: Tensor, img_real: Tensor) -> Tensor:
        try:
            features_gerada = self.vgg(img_gerada)
            gram_gerada = self.matriz_gram(features_gerada)

            with torch.no_grad():
                features_real = self.vgg(img_real)
                gram_real = self.matriz_gram(features_real)

            perda_estilo = self.criterio(gram_gerada, gram_real)

            return perda_estilo

        except Exception as e:
            raise RuntimeError(f"Erro ao calcular a perda de estilo: {str(e)}")