import torch
import torch.nn as nn
from config import REGISTRADOR

class Discriminador(nn.Module):
    """
    Discriminador inspirado em Pix2PixHD e BigGAN,
    com SpectralNorm, InstanceNorm, Dropout, profundidade real,
    e Minibatch Discrimination para reforçar diversidade de amostras
    e evitar mode collapse.
    """
    def __init__(self, nc=3, ndf=64, img_size=(925, 1225), dropout=True):
        super().__init__()
        layers = []

        # Primeira camada (sem normalização)
        layers.append(
            nn.utils.spectral_norm(nn.Conv2d(nc, ndf, kernel_size=4, stride=2, padding=1))
        )
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        # Camadas intermediárias
        n_layers = 6
        nf_mult = 1
        for i in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** i, 8)
            layers.append(
                nn.utils.spectral_norm(
                    nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, 4, 2, 1, bias=False)
                )
            )
            layers.append(nn.InstanceNorm2d(ndf * nf_mult))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            if dropout and i >= n_layers - 2:
                layers.append(nn.Dropout(0.35))

        # Última camada convolucional com PatchGAN
        layers.append(
            nn.utils.spectral_norm(
                nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1)
            )
        )

        self.model = nn.Sequential(*layers)

        # Módulo de Minibatch Discrimination
        self.minibatch_discrim = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=1),  # projeção para 10 features por patch
            nn.ReLU(inplace=True)
        )

        self._init_weights()
        REGISTRADOR.info(
            f"DiscriminadorProGAN+Minibatch iniciado (profundo, PatchGAN, SN+IN, dropout={dropout}) para imagem {img_size}"
        )

    def forward(self, x):
        out = self.model(x)  # [B, 1, H', W']
        mb_feat = self.minibatch_discrim(out)  # [B, 10, H', W']
        out_concat = torch.cat([out, mb_feat], dim=1)  # concatena canal original com os extras
        return out_concat  # [B, 11, H', W']

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)