import traceback
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from config import REGISTRADOR, IMAGE_WIDTH, IMAGE_HEIGHT

class AdaIN(nn.Module):
    """Adaptive Instance Normalization Layer"""
    def __init__(self, num_features, z_dim):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features)
        self.fc = nn.Linear(z_dim, num_features * 2)  # scale and shift

    def forward(self, x, z):
        params = self.fc(z)
        gamma, beta = params.chunk(2, dim=1)
        gamma = gamma[..., None, None]
        beta = beta[..., None, None]
        return self.norm(x) * (1 + gamma) + beta

class Gerador(nn.Module):
    def __init__(self, nc=3, ngf=64, z_dim=64):
        super().__init__()
        self.nc = nc
        self.ngf = ngf
        self.z_dim = z_dim

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(nc, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(ngf, ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(ngf*2, ngf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.enc4 = nn.Sequential(
            nn.Conv2d(ngf*4, ngf*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.enc5 = nn.Sequential(
            nn.Conv2d(ngf*8, ngf*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.enc6 = nn.Sequential(
            nn.Conv2d(ngf*8, ngf*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.enc7 = nn.Sequential(
            nn.Conv2d(ngf*8, ngf*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*8),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Bottleneck: recebe ruído concatenado
        self.bottleneck = nn.Sequential(
            nn.Conv2d(ngf*8 + z_dim, ngf*8, 4, 2, 1, bias=False),
            nn.ReLU(True)
        )

        # Decoder (profundo, skips, AdaIN, dropout)
        self.dec7 = nn.Sequential(
            nn.ConvTranspose2d(ngf*8, ngf*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*8),
            nn.ReLU(True),
            nn.Dropout(0.4)
        )
        self.adain7 = AdaIN(ngf*8, z_dim)

        self.dec6 = nn.Sequential(
            nn.ConvTranspose2d(ngf*16, ngf*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*8),
            nn.ReLU(True),
            nn.Dropout(0.4)
        )
        self.adain6 = AdaIN(ngf*8, z_dim)

        self.dec5 = nn.Sequential(
            nn.ConvTranspose2d(ngf*16, ngf*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*8),
            nn.ReLU(True),
            nn.Dropout(0.3)
        )
        self.adain5 = AdaIN(ngf*8, z_dim)

        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(ngf*16, ngf*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*8),
            nn.ReLU(True)
        )
        self.adain4 = AdaIN(ngf*8, z_dim)

        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(ngf*16, ngf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(True)
        )
        self.adain3 = AdaIN(ngf*4, z_dim)

        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(ngf*8, ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True)
        )
        self.adain2 = AdaIN(ngf*2, z_dim)

        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(ngf*4, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True)
        )
        self.adain1 = AdaIN(ngf, z_dim)

        self.final = nn.Sequential(
            nn.ConvTranspose2d(ngf*2, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

        self._inicializar_pesos()
        REGISTRADOR.info(f"Gerador U-Net robusto inicializado com AdaIN para {IMAGE_HEIGHT}×{IMAGE_WIDTH}")

    def forward(self, x: Tensor, ruido: Tensor = None) -> Tensor:
        try:
            # Encoder
            e1 = self.enc1(x)
            e2 = self.enc2(e1)
            e3 = self.enc3(e2)
            e4 = self.enc4(e3)
            e5 = self.enc5(e4)
            e6 = self.enc6(e5)
            e7 = self.enc7(e6)

            # Ruído no bottleneck e AdaIN
            batch, _, h, w = e7.shape
            if ruido is None:
                ruido_adain = torch.randn(batch, self.z_dim, device=x.device)
                ruido_bottleneck = ruido_adain[:, :, None, None].repeat(1, 1, h, w)
            else:
                if ruido.dim() == 2:
                    ruido_adain = ruido
                    ruido_bottleneck = ruido[:, :, None, None].repeat(1, 1, h, w)
                elif ruido.dim() == 4 and (ruido.shape[2] == 1 and ruido.shape[3] == 1):
                    ruido_adain = ruido.squeeze(-1).squeeze(-1)
                    ruido_bottleneck = ruido.repeat(1, 1, h, w)
                elif ruido.dim() == 4 and (ruido.shape[2] == h and ruido.shape[3] == w):
                    ruido_adain = ruido.mean(dim=(2, 3))
                    ruido_bottleneck = ruido
                else:
                    raise RuntimeError(f"Ruído shape inesperado: {ruido.shape}")

            b_input = torch.cat([e7, ruido_bottleneck], dim=1)
            b = self.bottleneck(b_input)

            # Decoder com skips, AdaIN, Dropout
            d7 = self.dec7(b)
            d7 = self.adain7(d7, ruido_adain)
            d7 = self._center_crop_and_concat(d7, e7)

            d6 = self.dec6(d7)
            d6 = self.adain6(d6, ruido_adain)
            d6 = self._center_crop_and_concat(d6, e6)

            d5 = self.dec5(d6)
            d5 = self.adain5(d5, ruido_adain)
            d5 = self._center_crop_and_concat(d5, e5)

            d4 = self.dec4(d5)
            d4 = self.adain4(d4, ruido_adain)
            d4 = self._center_crop_and_concat(d4, e4)

            d3 = self.dec3(d4)
            d3 = self.adain3(d3, ruido_adain)
            d3 = self._center_crop_and_concat(d3, e3)

            d2 = self.dec2(d3)
            d2 = self.adain2(d2, ruido_adain)
            d2 = self._center_crop_and_concat(d2, e2)

            d1 = self.dec1(d2)
            d1 = self.adain1(d1, ruido_adain)
            d1 = self._center_crop_and_concat(d1, e1)

            out = self.final(d1)
            # Correção robusta de shape no output:
            if out.shape[2] != IMAGE_HEIGHT or out.shape[3] != IMAGE_WIDTH:
                out = F.interpolate(out, size=(IMAGE_HEIGHT, IMAGE_WIDTH), mode='bilinear', align_corners=False)
            return out


        except Exception as e:
            REGISTRADOR.error(f"Falha no forward do Gerador U-Net AdaIN: {e}\n{traceback.format_exc()}")
            raise RuntimeError(f"Falha ao gerar imagens: {e}")

    def _center_crop_and_concat(self, upsampled, bypass):
        _, _, h1, w1 = upsampled.size()
        _, _, h2, w2 = bypass.size()
        diff_y = h2 - h1
        diff_x = w2 - w1
        if diff_y != 0 or diff_x != 0:
            bypass = bypass[:, :, diff_y // 2: diff_y // 2 + h1, diff_x // 2: diff_x // 2 + w1]
        return torch.cat([upsampled, bypass], dim=1)

    def _inicializar_pesos(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None: nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None: nn.init.constant_(m.weight, 1)
                if m.bias is not None:   nn.init.constant_(m.bias, 0)