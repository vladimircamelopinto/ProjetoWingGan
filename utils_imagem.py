import os
import time
import torch
import cv2
import numpy as np
from torch import Tensor
from PIL import Image, ExifTags
import torchvision.transforms as transforms
from config import REGISTRADOR, DISPOSITIVO, IMAGE_WIDTH, IMAGE_HEIGHT

def preprocessar_imagens_offline(dir_entrada: str, dir_saida: str) -> None:
    """
    Pré-processa imagens em batch, aplicando resize, normalização e exportando para saída.
    """
    funcao = "preprocessar_imagens_offline"
    REGISTRADOR.info(f"Iniciando {funcao} de '{dir_entrada}' para '{dir_saida}'")
    t_start = time.time()
    processados = 0

    try:
        if not os.path.isdir(dir_entrada):
            raise FileNotFoundError(f"Diretório de entrada não encontrado: {dir_entrada}")
        os.makedirs(dir_saida, exist_ok=True)

        # Pipeline de transforms
        transformacao = transforms.Compose([
            transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        arquivos = [f for f in os.listdir(dir_entrada)
                    if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        REGISTRADOR.info(f"{len(arquivos)} imagens encontradas.")

        with torch.no_grad():
            for nome in arquivos:
                try:
                    caminho = os.path.join(dir_entrada, nome)
                    img_pil = Image.open(caminho).convert("RGB")

                    # Corrige orientação via EXIF
                    try:
                        exif = img_pil._getexif() or {}
                        key = next(k for k, v in ExifTags.TAGS.items() if v == "Orientation")
                        orient = exif.get(key, None)
                        if orient == 3:
                            img_pil = img_pil.rotate(180, expand=True)
                        elif orient == 6:
                            img_pil = img_pil.rotate(-90, expand=True)
                        elif orient == 8:
                            img_pil = img_pil.rotate(90, expand=True)
                    except Exception:
                        pass

                    tensor_img = transformacao(img_pil)
                    assert tensor_img.shape == (3, IMAGE_HEIGHT, IMAGE_WIDTH), \
                        f"Shape inválido: {tensor_img.shape}"

                    # Converte tensor -> NumPy -> BGR e salva
                    arr = (tensor_img.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(os.path.join(dir_saida, nome), bgr)

                    processados += 1

                except Exception as e_img:
                    REGISTRADOR.error(f"Erro ao processar '{nome}': {e_img}", exc_info=True)

        t_end = time.time()
        REGISTRADOR.info(f"{funcao} finalizado: {processados} imagens em {t_end - t_start:.2f}s")

    except Exception as e:
        REGISTRADOR.error(f"Falha em {funcao}: {e}", exc_info=True)
        raise


def segmentar_figura(imagem: Tensor) -> Tensor:
    """
    Segmenta a figura principal da tensor-imagem de entrada e retorna
    uma máscara binária (1 canal) no shape (1, H, W).
    """
    try:
        with torch.no_grad():
            # Normaliza para HxWxC e converte para uint8
            if imagem.ndim == 4 and imagem.size(0) == 1:
                img_chw = imagem.squeeze(0)
            elif imagem.ndim == 3:
                img_chw = imagem
            else:
                # REGISTRADOR.warning(f"Formato inválido: {imagem.shape}")
                return torch.ones((1, IMAGE_HEIGHT, IMAGE_WIDTH), device=DISPOSITIVO)

            img_np = ((img_chw.permute(1, 2, 0).cpu().numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
            hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)

            # Exemplo simples de segmentação por cor (ajuste conforme seu caso)
            lower = np.array([0, 0, 0])
            upper = np.array([180, 255, 255])
            mask = cv2.inRange(hsv, lower, upper)  # cria máscara HxW

            mask_tensor = torch.from_numpy(mask).unsqueeze(0).float().to(DISPOSITIVO) / 255.0
            assert mask_tensor.shape == (1, IMAGE_HEIGHT, IMAGE_WIDTH), \
                f"Mask shape inválido: {mask_tensor.shape}"
            return mask_tensor

    except Exception as e:
        REGISTRADOR.error(f"Erro em segmentar_figura: {e}", exc_info=True)
        # fallback máscara cheia
        return torch.ones((1, IMAGE_HEIGHT, IMAGE_WIDTH), device=DISPOSITIVO)


def transformar_figura(imagem: Tensor, marca: Tensor) -> Tensor:
    """
    Recebe tensor de imagem (3×H×W) e máscara binária (1×H×W),
    aplica e retorna tensor modificado.
    """
    try:
        if marca.ndim == 3:
            # tudo ok
            pass
        elif marca.ndim == 2:
            marca = marca.unsqueeze(0)
        else:
            REGISTRADOR.warning(f"Formato inválido de máscara: {marca.shape}")
            return imagem

        marca_bin = (marca > 0.5).float()
        imagem_morf = imagem * marca_bin + (1 - marca_bin) * -1.0

        assert imagem_morf.shape == imagem.shape, \
            f"Shape diferente: {imagem.shape} vs {imagem_morf.shape}"
        return imagem_morf

    except Exception as e:
        REGISTRADOR.error(f"Erro em transformar_figura: {e}", exc_info=True)
        raise