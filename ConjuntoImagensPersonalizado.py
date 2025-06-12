import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.v2 as transforms_v2
from config import REGISTRADOR, CANAIS_IMAGEM, IMAGE_WIDTH, IMAGE_HEIGHT
from typing import Optional, Any, Tuple

torch._dynamo.config.suppress_errors = True

class ConjuntoImagensPersonalizado(Dataset):
    def __init__(self, diretorio: str, transformacao: Optional[Any] = None):

        self.diretorio = diretorio
        self.transformacao = transformacao

        if not os.path.isdir(diretorio):
            msg_erro_dir = f"O diretório especificado não existe ou não é um diretório: {diretorio}"
            REGISTRADOR.critical(msg_erro_dir)
            raise NotADirectoryError(msg_erro_dir)

        with torch.inference_mode():
            self.imagens = sorted([os.path.join(diretorio, f) for f in os.listdir(diretorio) if
                                   f.lower().endswith(('.png', '.jpg', '.jpeg'))])

            if not self.imagens:
                msg_erro_imgs = f"Nao foram encontradas imagens no diretorio: {diretorio}"
                REGISTRADOR.critical(msg_erro_imgs)
                raise ValueError(msg_erro_imgs)

            REGISTRADOR.info(f"Encontradas {len(self.imagens)} imagens em: {diretorio}")

    def __len__(self) -> int:
        return len(self.imagens)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, float]:

        if not self.imagens:
            REGISTRADOR.error(f"Tentativa de acessar __getitem__ com lista de imagens vazia.")
            return torch.zeros(CANAIS_IMAGEM, IMAGE_HEIGHT, IMAGE_WIDTH), 0.0

        caminho_imagem = self.imagens[idx % len(self.imagens)]

        try:
            with Image.open(caminho_imagem) as img:
                imagem_pil = img.convert("RGB")

            imagem_transformada: torch.Tensor
            with torch.no_grad():
                if self.transformacao:
                    imagem_transformada = self.transformacao(imagem_pil)
                else:
                    REGISTRADOR.debug(f"Nenhuma transformação fornecida para {caminho_imagem}. Aplicando ToTensor padrão.")
                    transform_to_tensor = transforms_v2.Compose([
                    transforms_v2.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
                    transforms_v2.ToTensor()
                ])
                    imagem_transformada = transform_to_tensor(imagem_pil)
            return imagem_transformada, 0.0

        except Exception as e:
            return torch.zeros(CANAIS_IMAGEM, IMAGE_HEIGHT, IMAGE_WIDTH), 0.0