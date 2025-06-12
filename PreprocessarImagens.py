import sys
import torch
from torchvision import transforms as transforms_v2
from torch.utils.data import DataLoader
from ConjuntoImagensPersonalizado import ConjuntoImagensPersonalizado
from RepeatSampler import RepeatSampler
from config import TAMANHO_LOTE, NUMERO_TRABALHADORES, IMAGE_WIDTH, IMAGE_HEIGHT, DISPOSITIVO
import logging

REGISTRADOR = logging.getLogger(__name__)

class PreprocessarImagens:
    @staticmethod
    def preprocessar_imagens(diretorio: str) -> tuple[DataLoader, transforms_v2.Compose]:
        # Cria a pipeline de transformações corretamente
        transformacao = transforms_v2.Compose([
            transforms_v2.ToTensor(),
            transforms_v2.Resize((IMAGE_HEIGHT, IMAGE_WIDTH), antialias=True),
            transforms_v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms_v2.Lambda(lambda x: x.contiguous())
        ])

        # Configuração de workers para DataLoader
        num_workers = 0 if sys.platform == 'win32' else NUMERO_TRABALHADORES
        loader_config = {
            'num_workers': num_workers,
            'persistent_workers': False,
            'multiprocessing_context': 'spawn' if sys.platform == 'win32' and num_workers > 0 else None
        }

        try:
            with torch.inference_mode():
                # Carrega o dataset
                conjunto = ConjuntoImagensPersonalizado(diretorio, transformacao=transformacao)
                if len(conjunto) == 0:
                    raise ValueError(f"O diretório {diretorio} não contém imagens válidas.")

                # Calcula quantas vezes repetir cada sample
                num_repeats_calculado = max(1, min(10, 500 // len(conjunto)))
                sampler = RepeatSampler(conjunto, num_repeats=num_repeats_calculado)

                # Cria o DataLoader
                carregador_dados = DataLoader(
                    conjunto,
                    batch_size=TAMANHO_LOTE,
                    sampler=sampler,
                    pin_memory=(DISPOSITIVO.type == 'cuda'),
                    drop_last=True,
                    **loader_config
                )

                return carregador_dados, transformacao

        except Exception as e:
            REGISTRADOR.error(f"Falha ao criar carregador de dados: {str(e)}", exc_info=True)
            # interrompe aqui para facilitar o debug
            raise RuntimeError(f"Falha ao criar carregador de dados: {str(e)}") from e