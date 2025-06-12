import os
import time
import traceback
from torch import Tensor
from typing import Optional, Any
from PIL import Image
import numpy as np
from config import REGISTRADOR
from Caminhos import Caminhos

class UtilitariosImagem:
    @staticmethod
    def salvar_imagens(caminhos: Caminhos, imagens_batch: Tensor, id_passo_epoca: Any,
                       imagens_reais_batch: Optional[Tensor] = None) -> None:

        if isinstance(caminhos, str):
            raise TypeError("Esperado objeto Caminhos em 'salvar_imagens', recebido str. Verifique a chamada.")

        REGISTRADOR.debug(f"Iniciando salvamento de imagem sintética para ID: {id_passo_epoca}...")
        t_inicio_batch_save = time.time()

        try:
            os.makedirs(caminhos.REPOSITORIO_GAN_GERADOR, exist_ok=True)

            for i in range(imagens_batch.size(0)):
                imagem_sintetica_chw = imagens_batch[i]
                try:
                    # Converte o tensor para um formato de imagem salvável (numpy uint8)
                    imagem_np = (imagem_sintetica_chw.permute(1, 2, 0).cpu().detach().numpy() + 1) * 127.5
                    imagem_np = np.clip(imagem_np, 0, 255).astype(np.uint8)

                    # Cria e salva a imagem PIL
                    img_pil = Image.fromarray(imagem_np)
                    caminho_saida = caminhos.obter_caminho_completo(caminhos.REPOSITORIO_GAN_GERADOR,
                                                                    f"imagem_sintetica_{id_passo_epoca}_{i}.{caminhos.EXTENSAO_PNG}")
                    img_pil.save(caminho_saida)
                    REGISTRADOR.info(f"Imagem sintética '{id_passo_epoca}_{i}' salva com sucesso em: {caminho_saida}")

                except Exception as e_save:
                    REGISTRADOR.error(
                        f"Erro ao salvar imagem sintética para '{id_passo_epoca}_{i}': {e_save}\n{traceback.format_exc()}")
                    continue

        except Exception as e_main:
            REGISTRADOR.error(
                f"Erro GERAL em salvar_imagens para ID '{id_passo_epoca}': {e_main}\n{traceback.format_exc()}")
        finally:
            t_fim_batch_save = time.time()
            REGISTRADOR.debug(
                f"Rotina de salvamento para ID {id_passo_epoca} finalizada. Tempo: {t_fim_batch_save - t_inicio_batch_save:.2f}s")