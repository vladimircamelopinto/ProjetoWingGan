import os
import time
import torch
import torch.nn as nn
from config import REGISTRADOR
import psutil
from typing import Optional

class GerenciadorCheckpoint:
    @staticmethod
    def salvar_checkpoint(modelo: nn.Module, caminho: str, nome_amigavel_modelo: str = "Modelo",
                          epoca: Optional[int] = None) -> bool:

        msg_epoca = f" (Época {epoca})" if epoca is not None else ""
        # Log de início do processo de salvamento
        REGISTRADOR.info(f"Iniciando salvamento do checkpoint para {nome_amigavel_modelo}{msg_epoca} em: {caminho}...")
        t_inicio_chkp = time.time()

        try:
            # Verificar memória disponível antes de salvar
            # Validação dos parâmetros de entrada
            def validar_entradas(modelo: nn.Module, caminho: str, nome_amigavel_modelo: str) -> bool:
                if modelo is None:
                    REGISTRADOR.error(f"Modelo para checkpoint ('{nome_amigavel_modelo}') é nulo.")
                    return False
                if not caminho or not caminho.strip():  # Checa se o caminho é nulo ou contém apenas espaços
                    REGISTRADOR.error(f"Caminho do checkpoint para '{nome_amigavel_modelo}' é nulo/vazio.")
                    return False
                return True

            with torch.inference_mode():  # Verificações sem gradientes
                REGISTRADOR.info("Usando validar_entradas sem compilação dinâmica.")
                if not validar_entradas(modelo, caminho, nome_amigavel_modelo):
                    return False

            # Cria o diretório pai se não existir
            diretorio_pai = os.path.dirname(caminho)
            if diretorio_pai:  # Se houver um diretório pai especificado
                # Verifica se o diretório já existe, se não, tenta criar
                if not os.path.exists(diretorio_pai):
                    REGISTRADOR.info(f"Diretório '{diretorio_pai}' não existe. Tentando criar...")
                    os.makedirs(diretorio_pai,
                                exist_ok=True)  # exist_ok=True evita erro se o diretório for criado entre o check e o makedirs
                    REGISTRADOR.info(f"Diretório '{diretorio_pai}' criado/verificado.")
            else:
                # Se não houver diretório pai, o arquivo será salvo no diretório atual de trabalho
                REGISTRADOR.debug(
                    f"Nenhum diretório pai especificado para o checkpoint '{caminho}'. Salvando no diretório de trabalho atual.")

            # Salva o estado do modelo
            with torch.no_grad():  # Evitar gradientes durante salvamento
                torch.save(modelo.state_dict(), caminho)

            REGISTRADOR.info(
                f"Checkpoint para {nome_amigavel_modelo}{msg_epoca} salvo com sucesso: {caminho}. Tempo: {time.time() - t_inicio_chkp:.2f}s")
            return True

        except Exception as e:
            return False