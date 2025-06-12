import os
from typing import Optional
from config import (REGISTRADOR)

class Caminhos:
    REPOSITORIO_IMAGENS_PADRAO = "C:\\Users\\vladi\\eclipse-workspace\\WingGAN\\ImagensNovas"
    REPOSITORIO_IMAGENS_PROC_PADRAO = "C:\\Users\\vladi\\eclipse-workspace\\WingGAN\\ImagensProcessadas"
    REPOSITORIO_GAN_GERADOR_PADRAO = "C:\\Users\\vladi\\eclipse-workspace\\WingGAN\\WingGan\\Gerador\\"
    REPOSITORIO_GAN_DISCRIMINADOR_PADRAO = "C:\\Users\\vladi\\eclipse-workspace\\WingGAN\\WingGan\\Discriminador\\"
    REPOSITORIO_GAN_MODELOS_PADRAO = "C:\\Users\\vladi\\eclipse-workspace\\WingGAN\\WingGan\\Modelos\\"
    REPOSITORIO_GAN_RELATORIOS_PADRAO = "C:\\Users\\vladi\\eclipse-workspace\\WingGAN\\WingGan\\Relatorios\\"
    REPOSITORIO_GAN_LOGS_PADRAO = "C:\\Users\\vladi\\eclipse-workspace\\WingGAN\\WingGan\\log\\"

    EXTENSAO_PNG = "png"
    EXTENSAO_ZIP = "zip"

    def __init__(self, dir_imagens: Optional[str] = None, dir_imagens_proc: Optional[str] = None,
                 dir_gerador: Optional[str] = None, dir_discriminador: Optional[str] = None,
                 dir_modelos: Optional[str] = None, dir_relatorios: Optional[str] = None,
                 dir_logs: Optional[str] = None):
        self.REPOSITORIO_IMAGENS = dir_imagens or self.REPOSITORIO_IMAGENS_PADRAO
        self.REPOSITORIO_IMAGENS_PROC = dir_imagens_proc or self.REPOSITORIO_IMAGENS_PROC_PADRAO
        self.REPOSITORIO_GAN_GERADOR = dir_gerador or self.REPOSITORIO_GAN_GERADOR_PADRAO
        self.REPOSITORIO_GAN_DISCRIMINADOR = dir_discriminador or self.REPOSITORIO_GAN_DISCRIMINADOR_PADRAO
        self.REPOSITORIO_GAN_MODELOS = dir_modelos or self.REPOSITORIO_GAN_MODELOS_PADRAO
        self.REPOSITORIO_GAN_RELATORIOS = dir_relatorios or self.REPOSITORIO_GAN_RELATORIOS_PADRAO
        self.REPOSITORIO_GAN_LOGS = dir_logs or self.REPOSITORIO_GAN_LOGS_PADRAO
        for name, path_val in vars(self).items():
            if isinstance(path_val, str) and ("REPOSITORIO" in name or "DIR" in name.upper()):
                if not os.path.splitext(path_val)[1] and not os.path.exists(path_val):
                    os.makedirs(path_val, exist_ok=True)
                    REGISTRADOR.debug(f"Diretório criado: {path_val}")

    @staticmethod
    def obter_caminho_completo(caminho: str, nome_arquivo: str) -> str:
        return os.path.join(caminho, nome_arquivo)