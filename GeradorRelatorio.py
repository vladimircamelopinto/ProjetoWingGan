import os
import time
import traceback
import matplotlib.pyplot as plt
import torch
from config import REGISTRADOR
import sys
from typing import List
import psutil

class GeradorRelatorio:
    @staticmethod
    def gerar_grafico_perda(perdas_g_lista: List[float],
                            perdas_d_lista: List[float],
                            caminho_arquivo: str) -> None:
        # Informações para logging detalhado em caso de erro
        nome_arquivo_atual = os.path.split(__file__)[1]
        classe_nome = GeradorRelatorio.__name__
        metodo_nome = "gerar_grafico_perda"

        if not perdas_g_lista and not perdas_d_lista:
            REGISTRADOR.info("Nenhuma perda registrada, gráfico de perdas não será gerado.")
            return

        REGISTRADOR.info(f"Iniciando geração do gráfico de perdas para: {caminho_arquivo}...")
        t_inicio_plot = time.time()
        fig = None

        try:
            # Verificar memória
            memoria = psutil.virtual_memory()
            if memoria.percent > 85:
                REGISTRADOR.warning(f"Memória alta ({memoria.percent}%). Liberando recursos antes de plotar.")

            # Estatísticas das perdas
            with torch.inference_mode():
                if perdas_g_lista:
                    g_media = sum(perdas_g_lista) / len(perdas_g_lista)
                    g_max = max(perdas_g_lista)
                    g_min = min(perdas_g_lista)
                    REGISTRADOR.info(f"Perdas do Gerador: Média={g_media:.4f}, Máx={g_max:.4f}, Mín={g_min:.4f}")
                if perdas_d_lista:
                    d_media = sum(perdas_d_lista) / len(perdas_d_lista)
                    d_max = max(perdas_d_lista)
                    d_min = min(perdas_d_lista)
                    REGISTRADOR.info(f"Perdas do Discriminador: Média={d_media:.4f}, Máx={d_max:.4f}, Mín={d_min:.4f}")

            fig = plt.figure(figsize=(12, 6))

            if perdas_g_lista:
                plt.plot(perdas_g_lista, label='Perda do Gerador (G)')
            if perdas_d_lista:
                plt.plot(perdas_d_lista, label='Perda do Discriminador (D)')

            plt.xlabel('Passos de Otimização (acumulados)')
            plt.ylabel('Perda')
            plt.title('Perdas de Treinamento (Gerador e Discriminador)')

            if perdas_g_lista or perdas_d_lista:
                plt.legend()

            plt.grid(True)
            plt.tight_layout()

            diretorio_destino = os.path.dirname(caminho_arquivo)
            if diretorio_destino:
                os.makedirs(diretorio_destino, exist_ok=True)

            plt.savefig(caminho_arquivo)
            REGISTRADOR.info(
                f"Gráfico de perda salvo com sucesso em: {caminho_arquivo}. Tempo: {time.time() - t_inicio_plot:.2f}s")

        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            linha_erro = exc_tb.tb_lineno if exc_tb is not None else "desconhecida"
            REGISTRADOR.error(
                f"Erro ao gerar gráfico de perda para '{caminho_arquivo}':\n"
                f"  Arquivo: {nome_arquivo_atual}\n"
                f"  Classe: {classe_nome}\n"
                f"  Método: {metodo_nome}\n"
                f"  Linha do erro no método: {linha_erro}\n"
                f"  Tipo do Erro: {exc_type.__name__ if exc_type else 'Desconhecido'}\n"
                f"  Mensagem do Erro: {str(e)}\n"
                f"  Traceback Completo:\n{traceback.format_exc()}"
            )
        finally:
            if fig is not None:
                plt.close(fig)
            elif plt.get_fignums():
                REGISTRADOR.debug("Fechando todas as figuras ativas do matplotlib como fallback.")
                plt.close('all')