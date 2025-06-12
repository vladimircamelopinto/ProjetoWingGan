import torch
import torch.nn as nn
from config import REGISTRADOR
import psutil

class MonitorGradientes:
    @staticmethod
    def registrar_normas_gradientes(modelo: nn.Module, passo_global: int, nome_modelo: str) -> None:

        if passo_global % 50 == 0:
            memoria = psutil.virtual_memory()
            if memoria.percent > 85:
                REGISTRADOR.warning(f"Memória alta ({memoria.percent}%). Liberando recursos antes de calcular gradientes.")

            def calcular_norma(modelo: nn.Module) -> tuple[float, int]:
                norma_total = 0.0
                num_params_com_grad = 0
                for p in modelo.parameters():
                    if p.grad is not None:
                        norma_param = p.grad.data.norm(2)
                        norma_total += norma_param.item() ** 2
                        num_params_com_grad += 1
                return norma_total, num_params_com_grad

            with torch.no_grad():
                norma_total, num_params_com_grad = calcular_norma(modelo)

            if num_params_com_grad > 0:
                norma_total = norma_total ** 0.5
                REGISTRADOR.debug(
                    f"Norma L2 dos gradientes de '{nome_modelo}' (Passo Global {passo_global}): {norma_total:.4f}"
                )