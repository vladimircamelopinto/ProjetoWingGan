import argparse
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import psutil
import traceback
import sys
import gc
from torch import Tensor
from Gerador import Gerador
from config import (REGISTRADOR, NUMERO_EPOCAS, DISPOSITIVO, TAMANHO_LOTE, EPOCAS_SALVAMENTO, DIMENSAO_RUIDO)
from Caminhos import Caminhos
from Discriminador import Discriminador
from PerdaPerceptual import PerdaPerceptual
from PerdaEstilo import PerdaEstilo
from PerdaVeias import PerdaVeias
from PerdaTextura import PerdaTextura
from PerdaAltaFrequencia import PerdaAltaFrequencia
from MonitorGradientes import MonitorGradientes
from UtilitariosImagem import UtilitariosImagem
from PreprocessarImagens import PreprocessarImagens
from GeradorRelatorio import GeradorRelatorio
from utils_imagem import preprocessar_imagens_offline, segmentar_figura

class WingGan:
    @staticmethod
    def registrar_metricas(caminhos: Caminhos, passo_g: int, p_d: float, p_g: float, acc_dr: float, acc_df: float, acc_gan_val: float) -> None:
        try:
            caminho_log_csv = caminhos.obter_caminho_completo(caminhos.REPOSITORIO_GAN_LOGS, "log_metricas.csv")
            REGISTRADOR.debug(f"Tentando registrar métricas no passo {passo_g} para {caminho_log_csv}")
            os.makedirs(os.path.dirname(caminho_log_csv), exist_ok=True)
            arquivo_novo = not os.path.exists(caminho_log_csv)
            with open(caminho_log_csv, 'a', newline='') as f:
                if arquivo_novo:
                    REGISTRADOR.info(f"Criando novo arquivo de métricas: {caminho_log_csv}")
                    f.write("passo_global,perda_discriminador,perda_gerador,acuracia_d_real,acuracia_d_falsa,acuracia_gan,cpu_percent,memory_percent,gpu_usage\n")
                with torch.no_grad():
                    gpu_usage = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() * 100 if torch.cuda.is_available() else 0
                    memoria = psutil.virtual_memory()
                    if memoria.percent > 85:
                        REGISTRADOR.warning(f"Memória alta ({memoria.percent}%). Liberando recursos antes de registrar métricas.")
                f.write(f"{passo_g},{p_d:.4f},{p_g:.4f},{acc_dr:.4f},{acc_df:.4f},{acc_gan_val:.4f},{psutil.cpu_percent()},{memoria.percent},{gpu_usage:.2f}\n")
            REGISTRADOR.debug(f"Métricas registradas com sucesso para passo {passo_g}.")
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            linha = exc_tb.tb_lineno
            REGISTRADOR.error(
                f"Erro em WingGan.py | Classe: WingGan | Método: registrar_metricas | Linha: {linha} | Detalhes: {str(e)} | Traceback: {''.join(traceback.format_tb(exc_tb))}"
            )

    @staticmethod
    def calcular_acuracia(previsoes_logits: Tensor, rotulos_alvo: Tensor) -> float:
        try:
            with torch.inference_mode():
                previsoes_probs = torch.sigmoid(previsoes_logits)
                previsoes_binarias = (previsoes_probs > 0.5).float()
                corretos = (previsoes_binarias == rotulos_alvo).float().sum()
                acuracia = corretos / rotulos_alvo.numel()
                return acuracia.item()
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            linha = exc_tb.tb_lineno
            REGISTRADOR.error(
                f"Erro em WingGan.py | Classe: WingGan | Método: calcular_acuracia | Linha: {linha} | "
                f"Detalhes: {str(e)} | Traceback: {''.join(traceback.format_tb(exc_tb))}"
            )
            return 0.0

    @staticmethod
    def perda_borda_sobel(imgs_geradas: Tensor, imgs_reais: Tensor) -> Tensor:
        try:
            dev = imgs_geradas.device
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=dev).view(1, 1, 3, 3)
            sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=dev).view(1, 1, 3, 3)
            criterio = nn.L1Loss()
            perda_total = 0.0
            num_canais = imgs_geradas.size(1)
            with torch.enable_grad():
                for c in range(num_canais):
                    canal_gerado = imgs_geradas[:, c:c + 1]
                    canal_real = imgs_reais[:, c:c + 1]
                    grad_gerado_x = F.conv2d(canal_gerado, sobel_x, padding=1)
                    grad_gerado_y = F.conv2d(canal_gerado, sobel_y, padding=1)
                    grad_real_x = F.conv2d(canal_real, sobel_x, padding=1)
                    grad_real_y = F.conv2d(canal_real, sobel_y, padding=1)
                    perda_total += criterio(grad_gerado_x, grad_real_x) + criterio(grad_gerado_y, grad_real_y)
            return perda_total / num_canais if num_canais > 0 else torch.tensor(0.0, device=dev)
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            linha = exc_tb.tb_lineno
            REGISTRADOR.error(
                f"Erro em WingGan.py | Classe: WingGan | Método: perda_borda_sobel | Linha: {linha} | "
                f"Detalhes: {str(e)} | Traceback: {''.join(traceback.format_tb(exc_tb))}"
            )
            return torch.tensor(0.0, device=imgs_geradas.device, requires_grad=True)

    @staticmethod
    def principal():
        try:
            REGISTRADOR.info("===== Iniciando WingGan Principal =====")
            # <<< NOVO >>> Instance noise schedule
            sigma_start = 0.1
            sigma_end   = 0.0

            tempo_inicio_total = time.time()

            parser = argparse.ArgumentParser(description='WingGan para geração de imagens')
            parser.add_argument('--epocas', '-e', type=int, default=NUMERO_EPOCAS, help='Número de épocas de treinamento')
            parser.add_argument('--dir_imagens', type=str, default='', help='Diretório de imagens originais')
            parser.add_argument('--dir_imagens_proc', type=str, default='', help='Diretório de imagens pré-processadas')
            parser.add_argument('--dir_gerador', type=str, default='', help='Diretório para salvar o gerador')
            parser.add_argument('--dir_discriminador', type=str, default='', help='Diretório para salvar o discriminador')
            parser.add_argument('--dir_modelos', type=str, default='', help='Diretório para salvar modelos')
            parser.add_argument('--dir_relatorios', type=str, default='', help='Diretório para relatórios')
            parser.add_argument('--dir_logs', type=str, default='', help='Diretório para logs')

            args = parser.parse_args()

            caminhos = Caminhos(
                dir_imagens=args.dir_imagens,
                dir_imagens_proc=args.dir_imagens_proc,
                dir_gerador=args.dir_gerador,
                dir_discriminador=args.dir_discriminador,
                dir_modelos=args.dir_modelos,
                dir_relatorios=args.dir_relatorios,
                dir_logs=args.dir_logs
            )
            REGISTRADOR.info(
                f"Configurações de Treinamento: Épocas={args.epocas}, Tam Lote={TAMANHO_LOTE}, Device='{DISPOSITIVO}'"
            )
            REGISTRADOR.info(f"Diretório de Imagens Originais: {caminhos.REPOSITORIO_IMAGENS}")
            REGISTRADOR.info(f"Diretório de Imagens Processadas: {caminhos.REPOSITORIO_IMAGENS_PROC}")

            with torch.no_grad():
                preprocessar_imagens_offline(caminhos.REPOSITORIO_IMAGENS, caminhos.REPOSITORIO_IMAGENS_PROC)

            carregador_dados, dataset = PreprocessarImagens.preprocessar_imagens(caminhos.REPOSITORIO_IMAGENS_PROC)

            REGISTRADOR.info("Inicializando modelos Gerador e Discriminador...")
            with torch.no_grad():
                gerador = Gerador(nc=3, ngf=64, z_dim=DIMENSAO_RUIDO).to(DISPOSITIVO)
                discriminador = Discriminador(nc=3, ndf=64).to(DISPOSITIVO)

            with torch.no_grad():
                perda_perceptual       = PerdaPerceptual().to(DISPOSITIVO)
                perda_estilo           = PerdaEstilo().to(DISPOSITIVO)
                perda_veias            = PerdaVeias().to(DISPOSITIVO)
                perda_textura          = PerdaTextura().to(DISPOSITIVO)
                perda_alta_frequencia  = PerdaAltaFrequencia().to(DISPOSITIVO)
                criterio_wgan          = nn.MSELoss()

            with torch.no_grad():
                # <<< NOVO >>> Rebalance learning rates
                otimizador_g = torch.optim.Adam(gerador.parameters(), lr=2e-4, betas=(0.5, 0.999))
                otimizador_d = torch.optim.Adam(discriminador.parameters(), lr=5e-5, betas=(0.5, 0.999))

            with torch.no_grad():
                escalonador_g = torch.optim.lr_scheduler.CosineAnnealingLR(otimizador_g, T_max=args.epocas)
                escalonador_d = torch.optim.lr_scheduler.CosineAnnealingLR(otimizador_d, T_max=args.epocas)

            scaler = None
            if DISPOSITIVO.type == 'cuda':
                scaler = torch.cuda.amp.GradScaler()

            def calcular_perda_discriminador(previsoes_reais: tuple[Tensor, Tensor], previsoes_falsas: tuple[Tensor, Tensor], lambda_gp: float = 10.0) -> Tensor:
                with torch.enable_grad():
                    # <<< NOVO >>> One-sided label smoothing for real
                    perda_d_real = -torch.mean(previsoes_reais[0]) * 0.9
                    perda_d_falsa = torch.mean(previsoes_falsas[0])
                    perda_d = perda_d_real + perda_d_falsa

                    alpha = torch.rand(TAMANHO_LOTE, 1, 1, 1).to(DISPOSITIVO)
                    interpolados = alpha * imagens_reais + (1 - alpha) * imagens_geradas.detach()
                    interpolados.requires_grad_(True)
                    saida_d_inter = discriminador(interpolados)
                    gradientes = torch.autograd.grad(outputs=saida_d_inter[0].sum(), inputs=interpolados,
                                                    create_graph=True, retain_graph=True)[0]
                    grad_norm = gradientes.view(TAMANHO_LOTE, -1).norm(2, dim=1)
                    perda_gp = lambda_gp * ((grad_norm - 1) ** 2).mean()
                    return perda_d + perda_gp

            def calcular_perda_gerador(imagens_geradas: Tensor, imagens_reais: Tensor, previsoes_falsas_g: tuple[Tensor, Tensor]) -> Tensor:
                with torch.enable_grad():
                    mascara_falsas = segmentar_figura(imagens_geradas)
                    if mascara_falsas is None or torch.isnan(mascara_falsas).any():
                        REGISTRADOR.error("Máscara inválida detectada em calcular_perda_gerador.")
                        return torch.tensor(0.0, device=DISPOSITIVO, requires_grad=True)
                    perda_g_gan       = -torch.mean(previsoes_falsas_g[0])
                    perda_g_perceptual = perda_perceptual(imagens_geradas, imagens_reais)
                    perda_g_veias     = perda_veias(imagens_geradas, imagens_reais)
                    perda_g_sobel     = WingGan.perda_borda_sobel(imagens_geradas, imagens_reais)
                    perda_g_mask      = torch.mean((mascara_falsas - 1) ** 2)
                    perda_total = (0.8 * perda_g_gan
                                  + 0.2 * perda_g_perceptual
                                  + 1.5 * perda_g_veias
                                  + 0.3 * perda_g_sobel
                                  + 0.1 * perda_g_mask)
                    REGISTRADOR.debug(f"Perdas - GAN: {perda_g_gan.item():.4f}, Perceptual: {perda_g_perceptual.item():.4f}, "
                                     f"Veias: {perda_g_veias.item():.4f}, Sobel: {perda_g_sobel.item():.4f}, Mask: {perda_g_mask.item():.4f}, Total: {perda_total.item():.4f}")
                    return perda_total

            REGISTRADOR.info(f"--- Iniciando Loop de Treinamento por {args.epocas} épocas ---")
            with torch.enable_grad():
                perdas_g_lista = []
                perdas_d_lista = []
                passo_global = 0
                monitor_grad = MonitorGradientes()
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                for epoca in range(1, args.epocas + 1):
                    REGISTRADOR.info(
                        f"--- Início Época [{epoca}/{args.epocas}] - "
                        f"LR Gerador: {otimizador_g.param_groups[0]['lr']:.1e}, "
                        f"LR Discriminador: {otimizador_d.param_groups[0]['lr']:.1e}"
                    )
                    t_inicio_epoca = time.time()
                    # <<< NOVO >>> compute decaying instance noise std
                    sigma = sigma_start + (sigma_end - sigma_start) * (epoca - 1) / (args.epocas - 1)

                    for i, (imagens_reais, _) in enumerate(carregador_dados):
                        try:
                            if not isinstance(imagens_reais, torch.Tensor):
                                REGISTRADOR.error(f"imagens_reais não é um tensor, tipo recebido: {type(imagens_reais)}. Pulando lote.")
                                continue
                            memoria = psutil.virtual_memory()
                            REGISTRADOR.debug(f"Lote {i+1}/{len(carregador_dados)} iniciado. RAM={memoria.percent:.2f}%, Shape imagens_reais={imagens_reais.shape}")
                            imagens_reais = imagens_reais.to(DISPOSITIVO)
                            tamanho_lote = imagens_reais.size(0)
                            if imagens_reais.shape[1] != 3 or imagens_reais.dim() != 4:
                                raise ValueError(f"Dimensão de imagens_reais inválida: esperado [batch, 3, h, w], recebido {imagens_reais.shape}")

                            with torch.no_grad():
                                imagens_iniciais = imagens_reais.clone()
                                ruido = torch.randn(tamanho_lote, DIMENSAO_RUIDO, 1, 1, device=DISPOSITIVO)
                            # <<< NOVO >>> add instance noise to real
                            if sigma > 0:
                                imagens_reais_noisy = imagens_reais + torch.randn_like(imagens_reais) * sigma
                            else:
                                imagens_reais_noisy = imagens_reais

                            # --- Treinar Discriminador (1 passo) ---
                            otimizador_d.zero_grad(set_to_none=True)
                            if DISPOSITIVO.type == 'cuda':
                                with torch.cuda.amp.autocast():
                                    imagens_geradas = gerador(imagens_iniciais, ruido)
                                    if imagens_geradas is None or imagens_geradas.shape[1] != 3:
                                        REGISTRADOR.error(f"imagens_geradas inválida: shape={imagens_geradas.shape if imagens_geradas is not None else 'None'}")
                                        continue
                                    previsoes_reais = discriminador(imagens_reais_noisy)
                                    previsoes_falsas = discriminador(imagens_geradas.detach())
                                    perda_d = calcular_perda_discriminador(previsoes_reais, previsoes_falsas)
                            else:
                                with torch.no_grad():
                                    imagens_geradas = gerador(imagens_iniciais, ruido)
                                    if imagens_geradas is None or imagens_geradas.shape[1] != 3:
                                        REGISTRADOR.error(f"imagens_geradas inválida: shape={imagens_geradas.shape if imagens_geradas is not None else 'None'}")
                                        continue
                                with torch.enable_grad():
                                    previsoes_reais = discriminador(imagens_reais_noisy)
                                    previsoes_falsas = discriminador(imagens_geradas.detach())
                                    perda_d = calcular_perda_discriminador(previsoes_reais, previsoes_falsas)

                            if scaler:
                                scaler.scale(perda_d).backward()
                                scaler.step(otimizador_d)
                                scaler.update()
                            else:
                                perda_d.backward()
                                otimizador_d.step()
                            REGISTRADOR.debug(f"Discriminador treinado. Perda d: {perda_d.item():.4f}")

                            # --- Treinar Gerador (2 passos) ---
                            for _ in range(2):  # <<< NOVO >>> 2 steps G for 1 step D
                                otimizador_g.zero_grad(set_to_none=True)
                                if DISPOSITIVO.type == 'cuda':
                                    with torch.cuda.amp.autocast():
                                        imagens_geradas = gerador(imagens_iniciais, ruido)
                                        if imagens_geradas is None or imagens_geradas.shape[1] != 3:
                                            REGISTRADOR.error(f"imagens_geradas inválida: shape={imagens_geradas.shape if imagens_geradas is not None else 'None'}")
                                            continue
                                        previsoes_falsas_g = discriminador(imagens_geradas)
                                        perda_g_total = calcular_perda_gerador(imagens_geradas, imagens_reais, previsoes_falsas_g)
                                        # <<< NOVO >>> diversity loss
                                        z2 = torch.randn_like(ruido)
                                        imgs2 = gerador(imagens_iniciais, z2)
                                        loss_div = -torch.mean(torch.abs(imagens_geradas - imgs2))
                                        perda_g_total = perda_g_total + 0.1 * loss_div
                                else:
                                    with torch.enable_grad():
                                        imagens_geradas = gerador(imagens_iniciais, ruido)
                                        if imagens_geradas is None or imagens_geradas.shape[1] != 3:
                                            REGISTRADOR.error(f"imagens_geradas inválida: shape={imagens_geradas.shape if imagens_geradas is not None else 'None'}")
                                            continue
                                        previsoes_falsas_g = discriminador(imagens_geradas)
                                        perda_g_total = calcular_perda_gerador(imagens_geradas, imagens_reais, previsoes_falsas_g)
                                        # <<< NOVO >>> diversity loss
                                        z2 = torch.randn_like(ruido)
                                        imgs2 = gerador(imagens_iniciais, z2)
                                        loss_div = -torch.mean(torch.abs(imagens_geradas - imgs2))
                                        perda_g_total = perda_g_total + 0.1 * loss_div

                                if torch.isnan(perda_g_total).any() or torch.isinf(perda_g_total).any():
                                    REGISTRADOR.error(f"Perda do Gerador contém NaN ou Inf: {perda_g_total}")
                                    continue

                                if scaler:
                                    scaler.scale(perda_g_total).backward()
                                    scaler.step(otimizador_g)
                                    scaler.update()
                                else:
                                    perda_g_total.backward()
                                    otimizador_g.step()
                                REGISTRADOR.debug(f"Gerador treinado. Perda: {perda_g_total.item():.4f}")

                            with torch.no_grad():
                                monitor_grad.registrar_normas_gradientes(gerador, passo_global, "Gerador")
                                monitor_grad.registrar_normas_gradientes(discriminador, passo_global, "Discriminador")

                            with torch.inference_mode():
                                acc_d_real = WingGan.calcular_acuracia(previsoes_reais[0], torch.ones_like(previsoes_reais[0], device=DISPOSITIVO))
                                acc_d_falsa = WingGan.calcular_acuracia(previsoes_falsas[0], torch.zeros_like(previsoes_falsas[0], device=DISPOSITIVO))
                                acc_gan    = WingGan.calcular_acuracia(previsoes_falsas_g[0], torch.ones_like(previsoes_falsas_g[0], device=DISPOSITIVO))

                            perdas_g_lista.append(perda_g_total.item())
                            perdas_d_lista.append(perda_d.item())
                            WingGan.registrar_metricas(
                                caminhos,
                                passo_g=passo_global,
                                p_d=perda_d.item(),
                                p_g=perda_g_total.item(),
                                acc_dr=acc_d_real,
                                acc_df=acc_d_falsa,
                                acc_gan_val=acc_gan
                            )

                            if epoca % 1 == 0:
                                UtilitariosImagem.salvar_imagens(
                                    caminhos,  # Passe o objeto Caminhos
                                    imagens_geradas,
                                    f"epoca_{epoca}_passo_{i}",
                                    imagens_reais
                                )
                                REGISTRADOR.info(f"Imagens sintéticas salvas para época {epoca}, passo {i}")

                            del imagens_geradas, previsoes_reais, previsoes_falsas, previsoes_falsas_g, perda_d, perda_g_total
                            if DISPOSITIVO.type == 'cuda':
                                torch.cuda.empty_cache()
                            else:
                                gc.collect()
                            passo_global += 1
                            REGISTRADOR.debug(f"Lote {i+1} concluído. RAM={psutil.virtual_memory().percent:.2f}%")

                        except Exception as e:
                            exc_type, exc_obj, exc_tb = sys.exc_info()
                            linha = exc_tb.tb_lineno
                            REGISTRADOR.error(
                                f"Erro no lote {i+1} da época {epoca}: {str(e)} | Linha: {linha} | "
                                f"Traceback: {''.join(traceback.format_tb(exc_tb))}"
                            )
                            continue

                    otimizador_g.step()
                    otimizador_d.step()
                    escalonador_g.step()
                    escalonador_d.step()

                    if epoca % EPOCAS_SALVAMENTO == 0:
                        caminho_gerador      = caminhos.obter_caminho_completo(caminhos.REPOSITORIO_GAN_MODELOS, f"Gerador_epoca_{epoca}.pth")
                        caminho_discriminador = caminhos.obter_caminho_completo(caminhos.REPOSITORIO_GAN_DISCRIMINADOR, f"Discriminador_epoca_{epoca}.pth")
                        torch.save(gerador.state_dict(), caminho_gerador)
                        torch.save(discriminador.state_dict(), caminho_discriminador)

                    caminho_grafico = caminhos.obter_caminho_completo(caminhos.REPOSITORIO_GAN_RELATORIOS, f"grafico_perdas_epoca_{epoca}.png")
                    GeradorRelatorio.gerar_grafico_perda(perdas_g_lista, perdas_d_lista, caminho_grafico)

                    t_fim_epoca = time.time()
                    REGISTRADOR.info(f"Fim da Época {epoca}. Tempo: {(t_fim_epoca - t_inicio_epoca):.2f}s")

        except KeyboardInterrupt:
            REGISTRADOR.info("Treinamento interrompido pelo usuário (Ctrl+C).")
            REGISTRADOR.info("Tentando salvar modelos atuais devido à interrupção...")
            caminho_gerador       = caminhos.obter_caminho_completo(caminhos.REPOSITORIO_GAN_MODELOS, "Gerador_INTERROMPIDO.pth")
            caminho_discriminador = caminhos.obter_caminho_completo(caminhos.REPOSITORIO_GAN_DISCRIMINADOR, "Discriminador_INTERROMPIDO.pth")
            torch.save(gerador.state_dict(), caminho_gerador)
            torch.save(discriminador.state_dict(), caminho_discriminador)
            raise Exception("Treinamento interrompido pelo usuário")
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            linha = exc_tb.tb_lineno
            REGISTRADOR.error(
                f"Erro em WingGan.py | Método: principal | Linha: {linha} | Detalhes: {str(e)} | "
                f"Traceback: {''.join(traceback.format_tb(exc_tb))}"
            )
            raise
        finally:
            if 'perdas_g_lista' in locals() and 'perdas_d_lista' in locals() and perdas_g_lista and perdas_d_lista:
                caminho_grafico = caminhos.obter_caminho_completo(caminhos.REPOSITORIO_GAN_RELATORIOS, "grafico_perdas_final.png")
                GeradorRelatorio.gerar_grafico_perda(perdas_g_lista, perdas_d_lista, caminho_grafico)

            tempo_total    = time.time() - tempo_inicio_total
            minutos_total  = int(tempo_total // 60)
            segundos_total = int(tempo_total % 60)
            REGISTRADOR.info(f"--- Finalizando Treinamento ---")
            REGISTRADOR.info(f"Tempo total de execução da aplicação: {minutos_total//60}h {minutos_total%60}m {segundos_total:.2f}s")
            REGISTRADOR.info("===== Aplicação WingGan Finalizada =====")

if __name__ == "__main__":
    try:
        print("Backends disponíveis para torch.compile (Dynamo):", torch._dynamo.list_backends())
    except Exception as e:
        print(f"Erro ao listar backends: {e}")
    WingGan.principal()
