
import os
import torch
import logging
import psutil
import numpy as np
from torch import Tensor

# Configurações
DIMENSAO_RUIDO = 100
CANAIS_IMAGEM = 3
IMAGE_WIDTH = 1825 #2450
IMAGE_HEIGHT = 1325 #1850
TAMANHO_LOTE = 2 #8
NUMERO_EPOCAS = 50 #10000
PASSOS_ACUMULACAO = 4
EPOCAS_SALVAMENTO = 10

DISPOSITIVO = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUMERO_TRABALHADORES = os.cpu_count() // 2 if os.cpu_count() is not None and os.cpu_count() > 0 else 0

if DISPOSITIVO.type == 'cuda':
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.set_float32_matmul_precision('high')
else:
    torch.backends.mkl.enabled = True
    torch.backends.mkldnn.enabled = True
    torch.jit.enable_onednn_fusion(True)

if NUMERO_TRABALHADORES > 0:
    torch.set_num_threads(max(1, NUMERO_TRABALHADORES))
torch.set_num_interop_threads(1)
torch.set_flush_denormal(True)

try:
    processo = psutil.Process()
    if os.name == 'nt':
        processo.nice(psutil.HIGH_PRIORITY_CLASS)
except Exception as e:
    print(f"Alerta: Falha ao definir a prioridade do processo: {e}")

log_file_full_path = os.path.join('C:\\', 'Users', 'vladi', 'eclipse-workspace', 'WingGAN', 'WingGan', 'Log', 'wgan.log')
log_directory = os.path.dirname(log_file_full_path)
os.makedirs(log_directory, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file_full_path, encoding='utf-8')
    ]
)
REGISTRADOR = logging.getLogger(__name__)

def tensor_para_numpy(tensor: Tensor) -> np.ndarray:
    return tensor.permute(0, 2, 3, 1).cpu().detach().numpy() * 0.5 + 0.5