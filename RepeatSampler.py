import torch
from torch.utils.data import Sampler

class RepeatSampler(Sampler):
    def __init__(self, dataset, num_repeats=10):
        try:
            with torch.inference_mode():  # Verificações sem índices
                self.dataset = dataset
                self.num_repeats = num_repeats
                self.indices = list(range(len(dataset))) * num_repeats
        except Exception as e:
            raise RuntimeError(f"Falha na inicialização do RepeatSampler: {str(e)}")

    def __iter__(self):
        try:
            return iter(self.indices)
        except Exception as e:
            raise RuntimeError(f"Falha ao iterar no RepeatSampler: {str(e)}")

    def __len__(self):
        try:
            return len(self.indices)
        except Exception as e:
            raise RuntimeError(f"Falha ao obter comprimento do RepeatSampler: {str(e)}")