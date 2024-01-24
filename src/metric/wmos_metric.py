import torch

from wvmos import get_wvmos

from src.metric.base_metric import BaseMetric


class WMOSMetric(BaseMetric):
    def __init__(self, cuda=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = get_wvmos(cuda=cuda)

    def __call__(self, generator_audio: torch.Tensor, **kwargs):
        x = generator_audio.detach()
        x = self.model.processor(x, return_tensors="pt", padding=True, sampling_rate=16000).input_values
        x = x.squeeze()
        # x = x.unsqueeze(1)
        with torch.no_grad():
            res = self.model.forward(x.to(generator_audio.device)).mean()
        return res.cpu().item()
