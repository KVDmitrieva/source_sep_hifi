from torch import Tensor
from torchmetrics.audio import ShortTimeObjectiveIntelligibility

from src.metric.base_metric import BaseMetric


class STOIMetric(BaseMetric):
    def __init__(self, fs=16000, extended=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stoi = ShortTimeObjectiveIntelligibility(fs, extended=extended)

    def __call__(self, generator_audio: Tensor, target_audio: Tensor, **kwargs):
        generator_audio = generator_audio.cpu().detach()
        target_audio = target_audio.cpu().detach()
        return self.stoi(generator_audio, target_audio)
