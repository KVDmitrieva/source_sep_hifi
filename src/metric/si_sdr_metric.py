from torch import Tensor
from torchmetrics import ScaleInvariantSignalDistortionRatio

from src.metric.base_metric import BaseMetric


class SISDRMetric(BaseMetric):
    def __init__(self, zero_mean=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.si_sdr = ScaleInvariantSignalDistortionRatio(zero_mean=zero_mean)

    def __call__(self, generator_audio: Tensor, target_audio: Tensor, **kwargs):
        generator_audio = generator_audio.cpu().detach()
        target_audio = target_audio.cpu().detach()
        return self.si_sdr(generator_audio, target_audio)
