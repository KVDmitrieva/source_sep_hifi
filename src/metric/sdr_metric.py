from torch import Tensor
from torchmetrics.audio import SignalDistortionRatio
from src.metric.base_metric import BaseMetric


class SDRMetric(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sdr = SignalDistortionRatio()

    def __call__(self, generator_audio: Tensor, target_audio: Tensor, **kwargs):
        generator_audio = generator_audio.cpu().detach()
        target_audio = target_audio.cpu().detach()
        return self.sdr(generator_audio, target_audio)



