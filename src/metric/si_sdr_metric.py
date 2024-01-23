from torch import Tensor
from torchmetrics import ScaleInvariantSignalDistortionRatio

from src.metric.base_metric import BaseMetric


class SISDRMetric(BaseMetric):
    def __init__(self, zero_mean=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.si_sdr = ScaleInvariantSignalDistortionRatio(zero_mean=zero_mean)

    def __call__(self, signal: Tensor, target: Tensor, **kwargs):
        signal = signal.cpu().detach()
        target = target.cpu().detach()
        return self.si_sdr(signal, target)
