from torch import Tensor

from hw_ss.metric.base_metric import BaseMetric
from torchmetrics import ScaleInvariantSignalDistortionRatio


class SISDRMetric(BaseMetric):
    def __init__(self, zero_mean=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.si_sdr = ScaleInvariantSignalDistortionRatio(zero_mean=zero_mean)

    def __call__(self, signal: Tensor, target: Tensor, **kwargs):
        signal = signal.cpu().detach()
        target = target.cpu().detach()
        return self.si_sdr(signal, target)
