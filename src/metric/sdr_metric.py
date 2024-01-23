from torch import Tensor
from torchmetrics.audio import SignalDistortionRatio
from hw_ss.metric.base_metric import BaseMetric


class SDRMetric(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sdr = SignalDistortionRatio()

    def __call__(self, signal: Tensor, target: Tensor, **kwargs):
        signal = signal.cpu().detach()
        target = target.cpu().detach()
        return self.sdr(signal, target)



