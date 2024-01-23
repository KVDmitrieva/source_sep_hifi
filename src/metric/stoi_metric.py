from torch import Tensor
from torchmetrics.audio import ShortTimeObjectiveIntelligibility

from hw_ss.metric.base_metric import BaseMetric


class STOIMetric(BaseMetric):
    def __init__(self, fs=16000, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stoi = ShortTimeObjectiveIntelligibility(fs)

    def __call__(self, signal: Tensor, target: Tensor, **kwargs):
        signal = signal.cpu().detach()
        target = target.cpu().detach()
        return self.stoi(signal, target)
