from torch import Tensor
from torchmetrics.audio import PerceptualEvaluationSpeechQuality

from hw_ss.metric.base_metric import BaseMetric


class PESQMetric(BaseMetric):
    def __init__(self, fs=16000, mode="wb", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pesq = PerceptualEvaluationSpeechQuality(fs, mode)

    def __call__(self, signal: Tensor, target: Tensor, **kwargs):
        signal = signal.cpu().detach()
        target = target.cpu().detach()
        try:
            return self.pesq(signal, target)
        except:
            return -0.5

