from torch import Tensor, tensor
from torchmetrics.audio import PerceptualEvaluationSpeechQuality

from src.metric.base_metric import BaseMetric


class PESQMetric(BaseMetric):
    def __init__(self, fs=16000, mode="wb", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pesq = PerceptualEvaluationSpeechQuality(fs, mode)

    def __call__(self, generator_audio: Tensor, target_audio: Tensor, **kwargs):
        generator_audio = generator_audio.cpu().detach()
        target_audio = target_audio.cpu().detach()
        try:
            return self.pesq(generator_audio, target_audio)
        except:
            return tensor(1)
