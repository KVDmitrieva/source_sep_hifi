from torch import Tensor, tensor, no_grad

from torchmetrics.audio import SignalDistortionRatio, ScaleInvariantSignalDistortionRatio
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility

from src.metric.base_metric import BaseMetric


__all__ = ['PESQMetric', 'SDRMetric', 'SISDRMetric', 'STOIMetric']


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
            return tensor(1.0)


class SDRMetric(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sdr = SignalDistortionRatio()

    def __call__(self, generator_audio: Tensor, target_audio: Tensor, **kwargs):
        generator_audio = generator_audio.cpu().detach()
        target_audio = target_audio.cpu().detach()
        return self.sdr(generator_audio, target_audio)
    

class SISDRMetric(BaseMetric):
    def __init__(self, zero_mean=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.si_sdr = ScaleInvariantSignalDistortionRatio(zero_mean=zero_mean)

    def __call__(self, generator_audio: Tensor, target_audio: Tensor, **kwargs):
        generator_audio = generator_audio.cpu().detach()
        target_audio = target_audio.cpu().detach()
        return self.si_sdr(generator_audio, target_audio)


class STOIMetric(BaseMetric):
    def __init__(self, fs=16000, extended=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stoi = ShortTimeObjectiveIntelligibility(fs, extended=extended)

    def __call__(self, generator_audio: Tensor, target_audio: Tensor, **kwargs):
        generator_audio = generator_audio.cpu().detach()
        target_audio = target_audio.cpu().detach()
        return self.stoi(generator_audio, target_audio)
