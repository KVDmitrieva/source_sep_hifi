import torchaudio.transforms as t
from torch import Tensor

from src.augmentations.base import AugmentationBase


class FrequencyMasking(AugmentationBase):
    def __init__(self, *args, **kwargs):
        """
        Apply masking to a spectrogram in the frequency domain.
        :param freq_mask_param: maximum possible length of the mask. Indices uniformly sampled from [0, freq_mask_param)
        :param iid_masks: whether to apply different masks to each example/channel in the batch
        """
        self._aug = t.FrequencyMasking(*args, **kwargs)

    def __call__(self, data: Tensor):
        return self._aug(data)
