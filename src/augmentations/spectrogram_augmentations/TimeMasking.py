import torchaudio.transforms as t
from torch import Tensor

from src.augmentations.base import AugmentationBase


class TimeMasking(AugmentationBase):
    def __init__(self, *args, **kwargs):
        """
        Apply masking to a spectrogram in the time domain.
        :param time_mask_param: maximum possible length of the mask. Indices uniformly sampled from [0, time_mask_param)
        :param iid_masks: whether to apply different masks to each example/channel in the batch
        :param p: maximum proportion of time steps that can be masked. Must be within range [0.0, 1.0]
        """
        self._aug = t.TimeMasking(*args, **kwargs)

    def __call__(self, data: Tensor):
        return self._aug(data)
