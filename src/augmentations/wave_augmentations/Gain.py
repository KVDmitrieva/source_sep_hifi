import torch_audiomentations
from torch import Tensor

from src.augmentations.base import AugmentationBase


class Gain(AugmentationBase):
    def __init__(self, *args, **kwargs):
        """
        Multiply the audio by a random amplitude factor to reduce or increase the volume.
        :param min_gain_in_db: float = -18.0
        :param max_gain_in_db: float = 6.0
        :param mode: str = "per_example"
        :param p: float = 0.5
        :param p_mode: Optional[str] = None
        :param sample_rate: Optional[int] = None
        :param target_rate: Optional[int] = None
        :param output_type: Optional[str] = None
        """
        self._aug = torch_audiomentations.Gain(*args, **kwargs)

    def __call__(self, data: Tensor):
        x = data.unsqueeze(1)
        return self._aug(x).squeeze(1)
