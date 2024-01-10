import torch_audiomentations
from torch import Tensor

from src.augmentations.base import AugmentationBase


class Shift(AugmentationBase):
    def __init__(self, *args, **kwargs):
        """
        Shift the audio forwards or backwards, with or without rollover.
        :param min_shift: Union[float, int] = -0.5
        :param max_shift: Union[float, int] = 0.5
        :param shift_unit: str = "fraction"
        :param rollover: bool = True
        :param mode: str = "per_example"
        :param p: float = 0.5
        :param p_mode: Optional[str] = None
        :param sample_rate: Optional[int] = None
        :param target_rate: Optional[int] = None
        :param output_type: Optional[str] = None
        """
        self._aug = torch_audiomentations.Shift(*args, **kwargs)

    def __call__(self, data: Tensor):
        x = data.unsqueeze(1)
        return self._aug(x).squeeze(1)
