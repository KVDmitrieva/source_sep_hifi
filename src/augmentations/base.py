from torch import Tensor


class AugmentationBase:
    def __call__(self, data: Tensor) -> Tensor:
        raise NotImplementedError()
