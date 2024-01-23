import torch
from torch import Tensor


def calc_si_sdr(preds: Tensor, targets: Tensor):
    eps = 1e-8
    alpha = (targets * preds).sum(dim=-1, keepdim=True) / torch.sum(targets ** 2, dim=-1, keepdim=True)
    scaled_target = alpha * targets
    val = (scaled_target ** 2).sum(dim=-1) / (torch.sum((scaled_target - preds) ** 2, dim=-1) + eps) + eps
    return torch.mean(10 * torch.log10(val))

