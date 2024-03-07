import torch.nn as nn
import torch.nn.functional as F


def init_weights(module, mean=0.0, std=0.01):
    if isinstance(module, nn.Conv1d) or isinstance(module, nn.ConvTranspose1d) or isinstance(module, nn.Conv2d):
        module.weight.data.normal_(mean=mean, std=std)


def fix_shapes_1d(x, skip, mode='crop'):
    assert mode in ['crop', 'pad']

    diff = skip.shape[-1] - x.shape[-1]
    if mode == 'crop' and diff > 0:
        diff1 = (diff + 1) // 2
        diff2 = diff // 2
        return x, skip[..., diff2:-diff1]

    return F.pad(x, (0, diff)), skip


def fix_shapes_2d(x, skip, mode='crop'):
    assert mode in ['crop', 'pad']

    diff2 = skip.shape[-2] - x.shape[-2]
    diff1 = skip.shape[-1] - x.shape[-1]
    if mode == 'crop':
        diff1 = diff1 // 2
        diff2 = diff2 // 2
        return x, skip[..., diff2:-diff2, diff1:-diff1]

    return F.pad(x, (0, diff1, 0, diff2)), skip


def closest_power_of_two(n):
    return 1 << (n - 1).bit_length()

