import torch
import torch.nn as nn


class DiscriminatorLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, gen_discriminator_out, real_discriminator_out, **batch):
        adv_loss = 0.0

        for real, gen in zip(real_discriminator_out, gen_discriminator_out):
            adv_loss += torch.mean((1 - real) ** 2) + torch.mean(gen ** 2)

        return {
            "discriminator_loss": adv_loss,
        }
