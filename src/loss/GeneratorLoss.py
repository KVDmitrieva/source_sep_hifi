import torch
import torch.nn as nn


class GeneratorLoss(nn.Module):
    def __init__(self, mel_lambda, fm_lambda):
        super().__init__()
        self.l1_loss = nn.L1Loss()
        self.mel_lambda = mel_lambda
        self.fm_lambda = fm_lambda

    def forward(self, gen_discriminator_out, mel, gen_mel, real_feature_map, gen_feature_map, **batch):
        adv_loss = 0.0
        for d_out in gen_discriminator_out:
            adv_loss += torch.mean((1 - d_out) ** 2)

        mel_loss = self.l1_loss(mel, gen_mel)

        feature_loss = 0.0
        for real, gen in zip(real_feature_map, gen_feature_map):
            for r, g in zip(real, gen):
                feature_loss += self.l1_loss(r, g)

        return {
            "generator_loss": adv_loss + self.mel_lambda * mel_loss + self.fm_lambda * feature_loss,
            "adv_loss": adv_loss,
            "mel_loss": mel_loss,
            "feature_loss": feature_loss
        }
