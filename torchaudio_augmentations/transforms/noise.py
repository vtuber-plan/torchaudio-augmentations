import random

import torch
from torch import nn
import torch.nn.functional as F

class GaussianNoise(torch.nn.Module):
    def __init__(self, min_snr=0.0001, max_snr=0.01):
        """
        :param min_snr: Minimum signal-to-noise ratio
        :param max_snr: Maximum signal-to-noise ratio
        """
        super().__init__()
        self.min_snr = min_snr
        self.max_snr = max_snr

    def forward(self, audio):
        std = torch.std(audio)
        noise_std = random.uniform(self.min_snr * std, self.max_snr * std)

        norm_dist = torch.distributions.normal.Normal(0.0, noise_std)
        noise = norm_dist.rsample(audio.shape).type(audio.dtype).to(audio.device)

        return audio + noise

class UniformWhiteNoise(torch.nn.Module):
    def __init__(self, max_snr=0.01):
        """
        :param max_snr: Maximum signal-to-noise ratio
        """
        super().__init__()
        self.max_snr = max_snr

    def forward(self, audio):
        noise_snr = random.random() * self.max_snr

        norm_dist = torch.distributions.uniform.Uniform(-noise_snr, noise_snr)
        noise = norm_dist.rsample(audio.shape).type(audio.dtype).to(audio.device)

        return audio + noise
