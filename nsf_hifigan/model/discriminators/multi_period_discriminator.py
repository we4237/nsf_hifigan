
from typing import List
import torch
from torch import nn
from torch.nn import functional as F

from .discriminator import DiscriminatorP, DiscriminatorS

class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self, periods=None):
        super(MultiPeriodDiscriminator, self).__init__()
        self.periods = periods if periods is not None else [2, 3, 5, 7, 11]
        self.discriminators = nn.ModuleList()
        for period in self.periods:
            self.discriminators.append(DiscriminatorP(period))

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


