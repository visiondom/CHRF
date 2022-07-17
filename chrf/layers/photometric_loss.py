import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _quadruple


def ssim_loss(x, y, c1=0.01 ** 2, c2=0.03**2):
    """
    https://github.com/nianticlabs/monodepth2/blob/ab2a1bf7d45ae53b1ae5a858f4451199f9c213b3/layers.py#L218

    """
    padding = _quadruple(1)
    x = F.pad(x, padding, mode="replicate")
    y = F.pad(y, padding, mode="replicate")

    mu_x = F.avg_pool2d(x, kernel_size=3, stride=1)
    mu_y = F.avg_pool2d(y, kernel_size=3, stride=1)

    sigma_x = F.avg_pool2d(x ** 2, 3, 1) - mu_x ** 2
    sigma_y = F.avg_pool2d(y ** 2, 3, 1) - mu_y ** 2
    sigma_xy = F.avg_pool2d(x * y, 3, 1) - mu_x * mu_y

    ssim_n = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    ssim_d = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2)

    return torch.clamp((1 - ssim_n / ssim_d) / 2, min=0, max=1.0)
