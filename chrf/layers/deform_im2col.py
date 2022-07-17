import torch
import torch.nn.functional as F
from chrf.nn.track_utils import grid_sampler_nomalize


def deform_im2col(tensor: torch.Tensor, offsets: torch.Tensor, kernel_size=(3, 3)):
    device = tensor.device
    H, W = tensor.shape[-2:]
    base_grid_y = torch.arange(start=0, end=H)
    base_grid_x = torch.arange(start=0, end=W)
    base_grid_y, base_grid_x = torch.meshgrid(base_grid_y, base_grid_x)
    # base_grid: [H, W, 2] (x, y) order
    base_grid = torch.stack([base_grid_x, base_grid_y], dim=-1).contiguous().to(device)

    kh, kw = kernel_size
    radius_h = (kh - 1) // 2
    radius_w = (kw - 1) // 2
    offset_y = torch.arange(start=-radius_h, end=radius_h + 1)
    offset_x = torch.arange(start=-radius_w, end=radius_w + 1)
    offset_y, offset_x = torch.meshgrid(offset_y, offset_x)
    # sd_offsets: [PH, PW, 2] (x, y) order
    sd_offsets = torch.stack([offset_x, offset_y], dim=-1).contiguous().to(device)
    # offsets: [N, H, W, 2]
    # sd_grid: [N, PH, PW, H, W, 2] (x, y) order
    sd_grid = (
        base_grid.unsqueeze(0).unsqueeze(1).unsqueeze(1)
        + offsets.unsqueeze(1).unsqueeze(1)
        + sd_offsets.unsqueeze(0).unsqueeze(-2).unsqueeze(-2)
    )
    N, PH, PW, H, W = sd_grid.shape[:-1]

    sd_grid = sd_grid.view(N, -1, H, W, 2).view(N, PH * PW, H * W, 2)
    sd_grid_norm = grid_sampler_nomalize(sd_grid, size=(H, W), align_corners=True)

    unfold_tensor = F.grid_sample(tensor, sd_grid_norm, align_corners=True)
    unfold_tensor = unfold_tensor.view(N, -1, PH * PW, H, W)

    return unfold_tensor
