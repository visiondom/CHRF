import torch
import torch.nn.functional as F

CCTT_LOSSES = dict()


def concentration_loss_v1(grid: torch.Tensor, coords: torch.Tensor, win_len, stride):
    assert grid.shape == coords.shape
    N, C, H, W = coords.shape

    unfolded_grid = F.unfold(grid, kernel_size=win_len, stride=stride)
    unfolded_coords = F.unfold(coords, kernel_size=win_len, stride=stride)

    unfolded_grid = unfolded_grid.reshape(N, C, win_len ** 2, -1)
    unfolded_coords = unfolded_coords.reshape(N, C, win_len ** 2, -1)

    center = unfolded_coords.mean(dim=2, keepdim=True)
    dis2center = (unfolded_coords - center) ** 2
    dis2center = dis2center.mean(dim=1)

    return dis2center.sum() / dis2center.numel()

CCTT_LOSSES["v1"] = concentration_loss_v1


def concentration_loss_v2(grid: torch.Tensor, coords: torch.Tensor, win_len, stride):
    assert grid.shape == coords.shape
    N, C, H, W = coords.shape

    unfolded_grid = F.unfold(grid, kernel_size=win_len, stride=stride)
    unfolded_coords = F.unfold(coords, kernel_size=win_len, stride=stride)

    unfolded_grid = unfolded_grid.reshape(N, C, win_len ** 2, -1)
    unfolded_coords = unfolded_coords.reshape(N, C, win_len ** 2, -1)

    center = unfolded_coords.mean(dim=2, keepdim=True)
    dis2center = (unfolded_coords - center) ** 2
    dis2center = dis2center.mean(dim=1)

    return dis2center.sum() / dis2center.numel()
    
CCTT_LOSSES["v2"] = concentration_loss_v2


def concentration_loss(grid: torch.Tensor, coords: torch.Tensor, win_len, stride, mode="v1"):
    return CCTT_LOSSES[mode](grid, coords, win_len, stride)
