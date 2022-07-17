import torch
import torch.nn.functional as F


def diff_crop(tensor, bbox, patch_size):
    """

    Args:
        tensor (Tensor):
        bbox (Tensor): 
        patch_size (Tuple[int]):

    Returns:

    """
    N, C, H, W = tensor.shape
    device = tensor.device
    bbox = bbox.float()
    x0, x1, y0, y1 = bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 3]
    patch_h, patch_w = patch_size

    a = ((x1 - x0) / (W - 1)).view(N, 1, 1)
    b = a.new_zeros(a.shape, device=device)
    c = (-1 + (x0 + x1) / (W - 1)).view(N, 1, 1)
    d = a.new_zeros(a.shape, device=device)
    e = ((y1 - y0) / (H - 1)).view(N, 1, 1)
    f = (-1 + (y1 + y0) / (H - 1)).view(N, 1, 1)

    theta = torch.cat(
        [
            torch.cat([a, b, c], dim=2),
            torch.cat([d, e, f], dim=2),
        ],
        dim=1
    )
    size = [N, C, patch_h, patch_w]
    grid = F.affine_grid(theta, size, align_corners=True)

    sampled_patch = F.grid_sample(tensor, grid, align_corners=True)

    return sampled_patch
