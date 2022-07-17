import torch
import torch.nn.functional as F


def transform(affinity, frame):
    """

    Args:
        affinity:
        frame:

    Returns:

    """
    N, C, H, W = frame.shape
    # frame: [N, C, H*W]
    # affinity: [N, H*W, H*W]
    frame_tf = torch.bmm(frame.view(N, C, -1), affinity).view(N, C, H, W)
    # frame_tf: [N, C, H, W]

    return frame_tf


def im2col(tensor, win_len=3, stride=8):
    """

    Args:
        tensor (Tensor):
        win_len (int):
        stride (int):

    Returns:

    """
    N, C = tensor.shape[:2]

    unfolded_tensor = F.unfold(tensor, kernel_size=win_len, stride=stride)
    unfolded_tensor = unfolded_tensor.view(N, C, win_len ** 2, -1).permute(0, 1, 3, 2)

    return unfolded_tensor


def aff2coord(grid, affinity=None, softmax=False, temp=1):
    """

    Args:
        grid (Tensor):
        affinity (Tensor):
        softmax (bool):
        temp (int):

    Returns:

    """
    if affinity is not None:
        if softmax:
            affinity = F.softmax(affinity * temp, dim=1)
        coordinates = transform(affinity, grid)
    else:
        coordinates = grid

    return coordinates


def create_grid(shape, device, flatten=False, align_corners=False):
    """

    Args:
        shape:
        device:
        flatten:
        align_corners:

    Returns:

    """
    N, _, H, W = shape

    theta = torch.tensor([[1, 0, 0], [0, 1, 0]]).float().to(device)
    theta = theta.unsqueeze(0).repeat(N, 1, 1)

    # grid [N, H, W, 2], in (x, y)
    grid = F.affine_grid(theta, shape, align_corners=align_corners)

    if flatten:
        grid = grid.view(N, -1, 2)

    return grid


def concentration_loss(grid, coords: torch.Tensor, win_len):
    N, C, H, W = coords.shape
    unfolded_coords = F.unfold(coords, kernel_size=win_len)
    unfolded_coords = unfolded_coords.reshape(N, C, win_len ** 2, -1)
    center = unfolded_coords.mean(dim=2)


def concentration_switch_loss(affinity, shape, temp, win_len, stride):
    """

    Args:
        affinity:
        shape:
        temp:
        win_len:
        stride:

    Returns:

    """
    N, _, H, W = shape
    grid = create_grid(shape, affinity.device).permute(0, 3, 1, 2)

    coord1 = aff2coord(grid, affinity, softmax=True, temp=temp)
    coord2 = aff2coord(grid, affinity.permute(0, 2, 1), softmax=True, temp=temp)

    unfolded_coords1 = im2col(coord1, win_len, stride=stride)
    unfolded_coords2 = im2col(coord2, win_len, stride=stride)

    # compute grids' center
    center1 = torch.mean(unfolded_coords1, dim=3)
    center1 = center1.view(N, 2, -1, 1)
    center2 = torch.mean(unfolded_coords2, dim=3)
    center2 = center2.view(N, 2, -1, 1)

    # compute point distance to center
    dis2center1 = (unfolded_coords1 - center1) ** 2
    dis2center2 = (unfolded_coords2 - center2) ** 2

    loss = (torch.sum(dis2center1) + torch.sum(dis2center2)) / dis2center1.numel()

    return loss
