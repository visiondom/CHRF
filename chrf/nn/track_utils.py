import torch
import torch.nn.functional as F


def grid_sampler_nomalize(grid, size, align_corners=True):
    """
    Unnormalizes a coordinate from the -1 to +1 scale to its pixel index value,
    where we view each pixel as an area between (idx - 0.5) and (idx + 0.5).
    if align_corners: -1 and +1 get sent to the centers of the corner pixels
            0 --> -1
            (size - 1) --> +1
            scale_factor = (size - 1) / 2
    if not align_corners: -1 and +1 get sent to the image edges
            -1 <-- -0.5
            +1 <-- (size - 1) + 0.5 == size - 0.5
            scale_factor = size / 2
    Args:
        grid:
        size:
        align_corners:

    Returns:

    """
    assert grid.shape[-1] == 2
    H, W = size

    grid_norm = torch.empty_like(grid)
    if align_corners:
        grid_norm[..., 0] = grid[..., 0] * 2 / (W - 1) - 1
        grid_norm[..., 1] = grid[..., 1] * 2 / (H - 1) - 1
    else:
        grid_norm[..., 0] = (grid[..., 0] * 2 + 1) / W - 1
        grid_norm[..., 1] = (grid[..., 1] * 2 + 1) / H - 1

    return grid_norm


def grid_sampler_unnormalize(grid, size, align_corners=True):
    """
    Unnormalizes a coordinate from the -1 to +1 scale to its pixel index value,
    where we view each pixel as an area between (idx - 0.5) and (idx + 0.5).
    if align_corners: -1 and +1 get sent to the centers of the corner pixels
            -1 --> 0
            +1 --> (size - 1)
            scale_factor = (size - 1) / 2
    if not align_corners: -1 and +1 get sent to the image edges
            -1 --> -0.5
            +1 --> (size - 1) + 0.5 == size - 0.5
            scale_factor = size / 2
    Args:
        grid:
        size:
        align_corners:

    Returns:

    """
    H, W = size
    grid_unnorm = torch.empty_like(grid)
    if align_corners:
        grid_unnorm[..., 0] = (grid[..., 0] + 1) / 2 * (W - 1)
        grid_unnorm[..., 1] = (grid[..., 1] + 1) / 2 * (H - 1)
    else:
        grid_unnorm[..., 0] = ((grid[..., 0] + 1) * W - 1) / 2
        grid_unnorm[..., 1] = ((grid[..., 1] + 1) * H - 1) / 2

    return grid_unnorm


def create_grid(shape, device, flatten=False, align_corners=True):
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
    grid = grid_sampler_unnormalize(grid, (H, W), align_corners=align_corners)

    if flatten:
        grid = grid.view(N, -1, 2)

    return grid


def get_affinity(query, target):
    N, C = query.shape[:2]
    query = query.view(N, C, -1).permute(0, 2, 1)
    target = target.view(N, C, -1)

    affinity = torch.bmm(query, target)

    return affinity


def coords_trans(points, hmat):
    assert isinstance(points, torch.Tensor)
    assert isinstance(hmat, torch.Tensor)
    assert points.dim() == hmat.dim()

    squeeze = False
    if points.dim() == 2:
        points = points.unsqueeze(0)
        hmat = hmat.unsqueeze(0)
        squeeze = True

    N, NC = points.shape[:2]
    ones = torch.ones(size=(N, NC, 1), device=points.device)
    points = torch.cat([points, ones], dim=2).permute(0, 2, 1)
    hmat = hmat.to(points.device)

    new_points = torch.bmm(hmat.float(), points).permute(0, 2, 1)
    new_points = new_points / new_points[:, :, -1:]

    if squeeze:
        new_points = new_points.squeeze(0)

    return new_points[:, :, :2]


def center_to_bbox(coordinates, patch_size, target_h, target_w):
    """

    Args:
        coordinates (Tensor):
        patch_size (tuple[int]): (h, w)
        target_h (int):
        target_w (int):

    Returns:

    """
    # 1. compute coordinates' center use mean
    center = torch.mean(coordinates, dim=1)

    # 2. compute box edge using fixed patch size
    left = center[:, 0:1] - patch_size[1] / 2
    right = left + patch_size[1]
    top = center[:, 1:2] - patch_size[0] / 2
    bottom = top + patch_size[0]

    # 3. clip box to target shape
    left[left < 0] = 0
    right[right > target_h] = target_h
    top[top < 0] = 0
    bottom[bottom > target_w] = target_w

    box = torch.cat([left, right, top, bottom], dim=1)

    return box


def coords_to_bbox(coordinates, target_h, target_w):
    """

    Args:
        coordinates:
        target_h:
        target_w:

    Returns:

    """
    N, PN = coordinates.shape[:2]
    # 1. compute coordinates' center use mean
    center = torch.mean(coordinates, dim=1)
    repeated_center = center.unsqueeze(1).repeat(1, PN, 1)

    dis_x = torch.sqrt(torch.pow(coordinates[:, :, 0] - repeated_center[:, :, 0], 2))
    dis_x = torch.mean(dis_x, dim=1).detach()
    dis_y = torch.sqrt(torch.pow(coordinates[:, :, 1] - repeated_center[:, :, 1], 2))
    dis_y = torch.mean(dis_y, dim=1).detach()

    # 2. compute box edge using adaptive size
    left = (center[:, 0] - dis_x * 2).view(N, 1)
    right = (center[:, 0] + dis_x * 2).view(N, 1)
    top = (center[:, 1] - dis_y * 2).view(N, 1)
    bottom = (center[:, 1] + dis_y * 2).view(N, 1)

    # 3. clip box to target shape
    left[left < 0] = 0
    right[right > target_w] = target_w
    top[top < 0] = 0
    bottom[bottom > target_h] = target_h

    box = torch.cat([left, right, top, bottom], dim=1)

    return box


def warp(seg, affinity, topk=1):
    """
    :param seg [N, C, H, W]:
    :param affinity [N, H*W, N2]:
    :param topk:
    :return:
    """
    topk_vals, topk_idxs = torch.topk(affinity, dim=1, k=topk)
    thresh, _ = torch.min(topk_vals, dim=1)
    affinity[affinity < thresh] = 0

    N, C, H, W = seg.shape
    _, _, N2 = affinity.shape
    assert H * W == N2, "shape check fails. seg: {}; affinity: {}".format(
        seg.shape, affinity.shape
    )

    seg_one_hot = seg.view(N, C, -1)
    warped_seg = torch.bmm(seg_one_hot, affinity).view(N, C, H, W)

    return warped_seg


@torch.no_grad()
def restrict_warp(seg, affinity, radius=12, topk=5):
    grid_query = create_grid(seg.shape, device=seg.device, flatten=True).squeeze()
    grid_source = create_grid(seg.shape, device=seg.device, flatten=True).squeeze()

    grid_query = grid_query.unsqueeze(1)
    grid_source = grid_source.unsqueeze(0)

    dis = torch.abs(grid_query - grid_source)
    mask = (dis[:, :, 0] <= radius) & (dis[:, :, 1] <= radius)

    masked_affinity = torch.empty_like(affinity)
    masked_affinity.fill_(-float("INF"))
    masked_affinity[:, mask] = affinity[:, mask]

    masked_affinity = F.softmax(masked_affinity, dim=1)

    N, C, H, W = seg.shape
    _, N1, N2 = masked_affinity.shape
    assert H * W == N1, "shape check fails. seg: {}; affinity: {}".format(
        seg.shape, masked_affinity.shape
    )

    topk_vals, topk_idxs = torch.topk(masked_affinity, dim=1, k=topk)
    thresh, _ = torch.min(topk_vals, dim=1)
    masked_affinity[masked_affinity < thresh] = 0

    seg = seg.view(N, C, -1)
    warped_seg = torch.bmm(seg, masked_affinity).view(N, C, H, W)

    return warped_seg


def l2_norm(tensor):
    return F.normalize(tensor, p=2, dim=1)


def norm_mask(mask: torch.Tensor, eps=1e-9):
    """Apply normalization to per class of the entire image."""
    squeeze = False
    if mask.dim() == 3:
        squeeze = True
        mask = mask.unsqueeze(0)

    assert mask.dim() == 4
    N, C, H, W = mask.shape
    mask = mask.view(N, C, -1)
    mask_max_per_class = mask.max(dim=-1, keepdim=True)[0].clamp(min=eps)
    mask_min_per_class = mask.min(dim=-1, keepdim=True)[0]

    mask = (mask - mask_min_per_class) / mask_max_per_class
    mask = mask.view(N, C, H, W)

    if squeeze:
        mask = mask.squeeze(0)

    return mask


def create_safe_grid(shape, device, flatten=False):
    """
    Returns:
        grid: [N, 2(x, y), H, W]
    """
    N, _, H, W = shape
    grid_x = torch.arange(start=0, end=W, step=1, device=device)
    grid_y = torch.arange(start=0, end=H, step=1, device=device)

    grid_y, grid_x = torch.meshgrid(grid_x, grid_y)
    grid = torch.cat([grid_x.unsqueeze(0), grid_y.unsqueeze(0)], dim=0)
    grid = grid.unsqueeze(0).repeat(N, 1, 1, 1)

    if flatten:
        grid = grid.reshape(N, 2, -1)

    return grid
