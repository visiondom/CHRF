import torch


def scatter(batch, N, dim=0):
    r"""Scatter each batched data into a tensor with outer dimension batch size"""

    def scatter_map(obj):
        if isinstance(obj, torch.Tensor):
            return [_.squeeze(0) for _ in obj.split(1, dim=dim)]
        if isinstance(obj, tuple) and len(obj) > 0:
            return list(zip(*map(scatter_map, obj)))
        if isinstance(obj, list) and len(obj) > 0:
            return list(map(list, zip(*map(scatter_map, obj))))
        if isinstance(obj, dict) and len(obj) > 0:
            return list(map(type(obj), zip(*map(scatter_map, obj.items()))))
        return [obj for _ in range(N)]

    try:
        res = scatter_map(batch)
    finally:
        scatter_map = None
    return res
