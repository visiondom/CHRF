import torch.nn as nn

from .build import HEADS_REGISTRY


class SelfSimFCHead(nn.Sequential):
    def __init__(self, in_channel, mid_depth, out_channel=128):
        layers = []
        dims = [in_channel] + [in_channel] * mid_depth + [out_channel]
        for d1, d2 in zip(dims, dims[1:]):
            h = nn.Linear(d1, d2)
            layers += [h, nn.ReLU()]
        super().__init__(*layers[:-1])


@HEADS_REGISTRY.register()
def build_selfsim_fc_head(cfg, input_shape):
    in_channel = input_shape[-3]
    depth = cfg.MODEL.TRACKER.HEAD.MID_DEPTH
    return SelfSimFCHead(in_channel, depth)
