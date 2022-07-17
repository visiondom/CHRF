import torch.nn as nn

from .build import HEADS_REGISTRY


class IDConvHead(nn.Sequential):
    def __init__(self, in_channel, out_channel):
        layers = [
            nn.Conv2d(in_channel, in_channel, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel, out_channel, 1, 1, 0),
        ]
        super().__init__(*layers)


@HEADS_REGISTRY.register()
def build_id_conv_head(cfg, input_shape=None):
    in_channels = cfg.MODEL.TRACKER.HEAD.DIM_IN
    out_channel = cfg.MODEL.TRACKER.HEAD.DIM_OUT
    return IDConvHead(in_channels, out_channel)
