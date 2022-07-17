import torch.nn as nn

from .build import HEADS_REGISTRY


class ClassifyHead(nn.Sequential):
    def __init__(self, in_channel, out_channel, drop_rate, bias=False):
        layers = [nn.Linear(in_channel, out_channel, bias=bias)]
        if drop_rate > 0:
            layers = [nn.Dropout(p=drop_rate)] + layers
        super().__init__(*layers)


@HEADS_REGISTRY.register()
def build_clsf_head(cfg, input_shape):
    in_channels = input_shape.channels
    out_channel = cfg.BENCHMARK.CLASSIFICATION.CATEGORY_NUM
    return ClassifyHead(in_channels, out_channel, cfg.MODEL.HEAD.IN_DROP)

@HEADS_REGISTRY.register()
def build_multih_clsf_head(cfg, input_shape):
    in_channels = input_shape.channels
    hiers = cfg.BENCHMARK.HIERARCHY
    outs_channel = cfg.MHBENCHMARK.CLASSIFICATION.CATEGORY_NUM
    assert len(hiers) == len(outs_channel)
    cls_head = {}
    for hier, out_channel in zip(hiers, outs_channel):
        cls_head[hier] = ClassifyHead(in_channels, out_channel, cfg.MODEL.HEAD.IN_DROP, cfg.MODEL.HEAD.CLS_BIAS)
    return cls_head

@HEADS_REGISTRY.register()
def build_multih_diffM_clsf_head(cfg, input_shape):
    in_channels = input_shape.channels
    hiers = cfg.BENCHMARK.HIERARCHY
    outs_channel = cfg.MHBENCHMARK.CLASSIFICATION.CATEGORY_NUM
    assert len(hiers) == len(outs_channel)
    cls_head = {}
    for hier, out_channel in zip(hiers, outs_channel):
        cls_head[hier] = ClassifyHead(in_channels[hier], out_channel, cfg.MODEL.HEAD.IN_DROP, cfg.MODEL.HEAD.CLS_BIAS)
    return cls_head