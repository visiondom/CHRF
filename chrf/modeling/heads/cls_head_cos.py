import torch
import torch.nn as nn
import torch.nn.functional as tF
from .build import HEADS_REGISTRY

MINI = 1e-10


class ClassifyHeadCos(nn.Module):
    def __init__(self, in_channel, out_channel, drop_rate, scale=20.0):
        super().__init__()
        self.scale = scale
        if self.scale == -1:
            self.scale = nn.Parameter(torch.ones(1) * 20.0)
        self.weight_ln = nn.Linear(in_channel, out_channel, bias=False)
        if drop_rate > 0:
            self.drop_layer = nn.Dropout(p=drop_rate)
        else:
            self.drop_layer = self._trivial_drop

    def _trivial_drop(self, x):
        return x

    def forward(self, feat_v):
        feat_v = self.drop_layer(feat_v)
        feat_norm = torch.norm(feat_v, p=2, dim=1).unsqueeze(1).expand_as(feat_v)
        feat_normed = feat_v / (feat_norm + MINI)
        ln_weight = self.weight_ln.weight
        ln_wnorm = torch.norm(ln_weight, p=2, dim=1).unsqueeze(1).expand_as(ln_weight)
        cls_weight = ln_weight / (ln_wnorm + MINI)
        cls_score = torch.einsum("bd,nd->bn", feat_normed, cls_weight)
        return cls_score * self.scale


@HEADS_REGISTRY.register()
def build_clsf_head_cos(cfg, input_shape):
    in_channels = input_shape.channels
    out_channel = cfg.BENCHMARK.CLASSIFICATION.CATEGORY_NUM
    scale = cfg.MODEL.HEAD.SCALE
    return ClassifyHeadCos(in_channels, out_channel, cfg.MODEL.HEAD.IN_DROP, scale)

@HEADS_REGISTRY.register()
def build_multih_clsf_head_cos(cfg, input_shape):
    in_channels = input_shape.channels
    hiers = cfg.BENCHMARK.HIERARCHY
    outs_channel = cfg.MHBENCHMARK.CLASSIFICATION.CATEGORY_NUM
    assert len(hiers) == len(outs_channel)
    scale = cfg.MODEL.HEAD.SCALE
    cls_head = {}
    for hier, out_channel in zip(hiers, outs_channel):
        cls_head[hier] = ClassifyHeadCos(in_channels, out_channel, cfg.MODEL.HEAD.IN_DROP, scale)
    return cls_head