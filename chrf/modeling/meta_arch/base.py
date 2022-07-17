import torch
import torch.nn as nn
import torch.nn.functional as F

from ..backbone import build_backbone
from .build import META_ARCH_REGISTRY


@META_ARCH_REGISTRY.register()
class BenchBase(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone = build_backbone(cfg)
        output_shapes = self.backbone.output_shape()
        self.strides = [output_shapes[in_feature].stride for in_feature in self.in_features]

        self.register_buffer("pixel_mean", torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.Tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1))
        assert (
            self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"

    @property
    def device(self):
        return self.pixel_mean.device

    def preprocess_image(self, batched_inputs):
        pass

    def forward(self, batched_inputs):
        if not self.training:
            return self.inference(batched_inputs)

        raise NotImplementedError

    def visualize_training(self, batched_inputs, results):
        raise NotImplementedError

    def inference(self, batched_inputs):
        assert not self.training
        pass

    def downsample(self, value, size):
        return F.interpolate(value, size=size, mode="bilinear", align_corners=True)

    def upsample(self, value, size):
        return F.interpolate(value, size=size, mode="bilinear", align_corners=True)
