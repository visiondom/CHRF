from chrf.layers import ShapeSpec
from chrf.utils.registry import Registry

NECKS_REGISTRY = Registry("NECKS")
NECKS_REGISTRY.__doc__ = """
Registry for necks, which refine feature maps from backbones and connect heads

The registered object must be a callable that accepts two arguments:

1. A :class:`detectron2.config.CfgNode`
2. A :class:`detectron2.layers.ShapeSpec`, which contains the input shape specification.

"""


def build_neck(cfg, input_shape=None):
    """
    Build a neck from `cfg.MODEL.TRACKER.HEAD.NAME`.

    Returns:
        head module
    """
    if input_shape is None:
        input_shape = ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN))

    neck_name = cfg.MODEL.NECK.NAME
    neck = NECKS_REGISTRY.get(neck_name)(cfg, input_shape)
    return neck
