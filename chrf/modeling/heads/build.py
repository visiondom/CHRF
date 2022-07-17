from chrf.layers import ShapeSpec
from chrf.utils.registry import Registry

HEADS_REGISTRY = Registry("HEADS")
HEADS_REGISTRY.__doc__ = """
Registry for heads, which refine feature maps from backbones

The registered object must be a callable that accepts two arguments:

1. A :class:`detectron2.config.CfgNode`
2. A :class:`detectron2.layers.ShapeSpec`, which contains the input shape specification.

"""


def build_head(cfg, input_shape=None):
    """
    Build a head from `cfg.MODEL.TRACKER.HEAD.NAME`.

    Returns:
        head module
    """
    if input_shape is None:
        input_shape = ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN))

    head_name = cfg.MODEL.HEAD.NAME
    head = HEADS_REGISTRY.get(head_name)(cfg, input_shape)
    return head
