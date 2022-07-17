from chrf.checkpoint import BMKCheckpointer
from .build import BACKBONE_REGISTRY

from .resnet import ResNet, BasicBlock, BasicStem, BottleneckBlock

from copy import deepcopy

class MultiHResNet(ResNet):

    def __init__(self,
                 stem,
                 stages,
                 num_classes=None,
                 out_features=None,
                 stage_divide_at=4,
                 resnet_weight="",
                 hierarchy=None,
                 ):
        super(MultiHResNet, self).__init__(
            stem=stem,
            stages=stages,
            num_classes=num_classes,
            out_features=out_features
        )

        # load resnet weight
        BMKCheckpointer(self).resume_or_load(resnet_weight, resume=True)

        self.hier = hierarchy
        self.stage_divide_at = stage_divide_at
        # self.stage_names, self.stages range between res2 to res5
        assert stage_divide_at>=3 and stage_divide_at<=5
        self.shared_stage_names = [name for name in self.stage_names[:stage_divide_at-2]]
        self.shared_stages = [stage for stage in self.stages[:stage_divide_at - 2]]

        self.hier_stage_names, self.hier_stages = {name: [] for name in self.hier[1:]},\
                                                  {stage: [] for stage in self.hier[1:]}

        # class
        self.hier_stage_names[self.hier[0]] = [name for name in self.stage_names[stage_divide_at-2:]]
        self.hier_stages[self.hier[0]] = [stage for stage in self.stages[stage_divide_at - 2:]]

        # genera, family, order
        for hier in self.hier[1:]:
            for class_name, class_stage in zip(self.hier_stage_names[self.hier[0]], self.hier_stages[self.hier[0]]):
                hier_name = hier + '_' + class_name
                hier_stage = deepcopy(class_stage)
                self.add_module(hier_name, hier_stage)
                self.hier_stage_names[hier].append(hier_name)
                self.hier_stages[hier].append(hier_stage)

    def forward(self, x):
        assert (
                x.dim() == 4
        ), f"ResNet takes an input of shape (N, C, H, W). Got {x.shape} instead!"
        outputs = {}
        x = self.stem(x)
        if "stem" in self._out_features:
            outputs["stem"] = x
        for name, stage in zip(self.shared_stage_names, self.shared_stages):
            x = stage(x)
            if name in self._out_features:
                outputs[name] = x
        for hier in self.hier:
            outputs[hier] = {}
            hier_x = x
            for name, stage in zip(self.hier_stage_names[hier], self.hier_stages[hier]):
                hier_x = stage(hier_x)
                res_name = "_".join(name.split('_')[1:]) if hier != self.hier[0] else name
                if res_name in self._out_features:
                    outputs[hier][res_name] = hier_x
        return outputs

    def freeze(self, shared_freeze_at=0, hier_freeze_at=[0, 0, 0, 0]):

        # freeze shared backbone
        freeze_at = min(shared_freeze_at, self.stage_divide_at-1)
        if freeze_at >= 1:
            self.stem.freeze()
        for idx, stage in enumerate(self.shared_stages, start=2):
            if freeze_at >= idx:
                for block in stage.children():
                    block.freeze()

        # freeze hier backbone
        for hier, hier_fa in zip(self.hier, hier_freeze_at):
            for idx, stage in enumerate(self.hier_stages[hier], start=self.stage_divide_at):
                if hier_fa >= idx:
                    for block in stage.children():
                        block.freeze()
        return self


@BACKBONE_REGISTRY.register()
def build_multhresnet_backbone(cfg, input_shape):
    """
    Create a ResNet instance from config.

    Returns:
        ResNet: a :class:`ResNet` instance.
    """
    # need registration of new blocks/stems?
    norm = cfg.MODEL.RESNETS.NORM
    stem = BasicStem(
        in_channels=input_shape.channels,
        out_channels=cfg.MODEL.RESNETS.STEM_OUT_CHANNELS,
        norm=norm,
    )

    # fmt: off
    stage_divide_at     = cfg.MODEL.MULTIHRESNETS.STAGE_DIVIDE_AT
    shared_freeze_at    = cfg.MODEL.MULTIHRESNETS.SHARED_FREEZE_AT
    hier_freeze_at      = cfg.MODEL.MULTIHRESNETS.HIER_FREEZE_AT
    resnet_weight       = cfg.MODEL.MULTIHRESNETS.RESNET_WEIGHT

    out_features        = cfg.MODEL.RESNETS.OUT_FEATURES
    depth               = cfg.MODEL.RESNETS.DEPTH
    num_groups          = cfg.MODEL.RESNETS.NUM_GROUPS
    width_per_group     = cfg.MODEL.RESNETS.WIDTH_PER_GROUP
    bottleneck_channels = num_groups * width_per_group
    in_channels         = cfg.MODEL.RESNETS.STEM_OUT_CHANNELS
    out_channels        = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
    stride_in_1x1       = cfg.MODEL.RESNETS.STRIDE_IN_1X1
    res5_dilation       = cfg.MODEL.RESNETS.RES5_DILATION
    # fmt: on
    assert res5_dilation in {1, 2}, "res5_dilation cannot be {}.".format(res5_dilation)

    num_blocks_per_stage = {
        18: [2, 2, 2, 2],
        34: [3, 4, 6, 3],
        50: [3, 4, 6, 3],
        101: [3, 4, 23, 3],
        152: [3, 8, 36, 3],
    }[depth]

    if depth in [18, 34]:
        assert out_channels == 64, "Must set MODEL.RESNETS.RES2_OUT_CHANNELS = 64 for R18/R34"
        assert res5_dilation == 1, "Must set MODEL.RESNETS.RES5_DILATION = 1 for R18/R34"
        assert num_groups == 1, "Must set MODEL.RESNETS.NUM_GROUPS = 1 for R18/R34"

    stages = []

    # Avoid creating variables without gradients
    # It consumes extra memory and may cause allreduce to fail
    out_stage_idx = [
        {"res2": 2, "res3": 3, "res4": 4, "res5": 5}[f] for f in out_features if f != "stem"
    ]
    max_stage_idx = max(out_stage_idx)
    for idx, stage_idx in enumerate(range(2, max_stage_idx + 1)):
        dilation = res5_dilation if stage_idx == 5 else 1
        first_stride = 1 if idx == 0 or (stage_idx == 5 and dilation == 2) else 2
        stage_kargs = {
            "num_blocks": num_blocks_per_stage[idx],
            "stride_per_block": [first_stride] + [1] * (num_blocks_per_stage[idx] - 1),
            "in_channels": in_channels,
            "out_channels": out_channels,
            "norm": norm,
        }
        # Use BasicBlock for R18 and R34.
        if depth in [18, 34]:
            stage_kargs["block_class"] = BasicBlock
        else:
            stage_kargs["bottleneck_channels"] = bottleneck_channels
            stage_kargs["stride_in_1x1"] = stride_in_1x1
            stage_kargs["dilation"] = dilation
            stage_kargs["num_groups"] = num_groups
            stage_kargs["block_class"] = BottleneckBlock
        blocks = ResNet.make_stage(**stage_kargs)
        in_channels = out_channels
        out_channels *= 2
        bottleneck_channels *= 2
        stages.append(blocks)
    return MultiHResNet(
        stem,
        stages,
        out_features=out_features,
        stage_divide_at=stage_divide_at,
        resnet_weight=resnet_weight,
        hierarchy=cfg.BENCHMARK.HIERARCHY).freeze(shared_freeze_at, hier_freeze_at)