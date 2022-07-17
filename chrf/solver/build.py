from typing import Any, Dict, List, Optional, Set

import torch
import torch.nn as nn
from chrf.config import CfgNode

from .lr_scheduler import (
    EpochBasedWarmupPolyLR,
    LambdaLR,
    MultiStepLR,
    WarmupCosineLR,
    WarmupMultiStepLR,
)

from .lr_scheduler1 import ContinuousExponentialLR


def build_optimizer(cfg: CfgNode, model: nn.Module) -> torch.optim.Optimizer:
    """
    Build an optimizer from config.
    """
    params = get_default_optimizer_params(
        model,
        base_lr=cfg.SOLVER.BASE_LR,
        weight_decay=cfg.SOLVER.WEIGHT_DECAY,
        weight_decay_norm=cfg.SOLVER.WEIGHT_DECAY_NORM,
        bias_lr_factor=cfg.SOLVER.BIAS_LR_FACTOR,
        weight_decay_bias=cfg.SOLVER.WEIGHT_DECAY_BIAS,
    )

    optim = cfg.SOLVER.OPTIM
    if optim == "SGD":
        return torch.optim.SGD(
            params,
            cfg.SOLVER.BASE_LR,
            momentum=cfg.SOLVER.MOMENTUM,
            nesterov=cfg.SOLVER.NESTEROV,
        )
    elif optim == "ADAM":
        return torch.optim.Adam(params, cfg.SOLVER.BASE_LR)
    elif optim == "ADAMW":
        return torch.optim.AdamW(params, cfg.SOLVER.BASE_LR)
    elif optim == "ADAGRAD":
        return torch.optim.Adagrad(
            params, cfg.SOLVER.BASE_LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY
        )
    else:
        raise ValueError


def get_default_optimizer_params(
    model: nn.Module,
    base_lr,
    weight_decay,
    weight_decay_norm,
    bias_lr_factor=1.0,
    weight_decay_bias=None,
    overrides: Optional[Dict[str, Dict[str, float]]] = None,
):
    """
    Get default param list for optimizer

    Args:
        overrides (dict: str -> (dict: str -> float)):
            if not `None`, provides values for optimizer hyperparameters
            (LR, weight decay) for module parameters with a given name; e.g.
            {"embedding": {"lr": 0.01, "weight_decay": 0.1}} will set the LR and
            weight decay values for all module parameters named `embedding` (default: None)
    """
    if weight_decay_bias is None:
        weight_decay_bias = weight_decay

    norm_module_types = (
        nn.BatchNorm1d,
        nn.BatchNorm2d,
        nn.BatchNorm3d,
        nn.SyncBatchNorm,
        # NaiveSyncBatchNorm inherits from BatchNorm2d
        nn.GroupNorm,
        nn.InstanceNorm1d,
        nn.InstanceNorm2d,
        nn.InstanceNorm3d,
        nn.LayerNorm,
        nn.LocalResponseNorm,
    )
    params: List[Dict[str, Any]] = []
    memo: Set[nn.parameter.Parameter] = set()
    for module in model.modules():
        for module_param_name, value in module.named_parameters(recurse=False):
            if not value.requires_grad:
                continue
            # Avoid duplicating parameters
            if value in memo:
                continue
            memo.add(value)

            schedule_params = {
                "lr": base_lr,
                "weight_decay": weight_decay,
            }
            if isinstance(module, norm_module_types):
                schedule_params["weight_decay"] = weight_decay_norm
            elif module_param_name == "bias":
                # NOTE: unlike Detectron v1, we now default BIAS_LR_FACTOR to 1.0
                # and WEIGHT_DECAY_BIAS to WEIGHT_DECAY so that bias optimizer
                # hyperparameters are by default exactly the same as for regular
                # weights.
                schedule_params["lr"] = base_lr * bias_lr_factor
                schedule_params["weight_decay"] = weight_decay_bias
            if overrides is not None and module_param_name in overrides:
                schedule_params.update(overrides[module_param_name])
            params += [
                {
                    "params": [value],
                    "lr": schedule_params["lr"],
                    "weight_decay": schedule_params["weight_decay"],
                }
            ]

    return params


def build_lr_scheduler(
    cfg: CfgNode, optimizer: torch.optim.Optimizer
) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Build a LR scheduler from config.
    """
    name = cfg.SOLVER.LR_SCHEDULER_NAME
    if name == "WarmupMultiStepLR":
        return WarmupMultiStepLR(
            optimizer,
            cfg.SOLVER.STEPS,
            cfg.SOLVER.GAMMA,
            warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
            warmup_iters=cfg.SOLVER.WARMUP_ITERS,
            warmup_method=cfg.SOLVER.WARMUP_METHOD,
        )
    elif name == "WarmupCosineLR":
        return WarmupCosineLR(
            optimizer,
            cfg.SOLVER.MAX_ITER,
            warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
            warmup_iters=cfg.SOLVER.WARMUP_ITERS,
            warmup_method=cfg.SOLVER.WARMUP_METHOD,
        )
    elif name == "MultiStepLR":
        return MultiStepLR(
            optimizer,
            cfg.SOLVER.STEPS,
            cfg.SOLVER.GAMMA,
        )
    elif name == "PolyLR":
        return LambdaLR(
            optimizer, lr_lambda=lambda iter: (1 - iter / cfg.SOLVER.MAX_ITER) ** 0.9
        )
    elif name == "EpochBasedWarmupPolyLR":
        return EpochBasedWarmupPolyLR(
            cfg.SOLVER.EPOCH_BASE, cfg.SOLVER.MAX_ITER, cfg.SOLVER.WARMUP_ITERS, optimizer
        )
    elif name == "ContinuousExponentialLR":
        return ContinuousExponentialLR(
            optimizer,
            cfg.SOLVER.GAMMA,
            cfg.SOLVER.BASE_DURATION,
            batches_per_epoch=cfg.DATASETS.TRAINSET_NUM // cfg.SOLVER.IMS_PER_BATCH
        )
    else:
        raise ValueError("Unknown LR scheduler: {}".format(name))
