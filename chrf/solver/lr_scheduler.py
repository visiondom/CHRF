# Copyright (c) Facebook, Inc. and its affiliates.
import math
from bisect import bisect_right
from typing import List
import torch
from torch.optim.lr_scheduler import LambdaLR, MultiStepLR

# NOTE: PyTorch's LR scheduler interface uses names that assume the LR changes
# only on epoch boundaries. We typically use iteration based schedules instead.
# As a result, "epoch" (e.g., as in self.last_epoch) should be understood to mean
# "iteration" instead.

# FIXME: ideally this would be achieved with a CombinedLRScheduler, separating
# MultiStepLR with WarmupLR but the current LRScheduler design doesn't allow it.


class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        milestones: List[int],
        gamma: float = 0.1,
        warmup_factor: float = 0.001,
        warmup_iters: int = 1000,
        warmup_method: str = "linear",
        last_epoch: int = -1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}", milestones
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        warmup_factor = _get_warmup_factor_at_iter(
            self.warmup_method, self.last_epoch, self.warmup_iters, self.warmup_factor
        )
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]

    def _compute_values(self) -> List[float]:
        # The new interface
        return self.get_lr()


class WarmupCosineLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        max_iters: int,
        warmup_factor: float = 0.001,
        warmup_iters: int = 1000,
        warmup_method: str = "linear",
        last_epoch: int = -1,
    ):
        self.max_iters = max_iters
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        warmup_factor = _get_warmup_factor_at_iter(
            self.warmup_method, self.last_epoch, self.warmup_iters, self.warmup_factor
        )
        # Different definitions of half-cosine with warmup are possible. For
        # simplicity we multiply the standard half-cosine schedule by the warmup
        # factor. An alternative is to start the period of the cosine at warmup_iters
        # instead of at 0. In the case that warmup_iters << max_iters the two are
        # very close to each other.
        return [
            base_lr
            * warmup_factor
            * 0.5
            * (1.0 + math.cos(math.pi * self.last_epoch / self.max_iters))
            for base_lr in self.base_lrs
        ]

    def _compute_values(self) -> List[float]:
        # The new interface
        return self.get_lr()


class EpochBasedLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self, epoch_base: int, optimizer: torch.optim.Optimizer, last_epoch: int = -1
    ) -> None:
        self.epoch_base = epoch_base
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self) -> float:
        if self.last_epoch + 1 > 0 and (self.last_epoch + 1) % self.epoch_base == 0:
            return self.new_epoch_lr()
        else:
            return [group["lr"] for group in self.optimizer.param_groups]

    def new_epoch_lr(self):
        raise NotImplementedError


class EpochBasedWarmupPolyLR(EpochBasedLR):
    def __init__(
        self,
        epoch_base: int,
        max_iters: int,
        warmup_iters: int,
        optimizer: torch.optim.Optimizer,
        last_epoch: int = -1,
    ) -> None:
        self.total_epoch = max_iters // epoch_base
        self.warmup_iters = warmup_iters
        super().__init__(epoch_base, optimizer, last_epoch=last_epoch)

    def new_epoch_lr(self):
        curr_epoch = (self.last_epoch + 1) // self.epoch_base  # iter idx // base = epoch idx
        if self.last_epoch + 1 > self.warmup_iters:
            epoch_diff = (
                curr_epoch - (self.warmup_iters - 1) // self.epoch_base
            )  # iter idx // base = epoch idx
            return [
                base_lr * (1 - epoch_diff / self.total_epoch) ** 0.9
                for base_lr in self.base_lrs
            ]
        else:
            return [
                base_lr * (1 - curr_epoch / self.total_epoch) ** 0.9
                for base_lr in self.base_lrs
            ]


def _get_warmup_factor_at_iter(
    method: str, iter: int, warmup_iters: int, warmup_factor: float
) -> float:
    """
    Return the learning rate warmup factor at a specific iteration.
    See :paper:`ImageNet in 1h` for more details.

    Args:
        method (str): warmup method; either "constant" or "linear".
        iter (int): iteration at which to calculate the warmup factor.
        warmup_iters (int): the number of warmup iterations.
        warmup_factor (float): the base warmup factor (the meaning changes according
            to the method used).

    Returns:
        float: the effective warmup factor at the given iteration.
    """
    if iter >= warmup_iters:
        return 1.0

    if method == "constant":
        return warmup_factor
    elif method == "linear":
        alpha = iter / warmup_iters
        return warmup_factor * (1 - alpha) + alpha
    else:
        raise ValueError("Unknown warmup method: {}".format(method))
