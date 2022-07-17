import torch
import warnings

class ContinuousExponentialLR(torch.optim.lr_scheduler._LRScheduler):
    """Decays the learning rate of each parameter group by gamma every epoch.
    When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        gamma (float): Multiplicative factor of learning rate decay.
        base_duration (int or float): period of learning rate decay.
        batches_per_epoch (float): batches of each epoch, equal to batches of training set.
        last_epoch (int): The index of last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.
    """

    def __init__(self, optimizer, gamma, base_duration, batches_per_epoch, last_epoch=-1, verbose=False):
        self.gamma = gamma
        self.base_duration = base_duration
        self.batches_per_epoch = batches_per_epoch
        super(ContinuousExponentialLR, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        if self.last_epoch == 0:
            return self.base_lrs
        return [base_lr * pow(self.gamma, (self.last_epoch / self.batches_per_epoch) / self.base_duration)
                for base_lr in self.base_lrs]

    def _get_closed_form_lr(self):
        return [base_lr * pow(self.gamma, (self.last_epoch / self.batches_per_epoch) / self.base_duration)
                for base_lr in self.base_lrs]