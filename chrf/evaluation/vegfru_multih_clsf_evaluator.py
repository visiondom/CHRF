import copy
import itertools
import logging
import os
from collections import OrderedDict

import chrf.utils.comm as comm
import numpy as np
import pandas as pd
import torch

from .evaluator import DatasetEvaluator


class VegFruMultihClsfEvaluator(DatasetEvaluator):
    def __init__(self, dataset_name, distributed, output_dir=None) -> None:
        self._distributed = distributed
        self._output_dir = output_dir
        if self._output_dir:
            self._output_dir = os.path.join(self._output_dir, dataset_name)
        self.dataset_name = dataset_name

        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)

        #There are 2 hierarchies including subclass, superclass
        self.gt_subclass = []
        self.gt_superclass = []

        self.score_subclass = []
        self.score_superclass = []

    def reset(self):
        self.gt_subclass = []
        self.gt_superclass = []

        self.score_subclass = []
        self.score_superclass = []

    def process(self, inputs, outputs):
        """
        inputs:a batch of dict
            {
                img: image tensor,
                category: name of category,
                cls_idx: index of class
            }
        """
        score_subclass, acc_subclass, score_superclass, acc_superclass = outputs
        for bi in range(len(inputs)):
            self.gt_subclass.append(inputs[bi]["subclass"])
            self.gt_superclass.append(inputs[bi]["superclass"])

            self.score_subclass.append(score_subclass[bi])
            self.score_superclass.append(score_superclass[bi])

    def evaluate(self):
        if self._distributed:
            comm.synchronize()

            gt_subclass = comm.gather(self.gt_subclass, dst=0)
            gt_superclass = comm.gather(self.gt_superclass, dst=0)

            score_subclass = comm.gather(self.score_subclass, dst=0)
            score_superclass = comm.gather(self.score_superclass, dst=0)

            gt_subclass = list(itertools.chain(*gt_subclass))
            gt_superclass = list(itertools.chain(*gt_superclass))

            score_subclass = list(itertools.chain(*score_subclass))
            score_superclass = list(itertools.chain(*score_superclass))

            if not comm.is_main_process():
                return {}
        else:
            gt_subclass = self.gt_subclass
            gt_superclass = self.gt_superclass

            score_subclass = self.score_subclass
            score_superclass = self.score_superclass

        # subclass
        gt_subclass_t = torch.tensor(gt_subclass, device="cpu")
        score_subclass_t = torch.stack([st.cpu() for st in score_subclass], dim=0)

        subclass_top1_idxs = torch.argmax(score_subclass_t, axis=1).view(-1)
        subclass_top5_idxs = torch.topk(score_subclass_t, k=5, dim=1, sorted=False)[1]
        subclass_top1_acc = (subclass_top1_idxs == gt_subclass_t).sum().float().item() / gt_subclass_t.shape[0]
        subclass_top5_acc = (subclass_top5_idxs == gt_subclass_t[:, None]).sum().float().item() / gt_subclass_t.shape[0]

        # superclass
        gt_superclass_t = torch.tensor(gt_superclass, device="cpu")
        score_superclass_t = torch.stack([st.cpu() for st in score_superclass], dim=0)
        superclass_top1_idxs = torch.argmax(score_superclass_t, axis=1).view(-1)
        superclass_top5_idxs = torch.topk(score_superclass_t, k=5, dim=1, sorted=False)[1]
        superclass_top1_acc = (superclass_top1_idxs == gt_superclass_t).sum().float().item() / gt_superclass_t.shape[0]
        superclass_top5_acc = (superclass_top5_idxs == gt_superclass_t[:, None]).sum().float().item() / gt_superclass_t.shape[0]

        result = OrderedDict()
        result["classification"] = pd.DataFrame(
            data=np.array([[subclass_top1_acc, subclass_top5_acc, superclass_top1_acc, superclass_top5_acc]]),
            columns=["subclass-top-1", "subclass-top-5", "superclass-top-1", "superclass-top-5"],
        )
        return copy.deepcopy(result)
