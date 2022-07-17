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


class CUBMultihClsfEvaluator(DatasetEvaluator):
    def __init__(self, dataset_name, distributed, output_dir=None) -> None:
        self._distributed = distributed
        self._output_dir = output_dir
        if self._output_dir:
            self._output_dir = os.path.join(self._output_dir, dataset_name)
        self.dataset_name = dataset_name

        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)

        #There are 4 hierarchies including class, genus, family and order
        self.gt_class = []
        self.gt_genus = []
        self.gt_family = []
        self.gt_order = []

        self.score_class = []
        self.score_genus = []
        self.score_family = []
        self.score_order = []

    def reset(self):
        self.gt_class = []
        self.gt_genus = []
        self.gt_family = []
        self.gt_order = []

        self.score_class = []
        self.score_genus = []
        self.score_family = []
        self.score_order = []

    def process(self, inputs, outputs):
        """
        inputs:a batch of dict
            {
                img: image tensor,
                category: name of category,
                cls_idx: index of class
            }
        """
        score_class, acc_class, score_genus, acc_genus,\
        score_family, acc_family, score_order, acc_order = outputs
        for bi in range(len(inputs)):
            self.gt_class.append(inputs[bi]["class"])
            self.gt_genus.append(inputs[bi]["genus"])
            self.gt_family.append(inputs[bi]["family"])
            self.gt_order.append(inputs[bi]["order"])

            self.score_class.append(score_class[bi])
            self.score_genus.append(score_genus[bi])
            self.score_family.append(score_family[bi])
            self.score_order.append(score_order[bi])

    def evaluate(self):
        if self._distributed:
            comm.synchronize()

            gt_class = comm.gather(self.gt_class, dst=0)
            gt_genus = comm.gather(self.gt_genus, dst=0)
            gt_family = comm.gather(self.gt_family, dst=0)
            gt_order = comm.gather(self.gt_order, dst=0)
            score_class = comm.gather(self.score_class, dst=0)
            score_genus = comm.gather(self.score_genus, dst=0)
            score_family = comm.gather(self.score_family, dst=0)
            score_order = comm.gather(self.score_order, dst=0)

            gt_class = list(itertools.chain(*gt_class))
            gt_genus = list(itertools.chain(*gt_genus))
            gt_family = list(itertools.chain(*gt_family))
            gt_order = list(itertools.chain(*gt_order))
            score_class = list(itertools.chain(*score_class))
            score_genus = list(itertools.chain(*score_genus))
            score_family = list(itertools.chain(*score_family))
            score_order = list(itertools.chain(*score_order))

            if not comm.is_main_process():
                return {}
        else:
            gt_class = self.gt_class
            gt_genus = self.gt_genus
            gt_family = self.gt_family
            gt_order = self.gt_order
            score_class = self.score_class
            score_genus = self.score_genus
            score_family = self.score_family
            score_order = self.score_order

        # class
        gt_class_t = torch.tensor(gt_class, device="cpu")
        score_class_t = torch.stack([st.cpu() for st in score_class], dim=0)
        class_top1_idxs = torch.argmax(score_class_t, axis=1).view(-1)
        class_top5_idxs = torch.topk(score_class_t, k=5, dim=1, sorted=False)[1]
        class_top1_acc = (class_top1_idxs == gt_class_t).sum().float().item() / gt_class_t.shape[0]
        class_top5_acc = (class_top5_idxs == gt_class_t[:, None]).sum().float().item() / gt_class_t.shape[0]

        # genus
        gt_genus_t = torch.tensor(gt_genus, device="cpu")
        score_genus_t = torch.stack([st.cpu() for st in score_genus], dim=0)
        genus_top1_idxs = torch.argmax(score_genus_t, axis=1).view(-1)
        genus_top5_idxs = torch.topk(score_genus_t, k=5, dim=1, sorted=False)[1]
        genus_top1_acc = (genus_top1_idxs == gt_genus_t).sum().float().item() / gt_genus_t.shape[0]
        genus_top5_acc = (genus_top5_idxs == gt_genus_t[:, None]).sum().float().item() / gt_genus_t.shape[0]

        # family
        gt_family_t = torch.tensor(gt_family, device="cpu")
        score_family_t = torch.stack([st.cpu() for st in score_family], dim=0)
        family_top1_idxs = torch.argmax(score_family_t, axis=1).view(-1)
        family_top5_idxs = torch.topk(score_family_t, k=5, dim=1, sorted=False)[1]
        family_top1_acc = (family_top1_idxs == gt_family_t).sum().float().item() / gt_family_t.shape[0]
        family_top5_acc = (family_top5_idxs == gt_family_t[:, None]).sum().float().item() / gt_family_t.shape[0]

        # order
        gt_order_t = torch.tensor(gt_order, device="cpu")
        score_order_t = torch.stack([st.cpu() for st in score_order], dim=0)
        order_top1_idxs = torch.argmax(score_order_t, axis=1).view(-1)
        order_top5_idxs = torch.topk(score_order_t, k=5, dim=1, sorted=False)[1]
        order_top1_acc = (order_top1_idxs == gt_order_t).sum().float().item() / gt_order_t.shape[0]
        order_top5_acc = (order_top5_idxs == gt_order_t[:, None]).sum().float().item() / gt_order_t.shape[0]


        result = OrderedDict()
        result["classification"] = pd.DataFrame(
            data=np.array([[class_top1_acc, class_top5_acc, genus_top1_acc, genus_top5_acc,
                            family_top1_acc, family_top5_acc, order_top1_acc, order_top5_acc,]]),
            columns=["class-top-1", "class-top-5", "genus-top-1", "genus-top-5",
                     "family-top-1", "family-top-5", "order-top-1", "order-top-5"],
        )
        return copy.deepcopy(result)
