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


class AirsMultihClsfEvaluator(DatasetEvaluator):
    def __init__(self, dataset_name, distributed, output_dir=None) -> None:
        self._distributed = distributed
        self._output_dir = output_dir
        if self._output_dir:
            self._output_dir = os.path.join(self._output_dir, dataset_name)
        self.dataset_name = dataset_name

        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)

        #There are 3 hierarchies including model, family, maker
        self.gt_model = []
        self.gt_family = []
        self.gt_maker = []

        self.score_model = []
        self.score_family = []
        self.score_maker = []

    def reset(self):
        self.gt_model = []
        self.gt_family = []
        self.gt_maker = []

        self.score_model = []
        self.score_family = []
        self.score_maker = []

    def process(self, inputs, outputs):
        """
        inputs:a batch of dict
            {
                img: image tensor,
                category: name of category,
                cls_idx: index of class
            }
        """
        score_model, acc_model, score_family, acc_family, score_maker, acc_maker = outputs
        for bi in range(len(inputs)):
            self.gt_model.append(inputs[bi]["model"])
            self.gt_family.append(inputs[bi]["family"])
            self.gt_maker.append(inputs[bi]["maker"])

            self.score_model.append(score_model[bi])
            self.score_family.append(score_family[bi])
            self.score_maker.append(score_maker[bi])

    def evaluate(self):
        if self._distributed:
            comm.synchronize()

            gt_model = comm.gather(self.gt_model, dst=0)
            gt_family = comm.gather(self.gt_family, dst=0)
            gt_maker = comm.gather(self.gt_maker, dst=0)

            score_model = comm.gather(self.score_model, dst=0)
            score_family = comm.gather(self.score_family, dst=0)
            score_maker = comm.gather(self.score_maker, dst=0)

            gt_model = list(itertools.chain(*gt_model))
            gt_family = list(itertools.chain(*gt_family))
            gt_maker = list(itertools.chain(*gt_maker))

            score_model = list(itertools.chain(*score_model))
            score_family = list(itertools.chain(*score_family))
            score_maker = list(itertools.chain(*score_maker))

            if not comm.is_main_process():
                return {}
        else:
            gt_model = self.gt_model
            gt_family = self.gt_family
            gt_maker = self.gt_maker

            score_model = self.score_model
            score_family = self.score_family
            score_maker = self.score_maker

        # model
        gt_model_t = torch.tensor(gt_model, device="cpu")
        score_model_t = torch.stack([st.cpu() for st in score_model], dim=0)

        model_top1_idxs = torch.argmax(score_model_t, axis=1).view(-1)
        model_top5_idxs = torch.topk(score_model_t, k=5, dim=1, sorted=False)[1]
        model_top1_acc = (model_top1_idxs == gt_model_t).sum().float().item() / gt_model_t.shape[0]
        model_top5_acc = (model_top5_idxs == gt_model_t[:, None]).sum().float().item() / gt_model_t.shape[0]

        # family
        gt_family_t = torch.tensor(gt_family, device="cpu")
        score_family_t = torch.stack([st.cpu() for st in score_family], dim=0)

        family_top1_idxs = torch.argmax(score_family_t, axis=1).view(-1)
        family_top5_idxs = torch.topk(score_family_t, k=5, dim=1, sorted=False)[1]
        family_top1_acc = (family_top1_idxs == gt_family_t).sum().float().item() / gt_family_t.shape[0]
        family_top5_acc = (family_top5_idxs == gt_family_t[:, None]).sum().float().item() / gt_family_t.shape[0]

        # maker
        gt_maker_t = torch.tensor(gt_maker, device="cpu")
        score_maker_t = torch.stack([st.cpu() for st in score_maker], dim=0)
        maker_top1_idxs = torch.argmax(score_maker_t, axis=1).view(-1)
        maker_top5_idxs = torch.topk(score_maker_t, k=5, dim=1, sorted=False)[1]
        maker_top1_acc = (maker_top1_idxs == gt_maker_t).sum().float().item() / gt_maker_t.shape[0]
        maker_top5_acc = (maker_top5_idxs == gt_maker_t[:, None]).sum().float().item() / gt_maker_t.shape[0]

        result = OrderedDict()
        result["classification"] = pd.DataFrame(
            data=np.array([[model_top1_acc, model_top5_acc, family_top1_acc, family_top5_acc, maker_top1_acc, maker_top5_acc]]),
            columns=["model-top-1", "model-top-5", "family-top-1", "family-top-5", "maker-top-1", "maker-top-5"],
        )
        return copy.deepcopy(result)
