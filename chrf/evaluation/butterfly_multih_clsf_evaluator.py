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


class ButterflyMultihClsfEvaluator(DatasetEvaluator):
    def __init__(self, dataset_name, distributed, output_dir=None) -> None:
        self._distributed = distributed
        self._output_dir = output_dir
        if self._output_dir:
            self._output_dir = os.path.join(self._output_dir, dataset_name)
        self.dataset_name = dataset_name

        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)

        #There are 4 hierarchies including species, genus, subfamily and family
        self.gt_species = []
        self.gt_genus = []
        self.gt_subfamily = []
        self.gt_family = []

        self.score_species = []
        self.score_genus = []
        self.score_subfamily = []
        self.score_family = []

    def reset(self):
        self.gt_species = []
        self.gt_genus = []
        self.gt_subfamily = []
        self.gt_family = []

        self.score_species = []
        self.score_genus = []
        self.score_subfamily = []
        self.score_family = []

    def process(self, inputs, outputs):
        """
        inputs:a batch of dict
            {
                img: image tensor,
                category: name of category,
                cls_idx: index of class
            }
        """
        score_species, acc_species, score_genus, acc_genus,\
        score_subfamily, acc_subfamily, score_family, acc_family = outputs
        for bi in range(len(inputs)):
            self.gt_species.append(inputs[bi]["species"])
            self.gt_genus.append(inputs[bi]["genus"])
            self.gt_subfamily.append(inputs[bi]["subfamily"])
            self.gt_family.append(inputs[bi]["family"])

            self.score_species.append(score_species[bi])
            self.score_genus.append(score_genus[bi])
            self.score_subfamily.append(score_subfamily[bi])
            self.score_family.append(score_family[bi])

    def evaluate(self):
        if self._distributed:
            comm.synchronize()

            gt_species = comm.gather(self.gt_species, dst=0)
            gt_genus = comm.gather(self.gt_genus, dst=0)
            gt_subfamily = comm.gather(self.gt_subfamily, dst=0)
            gt_family = comm.gather(self.gt_family, dst=0)
            score_species = comm.gather(self.score_species, dst=0)
            score_genus = comm.gather(self.score_genus, dst=0)
            score_subfamily = comm.gather(self.score_subfamily, dst=0)
            score_family = comm.gather(self.score_family, dst=0)
            gt_species = list(itertools.chain(*gt_species))
            gt_genus = list(itertools.chain(*gt_genus))
            gt_subfamily = list(itertools.chain(*gt_subfamily))
            gt_family = list(itertools.chain(*gt_family))
            score_species = list(itertools.chain(*score_species))
            score_genus = list(itertools.chain(*score_genus))
            score_subfamily = list(itertools.chain(*score_subfamily))
            score_family = list(itertools.chain(*score_family))

            if not comm.is_main_process():
                return {}
        else:
            gt_species = self.gt_species
            gt_genus = self.gt_genus
            gt_subfamily = self.gt_subfamily
            gt_family = self.gt_family
            score_species = self.score_species
            score_genus = self.score_genus
            score_subfamily = self.score_subfamily
            score_family = self.score_family

        # species
        gt_species_t = torch.tensor(gt_species, device="cpu")
        score_species_t = torch.stack([st.cpu() for st in score_species], dim=0)
        species_top1_idxs = torch.argmax(score_species_t, axis=1).view(-1)
        species_top5_idxs = torch.topk(score_species_t, k=5, dim=1, sorted=False)[1]
        species_top1_acc = (species_top1_idxs == gt_species_t).sum().float().item() / gt_species_t.shape[0]
        species_top5_acc = (species_top5_idxs == gt_species_t[:, None]).sum().float().item() / gt_species_t.shape[0]

        # genus
        gt_genus_t = torch.tensor(gt_genus, device="cpu")
        score_genus_t = torch.stack([st.cpu() for st in score_genus], dim=0)
        genus_top1_idxs = torch.argmax(score_genus_t, axis=1).view(-1)
        genus_top5_idxs = torch.topk(score_genus_t, k=5, dim=1, sorted=False)[1]
        genus_top1_acc = (genus_top1_idxs == gt_genus_t).sum().float().item() / gt_genus_t.shape[0]
        genus_top5_acc = (genus_top5_idxs == gt_genus_t[:, None]).sum().float().item() / gt_genus_t.shape[0]

        # subfamily
        gt_subfamily_t = torch.tensor(gt_subfamily, device="cpu")
        score_subfamily_t = torch.stack([st.cpu() for st in score_subfamily], dim=0)
        subfamily_top1_idxs = torch.argmax(score_subfamily_t, axis=1).view(-1)
        subfamily_top5_idxs = torch.topk(score_subfamily_t, k=5, dim=1, sorted=False)[1]
        subfamily_top1_acc = (subfamily_top1_idxs == gt_subfamily_t).sum().float().item() / gt_subfamily_t.shape[0]
        subfamily_top5_acc = (subfamily_top5_idxs == gt_subfamily_t[:, None]).sum().float().item() / gt_subfamily_t.shape[0]

        # family
        gt_family_t = torch.tensor(gt_family, device="cpu")
        score_family_t = torch.stack([st.cpu() for st in score_family], dim=0)
        family_top1_idxs = torch.argmax(score_family_t, axis=1).view(-1)
        family_top5_idxs = torch.topk(score_family_t, k=5, dim=1, sorted=False)[1]
        family_top1_acc = (family_top1_idxs == gt_family_t).sum().float().item() / gt_family_t.shape[0]
        family_top5_acc = (family_top5_idxs == gt_family_t[:, None]).sum().float().item() / gt_family_t.shape[0]


        result = OrderedDict()
        result["classification"] = pd.DataFrame(
            data=np.array([[species_top1_acc, species_top5_acc, genus_top1_acc, genus_top5_acc,
                            subfamily_top1_acc, subfamily_top5_acc, family_top1_acc, family_top5_acc,]]),
            columns=["species-top-1", "species-top-5", "genus-top-1", "genus-top-5",
                     "subfamily-top-1", "subfamily-top-5", "family-top-1", "family-top-5"],
        )
        return copy.deepcopy(result)
