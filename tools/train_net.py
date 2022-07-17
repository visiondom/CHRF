#!/usr/bin/env python
"""
Distributed Training Script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".
"""

import os
import sys

sys.path.append("./")


from chrf.checkpoint import BMKCheckpointer
from chrf.config import get_cfg
from chrf.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    launch,
)
from chrf.evaluation import (
    CUBMultihClsfEvaluator,
    ButterflyMultihClsfEvaluator,
    VegFruMultihClsfEvaluator,
    CarsMultihClsfEvaluator,
    AirsMultihClsfEvaluator,
    CUB2MultihClsfEvaluator,
    DatasetEvaluators,
)


class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can use the cleaner
    "SimpleTrainer", or write your own training loop. You can use
    "tools/plain_train_net.py" as an example.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []

        if "cub_multih" in dataset_name and cfg.TEST.MODE == "classify":
            evaluator_list.append(
                CUBMultihClsfEvaluator(
                    dataset_name, distributed=True, output_dir=output_folder
                )
            )
        elif "butterfly_multih" in dataset_name and cfg.TEST.MODE == "classify":
            evaluator_list.append(
                ButterflyMultihClsfEvaluator(
                    dataset_name, distributed=True, output_dir=output_folder
                )
            )
        elif "vegfru_multih" in dataset_name and cfg.TEST.MODE == "classify":
            evaluator_list.append(
                VegFruMultihClsfEvaluator(
                    dataset_name, distributed=True, output_dir=output_folder
                )
            )
        elif "cars_multih" in dataset_name and cfg.TEST.MODE == "classify":
            evaluator_list.append(
                CarsMultihClsfEvaluator(
                    dataset_name, distributed=True, output_dir=output_folder
                )
            )
        elif "airs_multih" in dataset_name and cfg.TEST.MODE == "classify":
            evaluator_list.append(
                AirsMultihClsfEvaluator(
                    dataset_name, distributed=True, output_dir=output_folder
                )
            )
        if "cub2_multih" in dataset_name and cfg.TEST.MODE == "classify":
            evaluator_list.append(
                CUB2MultihClsfEvaluator(
                    dataset_name, distributed=True, output_dir=output_folder
                )
            )

        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {}".format(dataset_name)
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        BMKCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop (see plain_train_net.py) or
    subclassing the trainer.
    """
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)

    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
