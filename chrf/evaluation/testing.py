import logging
import pandas as pd
from collections import OrderedDict
from collections.abc import Mapping


def print_csv_format(results):
    """
    Print main metrics in a format similar to Detectron,
    so that they are easy to copypaste into a spreadsheet.

    Args:
        results (OrderedDict[dict]): task_name -> {metric -> score}
    """
    assert isinstance(
        results, OrderedDict), results  # unordered results cannot be properly printed
    logger = logging.getLogger(__name__)
    for task, res in results.items():
        logger.info("copypaste: Task: {}".format(task))
        if isinstance(res, pd.DataFrame):
            logger.info("copypaste: \n" + res.to_string(index=False))
        elif isinstance(res, dict):
            logger.info("copypaste: {}\n".format(res))


def flatten_results_dict(results):
    """
    Expand a hierarchical dict of scalars into a flat dict of scalars.
    If results[k1][k2][k3] = v, the returned dict will have the entry
    {"k1/k2/k3": v}.

    Args:
        results (dict):
    """
    r = {}
    for k, v in results.items():
        if isinstance(v, pd.DataFrame):
            v = v.to_dict('records', into=dict)[0]
            for kk, vv in v.items():
                r[k + "/" + kk] = vv
        elif isinstance(v, Mapping):
            v = flatten_results_dict(v)
            for kk, vv in v.items():
                r[k + "/" + kk] = vv
        else:
            r[k] = v
    return r
