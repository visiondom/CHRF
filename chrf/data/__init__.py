# ensure the builtin datasets are registered
from . import datasets, mappers
from .build import build_test_loader, build_train_loader, get_dataset_dicts
from .catalog import DatasetCatalog, MapperCatalog

_all__ = [k for k in globals().keys() if not k.startswith("_")]
