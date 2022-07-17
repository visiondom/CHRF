import os

from .. import mappers
from ..catalog import DatasetCatalog, MapperCatalog

from .cub_multih import load_cub_data_multih
from .butterfly_multih import load_butterfly_data_multih
from .vegfru_multih import load_vegfru_data_multih
from .cars_multih import load_cars_data_multih
from .airs_multih import load_airs_data_multih
from .cub2_multih import load_cub2_data_multih


def register_cub_multih_clsf_h(datadir):
    train_name = "cub_multih_train"
    val_name = "cub_multih_val"
    DatasetCatalog.register(train_name, lambda: load_cub_data_multih(datadir)[0])
    MapperCatalog.register(train_name, mappers.CUBClassifierMapper)
    DatasetCatalog.register(val_name, lambda: load_cub_data_multih(datadir)[1])
    MapperCatalog.register(val_name, mappers.CUBClassifierMapper)


def register_cub2_mutih(datadir):
    train_name = "cub2_multih_train"
    val_name = "cub2_multih_val"
    DatasetCatalog.register(train_name, lambda: load_cub2_data_multih(datadir)[0])
    MapperCatalog.register(train_name, mappers.CUBClassifierMapper)
    DatasetCatalog.register(val_name, lambda: load_cub2_data_multih(datadir)[1])
    MapperCatalog.register(val_name, mappers.CUBClassifierMapper)

def register_butterfly_mutih(datadir):
    trainval_name = "butterfly_multih_trainval"
    train_name = "butterfly_multih_train"
    val_name = "butterfly_multih_val"
    test_name = "butterfly_multih_test"
    DatasetCatalog.register(trainval_name, lambda: load_butterfly_data_multih(datadir)[0])
    MapperCatalog.register(trainval_name, mappers.ButterflyClassifierMapper)
    DatasetCatalog.register(train_name, lambda: load_butterfly_data_multih(datadir)[1])
    MapperCatalog.register(train_name, mappers.ButterflyClassifierMapper)
    DatasetCatalog.register(val_name, lambda: load_butterfly_data_multih(datadir)[2])
    MapperCatalog.register(val_name, mappers.ButterflyClassifierMapper)
    DatasetCatalog.register(test_name, lambda: load_butterfly_data_multih(datadir)[3])
    MapperCatalog.register(test_name, mappers.ButterflyClassifierMapper)

def register_vegfru_mutih(datadir):
    trainval_name = "vegfru_multih_trainval"
    train_name = "vegfru_multih_train"
    val_name = "vegfru_multih_val"
    test_name = "vegfru_multih_test"
    DatasetCatalog.register(trainval_name, lambda: load_vegfru_data_multih(datadir)[0])
    MapperCatalog.register(trainval_name, mappers.VegFruClassifierMapper)
    DatasetCatalog.register(train_name, lambda: load_vegfru_data_multih(datadir)[1])
    MapperCatalog.register(train_name, mappers.VegFruClassifierMapper)
    DatasetCatalog.register(val_name, lambda: load_vegfru_data_multih(datadir)[2])
    MapperCatalog.register(val_name, mappers.VegFruClassifierMapper)
    DatasetCatalog.register(test_name, lambda: load_vegfru_data_multih(datadir)[3])
    MapperCatalog.register(test_name, mappers.VegFruClassifierMapper)


def register_cars_mutih(datadir):
    train_name = "cars_multih_train"
    val_name = "cars_multih_val"
    DatasetCatalog.register(train_name, lambda: load_cars_data_multih(datadir)[0])
    MapperCatalog.register(train_name, mappers.CarsClassifierMapper)
    DatasetCatalog.register(val_name, lambda: load_cars_data_multih(datadir)[1])
    MapperCatalog.register(val_name, mappers.CarsClassifierMapper)

def register_airs_mutih(datadir):
    train_name = "airs_multih_train"
    val_name = "airs_multih_val"
    DatasetCatalog.register(train_name, lambda: load_airs_data_multih(datadir)[0])
    MapperCatalog.register(train_name, mappers.AirsClassifierMapper)
    DatasetCatalog.register(val_name, lambda: load_airs_data_multih(datadir)[1])
    MapperCatalog.register(val_name, mappers.AirsClassifierMapper)


_root = os.getenv("BENCH_DATASETS", "Data/GattentionBmk")
register_cub_multih_clsf_h(os.path.join(_root, "CUB", "CUB_200_2011", "CUB_200_2011"))

register_butterfly_mutih(os.path.join(_root, "Butterfly200"))
register_vegfru_mutih(os.path.join(_root, "vegfru-dataset"))

register_cars_mutih(os.path.join(_root, "stanford_cars"))
register_airs_mutih(os.path.join(_root, "fgvc-aircraft-2013b/data"))
register_cub2_mutih(os.path.join(_root, "CUB", "CUB_200_2011", "CUB_200_2011"))