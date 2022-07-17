from typing import List
from collections import UserDict
from typing import List


class _DatasetCatalog(UserDict):
    """
    A global dictionary that stores information about the datasets and how to obtain them.

    It contains a mapping from strings
    (which are names that identify a dataset, e.g. "coco_2014_train")
    to a function which parses the dataset and returns the samples in the
    format of `list[dict]`.

    The returned dicts should be in Detectron2 Dataset format (See DATASETS.md for details)
    if used with the data loader functionalities in `data/build.py,data/detection_transform.py`.

    The purpose of having this catalog is to make it easy to choose
    different datasets, by just using the strings in the config.
    """

    def register(self, name, func):
        """
        Args:
            name (str): the name that identifies a dataset, e.g. "coco_2014_train".
            func (callable): a callable which takes no arguments and returns a list of dicts.
                It must return the same results if called multiple times.
        """
        assert callable(func), "You must register a function with `DatasetCatalog.register`!"
        assert name not in self, "Dataset '{}' is already registered!".format(name)
        self[name] = func

    def get(self, name):
        """
        Call the registered function and return its results.

        Args:
            name (str): the name that identifies a dataset, e.g. "coco_2014_train".

        Returns:
            list[dict]: dataset annotations.
        """
        try:
            f = self[name]
        except KeyError as e:
            raise KeyError(
                "Dataset '{}' is not registered! Available datasets are: {}".format(
                    name, ", ".join(list(self.keys()))
                )
            ) from e
        return f()

    def list(self) -> List[str]:
        """
        List all registered datasets.

        Returns:
            list[str]
        """
        return list(self.keys())

    def remove(self, name):
        """
        Alias of ``pop``.
        """
        self.pop(name)

    def __str__(self):
        return "DatasetCatalog(registered datasets: {})".format(", ".join(self.keys()))

    __repr__ = __str__


DatasetCatalog = _DatasetCatalog()
DatasetCatalog.__doc__ = (
    _DatasetCatalog.__doc__
    + """
    .. automethod:: tracking.data.catalog.DatasetCatalog.register
    .. automethod:: tracking.data.catalog.DatasetCatalog.get
"""
)


class _MapperCatalog(UserDict):
    """
    A catalog that stores information about the datasets and how to obtain them.
    """
    def register(self, name, mapper):
        """
        Args:
            name (str): the name that identifies a dataset, e.g. "coco_2014_train".
            func (callable): a callable which takes no arguments and returns a list of dicts.
                It must return the same results if called multiple times.
        """
        assert name not in self, "Dataset '{}' is already registered!".format(name)
        self[name] = mapper

    def get(self, name):
        """
        Call the registered function and return its results.

        Args:
            name (str): the name that identifies a dataset, e.g. "coco_2014_train".

        Returns:
            list[dict]: dataset annotations.
        """
        try:
            mapper = self[name]
        except KeyError as e:
            raise KeyError(
                "Dataset '{}' is not registered! Available datasets are: {}".format(
                    name, ", ".join(list(self.keys()))
                )
            ) from e
        return mapper

    def list(self) -> List[str]:
        """
        List all registered datasets.

        Returns:
            list[str]
        """
        return list(self.keys())

    def remove(self, name):
        """
        Alias of ``pop``.
        """
        self.pop(name)

    def __str__(self):
        return "DatasetCatalog(registered datasets: {})".format(", ".join(self.keys()))

    __repr__ = __str__

MapperCatalog = _MapperCatalog()
MapperCatalog.__doc__ = (
    _MapperCatalog.__doc__
    + """
    .. automethod:: tracking.data.catalog.MapperCatalog.register
    .. automethod:: tracking.data.catalog.MapperCatalog.get
"""
)
