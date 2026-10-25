"""
This module contains a collection of data related classes and functions to train the :mod:`cr.models`.
In a training loop a batch of data is structured as a dictionary on which the modules :mod:`cr.data.datasets`
and :mod:`cr.data.filters` allow to perform several operations.


Examples
########

Create a DataModule to train a model

.. code-block::python

    from cr.data import DataModule, DataModuleConfig
    from cr.data.filters import KeyFilter, KeyFilterConfig
    from cr.data.mappers import KeyRenameMapper, KeyRenameMapperConfig

    # Create the filters and mappers
    filters_mappers = [
        KeyFilter(KeyFilterConfig(keys=["image", "txt"])),
        KeyRenameMapper(
            KeyRenameMapperConfig(key_map={"jpg": "image", "txt": "text"})
        )
    ]

    # Create the DataModule
    data_module = DataModule(
        train_config=DataModuleConfig(
            shards_path_or_urls="your urls or paths",
            decoder="pil",
            shuffle_buffer_size=100,
            per_worker_batch_size=32,
            num_workers=4,
        ),
        train_filters_mappers=filters_mappers,
        eval_config=DataModuleConfig(
            shards_path_or_urls="your urls or paths",
            decoder="pil",
            shuffle_buffer_size=100,
            per_worker_batch_size=32,
            num_workers=4,
        ),
        eval_filters_mappers=filters_mappers,
    )

    # This can then be passed to a :mod:`pytorch_lightning.Trainer` to train a model





The :mod:`cr.data` includes the following submodules:

- :mod:`cr.data.datasets`: a collection of :mod:`pytorch_lightning.LightningDataModule` used to train the models. In particular,
    they can used to create the dataloaders and setup the data pipelines.
- :mod:`cr.data.filters`: a collection of filters used apply filters on a training batch of data/

"""

from .datasets import DataModule

__all__ = ["DataModule"]
