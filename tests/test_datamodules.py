from pathlib import Path

import pytest
import torch

from src.data.mocha_timit_datamodule import MTIMITDataModule


@pytest.mark.parametrize("batch_size", [1])
def test_mnist_datamodule(batch_size: int) -> None:
    """Tests `MNISTDataModule` to verify that it can be downloaded correctly, that the necessary
    attributes were created (e.g., the dataloader objects), and that dtypes and batch sizes
    correctly match.

    :param batch_size: Batch size of the data to be loaded by the dataloader.
    """
    data_dir = "/dataHDD/chsaikeerthi/2024-chirumamilla/data/"

    dm = MTIMITDataModule(data_dir=data_dir, batch_size=batch_size)
    dm.prepare_data()

    assert not dm.data_train and not dm.data_val and not dm.data_test
    assert Path(data_dir, "MOCHA_TIMIT").exists()
    #assert Path(data_dir, "MOCHA_TIMIT", "raw").exists()

    dm.setup()
    assert dm.data_train and dm.data_val and dm.data_test
    assert dm.train_dataloader() and dm.val_dataloader() and dm.test_dataloader()

    num_datapoints = len(dm.data_train) + len(dm.data_val) + len(dm.data_test)
    assert num_datapoints == 4028
    
    batch = next(iter(dm.train_dataloader()))
    x, y = batch
    assert len(x) == batch_size
    assert len(y) == batch_size
    
    assert y.dtype == torch.float64
    #assert x.dtype == torch.float32
