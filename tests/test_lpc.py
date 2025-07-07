# test the LPC module to see if we can reconstruct the original signal
import pytest
import torch

from pathlib import Path

from src.data.mocha_timit_datamodule import MTIMITDataModule
from src.models.components.extractors import LPCResidual

@pytest.mark.parametrize("batch_size", [1, 16])
def test_lpc_reconstruction(batch_size: int) -> None:
    """
        :param batch_size: Batch size of the data to be loaded by the dataloader.
    """
    data_dir = "./data/"

    dm = MTIMITDataModule(data_dir=data_dir, batch_size=batch_size)
    dm.prepare_data()
    dm.setup()
    batch = next(iter(dm.train_dataloader()))
    x, y, y_prob = batch

    lpc_extr = LPCResidual(N=320, H=160, TAU=257)
    lpcc = lpc_extr.lpc(x)
    lpc_res = lpc_extr.forward(x)
    reconstructed = lpc_extr.synthesis(lpc_res, lpcc)
    print(lpc_extr.lpc.frames(x).reshape(-1, lpc_extr.N).shape, reconstructed.shape)
    assert torch.allclose(lpc_extr.lpc.frames(x).reshape(-1, lpc_extr.N), reconstructed, atol=1e-7)