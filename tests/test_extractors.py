import torch
import numpy as np
from src.models.components.extractors import LPCResidual
from LPCTorch.lpctorch.lpc import LPCCoefficients
import librosa

def test_lpc_residual():
    # Create an instance of LPCResidual
    lpc_residual = LPCResidual(N=320, H=160, TAU=257)

    # Create a sample input tensor
    batch_size = 2
    num_samples = 1000
    input_tensor = torch.randn(batch_size, num_samples)
    lpc_cof = LPCCoefficients()
    x_cof_test=lpc_cof(input_tensor)
    x_cof = librosa.core.lpc(input_tensor, order=20)
    assert x_cof_test.shape == x_cof.shape
    # Call the extract method
    output_tensor = lpc_residual.extract(input_tensor)

    
    assert output_tensor.shape == (batch_size, 2 + np.floor((num_samples - lpc_residual.N) / lpc_residual.H).astype(int), lpc_residual.N)




    # Prepare the input tensor
    prepared_input = lpc_residual.prepare_input(input_tensor)

    # Check the shape of the prepared input tensor
    assert prepared_input.shape == (batch_size, num_samples)

    # Check the dtype of the prepared input tensor
    assert prepared_input.dtype == torch.float32