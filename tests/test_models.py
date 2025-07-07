import torch
import pytest
from src.models.components.extractors import FeatureExtractor, XCorrExt, IFExt
from src.models.components.encoders import Encoder, XCorrEncoder, IFEncoder
from src.models.components.decoders import Decoder, GRUDecoder


import pytest
from src.models.components.extractors import FeatureExtractor, XCorrExt, IFExt
from src.models.components.encoders import Encoder, XCorrEncoder, IFEncoder
from src.models.components.decoders import Decoder, GRUDecoder
from src.models.components.models import IFModel, XCorrModel

@pytest.fixture
def if_model():
    input_size = 10
    output_size = 5
    return IFModel(input_size=input_size, output_size=output_size)


def test_if_model_forward(if_model):
    batch_size = 32
    input_size = if_model.input_size
    x = torch.randn(batch_size, input_size)

    output = if_model.forward(x)

    assert output.shape == (batch_size, if_model.output_size)
    assert torch.is_tensor(output)
    assert output.dtype == torch.float32


def test_if_model_attributes(if_model):
    assert hasattr(if_model, 'extractor')
    assert hasattr(if_model, 'encoder')
    assert hasattr(if_model, 'decoder')
    assert isinstance(if_model.extractor, IFExt)
    assert isinstance(if_model.encoder, IFEncoder)
    assert isinstance(if_model.decoder, GRUDecoder)



@pytest.fixture
def if_model():
    input_size = 320
    output_size = 999
    return IFModel(input_size=input_size, output_size=output_size)


@pytest.fixture
def xcorr_model():
    input_size = 320
    output_size = 999
    return XCorrModel(input_size=input_size, output_size=output_size)


def test_if_model_forward(if_model):
    batch_size = 32
    input_size = if_model.input_size
    x = torch.randn(batch_size, input_size)

    output = if_model.forward(x)

    assert output.shape == (batch_size, if_model.output_size)
    assert torch.is_tensor(output)
    assert output.dtype == torch.float32


def test_if_model_attributes(if_model):
    assert hasattr(if_model, 'extractor')
    assert hasattr(if_model, 'encoder')
    assert hasattr(if_model, 'decoder')
    assert isinstance(if_model.extractor, IFExt)
    assert isinstance(if_model.encoder, IFEncoder)
    assert isinstance(if_model.decoder, GRUDecoder)


def test_xcorr_extractor():
    # Define synthetic input data with known properties
    batch_size = 32
    num_frames = 999
    num_elements_in_frame = 320
    input_data = torch.randn(batch_size, num_frames, num_elements_in_frame)

    # Instantiate XCorrExt extractor
    xcorr_extractor = XCorrExt(N=320, H=160, TAU=257)  # Example parameters, adjust as needed

    # Extract features
    features = xcorr_extractor.extract(input_data)

    # Perform assertions to check the correctness of extracted features
    # Here, you would compare the features with expected values
    # For example:
   # assert features.shape == (batch_size, 50)  # Check the shape of extracted features
    # Add more assertions as needed



def test_xcorr_model_attributes(xcorr_model):
    assert hasattr(xcorr_model, 'extractor')
    assert hasattr(xcorr_model, 'encoder')
    assert hasattr(xcorr_model, 'decoder')
    assert isinstance(xcorr_model.extractor, XCorrExt)
    assert isinstance(xcorr_model.encoder, XCorrEncoder)
    assert isinstance(xcorr_model.decoder, GRUDecoder)