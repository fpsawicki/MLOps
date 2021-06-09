import pytest
import torch
import pathlib
import numpy as np

from src.models.models.cnn_model import MyAwesomeModel


INVALID_IMAGE_1 = np.tile(np.array([1]), 28)
INVALID_IMAGE_2 = np.tile(np.array([1]), (29, 29))
INVALID_IMAGE_3 = np.tile(np.array([1]), (28, 28, 28))


@pytest.fixture
def model():
    model = MyAwesomeModel()
    return model


@pytest.fixture
def data():
    DATASET_PATH = pathlib.Path(__file__).parent.parent / 'data/MNIST/processed/test.pt'
    test_imgs, _ = torch.load(str(DATASET_PATH))
    return test_imgs


@pytest.mark.parametrize("input_idx", [3, 4, 500])
def test_forward_valid(model, data, input_idx):
    data = data[input_idx, :, :].view(1, 1, 28, 28).float()
    with torch.no_grad():
        output = model(data)
        assert list(output.shape) == [1, 10]


@pytest.mark.parametrize("input_img", [INVALID_IMAGE_1, INVALID_IMAGE_2, INVALID_IMAGE_3])
def test_forward_invalid(model, input_img):
    with pytest.raises(RuntimeError):
        input_img = torch.from_numpy(input_img).view(1, 1, 28, 28).float()
        model(input_img)
