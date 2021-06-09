import torch
import pathlib
from src.data import make_dataset


DATASET_PATH = pathlib.Path(__file__).parent.parent / 'data/MNIST'


def setup_module(module):
    print('setup')
    make_dataset.main()


def teardown_module(module):
    # not removing MNIST to save time
    print('teardown')


def test_dataset_exists():
    assert DATASET_PATH.exists()
    assert (DATASET_PATH / 'processed').is_dir()
    assert (DATASET_PATH / 'raw').is_dir()
    assert (DATASET_PATH / 'processed/training.pt').is_file()
    assert (DATASET_PATH / 'processed/test.pt').is_file()


def test_dataset_shape():
    train_imgs, train_labels = torch.load(str(DATASET_PATH / 'processed/training.pt'))
    assert train_imgs.shape[0] == 60000, 'Invalid number of training data images'
    assert train_labels.shape[0] == train_imgs.shape[0], 'Number of train labels do not match number of training images'  # noqa
    assert list(train_imgs.shape[1:3]) == [28, 28], 'Invalid dimensions of training images (should be 28x28)'
    assert set(torch.unique(train_labels).tolist()) == set(range(0, 10)), 'Missing labels in training data'

    test_imgs, test_labels = torch.load(str(DATASET_PATH / 'processed/test.pt'))
    assert test_imgs.shape[0] == 10000, 'Invalid number of test data images'
    assert test_labels.shape[0] == test_imgs.shape[0], 'Number of test labels do not match number of test images'
    assert list(test_imgs.shape[1:3]) == [28, 28], 'Invalid dimensions of test images (should be 28x28)'
    assert set(torch.unique(test_labels).tolist()) == set(range(0, 10)), 'Missing labels in testing data'



