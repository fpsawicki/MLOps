import pathlib
import argparse
import pickle
import sys

import numpy as np
from skimage.io import imread
from skimage.color import rgb2gray, rgba2rgb

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from models.cnn_model import MyAwesomeModel


class MyDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        if self.transform:
            x = self.transform(x)
        return x

    def __len__(self):
        return self.data.shape[0]


class Runner:
    def __init__(self):
        self.dir = pathlib.Path(__file__).parent
        parser = argparse.ArgumentParser(
            description="Script for either prediction",
            usage="python predict_model.py <command>"
        )
        parser.add_argument('--data', default=f'{str(self.dir.parent)}/data/PredictMNIST/raw')
        parser.add_argument('--model', default=f'{str(self.dir)}/save/model1')
        args = parser.parse_args(sys.argv[1:5])
        self.predict(args.model, args.data)

    def predict(self, model_path, data):
        # load model
        checkpoint = torch.load(f'{model_path}.pth')
        model = MyAwesomeModel()
        model.load_state_dict(checkpoint)

        # load data
        data = pathlib.Path(data)
        if data.suffix in ['pickle', 'pck', 'pcl']:
            images = pickle.load(str(data))
        elif data.suffix in ['pt']:
            images = torch.load(str(data))
        elif data.suffix == 'npy':
            images = np.load(str(data))
        elif data.is_dir():
            images = [rgb2gray(rgba2rgb(imread(f))) for f in data.iterdir()
                      if f.suffix in ['.jpg', '.jpeg', '.png']]
            images = np.array(images, dtype='uint8')
        else:
            raise Exception('Invalid input data format type!')
        # transform data
        transform = transforms.Compose([transforms.ToTensor()])
        dataset = MyDataset(images, transform=transform)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

        results = []
        with torch.no_grad():
            for images_data in dataloader:
                model.eval()
                # Forward
                output = model(images_data)
                ps = torch.exp(output)
                results.extend([*ps])
        for res in results:
            print(f'Output label: {np.argmax(res.numpy())}')


if __name__ == "__main__":
    Runner()
