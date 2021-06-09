import pathlib
import argparse
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import torch
from torch import nn, optim
from torchvision import datasets, transforms

from models.cnn_model import MyAwesomeModel


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Trainer:
    def __init__(self):
        self.dir = pathlib.Path(__file__).parent
        transform = transforms.Compose([transforms.ToTensor()])
        train = datasets.MNIST(
            str(self.dir.parent / 'data'),
            download=False, train=True, transform=transform
        )
        valid = datasets.MNIST(
            str(self.dir.parent / 'data'),
            download=False, train=False, transform=transform
        )
        self.train_set = torch.utils.data.DataLoader(train, batch_size=64, shuffle=True)
        self.valid_set = torch.utils.data.DataLoader(valid, batch_size=64, shuffle=True)

        self.train_acc = [0]
        self.valid_acc = [0]
        parser = argparse.ArgumentParser(
            description="Script for either training or evaluating",
            usage="python train_model.py <command>"
        )
        parser.add_argument("command", help="Run option (train or evaluate)")
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print('Unrecognized command')
            parser.print_help()
            exit(1)
        # use dispatch pattern to invoke method with same name
        if args.command == 'train':
            self.train()
        if args.command == 'evaluate':
            self.evaluate()
        else:
            raise Exception('Invalid command type (choose train or evaluate)')

    def measure_acc(self, ps, labels):
        _, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        return torch.mean(equals.type(torch.FloatTensor))

    def training_figure(self, name):
        train_acc = np.array(self.train_acc)
        valid_acc = np.array(self.valid_acc)
        epochs = np.array(list(range(0, train_acc.shape[0])))
        ax = plt.figure().gca()
        ax.set_xlim(0, epochs.max()+1)
        ax.set_ylim(0, 100)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.plot(epochs, train_acc, 'g', label='Training')
        ax.plot(epochs, valid_acc, 'b', label='Validation')
        ax.set_title('Training Metrics')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Accuracy %')
        ax.legend()
        plt.savefig(f'{self.dir.parent.parent}/reports/figures/{name}_train.png')

    def train(self):
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('--lr', default=0.001)
        parser.add_argument('--epochs', default=10)
        parser.add_argument('--path', default=f'{str(self.dir)}/save')
        parser.add_argument('--name', required=True)
        args = parser.parse_args(sys.argv[2:])
        print(f'Training with: {args}')

        model = MyAwesomeModel()
        model = model.to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        criterion = nn.NLLLoss()

        for e in range(int(args.epochs)):
            running_loss = 0
            running_acc = 0
            for images, labels in self.train_set:
                model.train()
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)
                # Forward
                optimizer.zero_grad()
                output = model(images)
                ps = torch.exp(output)
                # Backward
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
                # Metrics
                running_loss += loss.item()
                running_acc += self.measure_acc(ps, labels)
            torch.save(model.state_dict(), f'{args.path}/{args.name}.pth')
            print(f'========================[{e+1}/{int(args.epochs)}]========================')
            print(f'Training Loss: {running_loss / len(self.train_set)}')
            acc = (running_acc / len(self.train_set)) * 100
            self.train_acc.append(acc)
            print(f'Training Accuracy: {acc}%')
            self.evaluate(model=model)
        self.training_figure(args.name)
        print('Training Finished!')

    def evaluate(self, model=None):
        if not model:
            parser = argparse.ArgumentParser(description='Training arguments')
            parser.add_argument('--path', default=f'{str(self.dir)}/save')
            parser.add_argument('--name', required=True)
            args = parser.parse_args(sys.argv[2:])
            print(f'Evaluating with: {args}')

            checkpoint = torch.load(f'{args.path}/{args.name}.pth')
            model = MyAwesomeModel()
            model.load_state_dict(checkpoint)
            model = model.to(DEVICE)
        
        total_acc = 0
        with torch.no_grad():
            for images, labels in self.valid_set:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)
                model.eval()
                # Forward
                output = model(images)
                ps = torch.exp(output)
                # Metrics
                total_acc += self.measure_acc(ps, labels).cpu()
        acc = (total_acc / len(self.valid_set)) * 100
        self.valid_acc.append(acc)
        print(f'Test Accuracy: {(total_acc / len(self.valid_set)) * 100}%')


if __name__ == "__main__":
    Trainer()
