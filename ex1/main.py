import sys
import pathlib
import argparse

import torch
from torch import nn, optim

from data import mnist
from model import MyAwesomeModel


class TrainOREvaluate(object):
    """ Helper class that will help launch class methods as commands
        from a single script
    """
    def __init__(self):
        train_set, test_set = mnist()
        self.dir = str(pathlib.Path(__file__).parent)
        self.train_set = train_set
        self.test_set = test_set
        parser = argparse.ArgumentParser(
            description="Script for either training or evaluating",
            usage="python main.py <command>"
        )
        parser.add_argument("command", help="Subcommand to run")
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print('Unrecognized command')
            parser.print_help()
            exit(1)
        # use dispatch pattern to invoke method with same name
        getattr(self, args.command)()

    def measure_acc(self, ps, labels):
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        return torch.mean(equals.type(torch.FloatTensor))

    def train(self):
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('--lr', default=0.001)
        parser.add_argument('--epochs', default=10)
        parser.add_argument('--path', default=f'{self.dir}/save')
        parser.add_argument('--name', required=True)
        args = parser.parse_args(sys.argv[2:])
        print(f'Training with: {args}')

        model = MyAwesomeModel()
        model = model.cuda()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        criterion = nn.NLLLoss()

        for e in range(int(args.epochs)):
            running_loss = 0
            running_acc = 0
            for images, labels in self.train_set:
                model.train()
                images = images.cuda()
                labels = labels.cuda()
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
            print(f'Training Accuracy: {(running_acc / len(self.train_set)) * 100}%')
            self.evaluate(model=model)
        print('Training Finished!')

    def evaluate(self, model=None):
        if not model:
            parser = argparse.ArgumentParser(description='Training arguments')
            parser.add_argument('--path', default=f'{self.dir}/save')
            parser.add_argument('--name', required=True)
            args = parser.parse_args(sys.argv[2:])
            print(f'Evaluating with: {args}')

            checkpoint = torch.load(f'{args.path}/{args.name}.pth')
            model = MyAwesomeModel()
            model.load_state_dict(checkpoint)
        model = model.cuda()

        _, test_set = mnist()
        total_acc = 0
        with torch.no_grad():
            for images, labels in self.test_set:
                images = images.cuda()
                labels = labels.cuda()
                model.eval()
                # Forward
                output = model(images)
                ps = torch.exp(output)
                # Metrics
                total_acc += self.measure_acc(ps, labels).cpu()
        print(f'Test Accuracy: {(total_acc / len(self.test_set)) * 100}%')


if __name__ == '__main__':
    TrainOREvaluate()
