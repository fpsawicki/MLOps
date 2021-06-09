# -*- coding: utf-8 -*-
import logging
from pathlib import Path
# from dotenv import find_dotenv, load_dotenv

from torchvision import datasets, transforms


def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    transform = transforms.Compose([transforms.ToTensor()])
    cur_dir = Path(__file__).parent
    datasets.MNIST(f'{cur_dir}', download=True, train=True, transform=transform)
    datasets.MNIST(f'{cur_dir}', download=True, train=False, transform=transform)
    logger.info('finished!')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    # load_dotenv(find_dotenv())
    main()
