import torch
import hydra
from train import train
from models.base_model import Discriminator, Generator
from torch.utils.data import DataLoader
from utils import main_utils, train_utils
from utils.train_logger import TrainLogger
from omegaconf import DictConfig, OmegaConf
import torchvision.transforms as transforms
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import torchvision
from dataset import MyDataset

torch.backends.cudnn.benchmark = True

@hydra.main(config_path="config", config_name='config')
def reproduce_hw3():
        main_utils.init(cfg)
    logger = TrainLogger(exp_name_prefix=cfg['main']['experiment_name_prefix'], logs_dir=cfg['main']['paths']['logs'])
    logger.write(OmegaConf.to_yaml(cfg))

    # Set seed for results reproduction
    main_utils.set_seed(cfg['main']['seed'])

    #train_dataset = torchvision.datasets.ImageFolder(cfg['main']['paths']['train'], train_transformation)
    train_dataset = MyDataset(image_path=cfg['main']['paths']['train'], train=True)

    # Prepare Data Loaders for training and validation
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg['train']['batch_size'], shuffle=True,
                                               pin_memory=True, num_workers=cfg['main']['num_workers'])


    # Init model
    image_size = [cfg['train']['in_channel'],cfg['main']['image_shape'],cfg['main']['image_shape']]
    dis_model = Discriminator(in_size=image_size)
    gen_model = Generator(z_dim=cfg['train']['z_shape'])


    if cfg['main']['parallel']:
        dis_model = torch.nn.DataParallel(dis_model)
        gen_model = torch.nn.DataParallel(gen_model)

    if torch.cuda.is_available():
        dis_model = dis_model.cuda()
        gen_model = gen_model.cuda()

    logger.write(main_utils.get_model_string(dis_model))
    logger.write(main_utils.get_model_string(gen_model))

    # Run model
    train_params = train_utils.get_train_params(cfg)

    # Report metrics and hyper parameters to tensorboard
    train(dis_model, gen_model, train_loader, train_params, logger)

@hydra.main(config_path="config", config_name='config')
def main(cfg: DictConfig) -> None:
    """
    Run the code following a given configuration
    :param cfg: configuration file retrieved from hydra framework
    """
    main_utils.init(cfg)
    logger = TrainLogger(exp_name_prefix=cfg['main']['experiment_name_prefix'], logs_dir=cfg['main']['paths']['logs'])
    logger.write(OmegaConf.to_yaml(cfg))

    # Set seed for results reproduction
    main_utils.set_seed(cfg['main']['seed'])

    #train_dataset = torchvision.datasets.ImageFolder(cfg['main']['paths']['train'], train_transformation)
    train_dataset = MyDataset(image_path=cfg['main']['paths']['train'], train=True)

    # Prepare Data Loaders for training and validation
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg['train']['batch_size'], shuffle=True,
                                               pin_memory=True, num_workers=cfg['main']['num_workers'])


    # Init model
    image_size = [cfg['train']['in_channel'],cfg['main']['image_shape'],cfg['main']['image_shape']]
    dis_model = Discriminator(in_size=image_size)
    gen_model = Generator(z_dim=cfg['train']['z_shape'])


    if cfg['main']['parallel']:
        dis_model = torch.nn.DataParallel(dis_model)
        gen_model = torch.nn.DataParallel(gen_model)

    if torch.cuda.is_available():
        dis_model = dis_model.cuda()
        gen_model = gen_model.cuda()

    logger.write(main_utils.get_model_string(dis_model))
    logger.write(main_utils.get_model_string(gen_model))

    # Run model
    train_params = train_utils.get_train_params(cfg)

    # Report metrics and hyper parameters to tensorboard
    train(dis_model, gen_model, train_loader, train_params, logger)

if __name__ == '__main__':
    main()
