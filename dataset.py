"""
Here, we create a custom dataset
"""
import torch
import pickle
import argparse
import os
import sys
import json
import numpy as np
import re
import pickle
import utils
import tqdm
from utils.types import PathT
import torch.utils.data as data
from torch.utils.data import DataLoader
from typing import Any, Tuple, Dict, List
import torchvision.transforms as transforms
from PIL import Image
import h5py
from utils.image_preprocessing import image_preprocessing_master

class MyDataset(data.Dataset):
    """
    Custom dataset template. Implement the empty functions.
    """
    def __init__(self, image_path, train=True):
        # Set variables
        self.image_features_path = image_path

        #define train or val
        if (train):
            dataset_type = "train"
        else:
            dataset_type = "val"

        self.dataset_type = dataset_type

        self.num_of_pics = len(os.listdir(self.image_features_path))

        if not os.path.isfile("data/cache/"+dataset_type+".h5"):
            image_preprocessing_master()

    def __getitem__(self, item):
        images = h5py.File("data/cache/"+self.dataset_type+".h5", 'r')
        image = images['images'][item].astype('float32')
        image = torch.from_numpy(image)
        return image

    def __len__(self) -> int:
        """
        :return: the length of the dataset (number of sample).
        """
        return self.num_of_pics
