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

        self.images_features = self.loading_images_features()

    def __getitem__(self, item):
        if(item == 0):
            print("item 0 = ", item)
        images = h5py.File("data/cache/"+self.dataset_type+".h5", 'r')
        image = images['images'][item].astype('float32')
        image = torch.from_numpy(image)
        features = self.images_features[item]
        return (image, features) 

    def __len__(self) -> int:
        """
        :return: the length of the dataset (number of sample).
        """
        return self.num_of_pics

    def loading_images_features(self):
        features = []
        images_features_dict={}
        ok=0
        with open('data/list_attr_celeba.txt', "r") as file:
            for row_index, row in tqdm.tqdm(enumerate(file)):
                # split allows you to break a string apart with a string key
                temp_features=[]
                for col_index, value in enumerate(row.split(" ")):
                    if (row_index == 0):
                        number_of_images = int(value)
                        continue
                    if (row_index == 1):
                        features.append(value)
                        continue
                    if (value==''):
                        continue
                    if '/n' in value:
                        value.replace('/n', '')
                    ok=1
                    if (col_index==0):
                        id = int(value.split(".")[0])
                        continue
                    if (value == '-1'):
                        temp_features.append(-1)
                    else:
                        temp_features.append(int(value))
                if (row_index>1):
                    images_features_dict[id] = temp_features.copy()
        number_of_features = len(images_features_dict[1])
        images_features_tensor = torch.zeros((number_of_images,number_of_features))
        for i, features in enumerate(images_features_dict):
            images_features_tensor[i] = torch.FloatTensor(images_features_dict[features])
        return(images_features_tensor)
