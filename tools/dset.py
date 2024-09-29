import imageio.v3 as iio
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torchvision import models, transforms, datasets
from torchvision.io import read_image
from PIL import Image
import pandas as pd
import csv
import glob
torch.set_printoptions(edgeitems=2)
torch.manual_seed(123)
from functools import lru_cache
CSV_PATH = "../FaceDetection/rsrc/age_gender.csv"
IMG_PATH = "../FaceDetection/rsrc/images/"
IMG_TYPE = "*.jpg"

_MAX_AGE = 120
_MAX_ETHNICITY = 10
_MAX_GENDER = 2


class FaceDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.data_set = self.get_data()

    @lru_cache(1)
    def get_data(self):
        data_info = list()
        for data in (row for row in pd.read_csv(CSV_PATH, delimiter=',').itertuples()):
            img_pxls = np.array([pixel for pixel in data[-1].split(' ')], dtype=np.float32)
            pxls_tensor = torch.from_numpy(img_pxls).view(-1, 48) / 255.0
            (img_name, age, eth, gen) = (data[4],
                                       torch.tensor(data[1], dtype=torch.float32).div(_MAX_AGE),
                                       torch.tensor(data[2], dtype=torch.long).div(_MAX_ETHNICITY),
                                       torch.tensor(data[3], dtype=torch.float32).div(_MAX_GENDER))
            data_info.append(
                [{'image':      img_name,
                  'age':        age,
                  'ethnicity':  eth,
                  'gender':     gen},
                pxls_tensor])
        return data_info

    @lru_cache(1, typed=True)
    def __getitem__(self, index):
        return self.data_set[index]

    def __len__(self):
        return len(self.data_set)


class FaceImages(Dataset):
    def __init__(self):
        super().__init__()
        self.transform = transforms.Compose([   # <-- put it later in test data
            transforms.Resize(60),
            transforms.CenterCrop(48),
        ])
        self.images = self.get_images()

    def get_images(self):
        imgs_path = glob.glob(IMG_PATH + IMG_TYPE)
        assert len(imgs_path) >= 1, "No images found."
        imgs_list = []
        for img in imgs_path:
            img_t = read_image(img).float() / 255.0     # <--there must be a way to put this in transforms or something
            img_t = self.transform(img_t)
            img_t = torch.mean(img_t, dim=0).view(-1, 48)
            imgs_list.append({"image": img[len(IMG_PATH)::], "data": img_t})
        return imgs_list

    @lru_cache(1, typed=True)
    def __getitem__(self, index):
        return self.images[index]

    def __len__(self):
        return len(self.images)


"""
transforms.Normalize(
    (0.4915, 0.4823, 0.4468),
    (0.2470, 0.2435, 0.2616)),  <-- doing this while taking a mean (to merge channels) means we normalize
                                images two times.
    find another way to merge channels, then use this.
"""