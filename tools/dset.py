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
IMG_PATH = "../FaceDetection/rsrc/images/*.jpg"


class FaceDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.data_set = self.get_data()

    @lru_cache(1)
    def get_data(self):
        data_info = list()
        for data in (row for row in pd.read_csv(CSV_PATH, delimiter=',').itertuples()):
            img_pxls = np.array([pixel for pixel in data[-1].split(' ')], dtype=np.float32)
            pxls_tensor = torch.from_numpy(img_pxls)
            pxls_tensor = torch.reshape(pxls_tensor, (48, 48))
            pxls_tensor /= 255.0
            pxls_tensor = (torch.stack([pxls_tensor, pxls_tensor, pxls_tensor]))
            key, values = data[4], torch.tensor(data[1:4], dtype=torch.float32)
            data_info.append([{'image': key, 'age': values[0]/120, 'ethnicity': values[1]/10, 'gender': values[2]/2},
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
        self.images_dict = self.get_images()
        self.transform = transforms.Compose([   # <-- put it later in test data
            transforms.Resize(96),
            transforms.CenterCrop(48),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4915, 0.4823, 0.4468),
                (0.2470, 0.2435, 0.2616)),])

    def get_images(self):
        imgs_path = glob.glob(IMG_PATH)
        assert len(imgs_path) >= 1, "No images found."
        img_data = Image.open(imgs_path[0])
        img_data = self.transform(img_data)
        print(imgs_path)
        return img_data

"""        img_pixels = read_image(img_path[index]).float()
img_pixels /= 255.0
pxls_tensor = self.transform(img_pixels).permute(1, 2, 0)
# pxls_tensor = torch.mean(pxls_tensor, dim=-1)   # merge channels"""
