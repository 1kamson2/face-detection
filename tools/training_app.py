import imageio.v3 as iio
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms, datasets
from torchvision.io import read_image
from PIL import Image
import csv
import glob
from tools.model import *
from tools.dset import *
torch.set_printoptions(edgeitems=2)
torch.manual_seed(123)
from torch.optim import SGD


class TrainingApp:
    def __init__(self, what_model, lr=0.001, momentum=0.99):
        assert what_model in ['age', 'ethnicity', 'gender'], \
            f"{what_model} doesn't meet criteria."  # do case matching what to initialize
        self.what_model = what_model
        self.model = AgeModel(in_channels=3, out_channels=32)     # now only this.
        self.lr = lr
        self.momentum = momentum
        self.optimizer = self.init_optimizer(self)

    @staticmethod
    def init_optimizer(self):
        return SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum)

    @staticmethod
    def init_loss_fn(self):
        return nn.BCELoss()

    @staticmethod
    def init_training_data(self):
        train_ds = FaceDataset()
        batch_size = 32
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        return train_dl

    def training_loop(self):
        train_dl = self.init_training_data(self)
        loss_fn = self.init_loss_fn(self)
        self.model.train()

        for batch, (target, model_input) in enumerate(train_dl):
            try:
                prediction = self.model(model_input)  # sorts of work
                loss = loss_fn(prediction, target['age'].unsqueeze(1))   # hard coded, make tensor

                # back propagate
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if batch % 10 == 0:
                    print(f"Current loss: {loss.item():.4f}")
            except Exception as e:
                print(f"\n Error: {e} \n"
                    f"Couldn't access the required target: {target['image']}")
                continue
        torch.save(self.model.state_dict(), 'model_weights.pth')


"""
https://discuss.pytorch.org/t/strange-behaviour-of-linear-layer/41366

"""