from __future__ import print_function, division
import os
import torch

import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision import transforms, utils
import cv2

from torchvision import transforms


# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode


class Titanic(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, table_data, with_labels=True):
        """
        """
        self.table_data = table_data

        self.x1 = np.array(self.table_data.fillna(-100)['Age'].values / 100)
        self.x2 = np.array([int(type(item) == str) for item in self.table_data.Cabin.values])
        self.x3 = np.array([np.sign(ord(item) - ord('Q')) / 2 for item in self.table_data.fillna('A').Embarked])
        self.x4 = np.array(self.table_data.fillna(-600)['Fare'].values / 600)
        self.x5 = np.array([len(item) / 50 for item in self.table_data.fillna(-50)['Name'].values])
        self.x6 = np.array(self.table_data.Parch.values / 10)
        self.x7 = np.array(self.table_data.Pclass.values / 5)
        self.x8 = np.array([int(item == 'male') for item in self.table_data.Sex.values])
        self.x7 = np.array(self.table_data.SibSp.values / 10)

        if with_labels:
            self.labels = self.table_data.Survived.values.astype(np.float32)
        else:
            self.labels = self.table_data.PassengerId.values

        x = [getattr(self, f'x{i + 1}') for i in range(8)]
        self.data = np.stack(x, axis=-1).astype(np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        self.idx = idx
        return self.data[idx], float(self.labels[idx])

    def get_ids(self, idx):
        return self.idx
