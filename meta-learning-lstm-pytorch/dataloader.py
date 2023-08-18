from __future__ import division, print_function, absolute_import

import os
import re
import pdb
import glob
import pickle

import torch
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import PIL.Image as PILI
import numpy as np

from tqdm import tqdm
import os
import torch
import torch.utils.data as data
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

class EpisodeDataset(data.Dataset):

    def __init__(self, csv_file, phase='train', n_shot=5, n_eval=15):
        """Args:
            csv_file (str): path to csv data
            phase (str): train, val or test
            n_shot (int): how many examples per class for training (k/n_support)
            n_eval (int): how many examples per class for evaluation
        """
        data = pd.read_csv(csv_file)
        self.labels = sorted(data['label'].unique())
        grouped = data.groupby('label')

        self.episode_loader = [DataLoader(
            ClassDataset(data=grouped.get_group(label), label=label),
            batch_size=n_shot+n_eval, shuffle=True, num_workers=0) for label in self.labels]

    def __getitem__(self, idx):
        return next(iter(self.episode_loader[idx]))

    def __len__(self):
        return len(self.labels)


class ClassDataset(data.Dataset):

    def __init__(self, data, label):
        """Args:
            data (DataFrame): data for a single label
            label (int): the label of all the data
        """
        self.data = data.drop(columns=['label']).values
        self.label = label

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32), self.label

    def __len__(self):
        return len(self.data)


class EpisodicSampler(data.Sampler):

    def __init__(self, total_classes, n_class, n_episode):
        self.total_classes = total_classes
        self.n_class = n_class
        self.n_episode = n_episode

    def __iter__(self):
        for i in range(self.n_episode):
            yield torch.randperm(self.total_classes)[:self.n_class]

    def __len__(self):
        return self.n_episode


def prepare_data(args):

    train_set = EpisodeDataset(args.csv_train, 'train', args.n_shot, args.n_eval)
    val_set = EpisodeDataset(args.csv_val, 'val', args.n_shot, args.n_eval)
    test_set = EpisodeDataset(args.csv_test, 'test', args.n_shot, args.n_eval)

    train_loader = data.DataLoader(train_set, num_workers=args.n_workers,
        batch_sampler=EpisodicSampler(len(train_set), args.n_class, args.episode))

    val_loader = data.DataLoader(val_set, num_workers=2,
        batch_sampler=EpisodicSampler(len(val_set), args.n_class, args.episode_val))

    test_loader = data.DataLoader(test_set, num_workers=2,
        batch_sampler=EpisodicSampler(len(test_set), args.n_class, args.episode_val))

    return train_loader, val_loader, test_loader

