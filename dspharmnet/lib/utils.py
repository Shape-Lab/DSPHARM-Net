"""
May 2023

Seungeun Lee, selee@unist.ac.kr
Ilwoo Lyu, ilwoolyu@postech.ac.kr

3D Shape Analysis Lab
Department of Computer Science and Engineering
Pohang University of Science and Technology
"""

import os
import numpy as np
import random
import re
import itertools

import torch
from torch.utils.data import Dataset

from spharmnet.lib.io import read_dat
from spharmnet.lib.utils import normalize_data, squeeze_label


class SphericalDataset(Dataset):
    def __init__(
        self,
        data_dir,
        partition,
        fold,
        num_vert,
        classes,
        in_ch,
        seed,
        aug,
        n_splits,
        hemi,
        data_norm=True,
        preload=None,
    ):
        """
        Loader for DSPHARM-Net. This module subdivides the input dataset for cross-validation.

        Parameters
        __________
        data_dir : str
            Path to data directory. Data naming convention should be met.
            Geometry: data_dir/features/{subj}.{lh}.aug0.{feat}.dat.
            Label: data_dir/labels/{subj}.{lh}.aug0.label.dat.
        partition : str
            Partition = ['train', 'val', 'test']
        fold : int
            Cross-validation fold.
        num_vert : int
            # of vertices of the reference sphere mesh used for re-tessellation.
            The samples in the dataset are assumed to follow the same tessellation.
        classes : 1D int array (segmentation) or str array (registration)
            For segmentation, list of labels. Their numbers are not necessarily continuous.
            For registration, target feature.
        in_ch : 1D str array
            Input geometric features.
        seed : int
            Seed for data shuffling. This shuffling is deterministic.
        aug : int
            Not used unless training samples are augmented.
        n_splits : int
            A total of cross-validation folds.
        hemi : str
            Hemisphere = ['lh', 'rh']. Both hemispheres can be trained together.
        data_norm : bool, optional
            Z-score+prctile data normalization.
        preload : str, optional
            Data preloading on a specified device.
        """

        assert partition in ["train", "test", "val"]
        self.num_vert = num_vert
        self.partition = partition
        self.data_norm = data_norm
        self.preload = preload is not None

        feat_dir = os.path.join(data_dir, "features")
        feat_files = os.listdir(feat_dir)
        feat_files = [f for f in feat_files if f.split(".")[1] in hemi]
        feat_files = [f for ch in in_ch for f in feat_files if ".".join(f.split(".")[3:-1]) == ch]

        self.seg = not isinstance(classes[0], str)

        label_dir = os.path.join(data_dir, "labels")
        label_files = os.listdir(label_dir)
        label_files = [f for f in label_files if f.split(".")[1] in hemi]

        feat_dict = dict()
        for f in feat_files:
            temp = f.split(".")[0:2]
            subj = ".".join(temp)
            if subj not in feat_dict:
                feat_dict[subj] = dict()
            key = "aug" + re.sub("[^0-9]", "", f.split(".")[2])
            f_path = os.path.join(feat_dir, f)
            feat_dict[subj].setdefault(key, []).append(f_path)

        label_dict = dict()
        for f in label_files:
            temp = f.split(".")[0:2]
            subj = ".".join(temp)
            if subj not in label_dict:
                label_dict[subj] = dict()
            key = "aug" + re.sub("[^0-9]", "", f.split(".")[2])
            f_path = os.path.join(label_dir, f)
            label_dict[subj][key] = label_dict[subj].setdefault(key, f_path)

        subj_list = feat_dict.keys()
        subj_list = sorted(subj_list)

        random.seed(seed)
        random.shuffle(subj_list)
        train_subj, val_subj, test_subj = self.kfold(subj_list, n_splits, fold)

        self.feat_list = []
        self.name_list = []
        self.label_list = [] if self.seg else np.array([])
        if partition == "train":
            for subj in train_subj:
                for i in range(0, aug + 1):
                    self.feat_list.append(feat_dict[subj]["aug" + str(i)])
                    self.name_list.append(subj)
                    if self.seg:
                        self.label_list.append(label_dict[subj]["aug" + str(i)])

        if partition == "val":
            for subj in val_subj:
                self.feat_list.append(feat_dict[subj]["aug0"])
                self.name_list.append(subj)
                if self.seg:
                    self.label_list.append(label_dict[subj]["aug0"])

        if partition == "test":
            for subj in test_subj:
                self.feat_list.append(feat_dict[subj]["aug0"])
                self.name_list.append(subj)
                if self.seg:
                    self.label_list.append(label_dict[subj]["aug0"])

        self.lut, _ = squeeze_label(classes)

        if self.preload:
            self.data = []
            self.label = []
            for i in range(len(self.feat_list)):
                data, label, _ = self.read_data(i)
                data = torch.tensor(data, device=preload)
                label = torch.tensor(label, device=preload)
                self.data += [data]
                self.label += [label]

    def __len__(self):
        return len(self.feat_list)

    def __getitem__(self, idx):
        if self.preload:
            return self.data[idx], self.label[idx], self.name_list[idx]
        else:
            return self.read_data(idx)

    def read_data(self, idx):
        data = np.array([])
        for f in self.feat_list[idx]:
            temp = read_dat(f, self.num_vert)
            data = np.append(data, temp)

        data = np.reshape(data, (-1, self.num_vert)).astype(np.float32)
        if self.data_norm:
            data = normalize_data(data)

        label = read_dat(self.label_list[idx], self.num_vert)
        label = label.astype(int)
        label = [self.lut[l] for l in label]
        label = np.asarray(label)

        return data, label, self.name_list[idx]

    def kfold(self, subj, n_splits=5, fold=1):
        total_subj = len(subj)
        fold_size = total_subj // n_splits
        fold_residual = total_subj - fold_size * n_splits

        fold_size = [fold_size + 1 if i < fold_residual else fold_size for i in range(n_splits)]
        fold_idx = [0] + list(itertools.accumulate(fold_size))

        id_base = n_splits
        id_val = (id_base + fold - 1) % n_splits
        id_test = (id_base + fold) % n_splits

        val = subj[fold_idx[id_val] : fold_idx[id_val] + fold_size[id_val]]
        test = subj[fold_idx[id_test] : fold_idx[id_test] + fold_size[id_test]]
        if id_val > id_test:
            train = subj[fold_idx[id_test] + fold_size[id_test] : fold_idx[id_val]]
        else:
            train = subj[0 : fold_idx[id_val]] + subj[fold_idx[id_test] + fold_size[id_test] : None]

        return train, val, test

    def subj_list(self):
        return self.name_list


def eval_dice(input, target, n_class, area=None):
    """
    Dice score.

    Parameters
    __________
    input : torch.tensor, shape = [batch, n_vertex]
        Model inference.
    target : torch.tensor, shape = [batch, n_vertex]
        True labels.
    n_class : int
        # of classes.
    area : torch.tensor, shape = [n_vertex]
        Vertex-wise area.

    Returns
    _______
    dice : torch.tensor, shape = [batch, n_class]
        Batch-wise Dice score.
    """
    eps = 1e-8

    if area is None:
        area = 1
    num_batch = input.shape[0]
    batch_numer = torch.zeros(num_batch, n_class)
    batch_denom = torch.zeros(num_batch, n_class)

    for i in range(n_class):
        batch_numer[:, i] = torch.mul(area, ((input == i) & (target == i)).int()).sum(dim=1)
        batch_denom[:, i] = torch.mul(area, (target == i).int()).sum(dim=1) + torch.mul(area, (input == i).int()).sum(
            dim=1
        )

    return 2 * batch_numer / (batch_denom + eps)


def eval_accuracy(input, target, ignore_index=None):
    """
    Accuracy.

    Parameters
    __________
    input : torch.tensor, shape = [batch, n_vertex]
        Model inference.
    target : torch.tensor, shape = [batch, n_vertex]
        True labels.

    Returns
    _______
    n_correct : int
        # of correct vertices.
    n_vert : int
        # of vertices.
    """
    if ignore_index is not None:
        mask = target != ignore_index
        n_correct = (input[mask] == target[mask]).sum().item()
        n_vert = mask.sum().item()
    else:
        n_correct = (input == target).sum().item()
        n_vert = len(target.flatten(0))

    return n_correct, n_vert
