"""
May 2023

Seungeun Lee, selee@unist.ac.kr
Ilwoo Lyu, ilwoolyu@postech.ac.kr

3D Shape Analysis Lab
Department of Computer Science and Engineering
Pohang University of Science and Technology
"""

import torch
from torch.nn.modules.loss import _Loss
import numpy as np

from spharmnet.lib.io import read_mesh


class SILoss(_Loss):
    def __init__(self, device, sphere, k=4):
        super().__init__()

        self.v_orig, self.f_orig = read_mesh(sphere)
        self.f_reg = self.f_orig
        centroid = (
            self.v_orig[self.f_orig[:, 0], :] + self.v_orig[self.f_orig[:, 1], :] + self.v_orig[self.f_orig[:, 2], :]
        )
        centroid /= np.linalg.norm(centroid, axis=1, keepdims=True)

        a = self.v_orig[self.f_orig[:, 0], :]
        b = self.v_orig[self.f_orig[:, 1], :]
        c = self.v_orig[self.f_orig[:, 2], :]
        normal = np.cross(b - a, c - b)
        zero_normal = (normal == 0).all(axis=1)
        normal[zero_normal] = a[zero_normal]
        area = np.linalg.norm(normal, axis=1, keepdims=True)
        normal /= area

        inner_prod_orig = np.sum(centroid * normal, axis=1)
        self.inner_prod_orig = torch.from_numpy(inner_prod_orig).type(torch.float32).to(device)

        self.k = k

    def forward(self, vnew):
        v_reg = vnew.squeeze(0)

        centroid = v_reg[self.f_reg[:, 0], :] + v_reg[self.f_reg[:, 1], :] + v_reg[self.f_reg[:, 2], :]
        cent_norm = torch.norm(centroid, dim=1, keepdim=True).detach()
        centroid /= cent_norm

        a = v_reg[self.f_reg[:, 0], :]
        b = v_reg[self.f_reg[:, 1], :]
        c = v_reg[self.f_reg[:, 2], :]
        normal = torch.cross(b - a, c - b)
        zero_normal = (normal == 0).all(dim=1)
        normal[zero_normal] = a[zero_normal]
        area = torch.norm(normal, dim=1, keepdim=True).detach()
        normal /= area

        inner_prod = torch.sum(centroid * normal, dim=1)
        counts = 1 / (1 + torch.exp(-self.k * inner_prod))
        count = torch.mean(counts)
        sign = -count + 1
        return sign


class AreaLoss(_Loss):
    def __init__(self, v, f, reduction=None):
        super().__init__()

        self.f = f
        a = v[self.f[:, 0], :]
        b = v[self.f[:, 1], :]
        c = v[self.f[:, 2], :]
        self.base = torch.cross(a - b, a - c).norm(dim=-1)
        self.base = self.base.unsqueeze(0)
        self.log2 = torch.log(torch.tensor(2.0))

        if reduction == "mean":
            self.reduction = torch.mean
        elif reduction == "sum":
            self.reduction = torch.sum
        elif reduction == "max":
            self.reduction = torch.max
        else:
            self.reduction = torch.nn.Identity()

    def forward(self, x):
        a = x[..., self.f[:, 0], :]
        b = x[..., self.f[:, 1], :]
        c = x[..., self.f[:, 2], :]
        area = torch.cross(a - b, a - c).norm(dim=-1)
        loss = area / self.base
        loss = torch.clamp(loss, 1e-7)
        loss = (torch.abs(torch.log(loss))) / self.log2
        loss = self.reduction(loss)

        return loss


class CCLoss(_Loss):
    def __init__(self, reduction="mean"):
        super().__init__()

        if reduction == "mean":
            self.reduction = torch.mean
        elif reduction == "sum":
            self.reduction = torch.sum
        elif reduction == "max":
            self.reduction = torch.max
        else:
            self.reduction = torch.nn.Identity()

    def forward(self, input, target):
        loss = 1 - ((input - input.mean(-1, keepdim=True)) * (target - target.mean(-1, keepdim=True))) / input.std(
            -1, keepdim=True
        ) / target.std(-1, keepdim=True)
        loss = loss.mean(dim=-1)
        loss = self.reduction(loss)

        return loss
