"""
May 2023

Ilwoo Lyu, ilwoolyu@postech.ac.kr

3D Shape Analysis Lab
Department of Computer Science and Engineering
Pohang University of Science and Technology
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..lib.sphere import TriangleSearch


class Deform(nn.Module):
    def __init__(self, v, f, k=6):
        super().__init__()

        self.k = k
        self.v = v
        self.f = f
        self.tree = TriangleSearch(self.v, self.f)

    def forward(self, x):
        rot_x = F.normalize(x[..., :3], dim=-1)
        rot_z = F.normalize(rot_x.cross(x[..., 3:], dim=-1), dim=-1)
        rot_y = F.normalize(rot_z.cross(rot_x, dim=-1), dim=-1)

        if self.k > 0:
            axis = torch.cat(
                (
                    rot_y[..., 2:3] - rot_z[..., 1:2],
                    rot_z[..., 0:1] - rot_x[..., 2:3],
                    rot_x[..., 1:2] - rot_y[..., 0:1],
                ),
                dim=-1,
            )
            trace = rot_x[..., 0:1] + rot_y[..., 1:2] + rot_z[..., 2:3]
            trace = (trace - 1) * 0.5
            trace = torch.clamp(trace, -1 + 1e-7, 1 - 1e-7)
            angle = torch.arccos(trace)
            axis = F.normalize(axis, dim=-1)
            dot = (axis * self.v).sum(dim=-1, keepdim=True)
            cross = torch.cross(axis, self.v[None, ...])

            angle = -angle
            angle /= 2**self.k
            cos = torch.cos(angle)
            sin = torch.sin(angle)
            x = self.v * cos + cross * sin + axis * (dot * (1 - cos))

            for _ in range(self.k):
                x = F.normalize(x, dim=-1)
                y = x
                y = y.reshape(-1, 3)
                fid, bary = self.tree.query(x)
                x = (y[self.tree.f[fid]] * bary[..., None]).sum(dim=-2)
        else:
            rot_mat = torch.cat((rot_x, rot_y, rot_z), dim=-1)
            rot_mat = rot_mat.reshape(*rot_mat.shape[:-1], 3, 3)
            x = torch.matmul(rot_mat, self.v[..., None]).squeeze(-1)

        return F.normalize(x, dim=-1)
