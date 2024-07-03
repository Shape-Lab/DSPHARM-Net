"""
May 2023

Seungeun Lee, selee@unist.ac.kr
Ilwoo Lyu, ilwoolyu@postech.ac.kr

3D Shape Analysis Lab
Department of Computer Science and Engineering
Pohang University of Science and Technology
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import Deform
from ..lib.sphere import TriangleSearch

from spharmnet.core.models import SPHARM_Net
from spharmnet.core.layers import SHT, ISHT
from spharmnet.lib.sphere import vertex_area, spharm_real
from spharmnet.lib.io import read_mesh


class SHWarpBlock(nn.Module):
    def __init__(
        self, device, sphere, v, f, area, Y, in_ch=2, DL=40, k=6, C=16, L=80, D=3, interval=5, threads=1, verbose=False
    ):
        super().__init__()

        self.v, self.f = v, f
        self.Y = Y

        rot_param = 6
        self.velocity = SPHARM_Net(
            device=device,
            sphere=sphere,
            in_ch=in_ch,
            C=C,
            L=L,
            D=D,
            interval=interval,
            n_class=C,
            threads=threads,
            verbose=verbose,
        )
        self.linear = nn.Conv1d(C, rot_param, kernel_size=1, stride=1, bias=True)

        self.ISHT = ISHT(self.Y)

        # non-rigid alignment
        self.SHT = SHT(DL, self.Y.T, area)

        self.w = nn.Parameter(torch.zeros((rot_param, (DL + 1) ** 2)))
        self.b = nn.Parameter(torch.empty((rot_param, (DL + 1) ** 2)))
        self.b.data.uniform_(-5e-3, 5e-3)
        self.b.data[0, 0] = self.b.data[4, 0] = (4 * math.pi) ** 0.5

        self.deform = Deform(v, f, k)
        self.tree = self.deform.tree

        # rigid alignment
        self.w0 = nn.Parameter(area.reshape(-1, 1) / area.sum())

        self.linear0 = nn.Conv1d(in_ch, rot_param, kernel_size=1, stride=1, bias=True)
        self.linear0.weight.data.zero_()

        # a little perturbation
        a = torch.tensor([1, 0, 0]) * 0.05
        rot = torch.tensor([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
        rot = torch.matrix_exp(rot)
        self.linear0.bias.data = rot[:2, :].flatten()

        self.rigid = Deform(v, f, 0)

    def forward(self, input):
        x = input @ self.w0.abs() * 1e-3
        x = self.linear0(x)
        x = x.transpose(-1, -2)
        x = self.rigid(x)
        fid, bary = self.tree.query(x)
        fid = self.f[fid]
        fid = torch.repeat_interleave(fid[:, None], input.shape[1], dim=1).flatten(-2, -1)
        bary = torch.repeat_interleave(bary[:, None], input.shape[1], dim=1)
        x = (torch.gather(input, -1, fid).reshape(bary.shape) * bary).sum(dim=-1)

        x = self.velocity(x)
        x = F.relu(x)
        x = self.linear(x)

        x = self.SHT(x)
        x = x * self.w + self.b
        x = self.ISHT(x)
        x = x.transpose(-1, -2)

        x = self.deform(x)

        return x


class SPHARM_Reg(nn.Module):
    def __init__(
        self,
        device,
        sphere,
        in_ch=2,
        DL=40,
        k=6,
        C=16,
        L=80,
        D=3,
        interval=5,
        threads=1,
        verbose=False,
        n_class=None,
        Y=None,
    ):
        super().__init__()

        v, f = read_mesh(sphere)
        v = v.astype(float)

        area = vertex_area(v, f)
        area = torch.from_numpy(area).to(device=device, dtype=torch.float32)

        if Y is None:
            Y = spharm_real(v, L if L > DL else DL, threads)
            Y = torch.from_numpy(Y).to(device=device, dtype=torch.float32)

        self.v = torch.from_numpy(v).to(device, dtype=torch.float32)
        self.f = torch.from_numpy(f).to(device, dtype=torch.long)

        self.warp = SHWarpBlock(
            device, sphere, self.v, self.f, area, Y, in_ch, DL, k, C, L, D, interval, threads, verbose
        )

        self.n_class = n_class
        self.device = device

    def forward(self, input, mask=None):
        x = self.warp(input)
        if mask is not None:
            input = input * mask
        tree = TriangleSearch(x.squeeze(0), self.f)
        fid, bary = tree.query(self.v[None, ...])
        fid = self.f[fid]
        fid = torch.repeat_interleave(fid[:, None], input.shape[1], dim=1).flatten(-2, -1)
        bary = torch.repeat_interleave(bary[:, None], input.shape[1], dim=1)
        input = (torch.gather(input, -1, fid).reshape(bary.shape) * bary).sum(dim=-1)

        return x, input


class GuidedAtt(nn.Module):
    def __init__(self, f0, area0, in_ch=3, hd_ch=64, proj_ch=64, vfid_path="model_files/vfids.npy"):
        super().__init__()

        self.f0 = f0
        self.area0 = area0
        self.vfids = np.load(vfid_path)

        self.linear = nn.Linear(in_ch + 1, hd_ch)
        self.linear3 = nn.Linear(hd_ch, 1)
        self.proj = nn.Linear(in_ch, proj_ch)

        self.act = nn.Tanh()
        self.act2 = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()

    def varea_torch(self, v, f):
        u = v / torch.norm(v, dim=-1, keepdim=True)
        a = u[:, f[:, 0], :]
        b = u[:, f[:, 1], :]
        c = u[:, f[:, 2], :]

        normal = torch.cross(a - b, a - c)
        farea = torch.norm(normal, dim=-1)

        vareas = farea[:, self.vfids]
        varea = torch.sum(vareas, dim=-1)
        varea[:, :12] -= farea[:, 0]
        varea = varea / 6

        return varea.unsqueeze(1)

    def linear_mlp(self, x):
        x = self.linear(x.transpose(1, 2))
        x = self.act(x)
        x = self.linear3(x)
        x = self.sigmoid(x)

        return x.transpose(1, 2)

    def forward(self, vnew, dfeat):
        darea = self.varea_torch(vnew, self.f0)
        darea = darea - self.area0

        df = torch.cat((dfeat, darea), dim=1)
        sa = self.linear_mlp(df)

        dfeat_ = sa * dfeat
        Dfeat = dfeat_
        Dfeat = self.proj(Dfeat.transpose(1, 2)).transpose(1, 2)
        Dfeat = self.act2(Dfeat)

        return Dfeat


class DSPHARM_Net(nn.Module):
    def __init__(self, sphere, in_ch, classes, D_C, D_D, D_L, D_DL, D_k, C, D, L, interval, device, verbose=False, threads=1):
        super().__init__()

        v, f = read_mesh(sphere)
        v = v.astype(float)
        self.v = torch.from_numpy(v).to(device=device, dtype=torch.float32)
        self.f = torch.from_numpy(f).to(device=device, dtype=torch.long)

        Y = spharm_real(v, D_L if D_L > D_DL else D_DL, threads)
        Y = torch.from_numpy(Y).to(device=device, dtype=torch.float32)

        area = vertex_area(v, f)
        area_torch = torch.from_numpy(area)[None, None, ...]
        self.area_torch = area_torch.to(device=device, dtype=torch.float32)

        in_ch = len(in_ch)
        self.embed = None

        self.deform = SPHARM_Reg(
            device=device,
            sphere=sphere,
            DL=D_DL,
            k=D_k,
            C=D_C,
            L=D_L,
            D=D_D,
            in_ch=in_ch,
            interval=interval,
            threads=threads,
            verbose=verbose,
            n_class=len(classes),
            Y=Y,
        )
        self.deform.to(device)

        self.parc = SPHARM_Net(
            sphere=sphere,
            device=device,
            in_ch=64,
            n_class=len(classes),
            C=C,
            L=L,
            D=D,
            interval=interval,
            threads=threads,
            verbose=verbose,
        )
        self.parc.to(device)

        self.ga = GuidedAtt(f0=self.f, area0=self.area_torch).to(device)

        self.tree = TriangleSearch(self.v, self.f)

    def forward(self, input):
        vnew, Dfeat = self.deform(input)
        Dfeat = self.ga(vnew, Dfeat)
        Doutput = self.parc(Dfeat)
        fid, bary = self.tree.query(vnew)
        output = torch.multiply(Doutput[:, :, self.f[fid.squeeze(0)]], bary).sum(-1)

        return output, vnew, Dfeat
