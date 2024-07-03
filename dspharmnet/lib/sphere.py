"""
May 2023

Ilwoo Lyu, ilwoolyu@postech.ac.kr

3D Shape Analysis Lab
Department of Computer Science and Engineering
Pohang University of Science and Technology
"""

import torch
import TriangleSearchCUDA


class TriangleSearch:
    def __init__(self, v, f, norm=False):
        self.v = v
        self.f = f

        if norm:
            self.v = self.v / torch.norm(self.v, dim=1, keepdim=True)

        a = self.v[self.f[:, 0], :]
        b = self.v[self.f[:, 1], :]
        c = self.v[self.f[:, 2], :]

        normal = torch.cross(b - a, c - b)
        zero_normal = (normal == 0).all(dim=-1)
        normal[zero_normal] = a[zero_normal]
        area = torch.norm(normal, dim=1, keepdim=True)
        normal = normal / area

        self.inner = (normal * a).sum(dim=1, keepdim=True)
        self.normal = normal
        self.area = area

        self.tree = TriangleSearchCUDA

    def barycentric(self, v, f, q, fid, area, normal):
        n = normal[fid]
        area = area[fid]
        qa = v[f[fid, 0], :] - q
        qb = v[f[fid, 1], :] - q
        qc = v[f[fid, 2], :] - q

        u = (torch.cross(qb, qc) * n).sum(dim=1, keepdim=True) / area
        v = (torch.cross(qc, qa) * n).sum(dim=1, keepdim=True) / area
        w = 1 - u - v

        return torch.hstack((u, v, w))

    def query(self, q, tol=1e-4, norm=False, bary=True, cand=None, ncand=None, offset=None):
        dim = q.shape
        q = q.reshape(-1, 3)

        if norm:
            q = q / torch.norm(q, dim=1, keepdim=True)

        fid = self.tree.query(self.v, self.f, q, self.normal, self.inner, self.area, tol)

        if bary:
            q = q * (self.inner[fid] / (q * self.normal[fid]).sum(dim=1, keepdim=True))
            b = self.barycentric(self.v, self.f, q, fid, self.area, self.normal)
            fid = fid.reshape(dim[:-1])
            b = b.reshape(dim)

            return fid, b
        else:
            fid = fid.reshape(dim[:-1])

            return fid
