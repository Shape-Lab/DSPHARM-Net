"""
Example script for cortical parcellation.

Copyright 2024 Ilwoo Lyu

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
"""

import os
import argparse
import numpy as np
import torch

from dspharmnet.core.models import DSPHARM_Net

from spharmnet.lib.io import read_feat, read_mesh, write_vtk
from spharmnet.lib.utils import normalize_data, squeeze_label
from spharmnet.lib.sphere import TriangleSearch


class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sphere",
        type=str,
        default="./sphere/ico6.vtk",
        help="Path to a reference sphere mesh (vtk or FreeSurfer format)",
    )
    parser.add_argument("--subj-dir", type=str, help="Path to FreeSurfer's surf", required=True)
    parser.add_argument("--feat-dir", type=str, default="surf", help="Path to geometry for registration metrics")
    parser.add_argument("--native-sphere-dir", type=str, default="surf", help="Path to native sphere")
    parser.add_argument("--out-to-subj", action="store_true", help="Set output's root as the subject dir")
    parser.add_argument("--out-dir", type=str, default="./output", help="Path to inferred labels (output)")
    parser.add_argument("--surface", type=str, default=None, help="Native surface mesh (white, inflated, etc.)")
    parser.add_argument(
        "--native-sphere", type=str, default="sphere", help="Native sphere mesh (sphere, sphere.reg, etc.)"
    )
    parser.add_argument("--ckpt", type=str, default="./logs/best_model_fold1.pth", help="Trained model for inference")
    parser.add_argument("--hemi", type=str, choices=["lh", "rh"], help="Hemisphere for inference", required=True)

    parser.add_argument("--gpu", type=int, default=0, help="GPU ID for inference")
    parser.add_argument("--threads", type=int, default=1, help="# of CPU threads for basis reconstruction")

    args = parser.parse_args()

    return args


def read_subj(subj_dir, feat_dir, hemi, in_ch, data_norm):
    feat_dir = os.path.join(subj_dir, feat_dir)

    feat_list = []
    for ch in in_ch:
        f = os.path.join(feat_dir, hemi + "." + ch)
        feat_list.append(f)

    data = np.array([])
    for f in feat_list:
        feat = read_feat(f)
        data = np.append(data, feat)
    data = np.reshape(data, (len(feat_list), -1)).astype(np.float32)

    if data_norm:
        data = normalize_data(data)

    return data


def main(args):
    device = torch.device(f"cuda:{args.gpu}")

    print("Checkpoint: ", args.ckpt)
    checkpoint = torch.load(args.ckpt, map_location=device)
    ckpt_args = checkpoint["args"]

    _, lut = squeeze_label(ckpt_args["classes"])

    print("Loading data...")
    ckpt_args["in_ch"] = ["curv", "sulc", "inflated.H"]
    native_data = read_subj(args.subj_dir, args.feat_dir, args.hemi, ckpt_args["in_ch"], ckpt_args["data_norm"])
    sphere = os.path.join(args.subj_dir, args.native_sphere_dir, args.hemi + "." + args.native_sphere)
    native_v, native_f = read_mesh(sphere)
    native_v = native_v.astype(float)
    native_f = native_f.astype(int)

    ico_v, ico_f = read_mesh(args.sphere)

    tree = TriangleSearch(native_v, native_f)
    triangle_idx, bary_coeff = tree.query(ico_v)
    in_data = np.multiply(native_data[:, native_f[triangle_idx]], bary_coeff).sum(-1)
    in_data = torch.from_numpy(in_data).to(device=device, dtype=torch.float).unsqueeze(0)

    print("Loading model...")
    model = DSPHARM_Net(
        sphere=os.path.join(ckpt_args["sphere"]),
        in_ch=ckpt_args["in_ch"],
        classes=ckpt_args["classes"],
        D_C=ckpt_args["channel_deform"],
        D_D=ckpt_args["depth_deform"],
        D_L=ckpt_args["bandwidth_deform"],
        D_DL=ckpt_args["D_DL"],
        D_k=ckpt_args["D_k"],
        C=ckpt_args["channel"],
        D=ckpt_args["depth"],
        L=ckpt_args["bandwidth"],
        interval=ckpt_args["interval"],
        device=device,
        threads=args.threads,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print("Making inference...")
    with torch.no_grad():
        output, _, _ = model(in_data)
    output = torch.squeeze(output).cpu().detach().numpy()

    if args.sphere is not None:
        tree = TriangleSearch(ico_v, ico_f)
        triangle_idx, bary_coeff = tree.query(native_v)
        output = np.multiply(output[:, ico_f[triangle_idx]], bary_coeff).sum(-1)

    parc = np.argmax(output, axis=0)
    parc = [lut[p] for p in parc]
    out_dir = args.out_dir
    if args.out_to_subj:
        out_dir = os.path.join(args.subj_dir, out_dir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    print("Saving labels...")
    if args.out_to_subj:
        out_file = args.hemi + ".label.txt"
    else:
        out_prefix = list(filter(None, args.subj_dir.split("/")))
        out_prefix = out_prefix[-1] + "." + args.hemi if len(out_prefix) > 1 else args.hemi
        out_file = out_prefix + ".label.txt"
    np.savetxt(os.path.join(out_dir, out_file), parc, delimiter=",", fmt="%d")

    if args.surface is not None:
        if args.out_to_subj:
            out_file = args.hemi + ".label.vtk"
        else:
            out_file = out_prefix + ".label.vtk"
            surf_v, surf_f = read_mesh(os.path.join(args.subj_dir, args.feat_dir, args.hemi + "." + args.surface))
        write_vtk(os.path.join(out_dir, out_file), surf_v, surf_f, {"label": parc})


if __name__ == "__main__":
    args = get_args()
    main(args)
