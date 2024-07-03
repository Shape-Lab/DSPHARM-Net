"""
Example script for training DSPHARM-Net.

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
from tqdm import tqdm
from contextlib import ExitStack

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dspharmnet.core.models import DSPHARM_Net
from dspharmnet.lib.utils import SphericalDataset, eval_accuracy, eval_dice
from dspharmnet.lib.loss import SILoss

from spharmnet.lib.io import read_mesh
from spharmnet.lib.utils import Logger
from spharmnet.lib.loss import DiceLoss

from datetime import datetime

torch.autograd.set_detect_anomaly(True)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sphere", type=str, default="./sphere/ico6.vtk", help="Sphere mesh (vtk or FreeSurfer format)"
    )
    parser.add_argument("--data-dir", type=str, default="./dataset", help="Path to re-tessellated data")
    parser.add_argument("--data-norm", action="store_true", help="Z-score+prctile data normalization")
    parser.add_argument(
        "--preload", type=str, choices=["none", "cpu", "device"], default="device", help="Data preloading"
    )
    parser.add_argument("--in-ch", type=str, default=["curv", "sulc", "inflated.H"], nargs="+", help="List of geometry")
    parser.add_argument(
        "--hemi", type=str, default="lh", nargs="+", choices=["lh", "rh"], help="Hemisphere for learning"
    )
    parser.add_argument("--n-splits", type=int, default=5, required=False, help="A total of cross-validation folds")
    parser.add_argument("--fold", type=int, default=1, required=False, help="Cross-validation fold")
    parser.add_argument(
        "--classes", type=int, default=[0, 1, 2, 3, 4, 5, 6, 7, 8], nargs="+", help="List of regions of interest"
    )

    parser.add_argument("--seed", type=int, default=0, help="Random seed for data shuffling")
    parser.add_argument("--aug", type=int, default=0, help="Level of data augmentation")

    parser.add_argument("--epochs", type=int, default=40, help="Max epoch")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Initial learning rate")
    parser.add_argument("--no-decay", action="store_true", help="Disable decay (every 3 epochs if no progress)")
    parser.add_argument("--log-dir", type=str, default="./logs", help="Path to the log files (output)")
    parser.add_argument("--ckpt-dir", type=str, default=None, help="Path to the checkpoint file (output)")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint (pth) to resume training")

    parser.add_argument(
        "--loss",
        type=str,
        default="ce",
        choices=["dice", "ce"],
        help="dice: Dice loss, ce: cross entropy",
    )
    parser.add_argument("--wsl", type=float, default=1.0, help="Weight for similarity loss")
    parser.add_argument("--wsi", type=float, default=10.0, help="Weight for self-instersection loss (regularity)")

    parser.add_argument("-D", "--depth", type=int, default=3, help="Depth of sulcal labeling network")
    parser.add_argument(
        "-C", "--channel", type=int, default=128, help="# of channels in the entry layer of sulcal labeling network"
    )
    parser.add_argument("-L", "--bandwidth", type=int, default=120, help="Bandwidth of sulcal labeling network")
    parser.add_argument("--interval", type=int, default=5, help="Anchor interval of hamonic coefficients")

    parser.add_argument("-D-D", "--depth-deform", type=int, default=3, help="Depth of feature deformation module")
    parser.add_argument(
        "-D-C",
        "--channel-deform",
        type=int,
        default=64,
        help="# of channels in the entry layer of feature deformation module",
    )
    parser.add_argument(
        "-D-L", "--bandwidth-deform", type=int, default=120, help="Bandwidth of feature deformation module"
    )
    parser.add_argument("-D-DL", type=int, default=20, help="Bandwidth of velocity field refinement")
    parser.add_argument("-D-k", type=int, default=6, help="Numerical integration step for scaling and squaring")
    parser.add_argument("--interval-deform", type=int, default=5, help="Anchor interval of hamonic coefficients")

    parser.add_argument("--gpu", type=int, default=0, help="GPU ID for training (normally, starting with 0)")
    parser.add_argument("--threads", type=int, default=1, help="# of CPU threads for basis reconstruction")

    parser.add_argument("-exp", "--experiment-name", type=str, default="exp", help="Logger Title")

    args = parser.parse_args()

    return args


def step(
    model,
    train_loader,
    device,
    criterion,
    weight,
    epoch,
    logger,
    nclass,
    optimizer=None,
    pbar=False,
    args=None,
):
    if optimizer is not None:
        model.train()
    else:
        model.eval()

    progress = tqdm if pbar else lambda x: x

    running_si = 0.0
    running_sl = 0.0
    running_loss = 0.0

    total_correct = 0
    total_vertex = 0
    total_dice = torch.empty((0, nclass))

    iter = 0
    accuracy = 0

    with torch.no_grad() if optimizer is None else ExitStack():
        for _, (input, label, input_name) in enumerate(progress(train_loader)):

            input = input.to(device)
            label = label.to(device)

            args.subj_name = input_name[0].split(".")[0]
            args.cepoch = epoch

            if optimizer is not None:
                optimizer.zero_grad()

            output, v_new, _ = model(input)

            loss_si = criterion["si"](v_new)
            loss_sl = criterion["sl"](output, label)
            loss_t = loss_sl * weight["sl"] + loss_si * weight["si"]

            if optimizer is not None:
                loss_t.backward()
                optimizer.step()

            running_si += loss_si
            running_sl += loss_sl
            running_loss += loss_t

            _, output = torch.max(output, 1)
            correct, num_vertex = eval_accuracy(output, label)
            total_correct += correct
            total_vertex += num_vertex
            batch_dice = eval_dice(output, label, nclass)
            total_dice = torch.cat([total_dice, batch_dice], dim=0)

            iter += 1
    accuracy = total_correct / total_vertex
    total_dice_mean = torch.mean(total_dice).item()

    logger.write(
        [
            epoch + 1,
            running_loss.item() / iter,
            running_sl.item() / iter,
            running_si.item() / iter,
            accuracy,
            total_dice_mean,
        ]
    )

    return accuracy, total_dice_mean


def main(args):
    device = torch.device(f"cuda:{args.gpu}")
    preload = None if args.preload == "none" else device if args.preload == "device" else args.preload

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    print("Loading data...")
    sphere = os.path.join(args.sphere)
    v, _ = read_mesh(sphere)

    partition = ["train", "val", "test"]
    dataset = dict()

    for partition_type in partition:
        dataset[partition_type] = SphericalDataset(
            data_dir=args.data_dir,
            partition=partition_type,
            fold=args.fold,
            num_vert=v.shape[0],
            in_ch=args.in_ch,
            classes=args.classes,
            seed=args.seed,
            aug=args.aug,
            n_splits=args.n_splits,
            hemi=args.hemi,
            data_norm=args.data_norm,
            preload=preload,
        )

    loader = dict()
    loader["train"] = DataLoader(dataset["train"], batch_size=1, shuffle=False, drop_last=False)
    loader["val"] = DataLoader(dataset["val"], batch_size=1, shuffle=False, drop_last=False)
    loader["test"] = DataLoader(dataset["test"], batch_size=1, shuffle=False, drop_last=False)

    log_dir = os.path.join(args.log_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logger = dict()
    for partition_type in partition:
        logger[partition_type] = Logger(os.path.join(log_dir, partition_type + ".log"))
        logger[partition_type].write(f"\t{str(datetime.now())}:{args.experiment_name} ")

    print("Loading model...")
    start_epoch = 0

    model = DSPHARM_Net(
        sphere=sphere,
        in_ch=args.in_ch,
        classes=args.classes,
        D_C=args.channel_deform,
        D_D=args.depth_deform,
        D_L=args.bandwidth_deform,
        D_DL=args.D_DL,
        D_k=args.D_k,
        C=args.channel,
        D=args.depth,
        L=args.bandwidth,
        interval=args.interval,
        device=device,
        threads=args.threads,
    )
    model.to(device)
    logger["train"].write("{} paramerters in total".format(sum(x.numel() for x in model.parameters())))
    arguments = args.__dict__

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "max", factor=0.1, patience=3, threshold=1e-4, threshold_mode="abs", min_lr=1e-8
    )

    criterion = {}

    criterion["si"] = SILoss(device=device, sphere=sphere)

    if args.loss == "ce":
        criterion["sl"] = nn.CrossEntropyLoss()
    elif args.loss == "dice":
        criterion["sl"] = DiceLoss(eps=1e-8)

    weight = {
        "sl": args.wsl,
        "si": args.wsi,
    }

    total_weight = sum(weight.values())
    weight = {key: value / total_weight for key, value in weight.items()}

    for partition_type in partition:
        if partition_type == "train":
            logger[partition_type].write(arguments)
        logger[partition_type].write({"fold_data": dataset[partition_type].subj_list()})
        logger[partition_type].write(
            [
                "{:10}".format("Epoch"),
                "{:15}".format("Total-Loss"),
                "{:15}".format("SL-Loss"),
                "{:15}".format("SI-Loss"),
                "{:15}".format("Accuracy"),
                "{:15}".format("Dice"),
            ]
        )

    if args.ckpt_dir is None:
        best_acc = 0
        best_dice = 0
        for epoch in range(start_epoch, args.epochs):
            step(
                model,
                loader["train"],
                device,
                criterion,
                weight,
                epoch,
                logger["train"],
                len(args.classes),
                optimizer=optimizer,
                pbar=True,
                args=args,
            )
            val_acc, val_dice = step(
                model, loader["val"], device, criterion, weight, epoch, logger["val"], len(args.classes), args=args
            )
            if not args.no_decay:
                scheduler.step(val_acc)

            if val_dice > best_dice:

                best_dice = val_dice
                best_acc = val_acc
                print("Saving checkpoint...")
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "acc": best_acc,
                        "args": arguments,
                    },
                    os.path.join(log_dir, "best_model_fold{}.pth".format(args.fold)),
                )
        test_ckpt = torch.load(os.path.join(log_dir, "best_model_fold{}.pth".format(args.fold)))
        model.load_state_dict(test_ckpt["model_state_dict"])
        model.to(device)
        step(
            model,
            loader["test"],
            device,
            criterion,
            weight,
            test_ckpt["epoch"],
            logger["test"],
            len(args.classes),
            args=args,
        )
    else:
        test_ckpt = torch.load(os.path.join(args.ckpt_dir, "best_model_fold{}.pth".format(args.fold)))
        model.load_state_dict(test_ckpt["model_state_dict"])
        model.to(device)
        step(
            model,
            loader["test"],
            device,
            criterion,
            weight,
            test_ckpt["epoch"],
            logger["test"],
            len(args.classes),
            args=args,
        )


if __name__ == "__main__":
    args = get_args()
    main(args)
