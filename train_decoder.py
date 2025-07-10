import argparse
import os

from pathlib import Path
from random import choice

import numpy as np

import torch
from torch.backends import cudnn

from tqdm import tqdm

from datasets.get_dataloader import get_dataloader
from utils.checkpoint import save_checkpoint

from models.autoencoder import SDFTransformer, SDFLoss

import wandb


def parse_arguments():
    parser = argparse.ArgumentParser(prog="vessel SDF")

    # training params
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=9000)  # epochs * 11 steps
    parser.add_argument("--warmup_epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--regularize_latents", type=bool, default=True)
    parser.add_argument("--w_surface", type=float, default=1.0)
    parser.add_argument("--w_eikonal", type=float, default=0.1)
    parser.add_argument("--w_reg", type=float, default=1e-3)
    parser.add_argument("--encode_labels", type=int, default=1)
    parser.add_argument("--weight_labels", type=int, default=0)
    parser.add_argument(
        "--normal_loss", type=str, default="cosine", choices=["l1", "l2", "cosine"]
    )
    parser.add_argument("--label_loss", type=str, default="mse", choices=["mse", "ce"])

    parser.add_argument("--grid_size", type=int, default=1)

    # data params
    parser.add_argument(
        "--dataset",
        type=str,
        default="topcow",
        choices=["insist", "topcow", "vascusynth", "intra"],
    )
    parser.add_argument("--num_pointcloud", type=int, default=2048)
    parser.add_argument("--num_surface", type=int, default=2048)
    parser.add_argument("--num_volume", type=int, default=1024)
    parser.add_argument("--sigma_local", type=float, default=0.33)
    parser.add_argument("--random_rotation", type=int, default=1)
    parser.add_argument("--random_mirror", type=int, default=0)
    parser.add_argument("--random_translation", type=int, default=1)
    parser.add_argument("--max_rotation", type=float, default=0.2)  # in rad
    parser.add_argument("--max_translation_offset", type=float, default=0.0)

    # model params
    parser.add_argument("--semantic", type=int, default=1)
    parser.add_argument("--num_labels", type=int, default=13)
    parser.add_argument("--feature_dim", type=int, default=192)
    parser.add_argument("--num_latents", type=eval, default=(256,))
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--dim_heads", type=int, default=64)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--encoder_depth", type=int, default=6)
    parser.add_argument("--decoder_depth", type=int, default=0)
    parser.add_argument("--mlp_depth", type=int, default=1)
    parser.add_argument("--drop_path_rate", type=float, default=0)
    parser.add_argument("--variational", type=int, default=1)
    parser.add_argument("--probabilistic", type=int, default=0)
    parser.add_argument("--learnable_queries", type=bool, default=0)
    parser.add_argument(
        "--embed",
        type=str,
        default="linear",
        choices=["linear", "polynomial", "ff", "rff"],
    )

    # misc
    parser.add_argument("--model_id", type=str, default="lr")
    parser.add_argument("--data_path", type=str, default="")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints")
    parser.add_argument("--val_iter", type=int, default=10)
    parser.add_argument("--save_checkpoint_iter", type=int, default=20000)
    parser.add_argument(
        "--mode",
        type=str,
        default="disabled",
        choices=["disabled", "online", "offline"],
    )
    parser.add_argument("--logging_id", type=str, default="vessel-SDF-2")
    parser.add_argument("--detect_anomaly", action="store_true")

    args = parser.parse_args()

    if args.dataset == "vascusynth":
        args.semantic = 0
    if args.dataset == "intra":
        args.num_labels = 2
    if args.dataset == "topcow":
        args.num_labels = 13

    args.num_labels = args.num_labels if args.semantic else 0
    args.encode_labels = args.encode_labels if args.semantic else 0

    return args


def train_epoch(model, optimizer, criterion, scheduler, train_loader, args):
    model.train()

    pbar = tqdm(train_loader)

    for (
        points,
        volume,
        normals,
        labels_points,
    ) in pbar:
        optimizer.zero_grad()

        pointcloud = points[:, : args.num_pointcloud].to(args.device) * args.grid_size
        surface = points[:, args.num_pointcloud :].to(args.device) * args.grid_size
        normals = normals[:, args.num_pointcloud :].to(args.device)

        volume = volume.to(args.device) * args.grid_size

        labels_surface = labels_points[:, args.num_pointcloud :].to(args.device)
        labels_pointcloud = labels_points[:, : args.num_pointcloud].to(args.device)

        if args.semantic and args.encode_labels:
            pointcloud = torch.cat((pointcloud, labels_pointcloud), dim=-1)

        loss, loss_dict = criterion(
            model, pointcloud, surface, volume, normals, labels_surface
        )

        loss.backward()

        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        pbar.set_description(
            f"Training - " + ", ".join(f"{lt}: {l:.3f}" for lt, l in loss_dict.items())
        )

        loss_dict["lr"] = (
            scheduler.get_last_lr()[0] if scheduler is not None else args.lr
        )

        metrics = {"train": loss_dict}

        wandb.log(
            metrics,
            commit=False,
        )

    return metrics


def val_epoch(model, criterion, val_loader, args):
    model.eval()

    avg_losses = {}

    for i, (points, volume, normals, labels_points) in enumerate(tqdm(val_loader), 1):
        pointcloud = points[:, : args.num_pointcloud].to(args.device) * args.grid_size
        surface = points[:, args.num_pointcloud :].to(args.device) * args.grid_size
        normals = normals[:, args.num_pointcloud :].to(args.device)

        volume = volume.to(args.device) * args.grid_size

        labels_pointcloud = labels_points[:, : args.num_pointcloud].to(args.device)
        labels_surface = labels_points[:, args.num_pointcloud :].to(args.device)

        if args.semantic and args.encode_labels:
            pointcloud = torch.cat((pointcloud, labels_pointcloud), dim=-1)

        _, loss_dict = criterion(
            model,
            pointcloud,
            surface,
            volume,
            normals,
            labels_surface,
        )

        for loss_type, value in loss_dict.items():
            avg_losses[loss_type] = avg_losses.get(loss_type, 0) + value

    for loss_type in avg_losses:
        avg_losses[loss_type] /= i

    metrics = {"val": avg_losses}

    print(f"Validation -", ", ".join(f"{lt}: {l:.3f}" for lt, l in avg_losses.items()))

    wandb.log(
        metrics,
        commit=False,
    )

    return metrics


def train(model, optimizer, criterion, scheduler, train_loader, val_loader, args):
    val_epoch(model, criterion, val_loader, args)

    metrics = {}
    epoch = 0

    try:
        for epoch in range(1, args.epochs + 1):
            print(f"Epoch: {epoch}/{args.epochs}")

            metrics = train_epoch(
                model, optimizer, criterion, scheduler, train_loader, args
            )

            wandb.log({})

            if not args.combined and (
                epoch % args.val_iter == 0 or epoch == args.epochs
            ):
                metrics = val_epoch(model, criterion, val_loader, args)

            if epoch % args.save_checkpoint_iter == 0 or epoch == args.epochs:
                save_checkpoint(
                    model,
                    epoch,
                    args.checkpoint_path,
                    metrics=metrics,
                    model_kwargs=args.model_kwargs,
                )

    except KeyboardInterrupt:
        f"Training interrupted, saving model at epoch {epoch}."
        save_checkpoint(
            model,
            epoch,
            args.checkpoint_path,
            metrics=metrics,
            model_kwargs=args.model_kwargs,
        )
        wandb.log({})


def main():
    args = parse_arguments()

    #
    # Training environment setup
    #
    assert args.warmup_epochs <= args.epochs

    if args.detect_anomaly:
        torch.autograd.set_detect_anomaly(True, check_nan=True)

    default_device = "cuda" if torch.cuda.is_available() else None
    args.device = args.device if args.device else default_device

    cudnn.benchmark = True
    kl = "-kl" if args.variational else ""
    nl = "-".join(map(str, args.num_latents))
    header = "sSDF" if args.semantic else "SDF"
    header = f"p{header}" if args.probabilistic else header
    model_name = f"{header}{kl}-np{args.num_pointcloud}-ns{args.num_surface}-nv{args.num_volume}-nl{nl}-f{args.feature_dim}-e{args.encoder_depth}-d{args.decoder_depth}-mlp{args.mlp_depth}-l{args.latent_dim}-h{args.num_heads}"
    model_name += f"-{args.model_id}" if args.model_id else ""

    checkpoint_path = os.path.join(
        args.checkpoint_path, args.dataset, "decoder", model_name
    )

    Path(checkpoint_path).mkdir(parents=True, exist_ok=True)
    args.checkpoint_path = checkpoint_path

    wandb.init(
        project=args.logging_id,
        name=model_name,
        config=args,
        mode=args.mode,
        tags=[args.dataset],
    )

    train_loader = get_dataloader(
        args,
        split="train",
        shuffle=True,
        drop_last=True,
    )

    val_loader = get_dataloader(args, split="val", shuffle=False, drop_last=False)

    #
    # Model setup
    #
    model_kwargs = {
        "feature_dim": args.feature_dim,
        "dim_heads": args.dim_heads,
        "num_heads": args.num_heads,
        "latent_dim": args.latent_dim,
        "num_latents": args.num_latents,
        "decoder_depth": args.decoder_depth,
        "encoder_depth": args.encoder_depth,
        "drop_path_rate": args.drop_path_rate,
        "variational": args.variational,
        "probabilistic": args.probabilistic,
        "learnable_queries": args.learnable_queries,
        "embedding_type": args.embed,
        "num_labels": args.num_labels,
        "encode_labels": args.encode_labels,
        "semantic": args.semantic,
    }
    args.model_kwargs = model_kwargs

    model = SDFTransformer(**model_kwargs).to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-10)
    criterion = SDFLoss(
        args.num_surface,
        regularize_latents=args.regularize_latents,
        w_surface=args.w_surface,
        w_eikonal=args.w_eikonal,
        w_reg=args.w_reg,
        normal_loss_fn=args.normal_loss,
        weight_labels=args.weight_labels,
    )
    warmup_steps = args.warmup_epochs * len(train_loader)
    cooldown_steps = args.epochs * len(train_loader) - warmup_steps
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        [
            torch.optim.lr_scheduler.LinearLR(optimizer, 0.01, 1, warmup_steps),
            torch.optim.lr_scheduler.LinearLR(optimizer, 1, 0.01, cooldown_steps),
        ],
        [warmup_steps],
    )

    print(model)

    train(model, optimizer, criterion, scheduler, train_loader, val_loader, args)


if __name__ == "__main__":
    e = ""
    for _ in range(10):
        try:
            main()
        except OSError as _e:
            e = _e
            print("Connection lost, attempting to reestablish...")
        else:
            break
    else:
        print(e)
