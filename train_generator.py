import argparse
import os

from pathlib import Path

import torch
from torch.backends import cudnn

from tqdm import tqdm

from datasets.get_dataloader import get_dataloader
from utils.checkpoint import load_checkpoint, save_checkpoint

from models.autoencoder import SDFTransformer
from models.generator import EDMPrecond, EDMLoss


import wandb


def parse_arguments():
    parser = argparse.ArgumentParser(prog="vessel SDF")

    # training params
    parser.add_argument(
        "--decoder",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--decoder_path",
        type=str,
        default="checkpoints",
    )
    parser.add_argument("--decoder_epoch", type=int, default=-1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min_lr", type=float, default=1e-7)
    parser.add_argument("--epochs", type=int, default=6000)  # epochs * 12 steps
    parser.add_argument("--warmup_epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--num_workers", type=int, default=0)

    # data params
    parser.add_argument(
        "--dataset",
        type=str,
        default="topcow",
        choices=["topcow" "vascusynth"],
    )
    parser.add_argument("--num_pointcloud", type=int, default=2048)
    parser.add_argument("--random_rotation", type=int, default=1)
    parser.add_argument("--max_rotation", type=float, default=0.2)  # in rad
    parser.add_argument("--random_mirror", type=int, default=0)
    parser.add_argument("--random_translation", type=int, default=1)
    parser.add_argument("--max_translation_offset", type=int, default=0.0)

    # model params
    parser.add_argument("--feature_dim", type=int, default=192)
    parser.add_argument("--dim_heads", type=int, default=64)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--num_latents", type=int, default=256)
    parser.add_argument("--depth", type=int, default=6)

    # misc
    parser.add_argument("--model_id", type=str, default="")
    parser.add_argument("--data_path", type=str, default="")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints")
    parser.add_argument("--val_iter", type=int, default=10)
    parser.add_argument("--save_checkpoint_iter", type=int, default=4000)
    parser.add_argument("--save_figure_iter", type=int, default=500)
    parser.add_argument(
        "--mode",
        type=str,
        default="online",
        choices=["disabled", "online", "offline"],
    )
    parser.add_argument("--logging_id", type=str, default="vessel-SDF-2")
    parser.add_argument("--detect_anomaly", action="store_true")

    args = parser.parse_args()

    args.num_surface = 0
    args.num_volume = 16
    args.decoder_path = os.path.join(args.decoder_path, args.dataset, "decoder")
    args.checkpoint_path = os.path.join(args.checkpoint_path, args.dataset, "generator")

    # force zero labels if SDF is not semantic
    args.num_labels = args.num_labels if args.semantic else 0

    return args


def train_epoch(model, decoder, optimizer, criterion, scheduler, train_loader, args):
    model.train()
    decoder.eval()

    pbar = tqdm(train_loader)

    for pointcloud, _, _, label_points in pbar:
        optimizer.zero_grad()

        pointcloud = pointcloud.to(args.device)
        cond = None

        if decoder.semantic and decoder.encode_labels:
            pointcloud = torch.cat((pointcloud, label_points.to(args.device)), dim=-1)

        with torch.no_grad():
            latents, _ = decoder.encode(pointcloud)

        loss, loss_dict = criterion(model, latents, labels=cond)

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


@torch.no_grad()
def val_epoch(model, decoder, criterion, val_loader, args):
    model.eval()
    decoder.eval()

    avg_losses = {}

    for i, (pointcloud, _, _, label_points) in enumerate(val_loader):

        pointcloud = pointcloud.to(args.device)
        cond = None

        if decoder.semantic and decoder.encode_labels:
            pointcloud = torch.cat((pointcloud, label_points.to(args.device)), dim=-1)

        latents, _ = decoder.encode(pointcloud)

        _, loss_dict = criterion(model, latents, labels=cond)

        for loss_type, value in loss_dict.items():
            avg_losses[loss_type] = avg_losses.get(loss_type, 0) + value

    for loss_type in avg_losses:
        avg_losses[loss_type] /= i + 1

    metrics = {"val": avg_losses}

    print(f"Validation -", ", ".join(f"{lt}: {l:.3f}" for lt, l in avg_losses.items()))

    wandb.log(
        metrics,
        commit=False,
    )

    return metrics


def train(
    model, decoder, optimizer, criterion, scheduler, train_loader, val_loader, args
):
    val_epoch(model, decoder, criterion, val_loader, args)

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch: {epoch}/{args.epochs}")

        train_epoch(model, decoder, optimizer, criterion, scheduler, train_loader, args)

        wandb.log({})

        if epoch % args.val_iter == 0 or epoch == args.epochs:
            metrics = val_epoch(model, decoder, criterion, val_loader, args)

        if epoch % args.save_checkpoint_iter == 0 or epoch == args.epochs:
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

    kl = "-kl" if "kl" in args.decoder else ""
    model_name = f"SDF-gen-nl{args.num_latents}-f{args.feature_dim}-d{args.depth}-l{args.latent_dim}-h{args.num_heads}{kl}"
    model_name += f"-{args.model_id}" if args.model_id else ""

    checkpoint_path = os.path.join(args.checkpoint_path, model_name)

    Path(checkpoint_path).mkdir(parents=True, exist_ok=True)
    args.checkpoint_path = checkpoint_path

    wandb.init(
        project=args.logging_id,
        name=model_name,
        config=args,
        mode=args.mode,
        tags=["generator", args.dataset],
    )

    train_loader = get_dataloader(args, split="train", shuffle=True, drop_last=True)
    val_loader = get_dataloader(args, split="val", shuffle=False, drop_last=False)

    #
    # Model setup
    #
    model_kwargs = {
        "d_feature": args.feature_dim,
        "d_head": args.dim_heads,
        "n_heads": args.num_heads,
        "channels": args.latent_dim,
        "num_latents": args.num_latents,
        "depth": args.depth,
        "classes": 0,
    }
    args.model_kwargs = model_kwargs

    checkpoint_path = os.path.join(
        args.decoder_path, args.decoder, f"checkpoint-{args.checkpoint_epoch}.pth"
    )

    decoder = load_checkpoint(
        SDFTransformer,
        checkpoint_path,
    ).to(args.device)
    model = EDMPrecond(**model_kwargs).to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = EDMLoss()
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

    train(
        model, decoder, optimizer, criterion, scheduler, train_loader, val_loader, args
    )


if __name__ == "__main__":
    main()
    exit()
    while True:
        try:
            main()
        except OSError as e:
            if "No such file or directory" in str(e):
                raise e
            print("Connection lost, attempting to reestablish...")
        else:
            break
