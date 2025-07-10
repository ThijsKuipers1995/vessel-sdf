import os
import torch


def get_checkpoints_from_path(path: str, should_contain: str = ""):
    return [
        ckp
        for ckp in os.listdir(path)
        if os.path.isdir(os.path.join(path, ckp)) and should_contain in ckp
    ]


def get_latest_checkpoint(path: str):
    return [ckp for ckp in os.listdir(path) if "checkpoint" in ckp][-1]


@torch.no_grad()
def save_checkpoint(model, epoch, save_dir, metrics=None, model_kwargs=None, tag=None):
    if not save_dir:
        return

    name = f"checkpoint-{epoch}.pth"
    name = f"{tag}_" + name if tag is not None else name

    torch.save(
        {
            "model": model.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
            "model_kwargs": model_kwargs,
        },
        os.path.join(save_dir, name),
    )


def load_checkpoint(
    model_cls, checkpoint_dir, only_return_model=True, device=None, additional_kwargs={}
):
    checkpoint = torch.load(checkpoint_dir, map_location=device)
    model = model_cls(**checkpoint["model_kwargs"])
    model.load_state_dict(checkpoint["model"])

    if only_return_model:
        return model.to(device)
    else:
        return (
            model.to(device),
            checkpoint["model_kwargs"],
            checkpoint["metrics"],
        )
