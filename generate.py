import os
import torch
import numpy as np

from torch import nn, Tensor
from tqdm import trange

from pathlib import Path

from utils.config import parse_generate_config
from utils.checkpoint import load_checkpoint
from utils.sdf import create_grid, sdf_to_mesh, sdf_to_centerline

from models.generator import EDMPrecond
from models.autoencoder import SDFTransformer


@torch.no_grad()
def generate(
    generator: nn.Module, decoder: nn.Module, grid: Tensor, config: dict
) -> None:
    generator.eval()
    decoder.eval()

    # initialize seed for sampling
    config["sample"]["seed"] = (
        torch.randint(0, 99999, (1,))
        if config["sample"]["seed"] < 0
        else torch.tensor([config["sample"]["seed"]], dtype=int)
    ).to(config["device"])

    # sample latent codes
    latents = generator.sample(
        batch_seeds=config["sample"]["seed"],
        num_steps=config["sample"]["num_steps"],
        rho=config["sample"]["rho"],
        S_churn=config["sample"]["churn"],
    ).float()

    # compute the semantic SDF
    semantic_sdf = []
    for queries in grid.view(-1, 3).split(config["sdf"]["grid_batch_size"], dim=0):
        queries = queries.to(config["device"])[None]

        semantic_sdf.append(decoder.decode(latents, queries)[0].numpy(force=True))

    semantic_sdf = np.concatenate(semantic_sdf).reshape(*grid.shape[:-1], -1)

    # create save directory
    save_path = os.path.join(
        config["save"]["path"], f"seed_{config['sample']['seed'].item()}"
    )
    Path(save_path).mkdir(parents=True, exist_ok=True)

    # saving data
    if config["save"]["mesh"]:
        sdf_to_mesh(
            semantic_sdf,
            config["sdf"]["level"],
            config["sdf"]["offset"],
            config["sdf"]["scale"],
            decoder.probabilistic,
            save_path,
        )

    if config["save"]["centerline"]:
        sdf_to_centerline(
            semantic_sdf, config["sdf"]["scale"], decoder.probabilistic, save_path
        )

    if config["save"]["sdf"]:
        np.save(os.path.join(save_path, "sdf"), semantic_sdf)

    if config["save"]["latents"]:
        np.save(os.path.join(save_path, "latents"), latents.numpy(force=True))

    # update seed for next sample
    config["sample"]["num_steps"] += 1


def main() -> None:
    config = parse_generate_config()

    path, name, epoch = config["model"]["generator"]
    generator_path = f"{path}\\{name}\\checkpoint-{epoch}.pth"

    path, name, epoch = config["model"]["decoder"]
    decoder_path = f"{path}\\{name}\\checkpoint-{epoch}.pth"

    generator = load_checkpoint(EDMPrecond, generator_path).to(config["device"])
    decoder = load_checkpoint(SDFTransformer, decoder_path).to(config["device"])

    for _ in trange(config["sample"]["num_samples"], desc="Generating samples"):
        # grid for computing the SDF
        grid = torch.from_numpy(create_grid(config["sdf"]["resolution"])).float()

        generate(generator, decoder, grid, config)


if __name__ == "__main__":
    main()
