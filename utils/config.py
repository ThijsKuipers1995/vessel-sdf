import yaml
import argparse

from torch import device


def str_to_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def merge_args_with_config(
    args: argparse.Namespace, config: dict, delim: str = "."
) -> dict:
    # Convert argparse Namespace to dict
    args_dict = vars(args)

    # Override config values with non-None CLI args
    for key, value in args_dict.items():
        if value is not None:
            # Handle nested keys like "model_path" → config["model"]["path"]
            keys = key.split(delim)
            sub_config = config
            for k in keys[:-1]:
                sub_config = sub_config.setdefault(k, {})
            sub_config[keys[-1]] = value

    return config


def parse_generate_config():
    parser = argparse.ArgumentParser(description="Generate synthetic vessel trees.")

    parser.add_argument(
        "--config",
        type=str,
        default="config_files\\generate.yaml",
        help="Path to config .yaml file.",
    )

    # sdf arguments
    parser.add_argument("--sdf.resolution", type=int)
    parser.add_argument("--sdf.level", type=float)
    parser.add_argument("--sdf.offset", type=float)
    parser.add_argument("--sdf.scale", type=float)
    parser.add_argument("--sdf.grid_batch_size", type=int)

    # sample arguments
    parser.add_argument("--sample.num_samples", type=int)
    parser.add_argument("--sample.seed", type=int)
    parser.add_argument("--sample.num_steps", type=int)
    parser.add_argument("--sample.rho", type=int)
    parser.add_argument("--sample.churn", type=int)

    # model decoder arguments
    parser.add_argument("--model.decoder.path", type=str)
    parser.add_argument("--model.decoder.name", type=str)
    parser.add_argument("--model.decoder.epoch", type=int)

    # model generator arguments
    parser.add_argument("--model.generator.path", type=str)
    parser.add_argument("--model.generator.name", type=str)
    parser.add_argument("--model.generator.epoch", type=int)

    # save arguments
    parser.add_argument("--save.path", type=str)
    parser.add_argument("--save.mesh", type=str_to_bool)
    parser.add_argument("--save.centerline", type=str_to_bool)
    parser.add_argument("--save.latents", type=str_to_bool)
    parser.add_argument("--save.sdf", type=str_to_bool)

    parser.add_argument("--device", type=str)

    args = parser.parse_args()

    config = load_config(args.config)
    config = merge_args_with_config(args, config)

    return config
