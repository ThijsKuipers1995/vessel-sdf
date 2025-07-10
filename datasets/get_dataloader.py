from torch.utils.data import DataLoader

from datasets.vascusynth import VascuSynth
from datasets.topcow import TopCowSet


def get_dataloader(
    args, split, shuffle=False, drop_last=False, recalculate_radius=True
):
    if args.dataset == "topcow":
        if split == "train" or split == "combined":
            dataset = TopCowSet(
                num_surface=args.num_surface + args.num_pointcloud,
                num_volume=args.num_volume,
                split=split,
                volume_sigma=args.sigma_local,
                random_rotation=args.random_rotation,
                max_rotation_angle=args.max_rotation,  # 20 degrees
                random_mirror=args.random_mirror,
                point_distribution="uniform",
                random_translation=args.random_translation,
                max_translation_offset=args.max_translation_offset,
                erosion_factor=0.0,
                load_centerlines=False,
                dataset_name="mesh",
            )
        else:
            dataset = TopCowSet(
                num_surface=args.num_surface + args.num_pointcloud,
                num_volume=args.num_volume,
                split="val",
                volume_sigma=args.sigma_local,
                point_distribution="uniform",
                erosion_factor=0.0,
                load_centerlines=True,
                dataset_name="mesh",
            )

    elif args.dataset == "vascusynth":
        if split == "train" or split == "combined":
            dataset = VascuSynth(
                num_surface=args.num_surface + args.num_pointcloud,
                num_volume=args.num_volume,
                split=split,
                volume_sigma=args.sigma_local,
                random_rotation=args.random_rotation,
                max_rotation_angle=args.max_rotation,  # 20 degrees
                random_mirror=args.random_mirror,
                random_translation=args.random_translation,
                max_translation_offset=args.max_translation_offset,
                erosion_factor=0.0,
            )
        else:
            dataset = VascuSynth(
                num_surface=args.num_surface + args.num_pointcloud,
                num_volume=args.num_volume,
                split="val",
                volume_sigma=args.sigma_local,
                erosion_factor=0.0,
            )
    else:
        raise NotImplementedError(dataset)

    return DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=shuffle,
        drop_last=drop_last,
        pin_memory=True,
    )
