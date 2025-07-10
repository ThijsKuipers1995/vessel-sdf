try:
    from .topcow import TopCowSet
except ImportError:
    from topcow import TopCowSet

import numpy as np

import SimpleITK as sitk

import os

from tqdm import trange

import mcubes

import open3d as o3d
import trimesh

from pathlib import Path


class VascuSynth(TopCowSet):
    def __init__(
        self,
        num_surface=512,
        num_volume=512,
        volume_sigma=0.3,
        split="train",
        random_rotation=False,
        max_rotation_angle=0.2,
        random_mirror=False,
        random_translation=False,
        max_translation_offset=0.0,
        erosion_factor=0,
        use_fps=False,
        data_path="E:\VascuSynth",
        process_data=False,
        dataset_name="sdf_data",
        path="E:\VascuSynth\data",
    ):
        super().__init__(
            num_surface,
            num_volume,
            volume_sigma,
            split,
            False,
            random_rotation,
            max_rotation_angle,
            random_mirror,
            random_translation,
            max_translation_offset,
            "uniform",
            False,
            erosion_factor,
            False,
            use_fps,
            data_path,
            process_data,
            dataset_name,
            path,
            label_offset=0,
        )

    def process_data(self, num_points_per_tree=200000, smoothing_factor=0.25):
        for group_id in range(1, 10 + 1):
            group = f"Group{group_id}"

            scale_factor = 1 / 85

            for data_id in trange(1, 12 + 1, desc="Creating points"):
                data = f"data{data_id}"
                file_name = f"testVascuSynth{data_id}_101_101_101_uchar.mhd"

                dir = os.path.join(self.path, group, data, file_name)
                save_dir = os.path.join(
                    self.data_path,
                    self.dataset_name,
                    f"{group}_{data}",
                )

                img = sitk.ReadImage(dir)

                array = sitk.GetArrayFromImage(img)

                vertices, faces = mcubes.marching_cubes(array, 0)
                vertices = np.fliplr(vertices)
                faces = np.fliplr(faces)

                mean = vertices.mean(0)

                vertices -= mean
                vertices *= scale_factor

                mesh = o3d.geometry.TriangleMesh(
                    vertices=o3d.utility.Vector3dVector(vertices),
                    triangles=o3d.utility.Vector3iVector(faces),
                )
                mesh = mesh.filter_smooth_laplacian(
                    number_of_iterations=10, lambda_filter=0.3
                )

                mesh = trimesh.Trimesh(
                    vertices=np.asarray(mesh.vertices), faces=np.asarray(mesh.triangles)
                )

                points, normals = self.sample_mesh(mesh, num_points_per_tree)
                labels = np.zeros((points.shape[0], 1))

                point_data = np.concatenate((points, normals, labels), axis=-1)

                Path(save_dir).mkdir(parents=True, exist_ok=True)

                np.save(os.path.join(save_dir, "points_uniform.npy"), point_data)
                mesh.export(os.path.join(save_dir, "mesh.obj"))


def main():
    import matplotlib.pyplot as plt
    import torch

    dataset = VascuSynth(num_surface=256, process_data=True, split="combined")

    surface, volume, normals, _ = dataset[1]

    print(len(dataset))

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(*surface.T)

    for p, n in zip(surface, normals):
        line = torch.stack((p, p + 0.1 * n), dim=0)
        ax.plot(*line.T, c="red")

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    plt.show()


if __name__ == "__main__":
    main()
