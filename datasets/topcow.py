import torch
from torch.utils.data import Dataset
from torch import Tensor

import numpy as np

import SimpleITK as sitk

import os

from matplotlib import pyplot as plt
from matplotlib import colors, cm

from scipy.spatial import KDTree

import mcubes
from tqdm import tqdm, trange

import open3d as o3d
import trimesh


class MplColorHelper:

    def __init__(self, cmap_name, start_val, stop_val):
        self.cmap_name = cmap_name
        self.cmap = plt.get_cmap(cmap_name)
        self.norm = colors.Normalize(vmin=start_val, vmax=stop_val)
        self.scalarMap = cm.ScalarMappable(norm=self.norm, cmap=self.cmap)

    def get_rgb(self, val):
        return self.scalarMap.to_rgba(val)


CONNECTIVITY_MAP = {
    1: ((2, 3),),
    8: (2, 4),
    9: (3, 6),
    4: ((5, 11),),
    6: ((7, 12),),
    10: (11, 12),
}


LABEL_MAP = [
    "Background",
    "BA",
    "R-PCA",
    "L-PCA",
    "R-ICA",
    "R-MCA",
    "L-ICA",
    "L-MCA",
    "R-Pcom",
    "L-Pcom",
    "Acom",
    "R-ACA",
    "L-ACA",
    None,
    None,
    "3rd-A2",
]


def plot_points(points, labels=None, normals=None):
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(*points.T, c=labels)

    if normals is not None:
        for p, n in zip(points, normals):
            line = np.vstack((p, p + 0.1 * n))
            ax.plot(*line.T, color="red")

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    plt.show()


def rotation_matrix(alpha, beta, gamma):
    cosa, cosb, cosg = np.cos(alpha), np.cos(beta), np.cos(gamma)
    sina, sinb, sing = np.sin(alpha), np.sin(beta), np.sin(gamma)

    return torch.Tensor(
        [
            [
                cosa * cosb,
                cosa * sinb * sing - sina * cosg,
                cosa * sinb * cosg + sina * sing,
            ],
            [
                sina * cosb,
                sina * sinb * sing + cosa * cosg,
                sina * sinb * cosg - cosa * sing,
            ],
            [-sinb, cosb * sing, cosb * cosg],
        ]
    ).double()


def rotate(x: Tensor, rotation: Tensor):
    return x @ rotation.T


class TopCowSet(Dataset):
    __classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15]
    _vessel_types = [
        "Background",
        "BA",
        "R-PCA",
        "L-PCA",
        "R-ICA",
        "R-MCA",
        "L-ICA",
        "L-MCA",
        "R-Pcom",
        "L-Pcom",
        "Acom",
        "R-ACA",
        "L-ACA",
        None,
        None,
        "3rd-A2",
    ]
    __labels_mirrored = torch.tensor(
        [0, 1, 3, 2, 6, 7, 4, 5, 9, 8, 10, 12, 11, 13], dtype=int
    )

    LABEL_MAP_NO_BACKGROUND = [
        "BA",
        "R-PCA",
        "L-PCA",
        "R-ICA",
        "R-MCA",
        "L-ICA",
        "L-MCA",
        "R-Pcom",
        "L-Pcom",
        "Acom",
        "R-ACA",
        "L-ACA",
        "3rd-A2",
    ]

    def __init__(
        self,
        num_surface: int = 512,
        num_volume: int = 512,
        volume_sigma: float = 0.3,
        split: str = "train",
        include_background_label: bool = False,
        random_rotation: bool = False,
        max_rotation_angle: float = 0.2,
        random_mirror: bool = False,
        random_translation: bool = False,
        max_translation_offset: float = 0.0,
        point_distribution: str = "uniform",
        labels_as_onehot: bool = True,
        erosion_factor: float = 0.0,
        load_centerlines: bool = False,
        use_fps: bool = False,
        data_path: str = "datasets\\data\\TopCoW2024_Data_Release\\",
        process_data: bool = False,
        dataset_name: str = "sdf_data",
        path: str = "datasets\\data\\TopCoW2024_Data_Release\\cow_seg_labelsTr",
        label_offset: int = 1,
    ):
        assert point_distribution in ["uniform", "balanced"]
        assert split in ["train", "val", "combined"]

        self.label_offset = label_offset

        self.data_path = data_path
        self.dataset_name = dataset_name

        self.num_surface = num_surface
        self.num_volume = num_volume

        self.volume_sigma = volume_sigma

        self.split = split

        self.include_background_label = include_background_label

        self.random_rotation = random_rotation
        self.max_rotation_angle = max_rotation_angle

        self.random_mirror = random_mirror

        self.load_centerlines = load_centerlines

        self.random_translation = random_translation
        self.max_translation_offset = max_translation_offset

        self.labels_as_onehot = labels_as_onehot

        self.erosion_factor = erosion_factor

        self.point_distribution = point_distribution
        self.dirs = None
        self.path = path

        self.use_fps = use_fps

        if process_data:
            self.process_data()
            return

        self.get_data_paths()

        # train and validation splits
        if self.split != "combined":
            self.dirs = self.dirs[:104] if self.split == "train" else self.dirs[104:]

        self.data = [None for _ in range(len(self))]
        self.centerline_data = [None for _ in range(len(self))]

        self.pre_load()

        self.one_hot_labels = torch.eye(
            len(self.__classes) - 1 + self.include_background_label, dtype=int
        )

    def class_to_label(self, c: int) -> int:
        return c if c != 15 else 13

    def class_to_label_array(self, c: int) -> int:
        c[c == 15] = 13
        return c

    def class_to_vessel_type(self, c: int) -> str:
        return self.__vessel_types[c]

    def label_to_class(self, l: int) -> int:
        return self.__classes[l]

    def label_to_vessel_type(self, l: int) -> str:
        return self.class_to_vessel_type(self.label_to_class(l))

    def sample_normals(self, mesh, surface, faces):
        bary = trimesh.triangles.points_to_barycentric(
            triangles=mesh.triangles[faces], points=surface
        )
        normals = trimesh.unitize(
            (
                mesh.vertex_normals[mesh.faces[faces]]
                * trimesh.unitize(bary).reshape((-1, 3, 1))
            ).sum(axis=1)
        )

        return normals

    def sample_mesh(self, mesh, n=100_000):
        surface, faces = mesh.sample(n, return_index=True)
        normals = self.sample_normals(mesh, surface, faces)

        return [surface, normals]

    def get_data_paths(self):
        path = os.path.join(self.data_path, self.dataset_name)
        self.dirs = [os.path.join(path, sample_id) for sample_id in os.listdir(path)]

    def get_img_paths(self) -> list[str]:
        return [
            os.path.join(self.path, img)
            for img in os.listdir(self.path)
            if "ct" in img and ".nii" in img
        ]

    def load_item(self, index: int):
        if self.data[index] is None:
            self.data[index] = np.load(
                os.path.join(
                    self.dirs[index].split(".")[0],
                    f"points_{self.point_distribution}.npy",
                )
            )

        return self.data[index]

    def pre_load(self):
        for i in trange(
            len(self),
            desc=f"Loading data f({os.path.join(self.path, self.dataset_name)})",
        ):
            self.load_item(i)

    def process_data(
        self, num_points_per_tree: int = 200_000, smoothing_factor: float = 0.25
    ):
        mesh_data = []

        img_dirs = self.get_img_paths()
        scale_factor = 1 / 42.77645216180071
        # preparing meshes

        for dir in tqdm(img_dirs, desc="Processing meshes"):
            sample_id = dir.split(".")[0].split("\\")[-1]
            save_dir = os.path.join(
                self.data_path, self.dataset_name, sample_id, "centerline.npy"
            )

            img = sitk.ReadImage(dir)

            array = sitk.GetArrayFromImage(img)

            idx = np.where(array != 0)
            array_labels = array[idx]

            spacing = np.asarray(img.GetSpacing())

            vertices, faces = mcubes.marching_cubes(array, 0)
            vertices = np.fliplr(vertices)
            faces = np.fliplr(faces)

            array_idx = np.fliplr(np.stack(idx, axis=-1).astype(float))
            array_idx = array_idx * spacing
            vertices = vertices * spacing

            mean = vertices.mean(axis=0, keepdims=True)

            vertices -= mean
            vertices *= scale_factor

            array_idx -= mean
            array_idx *= scale_factor

            mesh = o3d.geometry.TriangleMesh(
                vertices=o3d.utility.Vector3dVector(vertices),
                triangles=o3d.utility.Vector3iVector(faces),
            )
            mesh = mesh.filter_smooth_laplacian(
                number_of_iterations=10, lambda_filter=0.2
            )

            mesh = trimesh.Trimesh(
                vertices=np.asarray(mesh.vertices), faces=np.asarray(mesh.triangles)
            )

            mesh_data.append((mesh, array_labels, array_idx))

        # sampling points and normals
        for dir, (mesh, array_labels, array_idx) in tqdm(
            zip(img_dirs, mesh_data),
            desc="Generating point data",
            total=len(img_dirs),
        ):
            sample_id = dir.split(".")[0].split("\\")[-1]
            save_dir = os.path.join(self.data_path, self.dataset_name, sample_id)

            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            points, normals = self.sample_mesh(mesh, num_points_per_tree)

            tree = KDTree(array_idx)
            _, idx = tree.query(points)
            points_labels = array_labels[idx]

            point_data = np.concatenate(
                (points, normals, points_labels.reshape(-1, 1)), axis=-1
            )

            # col = MplColorHelper("tab20", 0, 13)
            # colors = col.get_rgb(points_labels)[..., :3]

            # pc_mesh = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
            # pc_mesh.colors = o3d.utility.Vector3dVector(colors)

            # pc_cl = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(cl[..., :3]))

            # o3d.visualization.draw_geometries([pc_mesh, pc_cl])

            mesh.export(os.path.join(save_dir, "mesh.obj"))
            np.save(
                os.path.join(save_dir, "points_uniform.npy"),
                point_data,
            )
            # np.save(os.path.join(save_dir, "centerline.npy"), cl)

            with open(os.path.join(save_dir, "meta_data.txt"), "w") as f:
                f.write(f"scaling_factor: {scale_factor}\n")
                # f.write(f"classes: {list(np.unique(cl[..., -1]).astype(int))}\n")

    def sample_surface(self, tree, n):
        return np.random.choice(n, self.num_surface)

    def __getitem__(self, index: int) -> tuple[torch.Tensor]:
        tree = self.data[index]

        N = tree.shape[0]

        surface_idx = self.sample_surface(tree, N)
        volume_idx = np.random.choice(N, self.num_volume)

        surface_data = torch.from_numpy(tree[surface_idx])

        volume = torch.from_numpy(tree[volume_idx, :3])
        volume += self.volume_sigma * torch.randn_like(volume)

        # labels_volume = torch.zeros(self.num_volume, dtype=int)

        surface, normals, labels_surface = surface_data.split((3, 3, 1), dim=-1)
        labels_surface[labels_surface == 15] = 13
        labels_surface = labels_surface.int().view(-1)

        if self.erosion_factor > 0:
            surface = surface - (self.erosion_factor * normals)

        if self.random_mirror and np.random.rand() >= 0.5:
            reflection = torch.Tensor([-1, 1, 1])
            surface, volume, normals = map(
                lambda x: reflection * x, (surface, volume, normals)
            )
            labels_surface = self.__labels_mirrored[labels_surface].view(-1)

        if self.random_rotation:
            angles = self.max_rotation_angle * (2 * torch.rand(3) - 1)
            rotation = rotation_matrix(*angles)
            surface, volume, normals = map(
                lambda x: rotate(x, rotation), (surface, volume, normals)
            )

        if self.random_translation:
            translation = self.max_translation_offset * (2 * torch.rand(3) - 1)
            surface, volume = (
                surface + translation,
                volume + translation,
            )

        labels_surface -= self.label_offset

        if self.labels_as_onehot:
            labels_surface = self.one_hot_labels[labels_surface]
            # labels_volume = self.one_hot_labels[labels_volume]

        normals = normals.float()

        normals /= normals.norm(dim=-1, keepdim=True) + 1e-7

        return (
            surface.float(),
            volume.float(),
            normals,
            labels_surface,
        )

    def __len__(self):
        return len(self.dirs)
