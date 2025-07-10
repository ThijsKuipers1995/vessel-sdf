import pandas as pd
import numpy as np
import trimesh
import os

import scipy
from skimage import morphology, measure

from mcubes import marching_cubes


def create_grid(resolution: int, vmin: float = -1.0, vmax: float = 1.0) -> np.ndarray:
    x = np.linspace(vmin, vmax, resolution)

    return np.stack(np.meshgrid(x, x, x, indexing="ij"), axis=-1)


def keep_n_largest_components(
    mesh: trimesh.Trimesh, n_components: int
) -> trimesh.Trimesh:
    # Get the connected components of the mesh
    components = list(mesh.split(only_watertight=False))

    # Sort components by size (number of vertices)
    components.sort(key=lambda c: len(c.vertices), reverse=True)

    # Keep only the n largest components
    mesh = trimesh.util.concatenate(components[:n_components])

    return mesh


def sdf_to_mesh(
    semantic_sdf: np.ndarray,
    level: float = 0.0,
    offset: float = 0.0,
    scale: float = 1.0,
    probabilistic: bool = True,
    save_path: str | None = None,
    n_components: int = 2,
) -> trimesh.Trimesh:
    sdf = semantic_sdf[..., 0]
    half_resolution = sdf.shape[0] / 2

    labels = semantic_sdf[..., 1 + probabilistic :]

    # labels and log variance are optional
    labels = labels.argmax(-1).astype(np.uint8) if labels.size > 0 else None
    logvar = semantic_sdf[..., 1] if probabilistic else None

    # sign sdf to get normals pointing outwards
    vertices, faces = marching_cubes(-(sdf - offset), level)

    # normalize to [-1, 1]
    vertices -= half_resolution + 0.5
    vertices /= half_resolution

    # scale
    vertices *= scale

    # construct trimesh mesh
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    # post process mesh
    mesh = keep_n_largest_components(mesh, n_components)

    vertices = mesh.vertices  # processed vertices

    vertex_indices = tuple(vertices.round().astype(int).T)

    # gather labels and std or set to zero if not available
    labels = (
        np.zeros(vertices.shape[0], dtype=np.uint8)
        if labels is None
        else labels[vertex_indices]
    )
    std = (
        np.zeros(vertices.shape[0], dtype=np.float32)
        if logvar is None
        else scale * np.sqrt(np.exp(logvar[vertex_indices]))
    )  # convert log variance to standard deviation

    mesh.vertex_attributes = dict(labels=labels, std=std)

    # align axes to decoder world space
    mesh.vertices = vertices[..., [1, 0, 2]]

    if save_path is not None:
        mesh.export(os.path.join(save_path, "mesh.ply"))

    return mesh


def skeletonize_sdf(sdf, level=0.0, max_iter=100, refine_iter=2):
    def get_edges(sdf):
        edges = np.copy(sdf)
        edges[scipy.ndimage.binary_erosion(sdf != np.inf).astype(bool)] = np.inf
        return edges

    def remove_voxels(sdf, sdf_mask, candidates, _cached_offsets=[]):
        indices = np.argwhere(candidates)
        dim = indices.shape[-1]

        # cache offsets to find neighbors
        if not _cached_offsets:
            x = dim * [np.arange(-1, 2)]
            _cached_offsets.append(np.stack(np.meshgrid(*x), -1).reshape(-1, dim))

        offsets = _cached_offsets[0]

        neighbourhood = indices[:, None] + offsets
        neighbours = (
            sdf_mask[tuple(neighbourhood.reshape(-1, dim).T)]
            .reshape(-1, 3**dim)
            .sum(-1)
        )

        data = [(n, *index, i) for i, (n, index) in enumerate(zip(neighbours, indices))]
        data_sorted = sorted(data)
        sorted_indices = [i[-1] for i in data_sorted]

        neighbourhood = neighbourhood[sorted_indices]
        indices = indices[sorted_indices]

        n_removed = 0

        for index, neighbours in zip(indices, neighbourhood):
            remove_window = sdf_mask[tuple(neighbours.reshape(-1, dim).T)].reshape(
                *(dim * (3,))
            )
            remove_window[dim * (1,)] = 0
            _, n_components = measure.label(remove_window, return_num=True)

            if n_components < 2 and remove_window.sum() != 1:
                n_removed += 1
                remove_index = tuple(index)
                sdf_mask[remove_index] = 0
                sdf[remove_index] = np.inf

        return n_removed

    # initialize SDF for skeletonization
    sdf = np.copy(sdf)
    sdf[sdf > level] = np.inf
    import matplotlib.pyplot as plt

    for _ in range(max_iter):
        sdf_mask = sdf != np.inf

        # get edge voxels of the outer SDF level set
        edges = get_edges(sdf)

        points = np.argwhere(edges != np.inf)
        # print(points.shape)

        # fig = plt.figure()
        # ax = fig.add_subplot(projection="3d")
        # ax.scatter(*points.T)
        # ax.set_aspect("equal", "box")
        # plt.show()

        # any voxels with a ditance larger than the shortest distance on the edge
        # are schedules for removal
        shortest_distance = 1.1 * edges.min()
        candidates = sdf_mask & (sdf >= shortest_distance)

        # remove voxels from the sdf
        n = remove_voxels(sdf, sdf_mask, candidates)

        if not n:
            break

    skeleton = sdf != np.inf

    for _ in range(refine_iter):
        skeleton = morphology.binary_dilation(skeleton)
        skeleton = morphology.skeletonize(skeleton)

    return skeleton


def sdf_to_centerline(
    semantic_sdf: np.ndarray,
    scale: float,
    probabilistic: bool = True,
    save_path: str | None = None,
) -> pd.DataFrame:
    sdf = semantic_sdf[..., 0]

    labels = semantic_sdf[..., 1 + probabilistic :]
    labels = labels.argmax(-1).astype(np.uint8) if labels.size > 0 else None

    skeleton = np.argwhere(skeletonize_sdf(sdf))
    skeleton_labels = labels[tuple(skeleton.T)] if labels is not None else None

    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(*skeleton.T, c=skeleton_labels, cmap="tab20")
    ax.set_aspect("equal", "box")
    plt.show()

    exit()
