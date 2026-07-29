"""High-resolution cluster visualizations."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def plot_clusters(
    features: Any,
    labels: Any,
    representatives: dict[int, str],
    path: str | Path,
    *,
    kind: str,
    dpi: int = 400,
    random_state: int = 42,
    dim: int = 2,
) -> Path:
    """Plot a two- or three-dimensional SVD projection.

    NMR structure clusters show only a representative molecule image. Query
    clusters intentionally have no text annotations. The input is expected to
    be a bounded visualization sample rather than the complete multi-million-
    row NMRexp collection.
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.offsetbox import AnnotationBbox, OffsetImage
        from sklearn.decomposition import TruncatedSVD
    except ImportError as exc:
        raise RuntimeError(
            "cluster visualization requires numpy, scikit-learn, and matplotlib; install spectune[classifier]"
        ) from exc

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    matrix = np.asarray(features, dtype=np.float32)
    cluster_labels = np.asarray(labels, dtype=np.int32)
    if len(matrix) < 2:
        raise ValueError("at least two records are required for cluster visualization")
    if dim not in (2, 3):
        raise ValueError("cluster visualization dim must be 2 or 3")

    projection = TruncatedSVD(n_components=dim, random_state=random_state).fit_transform(matrix)
    unique_labels = sorted(int(label) for label in np.unique(cluster_labels))
    cmap = plt.get_cmap("tab20", max(len(unique_labels), 1))

    plt.style.use("seaborn-v0_8-whitegrid")
    if dim == 2:
        fig, ax = plt.subplots(figsize=(16, 10), constrained_layout=True)
    else:
        fig = plt.figure(figsize=(16, 10), constrained_layout=True)
        ax = fig.add_subplot(111, projection="3d")
    molecule_annotations: list[tuple[Any, Any, Any]] = []
    for color_index, label in enumerate(unique_labels):
        mask = cluster_labels == label
        points = projection[mask]
        scatter_kwargs = {
            "s": 20,
            "alpha": 0.58,
            "color": cmap(color_index),
            "edgecolors": "black",
            "linewidths": 0.25,
        }
        if dim == 2:
            ax.scatter(points[:, 0], points[:, 1], **scatter_kwargs)
        else:
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], **scatter_kwargs)
        center = points.mean(axis=0)
        representative = representatives.get(label, "")

        if kind == "nmr" and representative:
            image = _molecule_image(representative)
            if image is not None:
                molecule_annotations.append((center, image, cmap(color_index)))

    if molecule_annotations:
        fig.canvas.draw()
        for center, image, color in molecule_annotations:
            if dim == 2:
                annotation_point = center
                xycoords: Any = "data"
            else:
                from mpl_toolkits.mplot3d import proj3d

                x, y, _ = proj3d.proj_transform(center[0], center[1], center[2], ax.get_proj())
                annotation_point = (x, y)
                xycoords = ax.transData
            artist = AnnotationBbox(
                OffsetImage(image, zoom=0.28),
                annotation_point,
                xybox=(34, 34),
                xycoords=xycoords,
                boxcoords="offset points",
                arrowprops={"arrowstyle": "-", "color": color, "alpha": 0.7},
                frameon=True,
                bboxprops={"edgecolor": color, "facecolor": "white", "alpha": 0.92},
            )
            ax.add_artist(artist)
    if dim == 2:
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def _molecule_image(smiles: str) -> Any | None:
    try:
        from rdkit import Chem
        from rdkit.Chem import Draw
    except ImportError:
        return None
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None
    return Draw.MolToImage(molecule, size=(260, 190))


__all__ = ["plot_clusters"]
