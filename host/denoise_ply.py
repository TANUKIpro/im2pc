#!/usr/bin/env python3
"""Point cloud denoising module using scipy and scikit-learn.

Provides DBSCAN clustering and Statistical Outlier Removal (SOR)
for cleaning up noisy point clouds, especially suited for small objects.

Supports large point clouds (millions of points) via voxel downsampling.
"""

import argparse
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN


def voxel_downsample(
    points: np.ndarray,
    colors: np.ndarray | None,
    voxel_size: float,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """Downsample point cloud using voxel grid.

    Args:
        points: Point coordinates (N, 3).
        colors: Point colors (N, 3) or None.
        voxel_size: Size of voxel grid cells.

    Returns:
        Tuple of (downsampled_points, downsampled_colors, voxel_indices).
        voxel_indices maps each output point to a representative input index.
    """
    # Compute voxel indices for each point
    voxel_indices = np.floor(points / voxel_size).astype(np.int32)

    # Use dictionary to find unique voxels and representative points
    voxel_dict: dict[tuple, list[int]] = {}
    for i, voxel_idx in enumerate(map(tuple, voxel_indices)):
        if voxel_idx not in voxel_dict:
            voxel_dict[voxel_idx] = []
        voxel_dict[voxel_idx].append(i)

    # For each voxel, compute centroid and average color
    downsampled_points = []
    downsampled_colors = [] if colors is not None else None
    representative_indices = []

    for voxel_idx, point_indices in voxel_dict.items():
        indices = np.array(point_indices)
        centroid = points[indices].mean(axis=0)
        downsampled_points.append(centroid)
        representative_indices.append(point_indices[0])

        if colors is not None:
            avg_color = colors[indices].mean(axis=0)
            downsampled_colors.append(avg_color)

    result_points = np.array(downsampled_points, dtype=np.float32)
    result_colors = None
    if downsampled_colors is not None:
        result_colors = np.array(downsampled_colors, dtype=colors.dtype)

    return result_points, result_colors, np.array(representative_indices)


def statistical_outlier_removal(
    points: np.ndarray,
    colors: np.ndarray | None,
    nb_neighbors: int = 20,
    std_ratio: float = 2.0,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """Remove statistical outliers based on mean distance to neighbors.

    Args:
        points: Point coordinates (N, 3).
        colors: Point colors (N, 3) or None.
        nb_neighbors: Number of neighbors to analyze.
        std_ratio: Standard deviation multiplier for threshold.

    Returns:
        Tuple of (filtered_points, filtered_colors, inlier_indices).
    """
    if len(points) <= nb_neighbors:
        return points, colors, np.arange(len(points))

    # Build KD-tree for efficient neighbor search
    tree = cKDTree(points)

    # Query k+1 neighbors (includes self)
    distances, _ = tree.query(points, k=nb_neighbors + 1)

    # Mean distance to neighbors (excluding self at index 0)
    mean_distances = distances[:, 1:].mean(axis=1)

    # Compute threshold: mean + std_ratio * std
    global_mean = mean_distances.mean()
    global_std = mean_distances.std()
    threshold = global_mean + std_ratio * global_std

    # Filter inliers
    inlier_mask = mean_distances < threshold
    inlier_indices = np.where(inlier_mask)[0]

    filtered_points = points[inlier_mask]
    filtered_colors = colors[inlier_mask] if colors is not None else None

    return filtered_points, filtered_colors, inlier_indices


def dbscan_largest_cluster(
    points: np.ndarray,
    colors: np.ndarray | None,
    eps: float,
    min_samples: int = 10,
    max_points_for_dbscan: int = 500000,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, int]:
    """Extract the largest cluster using DBSCAN.

    For large point clouds, uses voxel downsampling to reduce memory usage.

    Args:
        points: Point coordinates (N, 3).
        colors: Point colors (N, 3) or None.
        eps: DBSCAN epsilon (maximum distance between samples).
        min_samples: Minimum samples for core point.
        max_points_for_dbscan: Maximum points before downsampling.
        verbose: Print progress.

    Returns:
        Tuple of (filtered_points, filtered_colors, inlier_indices, n_clusters).
    """
    if len(points) < min_samples:
        return points, colors, np.arange(len(points)), 0

    original_points = points
    original_colors = colors
    mapping = None

    # Downsample if too many points
    if len(points) > max_points_for_dbscan:
        # Compute voxel size to achieve target point count
        bbox_extent = points.max(axis=0) - points.min(axis=0)
        volume = np.prod(bbox_extent)
        target_density = max_points_for_dbscan / volume
        voxel_size = (1 / target_density) ** (1 / 3) * 0.8  # Slightly smaller for safety

        if verbose:
            print(f"  Downsampling {len(points):,} points to ~{max_points_for_dbscan:,} for DBSCAN...")

        points, colors, mapping = voxel_downsample(points, colors, voxel_size)

        if verbose:
            print(f"  Downsampled to {len(points):,} points (voxel size: {voxel_size:.4f})")

    # Run DBSCAN with KD-tree algorithm (more memory efficient)
    if verbose:
        print(f"  Running DBSCAN (eps={eps:.6f}, min_samples={min_samples})...")

    clustering = DBSCAN(eps=eps, min_samples=min_samples, algorithm='kd_tree', n_jobs=-1)
    labels = clustering.fit_predict(points)

    # Find unique labels (excluding noise label -1)
    unique_labels = np.unique(labels)
    cluster_labels = unique_labels[unique_labels >= 0]
    n_clusters = len(cluster_labels)

    if n_clusters == 0:
        # No clusters found, return original
        return original_points, original_colors, np.arange(len(original_points)), 0

    # Find largest cluster
    cluster_sizes = [(label, np.sum(labels == label)) for label in cluster_labels]
    largest_label = max(cluster_sizes, key=lambda x: x[1])[0]

    # If we downsampled, need to map back to original points
    if mapping is not None:
        # Build KD-tree of downsampled points
        tree = cKDTree(points)

        # For each original point, find its nearest downsampled point
        _, nearest_indices = tree.query(original_points, k=1)

        # Assign labels to original points based on their nearest downsampled point
        original_labels = labels[nearest_indices]

        inlier_mask = original_labels == largest_label
        inlier_indices = np.where(inlier_mask)[0]

        filtered_points = original_points[inlier_mask]
        filtered_colors = original_colors[inlier_mask] if original_colors is not None else None
    else:
        inlier_mask = labels == largest_label
        inlier_indices = np.where(inlier_mask)[0]

        filtered_points = points[inlier_mask]
        filtered_colors = colors[inlier_mask] if colors is not None else None

    return filtered_points, filtered_colors, inlier_indices, n_clusters


def denoise_point_cloud(
    points: np.ndarray,
    colors: np.ndarray | None = None,
    method: str = "dbscan+sor",
    # DBSCAN params
    dbscan_eps: float | None = None,
    dbscan_min_points: int = 10,
    # SOR params
    sor_neighbors: int = 20,
    sor_std_ratio: float = 2.0,
    # Performance params
    max_points_for_dbscan: int = 500000,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray | None, dict]:
    """Remove noise from a point cloud using DBSCAN and/or SOR.

    Args:
        points: Point coordinates (N, 3).
        colors: Point colors (N, 3), or None.
        method: Denoising method - "dbscan", "sor", or "dbscan+sor".
        dbscan_eps: DBSCAN epsilon (auto-tuned from object scale if None).
        dbscan_min_points: Minimum points for DBSCAN core point.
        sor_neighbors: Number of neighbors for SOR.
        sor_std_ratio: Standard deviation ratio threshold for SOR.
        max_points_for_dbscan: Maximum points before DBSCAN downsampling.
        verbose: Print statistics.

    Returns:
        Tuple of (denoised_points, denoised_colors, stats_dict).
    """
    if len(points) == 0:
        return points, colors, {"original": 0, "final": 0, "removed": 0, "removal_rate": 0.0}

    original_count = len(points)
    stats = {"original": original_count}

    current_points = points.copy()
    current_colors = colors.copy() if colors is not None else None

    # Auto-tune DBSCAN eps based on object scale
    if dbscan_eps is None and method in ("dbscan", "dbscan+sor"):
        bbox_min = current_points.min(axis=0)
        bbox_max = current_points.max(axis=0)
        bbox_extent = bbox_max - bbox_min
        object_scale = np.median(bbox_extent)
        dbscan_eps = object_scale * 0.02  # 2% of object scale
        if verbose:
            print(f"Auto-tuned DBSCAN eps: {dbscan_eps:.6f} (object scale: {object_scale:.3f}m)")

    # Step 1: DBSCAN clustering
    if method in ("dbscan", "dbscan+sor"):
        current_points, current_colors, _, n_clusters = dbscan_largest_cluster(
            current_points,
            current_colors,
            eps=dbscan_eps,
            min_samples=dbscan_min_points,
            max_points_for_dbscan=max_points_for_dbscan,
            verbose=verbose,
        )
        stats["after_dbscan"] = len(current_points)

        if verbose:
            removed = original_count - len(current_points)
            print(f"DBSCAN: {n_clusters} clusters found, kept largest ({len(current_points):,} points)")
            print(f"  Removed: {removed:,} points ({100*removed/original_count:.1f}%)")

    # Step 2: Statistical Outlier Removal
    if method in ("sor", "dbscan+sor"):
        before_sor = len(current_points)

        if verbose:
            print(f"Running SOR (k={sor_neighbors}, std_ratio={sor_std_ratio})...")

        current_points, current_colors, _ = statistical_outlier_removal(
            current_points,
            current_colors,
            nb_neighbors=sor_neighbors,
            std_ratio=sor_std_ratio,
        )
        stats["after_sor"] = len(current_points)

        if verbose:
            removed = before_sor - len(current_points)
            removal_pct = 100 * removed / before_sor if before_sor > 0 else 0
            print(f"SOR: {len(current_points):,} points remain")
            print(f"  Removed: {removed:,} points ({removal_pct:.1f}%)")

    # Final stats
    final_count = len(current_points)
    stats["final"] = final_count
    stats["removed"] = original_count - final_count
    stats["removal_rate"] = stats["removed"] / original_count if original_count > 0 else 0.0

    if verbose:
        print(f"\nSummary:")
        print(f"  Original:  {original_count:>10,} points")
        print(f"  Final:     {final_count:>10,} points")
        print(f"  Removed:   {stats['removed']:>10,} points ({100*stats['removal_rate']:.1f}%)")

    return current_points, current_colors, stats


def load_ply(path: str | Path) -> tuple[np.ndarray, np.ndarray | None]:
    """Load point cloud from PLY file.

    Returns:
        Tuple of (points, colors) where colors may be None.
    """
    ply_data = PlyData.read(str(path))
    vertex = ply_data["vertex"]

    # Extract XYZ coordinates
    points = np.vstack([vertex["x"], vertex["y"], vertex["z"]]).T.astype(np.float32)

    # Extract RGB colors if present
    colors = None
    if "red" in vertex.data.dtype.names:
        colors = np.vstack([vertex["red"], vertex["green"], vertex["blue"]]).T.astype(np.uint8)

    return points, colors


def save_ply(path: str | Path, points: np.ndarray, colors: np.ndarray | None = None) -> None:
    """Save point cloud to PLY file."""
    n_points = len(points)

    if colors is not None:
        # Ensure colors are uint8
        if colors.dtype != np.uint8:
            if colors.max() <= 1.0:
                colors = (colors * 255).astype(np.uint8)
            else:
                colors = colors.astype(np.uint8)

        dtype = [
            ("x", "f4"), ("y", "f4"), ("z", "f4"),
            ("red", "u1"), ("green", "u1"), ("blue", "u1"),
        ]
        vertex_data = np.empty(n_points, dtype=dtype)
        vertex_data["x"] = points[:, 0]
        vertex_data["y"] = points[:, 1]
        vertex_data["z"] = points[:, 2]
        vertex_data["red"] = colors[:, 0]
        vertex_data["green"] = colors[:, 1]
        vertex_data["blue"] = colors[:, 2]
    else:
        dtype = [("x", "f4"), ("y", "f4"), ("z", "f4")]
        vertex_data = np.empty(n_points, dtype=dtype)
        vertex_data["x"] = points[:, 0]
        vertex_data["y"] = points[:, 1]
        vertex_data["z"] = points[:, 2]

    vertex_element = PlyElement.describe(vertex_data, "vertex")
    PlyData([vertex_element], text=False).write(str(path))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Denoise point cloud PLY file using DBSCAN and/or Statistical Outlier Removal."
    )
    parser.add_argument("input_ply", help="Path to input PLY file")
    parser.add_argument(
        "-o", "--output",
        help="Output PLY path (default: input_denoised.ply)",
    )
    parser.add_argument(
        "--method",
        choices=["dbscan", "sor", "dbscan+sor"],
        default="dbscan+sor",
        help="Denoising method (default: dbscan+sor)",
    )
    parser.add_argument(
        "--dbscan-eps",
        type=float,
        default=None,
        help="DBSCAN epsilon distance (auto-tuned if not specified)",
    )
    parser.add_argument(
        "--dbscan-min-points",
        type=int,
        default=10,
        help="DBSCAN minimum points for core point (default: 10)",
    )
    parser.add_argument(
        "--sor-neighbors",
        type=int,
        default=20,
        help="SOR number of neighbors (default: 20)",
    )
    parser.add_argument(
        "--sor-std-ratio",
        type=float,
        default=2.0,
        help="SOR standard deviation ratio (default: 2.0)",
    )
    parser.add_argument(
        "--max-dbscan-points",
        type=int,
        default=500000,
        help="Maximum points for DBSCAN before downsampling (default: 500000)",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress verbose output",
    )

    args = parser.parse_args()

    # Determine output path
    input_path = Path(args.input_ply)
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.with_stem(input_path.stem + "_denoised")

    # Load input
    print(f"Loading: {input_path}")
    points, colors = load_ply(input_path)
    print(f"  {len(points):,} points loaded")

    # Denoise
    print(f"\nDenoising with method: {args.method}")
    denoised_points, denoised_colors, stats = denoise_point_cloud(
        points,
        colors,
        method=args.method,
        dbscan_eps=args.dbscan_eps,
        dbscan_min_points=args.dbscan_min_points,
        sor_neighbors=args.sor_neighbors,
        sor_std_ratio=args.sor_std_ratio,
        max_points_for_dbscan=args.max_dbscan_points,
        verbose=not args.quiet,
    )

    # Save output
    save_ply(output_path, denoised_points, denoised_colors)
    print(f"\nSaved: {output_path}")

    # Bounding box comparison
    if len(denoised_points) > 0 and len(points) > 0:
        orig_bbox = points.max(axis=0) - points.min(axis=0)
        new_bbox = denoised_points.max(axis=0) - denoised_points.min(axis=0)
        print(f"\nBounding box comparison:")
        print(f"  Original: {orig_bbox[0]:.3f} x {orig_bbox[1]:.3f} x {orig_bbox[2]:.3f} m")
        print(f"  Denoised: {new_bbox[0]:.3f} x {new_bbox[1]:.3f} x {new_bbox[2]:.3f} m")
        size_change = np.linalg.norm(new_bbox - orig_bbox) / np.linalg.norm(orig_bbox) * 100
        print(f"  Size change: {size_change:.1f}%")


if __name__ == "__main__":
    main()
