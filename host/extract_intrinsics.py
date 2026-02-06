#!/usr/bin/env python3
"""
Camera intrinsics estimation from colored point cloud + multi-view images.

Pi3X outputs cam-to-world poses but not explicit intrinsics (K matrix).
This script estimates fx, fy, cx, cy by projecting the RGB point cloud
onto camera frames and optimizing for color consistency.

Strategy:
  1. Grid search over FOV (40-80°) to find coarse estimate
  2. Refine with scipy.optimize.minimize (Nelder-Mead)
  3. Validate by reprojecting points onto frames and saving visualizations

Usage:
    python host/extract_intrinsics.py \
        --point-cloud data/output_sam2/object.ply \
        --poses data/output_sam2/camera_poses.json \
        --frames-dir data/output_sam2/frames \
        --masks-dir data/output_sam2/masks \
        --output intrinsics.json
"""

import argparse
import json
import sys
import functools
from pathlib import Path

# Force unbuffered output
print = functools.partial(print, flush=True)

import cv2
import numpy as np
from plyfile import PlyData


def load_point_cloud(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load point cloud, return (points (N,3), colors (N,3) in [0,1])."""
    ply = PlyData.read(path)
    v = ply["vertex"]
    points = np.column_stack([v["x"], v["y"], v["z"]]).astype(np.float64)
    colors = np.column_stack([v["red"], v["green"], v["blue"]]).astype(np.float64) / 255.0
    return points, colors


def load_poses(path: str) -> np.ndarray:
    """Load camera poses (cam-to-world 4x4 matrices). Returns (N, 4, 4)."""
    with open(path) as f:
        data = json.load(f)
    poses = np.array(data["poses"], dtype=np.float64)
    return poses


def load_frame(frames_dir: str, idx: int) -> np.ndarray:
    """Load frame as RGB float [0,1]. Returns (H, W, 3)."""
    path = Path(frames_dir) / f"{idx:05d}.jpg"
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Frame not found: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float64) / 255.0


def load_mask(masks_dir: str, idx: int) -> np.ndarray:
    """Load binary mask. Returns (H, W) bool."""
    path = Path(masks_dir) / f"{idx:05d}.png"
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Mask not found: {path}")
    return mask > 127


def project_points(
    points_world: np.ndarray,
    c2w: np.ndarray,
    K: np.ndarray,
    img_w: int,
    img_h: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Project 3D world points to 2D pixel coords given cam-to-world and intrinsics.

    Returns:
        uv: (M, 2) pixel coordinates (u, v)
        valid_mask: (N,) bool array
        depths: (N,) depth values in camera frame
    """
    # World to camera: invert cam-to-world
    w2c = np.linalg.inv(c2w)
    R = w2c[:3, :3]
    t = w2c[:3, 3]

    # Transform to camera coordinates
    pts_cam = (R @ points_world.T).T + t  # (N, 3)

    # Depth check: points must be in front of camera
    depths = pts_cam[:, 2]
    valid = depths > 0.01

    # Project to pixel coordinates
    pts_cam_valid = pts_cam[valid]
    x = pts_cam_valid[:, 0] / pts_cam_valid[:, 2]
    y = pts_cam_valid[:, 1] / pts_cam_valid[:, 2]

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    u = fx * x + cx
    v = fy * y + cy

    # Bounds check
    in_bounds = (u >= 0) & (u < img_w - 1) & (v >= 0) & (v < img_h - 1)

    uv = np.column_stack([u[in_bounds], v[in_bounds]])
    # Update valid mask
    valid_indices = np.where(valid)[0]
    full_valid = np.zeros(len(points_world), dtype=bool)
    full_valid[valid_indices[in_bounds]] = True

    return uv, full_valid, depths


def compute_color_score(
    K: np.ndarray,
    points: np.ndarray,
    colors: np.ndarray,
    poses: np.ndarray,
    frames_dir: str,
    masks_dir: str,
    frame_indices: list[int],
    img_w: int,
    img_h: int,
) -> float:
    """
    Compute average color match score (negative MSE) for given K across multiple views.
    Higher is better (less negative).
    """
    total_error = 0.0
    total_count = 0

    for idx in frame_indices:
        c2w = poses[idx]
        frame = load_frame(frames_dir, idx)
        mask = load_mask(masks_dir, idx)

        uv, valid, _ = project_points(points, c2w, K, img_w, img_h)
        if uv.shape[0] == 0:
            continue

        # Sample frame colors at projected locations (bilinear)
        u_int = uv[:, 0].astype(np.int32)
        v_int = uv[:, 1].astype(np.int32)

        # Check mask validity at projected pixels
        mask_valid = mask[v_int, u_int]
        if mask_valid.sum() == 0:
            continue

        # Get projected colors from frame
        frame_colors = frame[v_int[mask_valid], u_int[mask_valid]]
        # Get point cloud colors for valid projected points
        pc_colors = colors[valid][mask_valid]

        # MSE between point cloud colors and frame pixel colors
        diff = frame_colors - pc_colors
        mse = np.mean(diff ** 2)
        count = mask_valid.sum()

        total_error += mse * count
        total_count += count

    if total_count == 0:
        return -1.0  # worst case

    return -(total_error / total_count)


def make_K(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """Construct 3x3 intrinsics matrix."""
    return np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1],
    ], dtype=np.float64)


def estimate_intrinsics(
    points: np.ndarray,
    colors: np.ndarray,
    poses: np.ndarray,
    frames_dir: str,
    masks_dir: str,
    img_w: int = 1080,
    img_h: int = 1920,
    num_eval_frames: int = 10,
    subsample_points: int = 50000,
) -> dict:
    """
    Estimate camera intrinsics by grid search + optimization.

    Args:
        points: (N, 3) world coordinates
        colors: (N, 3) RGB [0,1]
        poses: (M, 4, 4) cam-to-world matrices
        frames_dir: path to frame images
        masks_dir: path to mask images
        img_w: frame width in pixels
        img_h: frame height in pixels
        num_eval_frames: number of frames to use for evaluation
        subsample_points: number of points to subsample for speed

    Returns:
        dict with fx, fy, cx, cy and metadata
    """
    from scipy.optimize import minimize

    # Subsample points for speed
    if len(points) > subsample_points:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(points), subsample_points, replace=False)
        pts_sub = points[idx]
        col_sub = colors[idx]
    else:
        pts_sub = points
        col_sub = colors

    # Select evaluation frames (evenly spaced)
    n_frames = len(poses)
    frame_indices = np.linspace(0, n_frames - 1, num_eval_frames, dtype=int).tolist()
    print(f"Using {len(frame_indices)} frames for evaluation: {frame_indices}")
    print(f"Using {len(pts_sub)} points (subsampled from {len(points)})")

    # Phase 1: Grid search over FOV (assuming square pixels, principal point at center)
    cx_init = img_w / 2.0
    cy_init = img_h / 2.0

    best_score = -np.inf
    best_fov = None

    print("\n--- Phase 1: Grid search over FOV ---")
    for fov_deg in range(35, 85, 1):
        fov_rad = np.radians(fov_deg)
        # FOV is horizontal FOV
        fx = img_w / (2.0 * np.tan(fov_rad / 2.0))
        fy = fx  # square pixels assumption
        K = make_K(fx, fy, cx_init, cy_init)

        score = compute_color_score(
            K, pts_sub, col_sub, poses, frames_dir, masks_dir,
            frame_indices, img_w, img_h,
        )
        if score > best_score:
            best_score = score
            best_fov = fov_deg
            print(f"  FOV={fov_deg}° → score={score:.6f} (best)")

    print(f"\nBest coarse FOV: {best_fov}° (score={best_score:.6f})")

    # Phase 2: Refine with optimizer
    fov_rad = np.radians(best_fov)
    fx_init = img_w / (2.0 * np.tan(fov_rad / 2.0))

    print("\n--- Phase 2: Refinement with Nelder-Mead ---")

    def objective(params):
        fx, fy, cx, cy = params
        if fx < 100 or fy < 100 or fx > 5000 or fy > 5000:
            return 1.0
        if cx < 0 or cx > img_w or cy < 0 or cy > img_h:
            return 1.0
        K = make_K(fx, fy, cx, cy)
        score = compute_color_score(
            K, pts_sub, col_sub, poses, frames_dir, masks_dir,
            frame_indices, img_w, img_h,
        )
        return -score  # minimize negative score

    x0 = [fx_init, fx_init, cx_init, cy_init]
    result = minimize(
        objective, x0,
        method="Nelder-Mead",
        options={"maxiter": 500, "xatol": 0.5, "fatol": 1e-7, "adaptive": True},
    )

    fx_opt, fy_opt, cx_opt, cy_opt = result.x
    final_score = -result.fun
    print(f"Optimized: fx={fx_opt:.2f}, fy={fy_opt:.2f}, cx={cx_opt:.2f}, cy={cy_opt:.2f}")
    print(f"Final score: {final_score:.6f}")

    # Compute equivalent FOV
    fov_h = 2.0 * np.degrees(np.arctan(img_w / (2.0 * fx_opt)))
    fov_v = 2.0 * np.degrees(np.arctan(img_h / (2.0 * fy_opt)))

    return {
        "fx": float(fx_opt),
        "fy": float(fy_opt),
        "cx": float(cx_opt),
        "cy": float(cy_opt),
        "image_width": img_w,
        "image_height": img_h,
        "fov_horizontal_deg": float(fov_h),
        "fov_vertical_deg": float(fov_v),
        "optimization_score": float(final_score),
        "K": make_K(fx_opt, fy_opt, cx_opt, cy_opt).tolist(),
    }


def visualize_reprojection(
    intrinsics: dict,
    points: np.ndarray,
    colors: np.ndarray,
    poses: np.ndarray,
    frames_dir: str,
    masks_dir: str,
    output_dir: str,
    frame_indices: list[int] | None = None,
    subsample: int = 100000,
):
    """Save reprojection visualizations for verification."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    K = np.array(intrinsics["K"], dtype=np.float64)
    img_w = intrinsics["image_width"]
    img_h = intrinsics["image_height"]

    # Subsample for visualization
    if len(points) > subsample:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(points), subsample, replace=False)
        pts = points[idx]
        cols = colors[idx]
    else:
        pts = points
        cols = colors

    if frame_indices is None:
        frame_indices = np.linspace(0, len(poses) - 1, 5, dtype=int).tolist()

    for fidx in frame_indices:
        frame = load_frame(frames_dir, fidx)
        frame_bgr = (frame[:, :, ::-1] * 255).astype(np.uint8).copy()

        uv, valid, _ = project_points(pts, poses[fidx], K, img_w, img_h)

        # Draw projected points
        for i in range(len(uv)):
            u, v = int(uv[i, 0]), int(uv[i, 1])
            pc_col = cols[valid][i]
            color = (int(pc_col[2] * 255), int(pc_col[1] * 255), int(pc_col[0] * 255))
            cv2.circle(frame_bgr, (u, v), 1, color, -1)

        out_path = out / f"reproj_{fidx:05d}.jpg"
        cv2.imwrite(str(out_path), frame_bgr)
        print(f"Saved reprojection: {out_path} ({len(uv)} points)")


def main():
    parser = argparse.ArgumentParser(description="Estimate camera intrinsics from point cloud + frames")
    parser.add_argument("--point-cloud", required=True, help="Path to colored point cloud (.ply)")
    parser.add_argument("--poses", required=True, help="Path to camera_poses.json")
    parser.add_argument("--frames-dir", required=True, help="Directory with frame images")
    parser.add_argument("--masks-dir", required=True, help="Directory with mask images")
    parser.add_argument("--output", default="intrinsics.json", help="Output JSON path")
    parser.add_argument("--img-width", type=int, default=1080, help="Frame width")
    parser.add_argument("--img-height", type=int, default=1920, help="Frame height")
    parser.add_argument("--num-eval-frames", type=int, default=10, help="Frames for evaluation")
    parser.add_argument("--subsample", type=int, default=50000, help="Points to subsample")
    parser.add_argument("--visualize", action="store_true", help="Save reprojection images")
    parser.add_argument("--vis-dir", default="data/output_sam2/reprojections", help="Visualization output dir")
    args = parser.parse_args()

    print("Loading point cloud...")
    points, colors = load_point_cloud(args.point_cloud)
    print(f"  {len(points)} points loaded")

    print("Loading camera poses...")
    poses = load_poses(args.poses)
    print(f"  {len(poses)} poses loaded")

    print("\nEstimating intrinsics...")
    intrinsics = estimate_intrinsics(
        points, colors, poses,
        args.frames_dir, args.masks_dir,
        img_w=args.img_width, img_h=args.img_height,
        num_eval_frames=args.num_eval_frames,
        subsample_points=args.subsample,
    )

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(intrinsics, f, indent=2)
    print(f"\nSaved intrinsics to {output_path}")
    print(f"  fx={intrinsics['fx']:.2f}, fy={intrinsics['fy']:.2f}")
    print(f"  cx={intrinsics['cx']:.2f}, cy={intrinsics['cy']:.2f}")
    print(f"  FOV: {intrinsics['fov_horizontal_deg']:.1f}° (H) x {intrinsics['fov_vertical_deg']:.1f}° (V)")

    if args.visualize:
        print("\nGenerating reprojection visualizations...")
        visualize_reprojection(
            intrinsics, points, colors, poses,
            args.frames_dir, args.masks_dir, args.vis_dir,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
