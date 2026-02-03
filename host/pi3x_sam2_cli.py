#!/usr/bin/env python3
"""
Pi3X + SAM2 CLI — Post-filtering pipeline for object-specific 3D reconstruction.

Runs Pi3X on full (unmasked) images for optimal pose estimation and depth quality,
then applies SAM2 masks as a post-filter to extract the target object's point cloud.

Pipeline:
    1. Extract frames from video -> JPEG directory (shared by SAM2 and Pi3X)
    2. SAM2: segment object in prompt frame -> propagate masks to all frames
    3. Pi3X: full-image inference -> per-pixel points + confidence + camera poses
    4. Post-filter: conf & depth-edge & SAM2 mask -> object PLY + camera poses

Usage (CLI):
    python host/pi3x_sam2_cli.py video.mp4 --point 512,384 --output-dir output

Usage (from Colab):
    from host.pi3x_sam2_cli import extract_frames, run_sam2_segmentation, ...
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

# Add repos to path
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "repos" / "pi3"))
sys.path.insert(0, str(PROJECT_ROOT / "repos" / "sam2"))


def _ensure_sam2_installed() -> None:
    """Ensure SAM2 and its Hydra dependency are pip-installed.

    SAM2 uses Hydra's instantiate() to construct models from YAML configs.
    This requires sam2 to be a proper installed package, not just on sys.path,
    and hydra-core + iopath to be available.
    If not installed, performs an editable install from the submodule.
    """
    import subprocess

    # Install hydra-core and iopath if missing (SAM2 deps not on Colab by default)
    missing_deps = []
    for pkg in ["hydra-core", "iopath"]:
        try:
            __import__(pkg.replace("-", "_").split("-")[0])
        except ImportError:
            missing_deps.append(pkg)

    # Check hydra specifically by its actual module name
    try:
        import hydra  # noqa: F401
    except ImportError:
        if "hydra-core" not in missing_deps:
            missing_deps.append("hydra-core")

    if missing_deps:
        print(f"Installing missing SAM2 dependencies: {missing_deps}")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install"] + missing_deps,
            stdout=subprocess.DEVNULL,
        )

    # Check if SAM2 itself is installed as a package
    sam2_repo = PROJECT_ROOT / "repos" / "sam2"
    try:
        from importlib.metadata import distribution
        distribution("SAM-2")
        return  # Already installed
    except Exception:
        pass

    print("SAM2 not installed. Running editable install (required for Hydra)...")
    env = {**__import__("os").environ, "SAM2_BUILD_CUDA": "0"}
    subprocess.check_call(
        [
            sys.executable, "-m", "pip", "install",
            "--no-deps", "--no-build-isolation", "-e", str(sam2_repo),
        ],
        stdout=subprocess.DEVNULL,
        env=env,
    )
    print("SAM2 installed successfully")

# SAM2 model configurations
SAM2_MODEL_CONFIGS = {
    "tiny": {
        "config": "configs/sam2.1/sam2.1_hiera_t.yaml",
        "checkpoint": "sam2.1_hiera_tiny.pt",
        "hf_model_id": "facebook/sam2.1-hiera-tiny",
    },
    "small": {
        "config": "configs/sam2.1/sam2.1_hiera_s.yaml",
        "checkpoint": "sam2.1_hiera_small.pt",
        "hf_model_id": "facebook/sam2.1-hiera-small",
    },
    "base": {
        "config": "configs/sam2.1/sam2.1_hiera_b+.yaml",
        "checkpoint": "sam2.1_hiera_base_plus.pt",
        "hf_model_id": "facebook/sam2.1-hiera-base-plus",
    },
    "large": {
        "config": "configs/sam2.1/sam2.1_hiera_l.yaml",
        "checkpoint": "sam2.1_hiera_large.pt",
        "hf_model_id": "facebook/sam2.1-hiera-large",
    },
}


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


def extract_frames(
    video_path: str,
    frame_interval: int = 10,
    max_frames: int = 50,
    output_dir: str = "output",
) -> Path:
    """Extract frames from a video as JPEGs.

    The extracted frame directory is shared by both SAM2 and Pi3X.
    Naming convention: 00000.jpg, 00001.jpg, ...

    Args:
        video_path: Path to input video (.mp4).
        frame_interval: Extract every N-th frame.
        max_frames: Maximum number of frames to extract.
        output_dir: Parent output directory.

    Returns:
        Path to the frames directory.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    frames_dir = Path(output_dir) / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video: {total_frames} frames, {fps:.1f} fps")

    frame_idx = 0
    saved_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            frame_path = frames_dir / f"{saved_idx:05d}.jpg"
            cv2.imwrite(str(frame_path), frame)
            saved_idx += 1
            if saved_idx >= max_frames:
                break
        frame_idx += 1

    cap.release()
    print(f"Extracted {saved_idx} frames to {frames_dir}")
    return frames_dir


def run_sam2_segmentation(
    frames_dir: str,
    points: list[list[float]],
    labels: list[int],
    output_dir: str = "output",
    prompt_frame: int = 0,
    model_type: str = "large",
    box: Optional[list[float]] = None,
) -> Path:
    """Run SAM2 segmentation and mask propagation.

    Loads SAM2, prompts on a single frame, propagates masks to all frames,
    saves mask PNGs, then releases the SAM2 model to free memory.

    Args:
        frames_dir: Directory of JPEG frames (from extract_frames).
        points: Click coordinates as [[x1,y1], [x2,y2], ...] in pixel space.
        labels: Per-point labels (1=positive, 0=negative).
        output_dir: Parent output directory.
        prompt_frame: Frame index to prompt on (default: 0).
        model_type: SAM2 model variant ("tiny"/"small"/"base"/"large").
        box: Optional bounding box [x1, y1, x2, y2] in pixel space.

    Returns:
        Path to the masks directory containing PNG masks.
    """
    _ensure_sam2_installed()
    from sam2.build_sam import build_sam2_video_predictor, build_sam2_video_predictor_hf

    if model_type not in SAM2_MODEL_CONFIGS:
        raise ValueError(
            f"Unknown SAM2 model type '{model_type}'. "
            f"Choose from: {list(SAM2_MODEL_CONFIGS.keys())}"
        )

    config = SAM2_MODEL_CONFIGS[model_type]
    device = get_device()

    # Try local checkpoint, fallback to HuggingFace
    sam2_dir = PROJECT_ROOT / "repos" / "sam2"
    ckpt_path = sam2_dir / "checkpoints" / config["checkpoint"]

    print(f"Loading SAM2 ({model_type})...")
    if ckpt_path.exists():
        predictor = build_sam2_video_predictor(
            config_file=config["config"],
            ckpt_path=str(ckpt_path),
            device=str(device),
        )
        print(f"SAM2 loaded from {ckpt_path}")
    else:
        print("Local checkpoint not found, loading from HuggingFace...")
        predictor = build_sam2_video_predictor_hf(
            model_id=config["hf_model_id"],
            device=str(device),
        )
        print("SAM2 loaded from HuggingFace")

    # Initialize video state
    with torch.inference_mode():
        inference_state = predictor.init_state(
            video_path=str(frames_dir),
            offload_video_to_cpu=True,
        )

        # Add prompt (points and/or box)
        points_np = np.array(points, dtype=np.float32) if points else None
        labels_np = np.array(labels, dtype=np.int32) if labels else None
        box_np = np.array(box, dtype=np.float32) if box is not None else None

        _, _, masks_out = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=prompt_frame,
            obj_id=1,
            points=points_np,
            labels=labels_np,
            box=box_np,
            clear_old_points=True,
            normalize_coords=True,  # SAM2 normalizes pixel coords internally
        )

        prompt_mask = (masks_out[0] > 0).cpu().numpy().squeeze()
        prompt_pixels = int(prompt_mask.sum())
        print(f"Prompt mask: {prompt_pixels} pixels on frame {prompt_frame}")

        # Propagate masks through all frames
        mask_dir = Path(output_dir) / "masks"
        mask_dir.mkdir(parents=True, exist_ok=True)

        print("Propagating masks...")
        num_frames = inference_state["num_frames"]
        for frame_idx, object_ids, masks_tensor in predictor.propagate_in_video(
            inference_state
        ):
            # masks_tensor: (num_objects, 1, H, W) logits
            mask = (masks_tensor[0] > 0).cpu().numpy().squeeze()  # (H, W) bool
            mask_path = mask_dir / f"{frame_idx:05d}.png"
            cv2.imwrite(str(mask_path), (mask * 255).astype(np.uint8))

    print(f"Saved {num_frames} masks to {mask_dir}")

    # Release SAM2 model to free memory
    del predictor, inference_state
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()
    print("SAM2 model released")

    return mask_dir


def run_pi3x_inference(
    frames_dir: str,
    pixel_limit: int = 255000,
    max_frames: int = 50,
) -> tuple[dict, torch.Tensor]:
    """Run Pi3X inference on full (unmasked) frames.

    Args:
        frames_dir: Directory of JPEG frames.
        pixel_limit: Max total pixels per frame for resize (default: 255000).
        max_frames: Maximum number of frames for Pi3X.

    Returns:
        Tuple of (results_dict, imgs_tensor).
        - results_dict: Pi3X output with 'points', 'conf', 'local_points', 'camera_poses'.
        - imgs_tensor: Input images tensor (N, 3, H, W) on device.
    """
    from pi3.utils.basic import load_images_as_tensor
    from pi3.models.pi3x import Pi3X

    # Load frames -- interval=1 because frames are already subsampled by extract_frames
    imgs = load_images_as_tensor(
        str(frames_dir),
        interval=1,
        PIXEL_LIMIT=pixel_limit,
    )  # (N, 3, H, W)

    if imgs.numel() == 0 or imgs.shape[0] < 2:
        raise RuntimeError("Need at least 2 frames for reconstruction.")

    if imgs.shape[0] > max_frames:
        print(f"Limiting to {max_frames} frames (from {imgs.shape[0]})")
        imgs = imgs[:max_frames]

    print(f"Pi3X input: {imgs.shape[0]} frames, {imgs.shape[2]}x{imgs.shape[3]} pixels")

    device = get_device()
    imgs = imgs.to(device)

    print("Loading Pi3X model...")
    model = Pi3X.from_pretrained("yyfz233/Pi3X").to(device).eval()
    print("Pi3X loaded")

    print(f"Running Pi3X inference on {imgs.shape[0]} frames...")
    with torch.no_grad():
        if device.type == "cuda":
            capability = torch.cuda.get_device_capability()[0]
            dtype = torch.bfloat16 if capability >= 8 else torch.float16
            with torch.amp.autocast("cuda", dtype=dtype):
                results = model(imgs[None])  # (1, N, 3, H, W)
        else:
            results = model(imgs[None])

    print("Pi3X inference complete")
    return results, imgs


def filter_and_save(
    results: dict,
    imgs: torch.Tensor,
    mask_dir: str,
    output_dir: str = "output",
    conf_threshold: float = 0.1,
    edge_rtol: float = 0.03,
) -> Path:
    """Apply triple filter (confidence + depth-edge + SAM2 mask) and save PLY.

    Args:
        results: Pi3X results dict.
        imgs: Input images tensor (N, 3, H, W).
        mask_dir: Directory of SAM2 mask PNGs.
        output_dir: Output directory for PLY and camera poses.
        conf_threshold: Confidence threshold for point filtering.
        edge_rtol: Relative tolerance for depth edge filtering.

    Returns:
        Path to the saved PLY file.
    """
    from pi3.utils.basic import write_ply
    from pi3.utils.geometry import depth_edge

    mask_dir = Path(mask_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    N, _, H_pi3x, W_pi3x = imgs.shape

    # --- Filter 1: Confidence ---
    conf_mask = torch.sigmoid(results["conf"][..., 0]) > conf_threshold  # (1, N, H, W)
    conf_count = conf_mask[0].sum().item()

    # --- Filter 2: Depth edge ---
    non_edge = ~depth_edge(results["local_points"][..., 2], rtol=edge_rtol)  # (1, N, H, W)
    conf_edge_mask = conf_mask & non_edge
    conf_edge_count = conf_edge_mask[0].sum().item()

    # --- Filter 3: SAM2 masks ---
    # Load SAM2 masks and resize to Pi3X resolution
    mask_files = sorted(mask_dir.glob("*.png"))
    if len(mask_files) != N:
        print(
            f"Warning: {len(mask_files)} masks found, but {N} frames in Pi3X. "
            f"Using min({len(mask_files)}, {N}) frames."
        )

    sam2_masks = []
    for i in range(N):
        if i < len(mask_files):
            mask_img = cv2.imread(str(mask_files[i]), cv2.IMREAD_GRAYSCALE)
            # Resize to Pi3X resolution with nearest-neighbor interpolation
            mask_resized = cv2.resize(
                mask_img, (W_pi3x, H_pi3x), interpolation=cv2.INTER_NEAREST
            )
            sam2_masks.append(mask_resized > 127)
        else:
            # No mask available -- exclude all points for this frame
            sam2_masks.append(np.zeros((H_pi3x, W_pi3x), dtype=bool))

    sam2_mask_tensor = torch.from_numpy(np.stack(sam2_masks)).to(imgs.device)  # (N, H, W)

    # Combine all three filters
    final_mask = conf_edge_mask[0] & sam2_mask_tensor  # (N, H, W)
    final_count = final_mask.sum().item()

    # --- Filtering stats ---
    total = N * H_pi3x * W_pi3x
    print("Filtering stats:")
    print(f"  Total pixels:     {total:>10,}")
    print(f"  After conf:       {conf_count:>10,} ({100*conf_count/total:.1f}%)")
    print(f"  After conf+edge:  {conf_edge_count:>10,} ({100*conf_edge_count/total:.1f}%)")
    print(f"  After +SAM2 mask: {final_count:>10,} ({100*final_count/total:.1f}%)")

    if final_count == 0:
        print("Warning: No points remain after filtering. Check mask quality.")
        ply_path = output_path / "object.ply"
        write_ply(torch.zeros(0, 3), torch.zeros(0, 3), str(ply_path))
        return ply_path

    # Per-frame point counts
    per_frame = final_mask.sum(dim=(1, 2))  # (N,)
    min_pts = per_frame.min().item()
    max_pts = per_frame.max().item()
    print(f"  Per-frame range:  {min_pts:,} - {max_pts:,}")
    for i in range(N):
        cnt = per_frame[i].item()
        if cnt < 100:
            print(f"  Warning: Frame {i} has only {cnt} points")

    # --- Extract points and colors ---
    points = results["points"][0][final_mask].cpu()  # (P, 3)
    colors = imgs.permute(0, 2, 3, 1)[final_mask]  # (P, 3)

    # --- Bounding box sanity check ---
    pts_np = points.numpy()
    bbox_min = pts_np.min(axis=0)
    bbox_max = pts_np.max(axis=0)
    bbox_size = bbox_max - bbox_min
    print(f"  Bounding box (m): {bbox_size[0]:.3f} x {bbox_size[1]:.3f} x {bbox_size[2]:.3f}")

    # --- Save PLY ---
    ply_path = output_path / "object.ply"
    write_ply(points, colors, str(ply_path))
    print(f"Saved PLY: {ply_path} ({final_count:,} points)")

    # --- Save camera poses ---
    poses = results["camera_poses"][0].cpu().numpy()  # (N, 4, 4)
    poses_path = output_path / "camera_poses.json"
    with open(poses_path, "w") as f:
        json.dump(
            {
                "poses": [pose.tolist() for pose in poses],
                "frame_indices": list(range(N)),
            },
            f,
            indent=2,
        )
    print(f"Saved camera poses: {poses_path}")

    return ply_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate object 3D point cloud from video using SAM2 + Pi3X (post-filtering)."
    )
    parser.add_argument("video_path", help="Path to input video (.mp4)")
    parser.add_argument(
        "--point",
        type=str,
        action="append",
        help="Click point as 'x,y' in pixel coords (repeatable, positive by default)",
    )
    parser.add_argument(
        "--neg-point",
        type=str,
        action="append",
        help="Negative click point as 'x,y' in pixel coords (repeatable)",
    )
    parser.add_argument(
        "--box",
        type=str,
        default=None,
        help="Bounding box as 'x1,y1,x2,y2' in pixel coords",
    )
    parser.add_argument(
        "--prompt-frame",
        type=int,
        default=0,
        help="Frame index to prompt SAM2 on (default: 0)",
    )
    parser.add_argument(
        "--sam2-model",
        type=str,
        default="large",
        choices=list(SAM2_MODEL_CONFIGS.keys()),
        help="SAM2 model variant (default: large)",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.1,
        help="Confidence threshold for point filtering (default: 0.1)",
    )
    parser.add_argument(
        "--frame-interval",
        type=int,
        default=10,
        help="Extract every N-th frame (default: 10)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Output directory (default: output)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=50,
        help="Maximum number of frames (default: 50)",
    )
    parser.add_argument(
        "--pixel-limit",
        type=int,
        default=255000,
        help="Max total pixels per frame for Pi3X resize (default: 255000)",
    )
    parser.add_argument(
        "--edge-rtol",
        type=float,
        default=0.03,
        help="Relative tolerance for depth edge filtering (default: 0.03)",
    )

    args = parser.parse_args()

    # Parse points
    points = []
    labels = []
    if args.point:
        for p in args.point:
            x, y = map(float, p.split(","))
            points.append([x, y])
            labels.append(1)
    if args.neg_point:
        for p in args.neg_point:
            x, y = map(float, p.split(","))
            points.append([x, y])
            labels.append(0)

    # Parse box
    box = None
    if args.box:
        box = list(map(float, args.box.split(",")))
        if len(box) != 4:
            print("Error: --box requires exactly 4 values: x1,y1,x2,y2", file=sys.stderr)
            sys.exit(1)

    if not points and box is None:
        print(
            "Error: At least one --point or --box is required for SAM2 prompting.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Step 1: Extract frames
    print("=" * 60)
    print("Step 1: Extracting frames")
    print("=" * 60)
    frames_dir = extract_frames(
        args.video_path,
        frame_interval=args.frame_interval,
        max_frames=args.max_frames,
        output_dir=args.output_dir,
    )

    # Step 2: SAM2 segmentation
    print()
    print("=" * 60)
    print("Step 2: SAM2 segmentation")
    print("=" * 60)
    mask_dir = run_sam2_segmentation(
        str(frames_dir),
        points=points,
        labels=labels,
        output_dir=args.output_dir,
        prompt_frame=args.prompt_frame,
        model_type=args.sam2_model,
        box=box,
    )

    # Step 3: Pi3X inference
    print()
    print("=" * 60)
    print("Step 3: Pi3X inference")
    print("=" * 60)
    results, imgs = run_pi3x_inference(
        str(frames_dir),
        pixel_limit=args.pixel_limit,
        max_frames=args.max_frames,
    )

    # Step 4: Filter and save
    print()
    print("=" * 60)
    print("Step 4: Post-filter and save")
    print("=" * 60)
    ply_path = filter_and_save(
        results,
        imgs,
        str(mask_dir),
        output_dir=args.output_dir,
        conf_threshold=args.confidence_threshold,
        edge_rtol=args.edge_rtol,
    )

    print()
    print("=" * 60)
    print("Done!")
    print(f"  PLY:    {ply_path}")
    print(f"  Poses:  {Path(args.output_dir) / 'camera_poses.json'}")
    print(f"  Masks:  {mask_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
