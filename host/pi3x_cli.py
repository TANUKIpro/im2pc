#!/usr/bin/env python3
"""
Pi3X CLI — Standalone 3D point cloud generation from video.

Extracts frames from a video, runs Pi3X inference (no SAM2),
filters points by confidence + depth-edge, and saves a PLY point cloud
plus camera poses as JSON.

Usage:
    python host/pi3x_cli.py <video_path> \
        --confidence-threshold 0.1 \
        --frame-interval 10 \
        --output-dir data/output \
        --max-frames 50
"""

import argparse
import json
import sys
from pathlib import Path

import torch

# Add repos to path
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "repos" / "pi3"))

from pi3.utils.basic import load_images_as_tensor, write_ply
from pi3.utils.geometry import depth_edge


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a 3D point cloud from video using Pi3X (no SAM2)."
    )
    parser.add_argument("video_path", help="Path to input video (.mp4) or image directory")
    parser.add_argument(
        "--confidence-threshold", type=float, default=0.1,
        help="Confidence threshold for point filtering (default: 0.1)",
    )
    parser.add_argument(
        "--frame-interval", type=int, default=10,
        help="Extract every N-th frame (default: 10)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/output",
        help="Output directory (default: data/output)",
    )
    parser.add_argument(
        "--max-frames", type=int, default=50,
        help="Maximum number of frames to feed Pi3X (default: 50)",
    )
    parser.add_argument(
        "--pixel-limit", type=int, default=255000,
        help="Max total pixels per frame for resize (default: 255000)",
    )
    parser.add_argument(
        "--edge-rtol", type=float, default=0.03,
        help="Relative tolerance for depth edge filtering (default: 0.03)",
    )

    args = parser.parse_args()

    # ---- Validate input ----
    video_path = Path(args.video_path)
    if not video_path.exists():
        print(f"Error: Input not found: {video_path}", file=sys.stderr)
        sys.exit(1)

    # ---- Load frames using official utility ----
    # load_images_as_tensor handles aspect-ratio-preserving resize to 14-multiple dims
    imgs = load_images_as_tensor(
        str(video_path),
        interval=args.frame_interval,
        PIXEL_LIMIT=args.pixel_limit,
    )  # (N, 3, H, W)

    if imgs.numel() == 0 or imgs.shape[0] < 2:
        print("Error: Need at least 2 frames for reconstruction.", file=sys.stderr)
        sys.exit(1)

    # Apply max_frames limit
    if imgs.shape[0] > args.max_frames:
        print(f"Limiting to {args.max_frames} frames (from {imgs.shape[0]})")
        imgs = imgs[:args.max_frames]

    print(f"Input tensor: {imgs.shape[0]} frames, {imgs.shape[2]}x{imgs.shape[3]} pixels")

    # ---- Device & model ----
    device = get_device()
    imgs = imgs.to(device)

    print("Loading Pi3X model...")
    from pi3.models.pi3x import Pi3X
    model = Pi3X.from_pretrained("yyfz233/Pi3X").to(device).eval()
    print("Pi3X loaded")

    # ---- Inference ----
    print(f"Running Pi3X inference on {imgs.shape[0]} frames...")
    with torch.no_grad():
        if device.type == "cuda":
            capability = torch.cuda.get_device_capability()[0]
            dtype = torch.bfloat16 if capability >= 8 else torch.float16
            with torch.amp.autocast("cuda", dtype=dtype):
                results = model(imgs[None])  # Add batch dimension
        else:
            results = model(imgs[None])

    # ---- Masking: confidence + depth edge ----
    conf_mask = torch.sigmoid(results['conf'][..., 0]) > args.confidence_threshold
    non_edge = ~depth_edge(results['local_points'][..., 2], rtol=args.edge_rtol)
    masks = torch.logical_and(conf_mask, non_edge)[0]  # (N, H, W)

    num_points = masks.sum().item()
    print(f"Extracted {num_points} points (conf>{args.confidence_threshold}, edge_rtol={args.edge_rtol})")

    # ---- Save PLY ----
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ply_path = output_dir / "object.ply"
    points = results['points'][0][masks].cpu()
    colors = imgs.permute(0, 2, 3, 1)[masks]  # (N, H, W, 3) -> masked
    write_ply(points, colors, str(ply_path))
    print(f"Saved PLY to {ply_path}")

    # ---- Save camera poses ----
    poses = results['camera_poses'][0].cpu().numpy()  # (N, 4, 4)
    poses_path = output_dir / "camera_poses.json"
    poses_list = [pose.tolist() for pose in poses]
    num_frames = imgs.shape[0]
    frame_indices = list(range(0, num_frames * args.frame_interval, args.frame_interval))
    with open(poses_path, 'w') as f:
        json.dump({
            "poses": poses_list,
            "frame_indices": frame_indices[:len(poses)],
        }, f, indent=2)
    print(f"Saved camera poses to {poses_path}")

    print("Done.")


if __name__ == "__main__":
    main()
