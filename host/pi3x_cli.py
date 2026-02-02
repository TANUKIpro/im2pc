#!/usr/bin/env python3
"""
Pi3X CLI — Standalone 3D point cloud generation from video.

Extracts frames from a video, runs Pi3X inference (no SAM2),
filters points by confidence threshold, and saves a PLY point cloud
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

import cv2
import numpy as np
import torch
from tqdm import tqdm

# Add repos to path
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "repos" / "pi3"))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


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
    frame_interval: int,
    max_frames: int,
) -> list[np.ndarray]:
    """Extract frames from video at the given interval.

    Returns a list of RGB float32 frames normalised to [0, 1].
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video: {total_frames} frames, {fps:.1f} fps")

    frames: list[np.ndarray] = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            frames.append(frame_rgb)
            if len(frames) >= max_frames:
                break
        frame_idx += 1

    cap.release()
    print(f"Extracted {len(frames)} frames (interval={frame_interval}, max={max_frames})")
    return frames


def prepare_tensor(
    frames: list[np.ndarray],
    device: torch.device,
) -> torch.Tensor:
    """Resize frames to 224x224 and stack into a (1, N, 3, 224, 224) tensor."""
    resized = [cv2.resize(f, (224, 224)) for f in frames]
    imgs = np.stack(resized, axis=0)          # (N, 224, 224, 3)
    imgs = np.transpose(imgs, (0, 3, 1, 2))  # (N, 3, 224, 224)
    tensor = torch.from_numpy(imgs).float().to(device)
    return tensor.unsqueeze(0)                # (1, N, 3, 224, 224)


def save_ply(points: np.ndarray, colors: np.ndarray, path: str) -> None:
    from plyfile import PlyData, PlyElement

    vertices = np.zeros(len(points), dtype=[
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
    ])
    vertices['x'] = points[:, 0]
    vertices['y'] = points[:, 1]
    vertices['z'] = points[:, 2]
    vertices['red'] = (colors[:, 0] * 255).astype(np.uint8)
    vertices['green'] = (colors[:, 1] * 255).astype(np.uint8)
    vertices['blue'] = (colors[:, 2] * 255).astype(np.uint8)

    el = PlyElement.describe(vertices, 'vertex')
    PlyData([el], text=True).write(path)
    print(f"Saved PLY to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a 3D point cloud from video using Pi3X (no SAM2)."
    )
    parser.add_argument("video_path", help="Path to input video")
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

    args = parser.parse_args()

    # ---- Validate input ----
    video_path = Path(args.video_path)
    if not video_path.exists():
        print(f"Error: Video not found: {video_path}", file=sys.stderr)
        sys.exit(1)

    # ---- Extract frames ----
    frames = extract_frames(str(video_path), args.frame_interval, args.max_frames)
    if len(frames) < 2:
        print("Error: Need at least 2 frames for reconstruction.", file=sys.stderr)
        sys.exit(1)

    # ---- Device & model ----
    device = get_device()

    print("Loading Pi3X model...")
    from pi3.models.pi3x import Pi3X
    model = Pi3X.from_pretrained("yyfz233/Pi3X").to(device).eval()
    print("Pi3X loaded")

    # ---- Prepare tensor ----
    imgs_tensor = prepare_tensor(frames, device)
    print(f"Running Pi3X inference on {imgs_tensor.shape[1]} frames...")

    # ---- Inference ----
    with torch.no_grad():
        results = model(imgs_tensor)

    points = results['points'][0].cpu().numpy()        # (N, 224, 224, 3)
    conf_logits = results['conf'][0].cpu().numpy()     # (N, 224, 224, 1)
    poses = results['camera_poses'][0].cpu().numpy()   # (N, 4, 4)

    # ---- Confidence filtering ----
    confidence = sigmoid(conf_logits).squeeze(-1)      # (N, 224, 224)
    conf_mask = confidence > args.confidence_threshold

    # Colors from the resized input images
    imgs_np = imgs_tensor[0].cpu().numpy()                      # (N, 3, 224, 224)
    colors = np.transpose(imgs_np, (0, 2, 3, 1))               # (N, 224, 224, 3)

    valid_points = points[conf_mask]
    valid_colors = colors[conf_mask]

    print(f"Extracted {len(valid_points)} points (threshold={args.confidence_threshold})")

    # ---- Save outputs ----
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ply_path = output_dir / "object.ply"
    save_ply(valid_points, valid_colors, str(ply_path))

    poses_path = output_dir / "camera_poses.json"
    poses_list = [pose.tolist() for pose in poses]
    frame_indices = list(range(0, len(frames) * args.frame_interval, args.frame_interval))
    with open(poses_path, 'w') as f:
        json.dump({
            "poses": poses_list,
            "frame_indices": frame_indices[:len(poses)],
        }, f, indent=2)
    print(f"Saved camera poses to {poses_path}")

    print("Done.")


if __name__ == "__main__":
    main()
