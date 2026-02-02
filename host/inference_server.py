#!/usr/bin/env python3
"""
Pi3X + SAM2 Inference Server
GPU inference API server running on macOS host with MPS support.

Endpoints:
- GET  /health              - Health check
- POST /sam2/init_video     - Load video and extract frames
- POST /sam2/add_prompt     - Add click prompt to generate mask
- POST /sam2/propagate      - Propagate mask through all frames
- POST /pi3x/reconstruct    - Run Pi3X reconstruction
"""

import os
import sys
import json
import tempfile
import shutil
from pathlib import Path
from typing import Optional
import base64

import numpy as np
import torch
import cv2
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from PIL import Image
from tqdm import tqdm

# Add repos to path
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "repos" / "sam2"))
sys.path.insert(0, str(PROJECT_ROOT / "repos" / "pi3"))

app = Flask(__name__)
CORS(app)

# SAM2 model variants
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

# Global state
class InferenceState:
    def __init__(self):
        self.device = None
        self.sam2_predictor = None
        self.sam2_model_type = None  # Track which model variant is loaded
        self.sam2_state = None
        self.pi3x_model = None
        self.video_frames_dir = None
        self.video_info = None
        self.masks = {}  # frame_idx -> mask array
        self.original_frames = []  # Store original frame paths

    def reset(self):
        self.sam2_state = None
        self.video_frames_dir = None
        self.video_info = None
        self.masks = {}
        self.original_frames = []


state = InferenceState()


def sigmoid(x):
    """Compute sigmoid function."""
    return 1.0 / (1.0 + np.exp(-x))


def get_device():
    """Determine the best available device."""
    if state.device is not None:
        return state.device

    if torch.backends.mps.is_available():
        state.device = torch.device("mps")
        print("Using MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        state.device = torch.device("cuda")
        print("Using CUDA GPU")
    else:
        state.device = torch.device("cpu")
        print("Using CPU")

    return state.device


def load_sam2_model(model_type: str = "large"):
    """Load SAM2 Video Predictor.

    Args:
        model_type: Model variant - "tiny", "small", "base", or "large".
                    If the requested model differs from the currently loaded one,
                    the old predictor is released and the new one is loaded.
    """
    if model_type not in SAM2_MODEL_CONFIGS:
        raise ValueError(
            f"Unknown SAM2 model type '{model_type}'. "
            f"Choose from: {list(SAM2_MODEL_CONFIGS.keys())}"
        )

    # Return existing predictor if the same model type is already loaded
    if state.sam2_predictor is not None and state.sam2_model_type == model_type:
        return state.sam2_predictor

    # Release old predictor if switching models
    if state.sam2_predictor is not None:
        print(f"Releasing SAM2 model ({state.sam2_model_type}) to load {model_type}...")
        del state.sam2_predictor
        state.sam2_predictor = None
        state.sam2_model_type = None
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()

    config = SAM2_MODEL_CONFIGS[model_type]
    print(f"Loading SAM2 Video Predictor ({model_type})...")
    device = get_device()

    # Try local checkpoint first
    sam2_dir = PROJECT_ROOT / "repos" / "sam2"
    ckpt_path = sam2_dir / "checkpoints" / config["checkpoint"]

    if ckpt_path.exists():
        try:
            from sam2.build_sam import build_sam2_video_predictor
            state.sam2_predictor = build_sam2_video_predictor(
                config_file=config["config"],
                ckpt_path=str(ckpt_path),
                device=str(device)
            )
            state.sam2_model_type = model_type
            print(f"SAM2 ({model_type}) loaded from {ckpt_path}")
            return state.sam2_predictor
        except Exception as e:
            print(f"Failed to load local checkpoint: {e}")

    # Fallback to HuggingFace (auto-download)
    print(f"Local checkpoint not found, loading {model_type} from HuggingFace...")
    from sam2.build_sam import build_sam2_video_predictor_hf
    state.sam2_predictor = build_sam2_video_predictor_hf(
        model_id=config["hf_model_id"],
        device=str(device)
    )
    state.sam2_model_type = model_type
    print(f"SAM2 ({model_type}) loaded from HuggingFace")

    return state.sam2_predictor


def load_pi3x_model():
    """Load Pi3X model if not already loaded."""
    if state.pi3x_model is not None:
        return state.pi3x_model

    print("Loading Pi3X model...")
    device = get_device()

    # Import Pi3X
    from pi3.models.pi3x import Pi3X

    state.pi3x_model = Pi3X.from_pretrained("yyfz233/Pi3X").to(device).eval()
    print("Pi3X loaded")

    return state.pi3x_model


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    device = get_device()
    return jsonify({
        "status": "ok",
        "device": str(device),
        "sam2_loaded": state.sam2_predictor is not None,
        "sam2_model_type": state.sam2_model_type,
        "pi3x_loaded": state.pi3x_model is not None,
        "video_loaded": state.video_info is not None
    })


@app.route('/sam2/init_video', methods=['POST'])
def init_video():
    """
    Initialize SAM2 with a video file.
    Extracts frames and prepares for prompting.

    Request body:
    {
        "video_path": "/path/to/video.mp4",
        "frame_interval": 5,  // optional, sample every N frames
        "model_type": "large"  // optional, "tiny"/"small"/"base"/"large"
    }
    """
    try:
        data = request.json
        video_path = data.get('video_path')
        frame_interval = data.get('frame_interval', 1)
        model_type = data.get('model_type', 'large')

        if not video_path or not os.path.exists(video_path):
            return jsonify({"error": f"Video not found: {video_path}"}), 400

        # Reset state
        state.reset()

        # Create temp directory for frames
        state.video_frames_dir = tempfile.mkdtemp(prefix="sam2_frames_")

        # Extract frames using OpenCV
        print(f"Extracting frames from {video_path}...")
        cap = cv2.VideoCapture(video_path)

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        frame_idx = 0
        saved_idx = 0
        state.original_frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                frame_path = os.path.join(state.video_frames_dir, f"{saved_idx:05d}.jpg")
                cv2.imwrite(frame_path, frame)
                state.original_frames.append(frame_path)
                saved_idx += 1

            frame_idx += 1

        cap.release()

        state.video_info = {
            "video_path": video_path,
            "fps": fps,
            "total_frames": total_frames,
            "extracted_frames": saved_idx,
            "frame_interval": frame_interval,
            "width": width,
            "height": height
        }

        print(f"Extracted {saved_idx} frames to {state.video_frames_dir}")

        # Initialize SAM2 state
        predictor = load_sam2_model(model_type=model_type)

        with torch.inference_mode():
            state.sam2_state = predictor.init_state(
                video_path=state.video_frames_dir,
            )

        # Return first frame as base64 for preview
        first_frame_path = state.original_frames[0]
        with open(first_frame_path, 'rb') as f:
            first_frame_b64 = base64.b64encode(f.read()).decode('utf-8')

        return jsonify({
            "success": True,
            "video_info": state.video_info,
            "first_frame": first_frame_b64
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/sam2/add_prompt', methods=['POST'])
def add_prompt():
    """
    Add click prompt to generate mask on first frame.

    Request body:
    {
        "frame_idx": 0,
        "points": [[x1, y1], [x2, y2]],
        "labels": [1, 0],  // 1=positive, 0=negative
        "obj_id": 1
    }
    """
    try:
        if state.sam2_state is None:
            return jsonify({"error": "No video loaded. Call /sam2/init_video first."}), 400

        data = request.json
        frame_idx = data.get('frame_idx', 0)
        points = np.array(data.get('points', []), dtype=np.float32)
        labels = np.array(data.get('labels', []), dtype=np.int32)
        obj_id = data.get('obj_id', 1)

        if len(points) == 0:
            return jsonify({"error": "No points provided"}), 400

        predictor = load_sam2_model()

        with torch.inference_mode():
            # フロントエンドから正規化座標 [0, 1] を受け取っている
            # normalize_coords=False を使用し、座標を自分でピクセル座標に変換する
            # SAM2は normalize_coords=True の場合、ピクセル座標を期待する
            video_H = state.sam2_state["video_height"]
            video_W = state.sam2_state["video_width"]
            points_pixel = points * np.array([video_W, video_H], dtype=np.float32)

            frame_idx_out, object_ids, masks = predictor.add_new_points_or_box(
                inference_state=state.sam2_state,
                frame_idx=frame_idx,
                obj_id=obj_id,
                points=points_pixel,
                labels=labels,
                clear_old_points=True,
                normalize_coords=True  # ピクセル座標を渡すので True
            )

        # Store mask - threshold logits to boolean (SAM2 returns logits, not binary masks)
        mask = (masks[0] > 0).cpu().numpy().squeeze()  # HxW boolean array
        state.masks[frame_idx] = mask

        # Create mask overlay image for preview
        frame_path = state.original_frames[frame_idx]
        frame = cv2.imread(frame_path)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize mask to frame size if needed
        if mask.shape != (frame_rgb.shape[0], frame_rgb.shape[1]):
            mask = cv2.resize(mask.astype(np.uint8),
                            (frame_rgb.shape[1], frame_rgb.shape[0]),
                            interpolation=cv2.INTER_NEAREST).astype(bool)

        # Create overlay
        overlay = frame_rgb.copy()
        overlay[mask] = overlay[mask] * 0.5 + np.array([0, 255, 0]) * 0.5

        # Encode as base64
        _, buffer = cv2.imencode('.jpg', cv2.cvtColor(overlay.astype(np.uint8), cv2.COLOR_RGB2BGR))
        overlay_b64 = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            "success": True,
            "frame_idx": frame_idx,
            "mask_pixels": int(mask.sum()),
            "mask_preview": overlay_b64
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/sam2/propagate', methods=['POST'])
def propagate():
    """
    Propagate masks through all frames.

    Request body:
    {
        "save_masks": true  // optional, save mask images
    }
    """
    try:
        if state.sam2_state is None:
            return jsonify({"error": "No video loaded. Call /sam2/init_video first."}), 400

        if len(state.masks) == 0:
            return jsonify({"error": "No prompts added. Call /sam2/add_prompt first."}), 400

        data = request.json or {}
        save_masks = data.get('save_masks', True)

        predictor = load_sam2_model()

        # Create output directory for masks
        output_dir = PROJECT_ROOT / "data" / "output" / "masks"
        output_dir.mkdir(parents=True, exist_ok=True)

        print("Propagating masks through video...")

        with torch.inference_mode():
            for frame_idx, object_ids, masks_tensor in tqdm(
                predictor.propagate_in_video(state.sam2_state),
                total=len(state.original_frames),
                desc="Propagating"
            ):
                # masks_tensor shape: (num_objects, 1, H, W)
                for i, obj_id in enumerate(object_ids):
                    mask = (masks_tensor[i] > 0).cpu().numpy().squeeze()  # Threshold logits to boolean
                    state.masks[frame_idx] = mask

                    if save_masks:
                        mask_path = output_dir / f"mask_{frame_idx:05d}.png"
                        cv2.imwrite(str(mask_path), (mask * 255).astype(np.uint8))

        print(f"Propagated masks for {len(state.masks)} frames")

        return jsonify({
            "success": True,
            "propagated_frames": len(state.masks),
            "mask_dir": str(output_dir) if save_masks else None
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/pi3x/reconstruct', methods=['POST'])
def reconstruct():
    """
    Run Pi3X reconstruction on masked frames.

    Request body:
    {
        "frame_interval": 10,  // sample every N frames for reconstruction
        "confidence_threshold": 0.1,
        "background_color": [1.0, 1.0, 1.0]  // white background
    }
    """
    try:
        if len(state.masks) == 0:
            return jsonify({"error": "No masks available. Run propagation first."}), 400

        data = request.json or {}
        frame_interval = data.get('frame_interval', 10)
        conf_threshold = data.get('confidence_threshold', 0.1)
        bg_color = np.array(data.get('background_color', [1.0, 1.0, 1.0]))

        device = get_device()
        model = load_pi3x_model()

        # Prepare masked images
        print("Preparing masked images for Pi3X...")
        masked_images = []
        frame_indices = sorted(state.masks.keys())[::frame_interval]

        for idx in tqdm(frame_indices, desc="Processing frames"):
            if idx >= len(state.original_frames):
                continue

            frame_path = state.original_frames[idx]
            frame = cv2.imread(frame_path)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

            mask = state.masks[idx]

            # Resize mask to frame size if needed
            if mask.shape != (frame_rgb.shape[0], frame_rgb.shape[1]):
                mask = cv2.resize(mask.astype(np.uint8),
                                (frame_rgb.shape[1], frame_rgb.shape[0]),
                                interpolation=cv2.INTER_NEAREST).astype(bool)

            # Apply mask with background color
            masked = frame_rgb * mask[:, :, None] + bg_color * (1 - mask[:, :, None])

            # Resize to 224x224 for Pi3X
            masked_resized = cv2.resize(masked, (224, 224))
            masked_images.append(masked_resized)

        if len(masked_images) < 2:
            return jsonify({"error": "Need at least 2 frames for reconstruction"}), 400

        # Stack and convert to tensor
        imgs = np.stack(masked_images, axis=0)  # (N, 224, 224, 3)
        imgs = np.transpose(imgs, (0, 3, 1, 2))  # (N, 3, 224, 224)
        imgs_tensor = torch.from_numpy(imgs).float().to(device)
        imgs_tensor = imgs_tensor.unsqueeze(0)  # (1, N, 3, 224, 224)

        print(f"Running Pi3X inference on {imgs_tensor.shape[1]} frames...")

        # Run inference
        with torch.no_grad():
            results = model(imgs_tensor)

        # Extract results
        points = results['points'][0].cpu().numpy()  # (N, 224, 224, 3)
        conf_logits = results['conf'][0].cpu().numpy()  # (N, 224, 224, 1)
        poses = results['camera_poses'][0].cpu().numpy()  # (N, 4, 4)

        # Apply confidence filtering using sigmoid
        confidence = sigmoid(conf_logits)
        confidence = confidence.squeeze(-1)  # (N, 224, 224)

        conf_mask = confidence > conf_threshold

        # Get colors from input images
        colors = np.transpose(imgs, (0, 2, 3, 1))  # (N, 224, 224, 3)

        # Filter and flatten
        valid_points = points[conf_mask]
        valid_colors = colors[conf_mask]

        print(f"Extracted {len(valid_points)} points (threshold={conf_threshold})")

        # Save PLY
        output_dir = PROJECT_ROOT / "data" / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        ply_path = output_dir / "object.ply"
        save_ply(valid_points, valid_colors, str(ply_path))

        # Save camera poses
        poses_path = output_dir / "camera_poses.json"
        poses_list = [pose.tolist() for pose in poses]
        with open(poses_path, 'w') as f:
            json.dump({
                "poses": poses_list,
                "frame_indices": frame_indices[:len(poses)]
            }, f, indent=2)

        return jsonify({
            "success": True,
            "num_points": len(valid_points),
            "num_frames": len(poses),
            "ply_path": str(ply_path),
            "poses_path": str(poses_path)
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/frame/<int:frame_idx>', methods=['GET'])
def get_frame(frame_idx):
    """Get a specific frame image."""
    if frame_idx >= len(state.original_frames):
        return jsonify({"error": "Frame index out of range"}), 404

    frame_path = state.original_frames[frame_idx]
    return send_file(frame_path, mimetype='image/jpeg')


@app.route('/download/ply', methods=['GET'])
def download_ply():
    """Download the generated PLY file."""
    ply_path = PROJECT_ROOT / "data" / "output" / "object.ply"
    if not ply_path.exists():
        return jsonify({"error": "PLY file not found"}), 404
    return send_file(str(ply_path), as_attachment=True)


def save_ply(points: np.ndarray, colors: np.ndarray, path: str):
    """Save point cloud as PLY file."""
    from plyfile import PlyData, PlyElement

    # Prepare vertex data
    vertices = np.zeros(len(points), dtype=[
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')
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


def cleanup():
    """Cleanup temporary files."""
    if state.video_frames_dir and os.path.exists(state.video_frames_dir):
        shutil.rmtree(state.video_frames_dir)
        print(f"Cleaned up {state.video_frames_dir}")


if __name__ == '__main__':
    import atexit
    atexit.register(cleanup)

    print("=" * 50)
    print("Pi3X + SAM2 Inference Server")
    print("=" * 50)

    # Pre-load models (optional, can be lazy)
    try:
        load_sam2_model()
    except Exception as e:
        print(f"Warning: Could not pre-load SAM2: {e}")

    try:
        load_pi3x_model()
    except Exception as e:
        print(f"Warning: Could not pre-load Pi3X: {e}")

    print("\nServer starting on http://localhost:5050")
    print("Endpoints:")
    print("  GET  /health           - Health check")
    print("  POST /sam2/init_video  - Load video")
    print("  POST /sam2/add_prompt  - Add click prompt")
    print("  POST /sam2/propagate   - Propagate masks")
    print("  POST /pi3x/reconstruct - Run reconstruction")
    print("")

    app.run(host='0.0.0.0', port=5050, debug=False)
