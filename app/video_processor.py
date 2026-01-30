"""
Video processing utilities for the Gradio UI.
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple
import cv2
import numpy as np


def list_videos(data_dir: str = "data") -> List[str]:
    """
    List available video files in the data directory.

    Args:
        data_dir: Path to data directory

    Returns:
        List of video file paths
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        return []

    video_extensions = {'.mp4', '.mov', '.avi', '.mkv', '.webm'}
    videos = []

    for f in data_path.iterdir():
        if f.is_file() and f.suffix.lower() in video_extensions:
            videos.append(str(f))

    return sorted(videos)


def get_video_info(video_path: str) -> dict:
    """
    Get video metadata.

    Args:
        video_path: Path to video file

    Returns:
        dict with fps, total_frames, width, height, duration
    """
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cap.release()

    duration = total_frames / fps if fps > 0 else 0

    return {
        "fps": fps,
        "total_frames": total_frames,
        "width": width,
        "height": height,
        "duration": duration
    }


def extract_first_frame(video_path: str) -> Optional[np.ndarray]:
    """
    Extract the first frame from a video.

    Args:
        video_path: Path to video file

    Returns:
        First frame as RGB numpy array, or None if failed
    """
    cap = cv2.VideoCapture(video_path)

    ret, frame = cap.read()
    cap.release()

    if not ret:
        return None

    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame_rgb


def create_click_visualization(
    image: np.ndarray,
    points: List[Tuple[float, float]],
    labels: List[int],
    point_radius: int = 8,
    positive_color: Tuple[int, int, int] = (0, 255, 0),
    negative_color: Tuple[int, int, int] = (255, 0, 0)
) -> np.ndarray:
    """
    Create an image with click points visualized.

    Args:
        image: Input image (RGB)
        points: List of (x, y) coordinates
        labels: List of labels (1=positive, 0=negative)
        point_radius: Radius of point circles
        positive_color: Color for positive clicks (RGB)
        negative_color: Color for negative clicks (RGB)

    Returns:
        Image with points drawn
    """
    vis_image = image.copy()

    for (x, y), label in zip(points, labels):
        color = positive_color if label == 1 else negative_color
        # Draw circle
        cv2.circle(vis_image, (int(x), int(y)), point_radius, color, -1)
        # Draw outline
        cv2.circle(vis_image, (int(x), int(y)), point_radius, (255, 255, 255), 2)

    return vis_image


def bytes_to_numpy(image_bytes: bytes) -> np.ndarray:
    """Convert image bytes to numpy array."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def numpy_to_bytes(image: np.ndarray, format: str = 'jpg') -> bytes:
    """Convert numpy array to image bytes."""
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode(f'.{format}', image_bgr)
    return buffer.tobytes()
