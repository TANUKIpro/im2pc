"""
API Client for communicating with the inference server.
"""

import os
import requests
from typing import Optional, List, Tuple
import base64


class InferenceClient:
    """Client for the Pi3X + SAM2 inference server."""

    def __init__(self, server_url: Optional[str] = None):
        """
        Initialize the inference client.

        Args:
            server_url: URL of the inference server.
                        Defaults to INFERENCE_SERVER_URL env var or localhost:5050
        """
        self.server_url = server_url or os.environ.get(
            'INFERENCE_SERVER_URL', 'http://localhost:5050'
        )
        self.server_url = self.server_url.rstrip('/')

        # Path conversion settings for Docker container -> host path mapping
        self.container_data_dir = os.environ.get('DATA_DIR', '/data')
        self.host_data_dir = os.environ.get('HOST_DATA_DIR', '')

    @staticmethod
    def _check_response(response: requests.Response) -> None:
        """Raise an error with the server's detail message if available."""
        if response.ok:
            return
        detail = ""
        try:
            body = response.json()
            detail = body.get("detail") or body.get("error") or ""
        except Exception:
            detail = response.text or ""
        raise requests.HTTPError(
            f"{response.status_code} {response.reason}: {detail}".strip(),
            response=response,
        )

    def _convert_path_to_host(self, container_path: str) -> str:
        """
        Convert a container-internal path to a host-side path.

        When the UI runs in Docker and the inference server runs on the host,
        file paths need to be converted from container paths (e.g., /data/video.mp4)
        to host paths (e.g., /Users/.../data/video.mp4).

        If HOST_DATA_DIR is not set, the path is returned unchanged.
        """
        if not self.host_data_dir:
            return container_path

        # Ensure we match at directory boundary (e.g., /data/ or /data exactly)
        prefix = self.container_data_dir.rstrip('/')
        if container_path == prefix or container_path.startswith(prefix + '/'):
            return container_path.replace(prefix, self.host_data_dir.rstrip('/'), 1)
        return container_path

    def health_check(self) -> dict:
        """Check if the server is healthy."""
        response = requests.get(f"{self.server_url}/health", timeout=10)
        self._check_response(response)
        return response.json()

    def init_video(self, video_path: str, frame_interval: int = 1) -> dict:
        """
        Initialize video for SAM2 processing.

        Args:
            video_path: Path to the video file
            frame_interval: Sample every N frames

        Returns:
            dict with video_info and first_frame (base64)
        """
        # Convert container path to host path for the inference server
        host_path = self._convert_path_to_host(video_path)
        response = requests.post(
            f"{self.server_url}/sam2/init_video",
            json={
                "video_path": host_path,
                "frame_interval": frame_interval
            },
            timeout=120
        )
        self._check_response(response)
        return response.json()

    def add_prompt(
        self,
        points: List[Tuple[float, float]],
        labels: List[int],
        frame_idx: int = 0,
        obj_id: int = 1
    ) -> dict:
        """
        Add click prompts to generate mask.

        Args:
            points: List of (x, y) coordinates
            labels: List of labels (1=positive, 0=negative)
            frame_idx: Frame index to add prompt on
            obj_id: Object ID

        Returns:
            dict with mask_preview (base64) and mask_pixels
        """
        response = requests.post(
            f"{self.server_url}/sam2/add_prompt",
            json={
                "frame_idx": frame_idx,
                "points": points,
                "labels": labels,
                "obj_id": obj_id
            },
            timeout=60
        )
        self._check_response(response)
        return response.json()

    def propagate(self, save_masks: bool = True) -> dict:
        """
        Propagate masks through all frames.

        Args:
            save_masks: Whether to save mask images

        Returns:
            dict with propagated_frames count
        """
        response = requests.post(
            f"{self.server_url}/sam2/propagate",
            json={"save_masks": save_masks},
            timeout=600  # Can take a while for long videos
        )
        self._check_response(response)
        return response.json()

    def reconstruct(
        self,
        frame_interval: int = 10,
        confidence_threshold: float = 0.1,
        background_color: List[float] = None
    ) -> dict:
        """
        Run Pi3X reconstruction.

        Args:
            frame_interval: Sample every N frames for reconstruction
            confidence_threshold: Minimum confidence to include points
            background_color: RGB values [0-1] for background

        Returns:
            dict with ply_path, poses_path, num_points
        """
        if background_color is None:
            background_color = [1.0, 1.0, 1.0]

        response = requests.post(
            f"{self.server_url}/pi3x/reconstruct",
            json={
                "frame_interval": frame_interval,
                "confidence_threshold": confidence_threshold,
                "background_color": background_color
            },
            timeout=600
        )
        self._check_response(response)
        return response.json()

    def get_frame(self, frame_idx: int) -> bytes:
        """Get a specific frame as image bytes."""
        response = requests.get(
            f"{self.server_url}/frame/{frame_idx}",
            timeout=30
        )
        self._check_response(response)
        return response.content

    def download_ply(self) -> bytes:
        """Download the generated PLY file."""
        response = requests.get(
            f"{self.server_url}/download/ply",
            timeout=60
        )
        self._check_response(response)
        return response.content


def decode_base64_image(b64_string: str) -> bytes:
    """Decode base64 string to image bytes."""
    return base64.b64decode(b64_string)
