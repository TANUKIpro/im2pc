#!/usr/bin/env python3
"""
Pi3X + SAM2 Gradio Web UI

A web interface for:
1. Loading videos
2. Selecting objects via click prompts
3. Propagating masks through video
4. Running 3D reconstruction with Pi3X
5. Downloading PLY files
"""

import os
import sys
from pathlib import Path
from typing import Optional, List, Tuple
import tempfile

import gradio as gr
import numpy as np

# Add app directory to path
APP_DIR = Path(__file__).parent
sys.path.insert(0, str(APP_DIR))

from api_client import InferenceClient, decode_base64_image
from video_processor import (
    list_videos,
    get_video_info,
    extract_first_frame,
    create_click_visualization,
    bytes_to_numpy
)


# Global state
class AppState:
    def __init__(self):
        self.client = InferenceClient()
        self.current_video = None
        self.first_frame = None
        self.click_points = []
        self.click_labels = []
        self.mask_preview = None
        self.video_info = None

    def reset(self):
        self.current_video = None
        self.first_frame = None
        self.click_points = []
        self.click_labels = []
        self.mask_preview = None
        self.video_info = None


app_state = AppState()


def check_server_status():
    """Check if inference server is running."""
    try:
        status = app_state.client.health_check()
        return f"Server OK - Device: {status['device']}, SAM2: {'Loaded' if status['sam2_loaded'] else 'Not loaded'}, Pi3X: {'Loaded' if status['pi3x_loaded'] else 'Not loaded'}"
    except Exception as e:
        return f"Server Error: {str(e)}"


def get_video_list():
    """Get list of available videos."""
    # Check multiple possible data directories
    possible_dirs = [
        os.environ.get('DATA_DIR', '/data'),  # Docker container (from env)
        "/data",                               # Docker container fallback
        str(APP_DIR.parent / "data"),          # Local development
        "data"                                 # Relative
    ]

    for data_dir in possible_dirs:
        videos = list_videos(data_dir)
        if videos:
            return videos

    return []


def on_video_select_event(evt: gr.SelectData):
    """Wrapper for .select() event which receives gr.SelectData."""
    return on_video_select(evt.value)


def on_video_select(video_path: str):
    """Handle video selection."""
    if not video_path:
        return None, "No video selected", None

    try:
        # Reset state
        app_state.reset()
        app_state.current_video = video_path

        # Get video info
        info = get_video_info(video_path)
        app_state.video_info = info
        info_text = (
            f"Resolution: {info['width']}x{info['height']}\n"
            f"FPS: {info['fps']:.2f}\n"
            f"Frames: {info['total_frames']}\n"
            f"Duration: {info['duration']:.1f}s"
        )

        # Extract first frame for preview
        first_frame = extract_first_frame(video_path)
        if first_frame is None:
            return None, "Failed to extract first frame", info_text

        app_state.first_frame = first_frame

        return first_frame, "Video loaded. Click on the object to segment.", info_text

    except Exception as e:
        return None, f"Error: {str(e)}", None


def on_image_click(image, evt: gr.SelectData):
    """Handle click on the image."""
    if app_state.first_frame is None:
        return image, "Load a video first"

    x, y = evt.index[0], evt.index[1]

    # 画像サイズを取得して正規化座標 [0, 1] に変換
    img_height, img_width = image.shape[:2]
    norm_x = x / img_width
    norm_y = y / img_height

    # 正規化座標を保存
    app_state.click_points.append((norm_x, norm_y))
    app_state.click_labels.append(1)

    # 可視化用にピクセル座標に変換（原動画解像度）
    vis_points = [(int(p[0] * app_state.first_frame.shape[1]),
                   int(p[1] * app_state.first_frame.shape[0]))
                  for p in app_state.click_points]

    vis_image = create_click_visualization(
        app_state.first_frame,
        vis_points,
        app_state.click_labels
    )

    points_text = f"Points: {len(app_state.click_points)} positive"

    return vis_image, points_text


def on_add_negative_click(image, evt: gr.SelectData):
    """Handle right-click (negative) on the image."""
    if app_state.first_frame is None:
        return image, "Load a video first"

    x, y = evt.index[0], evt.index[1]

    # 画像サイズを取得して正規化座標 [0, 1] に変換
    img_height, img_width = image.shape[:2]
    norm_x = x / img_width
    norm_y = y / img_height

    # 正規化座標を保存
    app_state.click_points.append((norm_x, norm_y))
    app_state.click_labels.append(0)

    # 可視化用にピクセル座標に変換（原動画解像度）
    vis_points = [(int(p[0] * app_state.first_frame.shape[1]),
                   int(p[1] * app_state.first_frame.shape[0]))
                  for p in app_state.click_points]

    vis_image = create_click_visualization(
        app_state.first_frame,
        vis_points,
        app_state.click_labels
    )

    neg_count = app_state.click_labels.count(0)
    pos_count = app_state.click_labels.count(1)
    points_text = f"Points: {pos_count} positive, {neg_count} negative"

    return vis_image, points_text


def clear_clicks():
    """Clear all click points."""
    app_state.click_points = []
    app_state.click_labels = []

    if app_state.first_frame is not None:
        return app_state.first_frame, "Clicks cleared"
    return None, "Clicks cleared"


def init_and_preview(frame_interval: int):
    """Initialize video and get mask preview."""
    if app_state.current_video is None:
        return None, "No video loaded"

    if len(app_state.click_points) == 0:
        return app_state.first_frame, "Add at least one click point"

    try:
        # Initialize video on server
        result = app_state.client.init_video(
            app_state.current_video,
            frame_interval=int(frame_interval)
        )

        if not result.get('success'):
            return app_state.first_frame, f"Error: {result.get('error', 'Unknown')}"

        # Add prompt
        prompt_result = app_state.client.add_prompt(
            points=app_state.click_points,
            labels=app_state.click_labels,
            frame_idx=0,
            obj_id=1
        )

        if not prompt_result.get('success'):
            return app_state.first_frame, f"Error: {prompt_result.get('error', 'Unknown')}"

        # Decode mask preview
        mask_b64 = prompt_result.get('mask_preview')
        if mask_b64:
            mask_bytes = decode_base64_image(mask_b64)
            mask_image = bytes_to_numpy(mask_bytes)
            app_state.mask_preview = mask_image
            return mask_image, f"Mask generated: {prompt_result['mask_pixels']:,} pixels"

        return app_state.first_frame, "Mask generated (no preview)"

    except Exception as e:
        return app_state.first_frame, f"Error: {str(e)}"


def propagate_masks(progress=gr.Progress()):
    """Propagate masks through all frames."""
    try:
        progress(0, desc="Starting propagation...")

        result = app_state.client.propagate(save_masks=True)

        if not result.get('success'):
            return f"Error: {result.get('error', 'Unknown')}"

        return f"Propagated masks to {result['propagated_frames']} frames"

    except Exception as e:
        return f"Error: {str(e)}"


def run_reconstruction(
    recon_frame_interval: int,
    confidence_threshold: float,
    progress=gr.Progress()
):
    """Run Pi3X reconstruction."""
    try:
        progress(0, desc="Running reconstruction...")

        result = app_state.client.reconstruct(
            frame_interval=int(recon_frame_interval),
            confidence_threshold=float(confidence_threshold),
            background_color=[1.0, 1.0, 1.0]
        )

        if not result.get('success'):
            return None, f"Error: {result.get('error', 'Unknown')}"

        ply_path = result.get('ply_path')
        status = (
            f"Reconstruction complete!\n"
            f"Points: {result['num_points']:,}\n"
            f"Frames: {result['num_frames']}\n"
            f"PLY: {ply_path}"
        )

        return ply_path, status

    except Exception as e:
        return None, f"Error: {str(e)}"


def create_ui():
    """Create the Gradio UI."""

    with gr.Blocks(title="Pi3X + SAM2 Reconstruction", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Pi3X + SAM2 Multi-View Reconstruction")
        gr.Markdown(
            "1. Select a video\n"
            "2. Click on the object to segment (green = include, red = exclude)\n"
            "3. Generate mask preview\n"
            "4. Propagate masks through video\n"
            "5. Run 3D reconstruction"
        )

        with gr.Row():
            with gr.Column(scale=2):
                # Video selection
                video_list = get_video_list()
                video_dropdown = gr.Dropdown(
                    choices=video_list,
                    value=video_list[0] if video_list else None,
                    label="Select Video",
                    interactive=True
                )
                refresh_btn = gr.Button("Refresh Video List", size="sm")

                # Image display for clicks
                image_display = gr.Image(
                    label="First Frame (Click to add points)",
                    interactive=False,
                    type="numpy"
                )

                # Secondary image for negative clicks
                with gr.Row():
                    clear_btn = gr.Button("Clear Clicks", size="sm")
                    negative_mode = gr.Checkbox(
                        label="Negative Click Mode (exclude region)",
                        value=False
                    )

            with gr.Column(scale=1):
                # Status and info
                server_status = gr.Textbox(
                    label="Server Status",
                    interactive=False,
                    value=check_server_status()
                )
                refresh_status_btn = gr.Button("Refresh Status", size="sm")

                video_info = gr.Textbox(
                    label="Video Info",
                    interactive=False,
                    lines=4
                )

                status_text = gr.Textbox(
                    label="Status",
                    interactive=False,
                    lines=2
                )

        gr.Markdown("---")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Step 1: Generate Mask")
                frame_interval = gr.Slider(
                    minimum=1, maximum=30, value=5, step=1,
                    label="Frame Sampling Interval"
                )
                preview_btn = gr.Button("Generate Mask Preview", variant="primary")

            with gr.Column():
                gr.Markdown("### Step 2: Propagate")
                propagate_btn = gr.Button("Propagate Masks", variant="primary")
                propagate_status = gr.Textbox(
                    label="Propagation Status",
                    interactive=False
                )

        gr.Markdown("---")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Step 3: Reconstruct")
                recon_frame_interval = gr.Slider(
                    minimum=1, maximum=30, value=10, step=1,
                    label="Reconstruction Frame Interval"
                )
                confidence_threshold = gr.Slider(
                    minimum=0.01, maximum=0.5, value=0.1, step=0.01,
                    label="Confidence Threshold"
                )
                reconstruct_btn = gr.Button("Run Reconstruction", variant="primary")

            with gr.Column():
                gr.Markdown("### Output")
                recon_status = gr.Textbox(
                    label="Reconstruction Status",
                    interactive=False,
                    lines=4
                )
                ply_file = gr.File(
                    label="Download PLY",
                    interactive=False
                )

        # Event handlers
        def refresh_video_list():
            videos = get_video_list()
            return gr.Dropdown(choices=videos, value=videos[0] if videos else None)

        refresh_btn.click(
            fn=refresh_video_list,
            outputs=video_dropdown
        ).then(
            fn=on_video_select,
            inputs=video_dropdown,
            outputs=[image_display, status_text, video_info]
        )

        refresh_status_btn.click(
            fn=check_server_status,
            outputs=server_status
        )

        video_dropdown.select(
            fn=on_video_select_event,
            outputs=[image_display, status_text, video_info]
        )

        # Handle clicks based on mode
        def handle_click(image, evt: gr.SelectData, is_negative: bool):
            if is_negative:
                return on_add_negative_click(image, evt)
            return on_image_click(image, evt)

        image_display.select(
            fn=handle_click,
            inputs=[image_display, negative_mode],
            outputs=[image_display, status_text]
        )

        clear_btn.click(
            fn=clear_clicks,
            outputs=[image_display, status_text]
        )

        preview_btn.click(
            fn=init_and_preview,
            inputs=frame_interval,
            outputs=[image_display, status_text]
        )

        propagate_btn.click(
            fn=propagate_masks,
            outputs=propagate_status
        )

        reconstruct_btn.click(
            fn=run_reconstruction,
            inputs=[recon_frame_interval, confidence_threshold],
            outputs=[ply_file, recon_status]
        )

        demo.load(
            fn=on_video_select,
            inputs=video_dropdown,
            outputs=[image_display, status_text, video_info]
        )

    return demo


if __name__ == "__main__":
    demo = create_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
