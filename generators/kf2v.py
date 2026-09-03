"""Keyframe-to-Video generator using DashScope wan2.2-kf2v-flash.

Generates video from first frame + last frame images.
Used for Shot 2+.
"""

import logging
import os
from typing import Optional

from . import dashscope_task
from . import video_utils

logger = logging.getLogger(__name__)

# wan2.2-kf2v-flash endpoint
KF2V_ENDPOINT = "/api/v1/services/aigc/image2video/video-synthesis"

# Default model
DEFAULT_MODEL = "wan2.2-kf2v-flash"


def generate_video(
    first_frame_path: str,
    last_frame_path: str,
    prompt: str,
    output_dir: str,
    shot_id: int,
    model: str = DEFAULT_MODEL,
    resolution: str = "480P",
    negative_prompt: Optional[str] = None,
    prompt_extend: bool = True,
    poll_interval: int = 15,
    timeout: int = 600,
) -> dict:
    """Generate a video from first and last frame images using wan2.2-kf2v-flash.

    Args:
        first_frame_path: Path to the first frame image
        last_frame_path: Path to the last frame image
        prompt: Text description of the motion/transition
        output_dir: Directory to save the video
        shot_id: Shot number for naming (1, 2, 3, ...)
        model: Model ID (default "wan2.2-kf2v-flash")
        resolution: Video resolution (default "480P")
        negative_prompt: Things to avoid in the video
        prompt_extend: Whether to auto-enhance the prompt
        poll_interval: Seconds between status polls
        timeout: Maximum wait time in seconds

    Returns:
        dict with:
            - video_path: Path to the generated video file
            - end_frame_path: Path to the ending frame (extracted from video)
    """
    logger.info(f"Generating keyframe-to-video: {prompt[:80]}...")
    logger.info(f"  First frame: {first_frame_path}")
    logger.info(f"  Last frame: {last_frame_path}")

    # Prepare output path
    os.makedirs(output_dir, exist_ok=True)
    video_filename = f"{shot_id:02d}.mp4"
    video_path = os.path.join(output_dir, video_filename)
    frame_filename = f"{shot_id:02d}_endframe.png"
    frame_path = os.path.join(output_dir, frame_filename)

    # Convert local files to data URIs
    first_frame_uri = dashscope_task.file_to_data_uri(first_frame_path)
    last_frame_uri = dashscope_task.file_to_data_uri(last_frame_path)

    # Enhance prompt to prevent transitions and sudden changes
    enhanced_prompt = f"{prompt}. Maintain smooth, continuous motion between the first and last frame."

    # Build request payload
    payload = {
        "model": model,
        "input": {
            "first_frame_url": first_frame_uri,
            "last_frame_url": last_frame_uri,
            "prompt": enhanced_prompt
        },
        "parameters": {
            "resolution": resolution,
            "prompt_extend": prompt_extend
        }
    }

    if negative_prompt:
        payload["input"]["negative_prompt"] = negative_prompt

    # Submit task
    task_id = dashscope_task.submit_task(KF2V_ENDPOINT, payload)

    # Poll until complete
    result = dashscope_task.poll_task(task_id, poll_interval, timeout)

    # Extract video URL
    video_url = result.get("output", {}).get("video_url")
    if not video_url:
        raise RuntimeError(f"No video_url in result: {result}")

    # Download video
    dashscope_task.download_file(video_url, video_path)

    # Extract ending frame (should match last_frame but extract to be safe)
    video_utils.extract_ending_frame(video_path, frame_path)

    logger.info(f"Keyframe-to-video complete: {video_path}")
    return {
        "video_path": video_path,
        "end_frame_path": frame_path
    }
