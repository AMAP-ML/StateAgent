"""Text-to-Video generator using DashScope wan2.2-t2v-plus.

Generates video directly from text prompt, then extracts the ending frame.
Used for Shot 1.
"""

import logging
import os
from typing import Optional

from . import dashscope_task
from . import video_utils

logger = logging.getLogger(__name__)

# wan2.2-t2v-plus endpoint
T2V_ENDPOINT = "/api/v1/services/aigc/video-generation/video-synthesis"

# Default model
DEFAULT_MODEL = "wan2.2-t2v-plus"


def generate_video(
    prompt: str,
    output_dir: str,
    shot_id: int,
    model: str = DEFAULT_MODEL,
    size: str = "832*480",
    negative_prompt: Optional[str] = None,
    prompt_extend: bool = True,
    poll_interval: int = 15,
    timeout: int = 600,
) -> dict:
    """Generate a video from text prompt using wan2.2-t2v-plus.

    Args:
        prompt: Text description of the video
        output_dir: Directory to save video and frame
        shot_id: Shot number for naming (1, 2, 3, ...)
        model: Model ID (default "wan2.2-t2v-plus")
        size: Video resolution (default "832*480" for 480P)
        negative_prompt: Things to avoid in the video
        prompt_extend: Whether to auto-enhance the prompt
        poll_interval: Seconds between status polls
        timeout: Maximum wait time in seconds

    Returns:
        dict with:
            - video_path: Path to the generated video file
            - end_frame_path: Path to the extracted ending frame
    """
    logger.info(f"Generating text-to-video: {prompt[:80]}...")

    # Prepare output paths
    os.makedirs(output_dir, exist_ok=True)
    video_filename = f"{shot_id:02d}.mp4"
    video_path = os.path.join(output_dir, video_filename)
    frame_filename = f"{shot_id:02d}_endframe.png"
    frame_path = os.path.join(output_dir, frame_filename)

    # Build request payload
    payload = {
        "model": model,
        "input": {
            "prompt": prompt
        },
        "parameters": {
            "size": size,
            "prompt_extend": prompt_extend
        }
    }

    if negative_prompt:
        payload["input"]["negative_prompt"] = negative_prompt

    # Submit task
    task_id = dashscope_task.submit_task(T2V_ENDPOINT, payload)

    # Poll until complete
    result = dashscope_task.poll_task(task_id, poll_interval, timeout)

    # Extract video URL
    video_url = result.get("output", {}).get("video_url")
    if not video_url:
        raise RuntimeError(f"No video_url in result: {result}")

    # Download video
    dashscope_task.download_file(video_url, video_path)

    # Extract ending frame
    video_utils.extract_ending_frame(video_path, frame_path)

    logger.info(f"Text-to-video complete: {video_path}")
    return {
        "video_path": video_path,
        "end_frame_path": frame_path
    }
