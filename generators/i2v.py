"""Image-to-Video (I2V) generator using DashScope wan2.2-i2v-plus.

Generates video from a single image + text prompt.
"""

import logging
import os
from typing import Optional

from . import dashscope_task
from . import video_utils

logger = logging.getLogger(__name__)

I2V_ENDPOINT = "/api/v1/services/aigc/video-generation/video-synthesis"
DEFAULT_MODEL = "wan2.2-i2v-plus"


def generate_video(
    image_path: str,
    prompt: str,
    output_dir: str,
    shot_id: int,
    model: str = DEFAULT_MODEL,
    size: str = "832*480",
    negative_prompt: Optional[str] = None,
    prompt_extend: bool = False,
    poll_interval: int = 15,
    timeout: int = 600,
) -> dict:
    """Generate a video from a single image using wan2.2-i2v-plus.

    Args:
        image_path: Path to the input image (first frame)
        prompt: Text description of the motion
        output_dir: Directory to save the video
        shot_id: Shot number for naming
        model: Model ID
        size: Video size (e.g. "832*480")
        negative_prompt: Things to avoid
        prompt_extend: Whether to auto-enhance the prompt
        poll_interval: Seconds between status polls
        timeout: Maximum wait time in seconds

    Returns:
        dict with video_path and end_frame_path
    """
    logger.info(f"Generating image-to-video: {prompt[:80]}...")
    logger.info(f"  Image: {image_path}")

    os.makedirs(output_dir, exist_ok=True)
    video_filename = f"{shot_id:02d}.mp4"
    video_path = os.path.join(output_dir, video_filename)
    frame_filename = f"{shot_id:02d}_endframe.png"
    frame_path = os.path.join(output_dir, frame_filename)

    image_uri = dashscope_task.file_to_data_uri(image_path)

    payload = {
        "model": model,
        "input": {
            "img_url": image_uri,
            "prompt": prompt,
        },
        "parameters": {
            "size": size,
            "prompt_extend": prompt_extend,
        },
    }

    if negative_prompt:
        payload["input"]["negative_prompt"] = negative_prompt

    task_id = dashscope_task.submit_task(I2V_ENDPOINT, payload)
    result = dashscope_task.poll_task(task_id, poll_interval, timeout)

    video_url = result.get("output", {}).get("video_url")
    if not video_url:
        raise RuntimeError(f"No video_url in result: {result}")

    dashscope_task.download_file(video_url, video_path)
    video_utils.extract_ending_frame(video_path, frame_path)

    logger.info(f"Image-to-video complete: {video_path}")
    return {
        "video_path": video_path,
        "end_frame_path": frame_path,
    }
