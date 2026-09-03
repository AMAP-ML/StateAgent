"""Image generator using DashScope wan2.7-image.

Supports two modes:
1. Text-to-image (Shot 1): Generate image from prompt only
2. Image editing (Shot 2+): Generate image from prompt + reference images

Reference images are ordered:
- [0..N-1]: Entity crop images (for identity preservation)
- [N] (last): Previous ending frame (maintains resolution/composition)
"""

import logging
import os
import subprocess
from typing import Optional

import requests

from . import dashscope_task

logger = logging.getLogger(__name__)

# wan2.7-image endpoint (synchronous mode)
IMAGE_ENDPOINT = "/api/v1/services/aigc/multimodal-generation/generation"

# Default model
DEFAULT_MODEL = "wan2.7-image"


def generate_image(
    prompt: str,
    output_path: str,
    reference_images: Optional[list[str]] = None,
    model: str = DEFAULT_MODEL,
    size: str = "1K",
    target_size: Optional[str] = "832*480",
    n: int = 1,
    watermark: bool = False,
) -> str:
    """Generate an image using wan2.7-image.

    Args:
        prompt: Text description of the image
        output_path: Path to save the generated image
        reference_images: Optional list of reference image paths
        model: Model ID (default "wan2.7-image")
        size: Generation size, wan2.7-image uses "1K" or "2K" (default "1K")
        target_size: If set, resize to this size after generation (e.g., "832*480")
        n: Number of images to generate
        watermark: Whether to add watermark

    Returns:
        Path to the generated image
    """
    logger.info(f"Generating image: {prompt[:80]}...")
    if reference_images:
        logger.info(f"  With {len(reference_images)} reference images")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Build content array for messages
    content = []

    # Add reference images (if any)
    if reference_images:
        for img_path in reference_images:
            if not os.path.exists(img_path):
                logger.warning(f"Reference image not found: {img_path}")
                continue
            img_uri = dashscope_task.file_to_data_uri(img_path)
            content.append({"image": img_uri})
            logger.info(f"  Added reference: {os.path.basename(img_path)}")

    # Add text prompt
    content.append({"text": prompt})

    # Build request payload
    payload = {
        "model": model,
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": content
                }
            ]
        },
        "parameters": {
            "size": size,
            "n": n,
            "watermark": watermark
        }
    }

    # Submit synchronous request
    url = f"{dashscope_task.DASHSCOPE_BASE}{IMAGE_ENDPOINT}"
    headers = dashscope_task.auth_headers()
    headers["Content-Type"] = "application/json"

    resp = requests.post(url, json=payload, headers=headers, timeout=120)

    if resp.status_code != 200:
        raise RuntimeError(f"Image generation failed: {resp.status_code} {resp.text}")

    data = resp.json()

    # Extract image URL from response
    choices = data.get("output", {}).get("choices", [])
    if not choices:
        raise RuntimeError(f"No choices in response: {data}")

    message_content = choices[0].get("message", {}).get("content", [])
    if not message_content:
        raise RuntimeError(f"No content in message: {data}")

    # Find the image in content
    image_url = None
    for item in message_content:
        if "image" in item:
            image_url = item["image"]
            break

    if not image_url:
        raise RuntimeError(f"No image found in response: {data}")

    # Download image
    dashscope_task.download_file(image_url, output_path)

    # Resize if target_size is specified
    if target_size and target_size != size:
        _resize_image(output_path, target_size)

    logger.info(f"Image generated: {output_path}")
    return output_path


def _resize_image(image_path: str, target_size: str) -> None:
    """Resize an image to target size using ffmpeg.

    Args:
        image_path: Path to the image file (modified in-place)
        target_size: Target size in format "WIDTH*HEIGHT" (e.g., "832*480")
    """
    logger.info(f"Resizing image to {target_size}")

    # Parse target size
    try:
        width, height = target_size.split("*")
        width = int(width)
        height = int(height)
    except (ValueError, AttributeError) as e:
        logger.warning(f"Invalid target_size format: {target_size}, skipping resize")
        return

    # Create temporary output path
    temp_path = image_path + ".tmp.png"

    # Use ffmpeg to resize
    cmd = [
        "ffmpeg",
        "-y",
        "-i", image_path,
        "-vf", f"scale={width}:{height}",
        temp_path
    ]

    try:
        subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
            check=True
        )

        # Replace original with resized
        os.replace(temp_path, image_path)
        logger.info(f"Resized to {width}x{height}")

    except subprocess.CalledProcessError as e:
        logger.error(f"Resize failed: {e.stderr}")
        # Clean up temp file if it exists
        if os.path.exists(temp_path):
            os.remove(temp_path)
        # Continue with original image
    except subprocess.TimeoutExpired:
        logger.error("Resize timed out")
        if os.path.exists(temp_path):
            os.remove(temp_path)
    except Exception as e:
        logger.error(f"Resize failed: {e}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
