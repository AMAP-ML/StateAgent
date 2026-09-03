"""Shared DashScope async task helper.

All video generation models (t2v, kf2v) use the same async pattern:
  1. POST submit task with X-DashScope-Async: enable
  2. GET poll task status every N seconds
  3. On SUCCEEDED, download result
"""

import logging
import os
import time

import requests

logger = logging.getLogger(__name__)

DASHSCOPE_BASE = "https://dashscope.aliyuncs.com"
SUBMIT_HEADERS = {
    "Content-Type": "application/json",
    "X-DashScope-Async": "enable",
}

# Task status values
STATUS_PENDING = "PENDING"
STATUS_RUNNING = "RUNNING"
STATUS_SUCCEEDED = "SUCCEEDED"
STATUS_FAILED = "FAILED"
STATUS_CANCELED = "CANCELED"
STATUS_UNKNOWN = "UNKNOWN"


def get_api_key() -> str:
    """Get DashScope API key from environment."""
    key = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("VLM_KEY") or os.environ.get("VLM_API_KEY", "")
    if not key:
        raise ValueError("DASHSCOPE_API_KEY, VLM_KEY, or VLM_API_KEY not set")
    return key


def auth_headers() -> dict:
    """Get authorization headers."""
    return {"Authorization": f"Bearer {get_api_key()}"}


def submit_task(endpoint: str, payload: dict) -> str:
    """Submit an async task and return the task_id.

    Args:
        endpoint: Full URL path (e.g., '/api/v1/services/aigc/video-generation/video-synthesis')
        payload: Request body JSON

    Returns:
        task_id string

    Raises:
        RuntimeError if submission fails
    """
    url = f"{DASHSCOPE_BASE}{endpoint}"
    headers = {**auth_headers(), **SUBMIT_HEADERS}

    logger.info(f"Submitting task to {endpoint}")
    resp = requests.post(url, json=payload, headers=headers, timeout=60)

    if resp.status_code != 200:
        raise RuntimeError(
            f"Task submission failed: {resp.status_code} {resp.text}"
        )

    data = resp.json()
    task_id = data.get("output", {}).get("task_id")
    if not task_id:
        raise RuntimeError(f"No task_id in response: {data}")

    logger.info(f"Task submitted: {task_id}")
    return task_id


def poll_task(task_id: str, poll_interval: int = 15, timeout: int = 600) -> dict:
    """Poll a task until completion.

    Args:
        task_id: The task ID to poll
        poll_interval: Seconds between polls (default 15)
        timeout: Maximum seconds to wait (default 600 = 10 min)

    Returns:
        The full task response dict on success

    Raises:
        RuntimeError if task fails or times out
    """
    url = f"{DASHSCOPE_BASE}/api/v1/tasks/{task_id}"
    headers = auth_headers()

    start_time = time.time()
    while True:
        elapsed = time.time() - start_time
        if elapsed > timeout:
            raise RuntimeError(f"Task {task_id} timed out after {timeout}s")

        resp = requests.get(url, headers=headers, timeout=30)
        if resp.status_code != 200:
            logger.warning(f"Poll failed: {resp.status_code}, retrying...")
            time.sleep(poll_interval)
            continue

        data = resp.json()
        status = data.get("output", {}).get("task_status", STATUS_UNKNOWN)

        if status == STATUS_SUCCEEDED:
            logger.info(f"Task {task_id} succeeded ({elapsed:.0f}s)")
            return data

        elif status in (STATUS_FAILED, STATUS_CANCELED, STATUS_UNKNOWN):
            error_msg = data.get("output", {}).get("message", "Unknown error")
            error_code = data.get("output", {}).get("code", "")
            raise RuntimeError(
                f"Task {task_id} {status}: [{error_code}] {error_msg}"
            )

        else:
            logger.info(f"Task {task_id}: {status} ({elapsed:.0f}s)")
            time.sleep(poll_interval)


def download_file(url: str, output_path: str, timeout: int = 120, max_retries: int = 3) -> str:
    """Download a file from URL to local path with retry for transient network errors.

    Args:
        url: The URL to download from
        output_path: Local path to save the file
        timeout: Download timeout in seconds
        max_retries: Number of retry attempts for transient failures

    Returns:
        The output_path
    """
    logger.info(f"Downloading {url[:80]}... to {output_path}")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, timeout=timeout, stream=True)
            resp.raise_for_status()
            with open(output_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            break
        except (requests.ConnectionError, requests.Timeout, OSError) as e:
            last_err = e
            if attempt < max_retries:
                wait = 5 * attempt
                logger.warning(
                    f"Download attempt {attempt}/{max_retries} failed: {e}, "
                    f"retrying in {wait}s..."
                )
                time.sleep(wait)
            else:
                raise RuntimeError(
                    f"Download failed after {max_retries} attempts: {e}"
                ) from last_err

    size = os.path.getsize(output_path)
    logger.info(f"Downloaded: {output_path} ({size} bytes)")
    return output_path


def file_to_data_uri(file_path: str) -> str:
    """Convert a local file to a base64 data URI.

    Args:
        file_path: Path to the local file

    Returns:
        Data URI string (e.g., 'data:image/png;base64,...')
    """
    import base64
    from pathlib import Path

    ext = Path(file_path).suffix.lower()
    mime_map = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
        ".mp4": "video/mp4",
    }
    mime = mime_map.get(ext, "image/png")

    with open(file_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")

    return f"data:{mime};base64,{data}"
