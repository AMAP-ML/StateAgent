"""Video utility functions for frame extraction."""

import logging
import os
import subprocess

logger = logging.getLogger(__name__)


def get_video_duration(video_path: str) -> float:
    """Get video duration in seconds using ffprobe."""
    probe_cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ]
    try:
        result = subprocess.run(
            probe_cmd, capture_output=True, text=True, timeout=30, check=True
        )
        return float(result.stdout.strip())
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError) as e:
        logger.warning(f"Could not get video duration: {e}, defaulting to 5s")
        return 5.0


def extract_frame_at(video_path: str, timestamp: float, output_path: str) -> str:
    """Extract a single frame at the given timestamp."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{timestamp:.2f}",
        "-i", video_path,
        "-vframes", "1",
        "-q:v", "2",
        output_path
    ]
    subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=True)

    if not os.path.exists(output_path):
        raise RuntimeError(f"Frame not found after extraction: {output_path}")
    return output_path


def extract_ending_frame(video_path: str, output_path: str) -> str:
    """Extract the last frame from a video file."""
    logger.info(f"Extracting ending frame from {video_path}")
    duration = get_video_duration(video_path)
    seek_time = max(0, duration - 0.1)
    return extract_frame_at(video_path, seek_time, output_path)


def extract_first_frame(video_path: str, output_path: str) -> str:
    """Extract the first frame from a video file."""
    logger.info(f"Extracting first frame from {video_path}")
    return extract_frame_at(video_path, 0.0, output_path)


def extract_keyframes(
    video_path: str,
    output_dir: str,
    num_frames: int = 6,
) -> list[dict]:
    """Extract multiple keyframes at regular intervals from a video.

    Args:
        video_path: Path to the input video file
        output_dir: Directory to save extracted frames
        num_frames: Number of frames to extract (default 6)

    Returns:
        List of dicts with keys:
            - path: frame file path
            - timestamp: seconds into the video
    """
    logger.info(f"Extracting {num_frames} keyframes from {video_path}")
    os.makedirs(output_dir, exist_ok=True)

    duration = get_video_duration(video_path)

    # Evenly spaced timestamps, avoiding very start and very end
    if num_frames <= 1:
        timestamps = [duration * 0.5]
    else:
        step = duration / (num_frames + 1)
        timestamps = [step * (i + 1) for i in range(num_frames)]

    frames = []
    for i, ts in enumerate(timestamps):
        frame_filename = f"keyframe_{i:02d}_{ts:.1f}s.png"
        frame_path = os.path.join(output_dir, frame_filename)

        try:
            extract_frame_at(video_path, ts, frame_path)
            if os.path.exists(frame_path) and os.path.getsize(frame_path) > 1000:
                frames.append({"path": frame_path, "timestamp": ts})
                logger.debug(f"  Extracted frame at {ts:.1f}s: {frame_path}")
            else:
                logger.warning(f"  Frame at {ts:.1f}s too small or missing")
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, RuntimeError) as e:
            logger.warning(f"  Failed to extract frame at {ts:.1f}s: {e}")

    logger.info(f"Extracted {len(frames)}/{num_frames} keyframes")
    return frames
