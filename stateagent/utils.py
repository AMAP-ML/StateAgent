"""Shared utility functions for StateAgent."""

import base64
import json
import logging
import os

from stateagent.models import MemoryBank

logger = logging.getLogger(__name__)


def encode_image(path: str) -> str | None:
    """Read and base64-encode an image file. Returns None on failure."""
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            data = f.read()
        if len(data) < 100:
            return None
        ext = os.path.splitext(path)[1].lower()
        mime = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
            ".bmp": "image/bmp",
        }.get(ext, "image/png")
        encoded = base64.b64encode(data).decode("utf-8")
        return f"data:{mime};base64,{encoded}"
    except Exception as e:
        logger.warning(f"Failed to encode image {path}: {e}")
        return None


def format_memory_for_prompt(memory: MemoryBank) -> str:
    """Format MemoryBank as readable JSON for VLM prompts.

    Used by video_observer and state_predictor to pass current state to VLM.
    """
    data = {
        "time": memory.time,
        "entities": {},
    }
    for eid, entity in memory.entities.items():
        data["entities"][eid] = {
            "type": entity.type.value,
            "attributes": entity.attributes,
            "visibility": entity.visibility.value,
            "current_location": entity.current_location,
            "open_state": entity.open_state.value,
            "relations": [r.model_dump() for r in entity.relations],
            "has_appearance": entity.appearance_image is not None,
            "size_description": entity.size_description,
            "shape_description": entity.shape_description,
            "state_description": entity.state_description,
        }
    return json.dumps(data, ensure_ascii=False, indent=2)
