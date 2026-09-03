"""Video observer — VLM-based state observation from generated video frames.

After each shot's video is generated, the observer uses VLM to look at the
ending frame and confirm what actually happened. This implements closed-loop
feedback: state is based on actual generated content, not just text prediction.

Outputs a MemoryBank with observed entities and their states.
"""

import base64
import json
import logging
from pathlib import Path
from typing import Optional

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from stateagent.models import (
    EntityMemory,
    EntityType,
    MemoryBank,
    OpenState,
    Relation,
    VisibilityState,
)
from stateagent.utils import format_memory_for_prompt

logger = logging.getLogger(__name__)

OBSERVER_SYSTEM_PROMPT = """You are a video state observer. Your task is to analyze the video ending frame image and confirm the actual state of all entities in the current world.

You should focus on:
1. Which entities are visible? Their appearance attributes (color, material, shape, size)
2. The position of each visible entity (which container it is in, who is holding it, where it is)
3. The open/closed state of containers
4. Spatial relationships between entities (inside, holding, on_top_of, covered_by, wearing, occurred_by, behind, etc., not limited to these.)
5. Which entities are not visible (occluded or inside closed containers) — infer based on logic
6. The current physical state description of each entity (state_description): use a brief natural language sentence to describe the entity's current physical state, contents, surface changes, etc.

Compare the "previously known state" with "this shot's description" to determine what actually changed.
Note: video generation may not fully follow the description; you should report what you actually observe."""

OBSERVER_USER_TEMPLATE = """This is the video ending frame. Please analyze the image and describe the current state.

This shot's description: "{prompt}"

Previously known state:
{previous_state_json}

Please return the complete observed state in JSON format:
{{
  "entities": {{
    "entity_id": {{
      "type": "person|object|container|clothing|unknown",
      "attributes": {{"color": "...", "material": "...", "shape": "...", "size": "..."}},
      "visibility": "visible|hidden|partially_visible",
      "current_location": "inside:box_1|holding:person_1|none",
      "open_state": "open|closed|n/a",
      "relations": [
        {{"type": "inside|holding|wearing|covered_by|on_top_of", "subject": "entity_id", "object": "entity_id", "value": true}}
      ],
      "size_description": "natural language description of physical size",
      "shape_description": "natural language description of shape",
      "state_description": "brief description of the entity's current physical state (contents, surface changes, morphological changes, etc.)"
    }}
  }},
  "observations": "briefly explain what you observed and whether it is consistent with expectations"
}}

Important:
- If the image is not a real image (e.g., a text file or placeholder), return "mock": true and keep the state unchanged
- Try to keep entity_id consistent with IDs from the previous state
- For invisible entities, infer based on logic (e.g., objects inside closed containers)"""


class VideoObserver:
    """VLM-based video frame observer for closed-loop state tracking."""

    def __init__(self, vlm_client: OpenAI, model: str):
        self.vlm_client = vlm_client
        self.model = model

    def observe(
        self,
        video_frame_path: Optional[str],
        previous_memory: MemoryBank,
        prompt: str,
    ) -> MemoryBank:
        """Observe the ending frame of a generated video.

        Args:
            video_frame_path: Path to the ending frame image
            previous_memory: The memory bank before this shot
            prompt: The prompt used to generate this shot

        Returns:
            Updated MemoryBank reflecting what actually happened in the video
        """
        previous_state_json = format_memory_for_prompt(previous_memory)

        user_content = OBSERVER_USER_TEMPLATE.format(
            prompt=prompt,
            previous_state_json=previous_state_json,
        )

        # Try to read and encode the image
        image_base64 = None
        if video_frame_path and Path(video_frame_path).exists():
            try:
                with open(video_frame_path, "rb") as f:
                    image_data = f.read()
                if len(image_data) > 100:  # Real images are > 100 bytes
                    image_base64 = base64.b64encode(image_data).decode("utf-8")
            except Exception as e:
                logger.warning(f"Failed to read image {video_frame_path}: {e}")

        # Build VLM request
        if image_base64:
            messages = [
                {"role": "system", "content": OBSERVER_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_content},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_base64}",
                                "detail": "high",
                            },
                        },
                    ],
                },
            ]
        else:
            messages = [
                {"role": "system", "content": OBSERVER_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": user_content + "\n\nNote: No image provided, please return {\"mock\": true} and keep the state unchanged.",
                },
            ]

        try:
            result = self._call_vlm(messages)

            if result.get("mock"):
                logger.info("Video observer: mock mode, keeping state unchanged")
                return previous_memory.model_copy(deep=True)

            observed_memory = self._parse_observation(result, previous_memory)

            observations_text = result.get("observations", "")
            if observations_text:
                logger.info(f"Video observation result: {observations_text}")

            return observed_memory

        except Exception as e:
            logger.error(f"Video observation failed: {e}, keeping previous state")
            return previous_memory.model_copy(deep=True)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=30))
    def _call_vlm(self, messages: list[dict]) -> dict:
        """Call VLM with retry. Parse JSON from response text."""
        response = self.vlm_client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.1,
            extra_body={"enable_thinking": False},
        )
        content = response.choices[0].message.content or ""

        # Try parsing as JSON directly
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        # Try extracting JSON from markdown code block
        import re
        json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', content, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try finding JSON object in text
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        raise ValueError(f"Cannot parse JSON from VLM response: {content[:200]}")

    def _parse_observation(self, result: dict, previous_memory: MemoryBank) -> MemoryBank:
        """Parse VLM observation into MemoryBank."""
        new_memory = MemoryBank(
            time=previous_memory.time + 1,
            entities={},
            history=list(previous_memory.history),
        )

        entities_data = result.get("entities", {})
        for eid, entity_data in entities_data.items():
            # Parse relations
            relations = []
            for rel_data in entity_data.get("relations", []):
                relations.append(Relation(
                    type=rel_data["type"],
                    subject=rel_data["subject"],
                    object=rel_data["object"],
                    value=rel_data.get("value", True),
                ))

            # Carry over appearance from previous memory
            prev_entity = previous_memory.entities.get(eid)
            appearance_image = None
            state_description = None
            source_shot = None
            if prev_entity:
                appearance_image = prev_entity.appearance_image
                state_description = prev_entity.state_description
                source_shot = prev_entity.source_shot

            entity = EntityMemory(
                entity_id=eid,
                type=EntityType(entity_data.get("type", "unknown")),
                attributes=entity_data.get("attributes", {}),
                visibility=VisibilityState(entity_data.get("visibility", "visible")),
                current_location=entity_data.get("current_location"),
                open_state=OpenState(entity_data.get("open_state", "n/a")),
                relations=relations,
                appearance_image=appearance_image,
                size_description=entity_data.get("size_description"),
                shape_description=entity_data.get("shape_description"),
                state_description=entity_data.get("state_description") or state_description,
                source_shot=source_shot,
            )
            new_memory.entities[eid] = entity

        return new_memory

    def analyze_frame_visibility(
        self,
        frame_path: str,
        entity_ids: list[str],
    ) -> dict[str, bool]:
        """Analyze which entities are visible in a frame.

        Args:
            frame_path: Path to the frame image
            entity_ids: List of entity IDs to check

        Returns:
            Dict mapping entity_id to True (visible) or False (not visible)
        """
        if not frame_path or not Path(frame_path).exists():
            return {eid: False for eid in entity_ids}

        try:
            with open(frame_path, "rb") as f:
                image_data = f.read()
            if len(image_data) < 100:
                return {eid: False for eid in entity_ids}
            image_base64 = base64.b64encode(image_data).decode("utf-8")
        except Exception as e:
            logger.warning(f"Failed to read image {frame_path}: {e}")
            return {eid: False for eid in entity_ids}

        entity_list = ", ".join(entity_ids)
        prompt = f"""Determine whether the following entities are clearly visible in this image: {entity_list}

Return JSON format:
{{
  "visibility": {{
    "entity_id": true/false
  }}
}}

Only judge visibility. true means clearly visible, false means not visible or occluded. Return only JSON."""

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}",
                            "detail": "high",
                        },
                    },
                ],
            }
        ]

        try:
            result = self._call_vlm(messages)
            visibility = result.get("visibility", {})
            return {eid: visibility.get(eid, False) for eid in entity_ids}
        except Exception as e:
            logger.warning(f"Visibility analysis failed: {e}")
            return {eid: False for eid in entity_ids}
