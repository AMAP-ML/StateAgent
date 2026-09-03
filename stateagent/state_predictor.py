"""State predictor — VLM-based ending state prediction.

Given the current MemoryBank (from video observation) and the next shot's prompt,
uses VLM to predict what the world state will look like at the end of that shot.

All reasoning is done by VLM — no symbolic rules.
"""

import json
import logging

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from stateagent.models import (
    EntityMemory,
    EntityType,
    HistoryEvent,
    MemoryBank,
    OpenState,
    PredictionResult,
    Relation,
    VisibilityState,
)
from stateagent.utils import format_memory_for_prompt

logger = logging.getLogger(__name__)

PREDICTOR_SYSTEM_PROMPT = """You are a video state predictor. Your task: given the complete current world state and the next shot's description, predict what the world state should be at the end of that shot.

You should focus on:
1. State changes for each entity (position, visibility, container open/closed state)
2. Changes in relationships between entities
3. Which new entities will appear
4. Which entities will transition from invisible to visible (this is very important for image generation)
5. How each entity's physical state description (state_description) changes:
   - Mixture color changes (blue + yellow -> green)
   - Container content changes (empty -> has liquid, dissolution complete)
   - Surface changes (blank -> has text/stains)
   - Physical form changes (intact -> broken, folded)
   - Quantity/liquid level changes (full cup -> half cup)

Important rules about visibility:
- visibility refers to "whether visible from the current camera perspective", not "whether it physically exists"
- You must consider the camera angle described in the prompt to determine visibility
- For example: in a fixed medium shot, even if a box is opened, if the camera is not shooting from the opening direction, objects inside the box are still hidden
- Only when the prompt explicitly says "show interior", "tilts toward camera", etc., does it mean the camera can see inside the container, and the interior objects become visible
- When an entity is occluded by another entity, it is also hidden

Please return the complete predicted state as JSON."""

PREDICTOR_USER_TEMPLATE = """Current world state:
{current_state_json}

Next shot description: "{prompt}"

Predict the complete world state at the end of this shot. Return JSON format:
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
      "state_description": "predicted entity physical state (contents, surface changes, morphological changes, etc.)"
    }}
  }},
  "entities_becoming_visible": ["entity_id1", "entity_id2"],
  "description": "briefly describe what changes occurred in this shot"
}}

Important:
- entities_becoming_visible includes entities transitioning from hidden or partially_visible to visible (not including newly appearing entities)
- Newly appearing entities should be set to visible directly, no need to add to entities_becoming_visible
- Keep entity_id consistent with the previous state
- Carefully consider camera perspective: container opening does not mean contents are visible to the camera; the shooting angle must be considered"""


class StatePredictor:
    """VLM-based state predictor for ending frame generation."""

    def __init__(self, vlm_client: OpenAI, model: str):
        self.vlm_client = vlm_client
        self.model = model

    def predict(self, current_memory: MemoryBank, prompt: str) -> PredictionResult:
        """Predict the ending state for the next shot.

        Args:
            current_memory: Current world state (from video observation)
            prompt: Description of the next shot

        Returns:
            PredictionResult with predicted MemoryBank and entities_becoming_visible
        """
        logger.info(f"Predicting state: prompt='{prompt}'")

        current_state_json = format_memory_for_prompt(current_memory)

        user_content = PREDICTOR_USER_TEMPLATE.format(
            current_state_json=current_state_json,
            prompt=prompt,
        )

        messages = [
            {"role": "system", "content": PREDICTOR_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        try:
            result = self._call_vlm(messages)
            predicted_memory = self._parse_prediction(result, current_memory)
            entities_becoming_visible = result.get("entities_becoming_visible", [])
            description = result.get("description", "")

            logger.info(
                f"Prediction complete: {len(predicted_memory.entities)} entities, "
                f"{len(entities_becoming_visible)} becoming visible"
            )

            return PredictionResult(
                predicted_memory=predicted_memory,
                entities_becoming_visible=entities_becoming_visible,
                description=description,
            )

        except Exception as e:
            logger.error(f"VLM prediction failed: {e}, returning current state")
            return PredictionResult(
                predicted_memory=current_memory.model_copy(deep=True),
                entities_becoming_visible=[],
                description=f"Prediction failed: {e}",
            )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=30))
    def _call_vlm(self, messages: list[dict]) -> dict:
        """Call VLM with retry. Parse JSON from response text."""
        response = self.vlm_client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.1,
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

    def _parse_prediction(self, result: dict, current_memory: MemoryBank) -> MemoryBank:
        """Parse VLM prediction result into MemoryBank."""
        new_memory = MemoryBank(
            time=current_memory.time + 1,
            entities={},
            history=list(current_memory.history),
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

            # Carry over appearance from current memory if entity exists
            current_entity = current_memory.entities.get(eid)
            appearance_image = None
            size_description = None
            shape_description = None
            state_description = None
            source_shot = None
            if current_entity:
                appearance_image = current_entity.appearance_image
                size_description = current_entity.size_description
                shape_description = current_entity.shape_description
                state_description = current_entity.state_description
                source_shot = current_entity.source_shot

            entity = EntityMemory(
                entity_id=eid,
                type=EntityType(entity_data.get("type", "unknown")),
                attributes=entity_data.get("attributes", {}),
                visibility=VisibilityState(entity_data.get("visibility", "visible")),
                current_location=entity_data.get("current_location"),
                open_state=OpenState(entity_data.get("open_state", "n/a")),
                relations=relations,
                appearance_image=appearance_image,
                size_description=size_description or entity_data.get("size_description"),
                shape_description=shape_description or entity_data.get("shape_description"),
                state_description=entity_data.get("state_description") or state_description,
                source_shot=source_shot,
            )
            new_memory.entities[eid] = entity

        # Record history
        description = result.get("description", "")
        if description:
            new_memory.history.append(HistoryEvent(
                time=new_memory.time,
                description=description,
            ))

        return new_memory
