"""Frame spec writer — generates image editing prompt from state diff.

Compares the current ending frame's visible content with the predicted
ending state, and generates a concrete image editing prompt that tells
the image generator exactly how to transform the current frame into
the next shot's ending frame.

The VLM receives:
- The current frame image (labeled as the BASE to edit)
- Reference images of entities becoming visible (for appearance matching)
- Text descriptions of state changes

The editing prompt covers:
1. Entities to REMOVE from the frame (became hidden/occluded)
2. Entities to ADD to the frame (became visible, with reference appearance)
3. Visible entities to MODIFY (position/pose/state changes)
4. Elements to KEEP unchanged (camera, lighting, background, stable entities)
"""

import json
import logging
import re

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from stateagent.models import MemoryBank, VisibilityState
from stateagent.utils import encode_image, format_memory_for_prompt

logger = logging.getLogger(__name__)

WRITER_SYSTEM_PROMPT = """You are an expert at writing image editing prompts for an AI image generation model.

Your task: given numbered images, an action description, and state information, write a natural English description of what the result image should look like.

IMAGE NUMBERING:
- The last image is the current frame, Image N, and must be edited.
- Earlier images are references for entity appearance.
- Always refer to images using their labels: Image 1, Image 2, Image N.

STYLE: Describe the expected result image as if you were describing a photograph to a photographer. Use natural, flowing language rather than rigid edit instructions.

TWO SCENARIOS:

1. When the action involves camera movement, view change, or perspective shift:
   - Describe the entire expected scene from the new viewpoint
   - Do NOT say "background remains the same" — the background MUST change to reflect the new camera angle
   - Example: "The camera has moved forward and turned right, now showing the back of the house. A dog is visible behind the house, standing on the grass. The scene is viewed from behind the house looking at its rear wall, with the garden fence visible in the background."

2. When the action only involves object state changes (no camera movement):
   - Describe the specific changes that should be visible
   - Add: "All other elements (background, lighting, unchanged objects) in Image N remain the same."
   - Example: "The wooden drawer is now pulled open, revealing a red apple inside. The person’s hand is resting on the edge of the drawer. All other elements in Image N remain the same."

IMPORTANT DISTINCTIONS:
- Occlusion is not deletion: if one entity moves in front of another, describe the occluding entity’s new position and note that the rear entity is now hidden behind it
- When referencing an entity’s appearance (color, shape, material), you can refer to its reference image: "matching the appearance in Image M"
- Describe what IS visible in the result, not what was removed

OUTPUT:
Return only a JSON object:
{"image_prompt": "..."}"""

WRITER_USER_TEMPLATE = """## Action Description
"{prompt}"

## Image Layout (..., Image {current_frame_number})
{image_layout}

## Current Frame State (Image {current_frame_number})
Visible entities:
{current_visible_desc}

Hidden entities (not visible in this frame but exist in the scene):
{current_hidden_desc}

## Predicted Next State
{predicted_state_json}

## Task
Describe what the result image should look like after transforming Image {current_frame_number} according to the action above.

Use natural language to describe the expected result scene. Use Image 1, Image 2, etc. to refer to specific reference images.

If the action involves camera movement or perspective change, describe the entire new scene.
If only object states change, describe the changes and note that other elements in Image {current_frame_number} remain the same.

Return ONLY: {{"image_prompt": "..."}}"""


class FrameSpecWriter:
    """Generates image editing prompts with visual context."""

    def __init__(self, vlm_client: OpenAI, model: str):
        self.vlm_client = vlm_client
        self.model = model

    def write(
        self,
        prompt: str,
        previous_memory: MemoryBank,
        predicted_memory: MemoryBank,
        current_frame_path: str,
        reference_images: dict[str, str] | None = None,
        entities_becoming_visible: list[str] | None = None,
    ) -> str:
        """Generate an image editing prompt with visual context.

        Args:
            prompt: The shot/action description
            previous_memory: Current frame state (from video observation)
            predicted_memory: Predicted next frame state
            current_frame_path: Path to the current ending frame (BASE IMAGE)
            reference_images: Dict of entity_id → appearance image path
            entities_becoming_visible: Entity IDs going hidden → visible

        Returns:
            English image editing prompt
        """
        # Build text descriptions
        current_visible_desc = self._describe_visible(previous_memory)
        current_hidden_desc = self._describe_hidden(previous_memory)
        predicted_state_json = format_memory_for_prompt(predicted_memory)

        # Build numbered image list: [(path, label), ...]
        ref_imgs = reference_images or {}
        numbered_images = []

        # Add entity reference images (Image 1, Image 2, ...) — deduplicate by path
        seen_paths: dict[str, int] = {}  # path → index in numbered_images
        for eid, img_path in ref_imgs.items():
            entity = predicted_memory.entities.get(eid) or previous_memory.entities.get(eid)
            attrs = ", ".join(entity.attributes.values()) if entity and entity.attributes else eid
            label = f"{attrs} ({eid})"

            if img_path in seen_paths:
                idx = seen_paths[img_path]
                existing_path, existing_label = numbered_images[idx]
                numbered_images[idx] = (existing_path, f"{existing_label}, {label}")
            else:
                seen_paths[img_path] = len(numbered_images)
                numbered_images.append((img_path, label))

        # Add current frame last (Image N)
        numbered_images.append((current_frame_path, "current frame (to be edited)"))

        current_frame_number = len(numbered_images)

        # Build image layout description
        image_layout_lines = []
        for i, (path, label) in enumerate(numbered_images, 1):
            image_layout_lines.append(f"- Image {i}: {label}")
        image_layout = "\n".join(image_layout_lines)

        user_text = WRITER_USER_TEMPLATE.format(
            prompt=prompt,
            image_layout=image_layout,
            current_frame_number=current_frame_number,
            current_visible_desc=current_visible_desc,
            current_hidden_desc=current_hidden_desc,
            predicted_state_json=predicted_state_json,
        )

        # Build multimodal message content with numbered labels
        content_parts = []
        for i, (img_path, label) in enumerate(numbered_images, 1):
            data_uri = encode_image(img_path)
            if data_uri:
                content_parts.append({
                    "type": "text",
                    "text": f"[Image {i} — {label}]:",
                })
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": data_uri, "detail": "high"},
                })
            else:
                logger.warning(f"Could not encode image {i}: {img_path}")

        # Add the text prompt
        content_parts.append({"type": "text", "text": user_text})

        messages = [
            {"role": "system", "content": WRITER_SYSTEM_PROMPT},
            {"role": "user", "content": content_parts},
        ]

        try:
            image_prompt = self._call_llm(messages)
            if image_prompt:
                logger.info(f"Image prompt: {image_prompt[:120]}...")
                return image_prompt
            else:
                logger.warning("LLM returned empty image_prompt, using fallback")
                return self._fallback_prompt(prompt, previous_memory, predicted_memory)
        except Exception as e:
            logger.warning(f"LLM prompt generation failed: {e}, using fallback")
            return self._fallback_prompt(prompt, previous_memory, predicted_memory)

    def refine(
        self,
        original_prompt: str,
        feedback: str,
        base_frame_path: str,
        predicted_memory: MemoryBank,
    ) -> str:
        """Refine an image editing prompt based on verification feedback.

        Args:
            original_prompt: The original image editing prompt
            feedback: Feedback from EditVerifier about what needs to change
            base_frame_path: Path to the base frame
            predicted_memory: Predicted future state

        Returns:
            Refined image editing prompt
        """
        logger.info(f"Refining prompt based on feedback: {feedback[:100]}")

        predicted_state_json = format_memory_for_prompt(predicted_memory)

        base_uri = encode_image(base_frame_path)

        content_parts = []
        if base_uri:
            content_parts.append({"type": "text", "text": "[Base Image]:"})
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": base_uri, "detail": "high"},
            })

        refine_text = (
            f"## Original Edit Prompt\n\n{original_prompt}\n\n"
            f"## Verification Feedback\n\n{feedback}\n\n"
            f"## Predicted Future State\n\n{predicted_state_json}\n\n"
            f"## Task\n\n"
            f"Please refine the original prompt based on the verification feedback to address the identified issues. "
            f"Maintain a natural language style, describing the expected result scene as if describing a photograph.\n\n"
            f"Return ONLY: {{\"image_prompt\": \"...\"}}"
        )
        content_parts.append({"type": "text", "text": refine_text})

        messages = [
            {"role": "system", "content": WRITER_SYSTEM_PROMPT},
            {"role": "user", "content": content_parts},
        ]

        try:
            refined_prompt = self._call_llm(messages)
            if refined_prompt:
                logger.info(f"Refined prompt: {refined_prompt[:120]}...")
                return refined_prompt
            else:
                logger.warning("Refinement returned empty, using original prompt")
                return original_prompt
        except Exception as e:
            logger.warning(f"Refinement failed: {e}, using original prompt")
            return original_prompt

    # ------------------------------------------------------------------
    # Description helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _describe_visible(memory: MemoryBank) -> str:
        visible = [e for e in memory.entities.values()
                   if e.visibility == VisibilityState.VISIBLE]
        if not visible:
            return "(nothing visible)"
        lines = []
        for e in visible:
            attrs = ", ".join(f"{k}: {v}" for k, v in e.attributes.items()) if e.attributes else ""
            loc = e.current_location or "in the scene"
            line = f"- {e.entity_id} ({e.type.value}): [{attrs}], location: {loc}"
            if e.state_description:
                line += f", state: {e.state_description}"
            lines.append(line)
        return "\n".join(lines)

    @staticmethod
    def _describe_hidden(memory: MemoryBank) -> str:
        hidden = [e for e in memory.entities.values()
                  if e.visibility != VisibilityState.VISIBLE]
        if not hidden:
            return "(nothing hidden)"
        lines = []
        for e in hidden:
            attrs = ", ".join(f"{k}: {v}" for k, v in e.attributes.items()) if e.attributes else ""
            loc = e.current_location or "unknown"
            has_ref = "yes" if e.appearance_image else "no"
            line = f"- {e.entity_id} ({e.type.value}): [{attrs}], location: {loc}, has_reference: {has_ref}"
            if e.state_description:
                line += f", state: {e.state_description}"
            lines.append(line)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # LLM call
    # ------------------------------------------------------------------

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=30))
    def _call_llm(self, messages: list[dict]) -> str:
        response = self.vlm_client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.3,
            extra_body={"enable_thinking": False},
        )
        content = response.choices[0].message.content or ""

        try:
            data = json.loads(content)
            return data.get("image_prompt", "")
        except json.JSONDecodeError:
            pass

        json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', content, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                return data.get("image_prompt", "")
            except json.JSONDecodeError:
                pass

        json_match = re.search(r'\{[^{}]*"image_prompt"[^{}]*\}', content)
        if json_match:
            try:
                data = json.loads(json_match.group())
                return data.get("image_prompt", "")
            except json.JSONDecodeError:
                pass

        if len(content) > 20:
            logger.warning("Could not parse JSON from LLM, using raw text")
            return content.strip()
        return ""

    @staticmethod
    def _fallback_prompt(prompt: str, previous: MemoryBank, predicted: MemoryBank) -> str:
        parts = [f"The scene shows: {prompt}"]

        visible = [e for e in predicted.entities.values()
                   if e.visibility == VisibilityState.VISIBLE]
        if visible:
            descs = []
            for e in visible:
                attrs = " ".join(e.attributes.values())
                descs.append(f"a {attrs} {e.type.value}")
            parts.append(f"Visible: {', '.join(descs)}.")

        parts.append("All other elements in the base image remain the same.")
        return " ".join(parts)
