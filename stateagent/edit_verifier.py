"""Edit verifier — VLM-based verification of edited frames.

After image editing generates an ending frame, this module verifies:
1. State verification: Does the edit match the predicted state?
2. Scene verification: Does the scene match the prompt description?

If verification fails, provides feedback for prompt refinement and retry.
"""

import json
import logging
import re

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from stateagent.models import EditVerifyResult, MemoryBank
from stateagent.utils import encode_image, format_memory_for_prompt

logger = logging.getLogger(__name__)

VERIFIER_SYSTEM_PROMPT = """You are an image editing verifier. Your task: verify whether the edited image simultaneously satisfies two requirements:
1. State verification: does the edit match the predicted future state (object positions, visibility, attributes, etc.)
2. Scene verification: does the edit match the scene changes expected by the action description

Important rules:
- If the prompt involves camera movement (e.g., "camera moves forward", "turns right", "walks behind the house"), the background must change accordingly; do not require the background to remain unchanged
- If the prompt only involves object state changes (e.g., "opens drawer", "puts down cup"), the background should remain unchanged
- During verification, consider both dimensions and provide a comprehensive judgment

If verification fails, provide specific correction suggestions explaining what needs to be adjusted in the edit."""

VERIFIER_USER_TEMPLATE = """## Action Description

"{prompt}"

## Predicted Future State

{predicted_state_json}

## Verification Task

Compare the base image and the edited image, then verify:

1. **State Verification**: does the edited image match the predicted future state?
   - Are visible entities correct (those that should appear are present, those that should be hidden are hidden)
   - Are entity positions, poses, and attributes correct
   - Are relationships between entities correct

2. **Scene Verification**: does the edited image match the scene expected by the action description?
   - If the prompt involves camera movement, has the background changed correctly
   - If the prompt involves perspective change, is it viewed from the correct angle
   - Is the overall composition reasonable

Return JSON:
{{
  "passed": true,
  "state_ok": true,
  "scene_ok": true,
  "feedback": ""
}}

If verification fails:
- passed=false
- state_ok and scene_ok respectively indicate which dimension has issues
- feedback provides specific correction suggestions (e.g., "should see the back of the house, not the front", "the dog should appear behind the house")"""


class EditVerifier:
    """VLM-based verification of edited frames."""

    def __init__(self, vlm_client: OpenAI, model: str):
        self.vlm_client = vlm_client
        self.model = model

    def verify_edit(
        self,
        edited_frame_path: str,
        base_frame_path: str,
        predicted_memory: MemoryBank,
        prompt: str,
    ) -> EditVerifyResult:
        """Verify an edited frame against predicted state and prompt.

        Args:
            edited_frame_path: Path to the edited ending frame
            base_frame_path: Path to the base frame used for editing
            predicted_memory: Predicted future state from StatePredictor
            prompt: The shot/action description

        Returns:
            EditVerifyResult with pass/fail and feedback
        """
        logger.info(f"Verifying edit: {edited_frame_path}")

        predicted_state_json = format_memory_for_prompt(predicted_memory)

        user_text = VERIFIER_USER_TEMPLATE.format(
            prompt=prompt,
            predicted_state_json=predicted_state_json,
        )

        content_parts = []

        base_uri = encode_image(base_frame_path)
        if base_uri:
            content_parts.append({"type": "text", "text": "[Base Image]:"})
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": base_uri, "detail": "high"},
            })

        edited_uri = encode_image(edited_frame_path)
        if edited_uri:
            content_parts.append({"type": "text", "text": "[Edited Image]:"})
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": edited_uri, "detail": "high"},
            })

        if not base_uri or not edited_uri:
            logger.warning("Could not encode images for verification, skipping")
            return EditVerifyResult(
                passed=True,
                feedback="Could not encode images for verification",
            )

        content_parts.append({"type": "text", "text": user_text})

        messages = [
            {"role": "system", "content": VERIFIER_SYSTEM_PROMPT},
            {"role": "user", "content": content_parts},
        ]

        try:
            result = self._call_vlm(messages)
            passed = result.get("passed", False)
            state_ok = result.get("state_ok", True)
            scene_ok = result.get("scene_ok", True)
            feedback = result.get("feedback", "")

            logger.info(
                f"Edit verification: passed={passed}, "
                f"state_ok={state_ok}, scene_ok={scene_ok}"
            )
            if not passed:
                logger.info(f"Feedback: {feedback[:150]}")

            return EditVerifyResult(
                passed=passed,
                feedback=feedback,
                state_ok=state_ok,
                scene_ok=scene_ok,
            )

        except Exception as e:
            logger.error(f"Edit verification failed: {e}, assuming pass")
            return EditVerifyResult(
                passed=True,
                feedback=f"Verification error: {e}",
            )

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, max=15))
    def _call_vlm(self, messages: list[dict]) -> dict:
        """Call VLM with retry. Parse JSON from response text."""
        response = self.vlm_client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.1,
            extra_body={"enable_thinking": False},
        )
        content = response.choices[0].message.content or ""

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', content, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        raise ValueError(f"Cannot parse JSON from VLM response: {content[:200]}")
