"""VLM-based frame selector for image editing base frame.

Given the previous video, predicted future state, and action prompt,
extracts multiple candidate keyframes and asks VLM to select the one
closest to the desired future state. If a candidate already satisfies
the future state, editing can be skipped entirely.
"""

import json
import logging
import re

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from stateagent.models import FrameSelectionResult, MemoryBank
from stateagent.utils import encode_image, format_memory_for_prompt

logger = logging.getLogger(__name__)

SELECTOR_SYSTEM_PROMPT = """You are a video frame selector. Your task: given multiple candidate keyframes from a video, the predicted future state, and the action description, select the frame most suitable as the base for image editing.

Selection criteria:
1. Camera angle continuity: the selected frame should be closest to the desired future perspective
2. Object visibility: the visible/hidden state of objects in the frame should be as close as possible to the predicted future state
3. Occlusion relationships: the front-to-back occlusion relationships between objects should be reasonable
4. Background consistency: if the future state involves camera movement, the background should change accordingly

Important rules:
- If a frame already fully satisfies the future state requirements (correct object positions, correct perspective, correct visibility), mark needs_editing=false
- If no frame fully satisfies, select the closest one as the editing base, mark needs_editing=true
- satisfaction_score: 0-10, where 10 means fully satisfied with no editing needed, 0 means very far off

Regarding camera movement:
- If the prompt describes camera movement (e.g., "camera moves forward", "turns right"), you should select the frame that best reflects this movement direction
- Even if that frame's background differs from before, this is correct — camera movement inevitably causes background changes

Please return the selection result in JSON format."""

SELECTOR_USER_TEMPLATE = """## Candidate Frames

There are {num_frames} candidate frames (Frame 1 to Frame {num_frames}), arranged in chronological order.

## Predicted Future State

{predicted_state_json}

## Action Description

"{prompt}"

## Task

Evaluate each candidate frame and select the one most suitable as the editing base.

Return JSON:
{{
  "selected_index": 0,
  "reasoning": "selection reasoning",
  "needs_editing": true,
  "satisfaction_score": 5
}}

Note:
- selected_index is 0-based (0 = Frame 1, 1 = Frame 2, ...)
- If the selected frame already fully satisfies the future state, set needs_editing=false, satisfaction_score=10
- If editing is needed to reach the future state, set needs_editing=true and provide satisfaction_score"""


class FrameSelector:
    """VLM-based frame selection from video history."""

    def __init__(self, vlm_client: OpenAI, model: str):
        self.vlm_client = vlm_client
        self.model = model

    def select_base_frame(
        self,
        video_path: str,
        predicted_memory: MemoryBank,
        prompt: str,
        frames_dir: str,
        num_candidates: int = 6,
    ) -> FrameSelectionResult:
        """Select the best base frame from video for image editing.

        Args:
            video_path: Path to the previous shot's video
            predicted_memory: Predicted future state from StatePredictor
            prompt: Description of the next shot's action
            frames_dir: Directory to save extracted candidate frames
            num_candidates: Number of candidate frames to extract

        Returns:
            FrameSelectionResult with selected frame path and metadata
        """
        from generators.video_utils import extract_keyframes

        logger.info(f"Selecting base frame from {video_path}")

        keyframes = extract_keyframes(
            video_path=video_path,
            output_dir=frames_dir,
            num_frames=num_candidates,
        )

        if not keyframes:
            logger.warning("No keyframes extracted, returning empty result")
            return FrameSelectionResult(
                selected_frame_path="",
                reasoning="No keyframes could be extracted from video",
                needs_editing=True,
                satisfaction_score=0,
            )

        predicted_state_json = format_memory_for_prompt(predicted_memory)

        user_text = SELECTOR_USER_TEMPLATE.format(
            num_frames=len(keyframes),
            predicted_state_json=predicted_state_json,
            prompt=prompt,
        )

        content_parts = []
        for i, frame_info in enumerate(keyframes, 1):
            data_uri = encode_image(frame_info["path"])
            if data_uri:
                content_parts.append({
                    "type": "text",
                    "text": f"[Frame {i} — {frame_info['timestamp']:.1f}s]:",
                })
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": data_uri, "detail": "high"},
                })
            else:
                logger.warning(f"Could not encode frame {i}: {frame_info['path']}")

        content_parts.append({"type": "text", "text": user_text})

        messages = [
            {"role": "system", "content": SELECTOR_SYSTEM_PROMPT},
            {"role": "user", "content": content_parts},
        ]

        try:
            result = self._call_vlm(messages)
            selected_index = result.get("selected_index", 0)
            selected_index = max(0, min(selected_index, len(keyframes) - 1))

            selected_path = keyframes[selected_index]["path"]
            reasoning = result.get("reasoning", "")
            needs_editing = result.get("needs_editing", True)
            satisfaction_score = result.get("satisfaction_score", 0)

            logger.info(
                f"Selected frame {selected_index + 1}/{len(keyframes)} "
                f"(score={satisfaction_score}, needs_editing={needs_editing}): "
                f"{reasoning[:100]}"
            )

            return FrameSelectionResult(
                selected_frame_path=selected_path,
                reasoning=reasoning,
                needs_editing=needs_editing,
                satisfaction_score=satisfaction_score,
            )

        except Exception as e:
            logger.error(f"VLM frame selection failed: {e}, falling back to last frame")
            last_frame = keyframes[-1]["path"]
            return FrameSelectionResult(
                selected_frame_path=last_frame,
                reasoning=f"Selection failed ({e}), using last frame",
                needs_editing=True,
                satisfaction_score=0,
            )

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
