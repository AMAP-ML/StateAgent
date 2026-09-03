"""VLM-based verifier for generated content.

Runs verification checks against the predicted MemoryBank:
1. SCS (State-Content Satisfaction) — does ending frame match predicted state?
2. Identity — do entities maintain their visual appearance?
3. Visibility — are hidden objects properly hidden?

Each check returns pass/fail/partial/skip independently.
"""

import base64
import json
import logging
import os
import re
from pathlib import Path
from typing import Optional

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from stateagent.models import (
    CheckResult,
    MemoryBank,
    VerificationResult,
    VisibilityState,
)
from stateagent.utils import format_memory_for_prompt

logger = logging.getLogger(__name__)


class Verifier:
    """VLM-based verification of generated video/frame content."""

    def __init__(self, config: dict | None = None, llm_client: Optional[OpenAI] = None):
        self.config = config or {}
        self.client = llm_client
        self.model = self.config.get("vlm_model", os.environ.get("VLM_MODEL", "qwen3.5-plus"))
        self.enabled_checks = set(
            self.config.get("verifier_checks", ["scs", "identity", "visibility"])
        )

    def verify(
        self,
        end_frame_path: str,
        predicted_memory: MemoryBank,
        prompt: str,
    ) -> VerificationResult:
        """Run all enabled verification checks.

        Args:
            end_frame_path: Path to the generated ending frame
            predicted_memory: The predicted state from StatePredictor
            prompt: The shot description

        Returns:
            VerificationResult with individual check results
        """
        result = VerificationResult()

        if self.client is None:
            logger.warning("No LLM client for verification, skipping")
            return result

        # Check if we have a real image
        if not self._is_real_image(end_frame_path):
            logger.info("No real image to verify, skipping verification")
            return result

        if "scs" in self.enabled_checks:
            result.scs = self._check_scs(end_frame_path, predicted_memory, prompt)

        if "identity" in self.enabled_checks:
            result.identity = self._check_identity(end_frame_path, predicted_memory)

        if "visibility" in self.enabled_checks:
            result.visibility = self._check_visibility(end_frame_path, predicted_memory)

        # Aggregate failure reason
        failures = []
        if result.scs == CheckResult.FAIL:
            failures.append("SCS: state not satisfied")
        if result.identity == CheckResult.FAIL:
            failures.append("Identity: appearance mismatch")
        if result.visibility == CheckResult.FAIL:
            failures.append("Visibility: hidden object visible")
        result.failure_reason = "; ".join(failures)

        return result

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, max=10))
    def _check_scs(self, end_frame_path: str, predicted: MemoryBank,
                   prompt: str) -> CheckResult:
        """SCS: Does the ending frame match the predicted state?"""
        state_desc = format_memory_for_prompt(predicted)
        visible = [e for e in predicted.entities.values()
                    if e.visibility == VisibilityState.VISIBLE]
        visible_desc = ", ".join(
            f"{e.entity_id}({', '.join(e.attributes.values())})" for e in visible
        ) or "(none)"

        text_prompt = f"""Look at this image (ending frame of a video shot).

Shot description: "{prompt}"

Expected visible entities: {visible_desc}

Expected state (JSON):
{state_desc}

Does this image accurately reflect the expected state?
Are the right objects visible, in the right positions, with the right attributes?

Respond with JSON: {{"result": "pass" | "fail" | "partial", "reason": "brief explanation"}}"""

        return self._vlm_check(end_frame_path, text_prompt)

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, max=10))
    def _check_identity(self, end_frame_path: str,
                        predicted: MemoryBank) -> CheckResult:
        """Identity: Do entities maintain their visual appearance?"""
        entities_with_appearance = [
            e for e in predicted.entities.values()
            if e.appearance_image and e.visibility == VisibilityState.VISIBLE
        ]
        if not entities_with_appearance:
            return CheckResult.SKIP

        ref_descs = []
        for e in entities_with_appearance:
            attrs = ", ".join(f"{k}={v}" for k, v in e.attributes.items())
            size = e.size_description or "unknown size"
            shape = e.shape_description or "unknown shape"
            ref_descs.append(f"- {e.entity_id}: [{attrs}], {size}, {shape}")

        text_prompt = f"""Look at this image (ending frame of a video shot).

These entities have stored visual references and should maintain their appearance:
{chr(10).join(ref_descs)}

Do the objects in this image match these descriptions?
Same color, material, shape, size?

Respond with JSON: {{"result": "pass" | "fail" | "partial", "reason": "brief explanation"}}"""

        return self._vlm_check(end_frame_path, text_prompt)

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, max=10))
    def _check_visibility(self, end_frame_path: str,
                          predicted: MemoryBank) -> CheckResult:
        """Visibility: Are hidden objects properly hidden?"""
        hidden = [e for e in predicted.entities.values()
                  if e.visibility == VisibilityState.HIDDEN]
        if not hidden:
            return CheckResult.PASS

        hidden_descs = []
        for e in hidden:
            attrs = ", ".join(f"{k}={v}" for k, v in e.attributes.items())
            hidden_descs.append(f"- {e.entity_id}: [{attrs}]")

        text_prompt = f"""Look at this image (ending frame of a video shot).

The following objects should NOT be visible in this image:
{chr(10).join(hidden_descs)}

Are any of these objects visible in the image?

Respond with JSON: {{"result": "pass" | "fail" | "partial", "reason": "brief explanation"}}"""

        return self._vlm_check(end_frame_path, text_prompt)

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def _vlm_check(self, image_path: str, text_prompt: str) -> CheckResult:
        """Run a single VLM-based check with image."""
        try:
            # Read and encode image
            with open(image_path, "rb") as f:
                image_data = f.read()
            image_base64 = base64.b64encode(image_data).decode("utf-8")

            # Determine image type from extension
            ext = Path(image_path).suffix.lower()
            mime = {
                ".png": "image/png",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".webp": "image/webp",
            }.get(ext, "image/png")

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": text_prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime};base64,{image_base64}",
                                    "detail": "high",
                                },
                            },
                        ],
                    }
                ],
                temperature=0.1,
                extra_body={"enable_thinking": False},
            )

            content = response.choices[0].message.content or ""

            # Parse JSON result
            json_match = re.search(r"\{[\s\S]*?\}", content)
            if json_match:
                data = json.loads(json_match.group())
                result_str = data.get("result", "skip").lower()
                try:
                    return CheckResult(result_str)
                except ValueError:
                    return CheckResult.SKIP

            # Fallback: keyword matching
            content_lower = content.lower()
            if "pass" in content_lower:
                return CheckResult.PASS
            elif "fail" in content_lower:
                return CheckResult.FAIL
            elif "partial" in content_lower:
                return CheckResult.PARTIAL
            return CheckResult.SKIP

        except Exception as e:
            logger.warning(f"VLM check failed: {e}")
            return CheckResult.SKIP

    @staticmethod
    def _is_real_image(path: str) -> bool:
        """Check if path points to a real image file."""
        if not path or not os.path.exists(path):
            return False
        valid_ext = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
        if Path(path).suffix.lower() not in valid_ext:
            return False
        try:
            return os.path.getsize(path) > 1000
        except OSError:
            return False
