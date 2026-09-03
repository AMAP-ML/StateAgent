"""StateAgent Pipeline Runner.

Main entry point. Implements the closed-loop video generation pipeline:
- Shot 1: Text → VLM entity extraction → text-to-video → select best frame per entity
- Shot 2+: Video observation → VLM prediction → image generation with refs → select frames → Keyframe-to-Video

Usage:
    python run_pipeline.py examples/containment_001.json
    python run_pipeline.py examples/containment_001.json --config configs/default.yaml
"""

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DEFAULT_CONFIG = os.path.join(PROJECT_ROOT, "configs", "default.yaml")
# Add project root so `stateagent` and `generators` are importable
# regardless of the working directory the script is launched from.
sys.path.append(PROJECT_ROOT)

import yaml
from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table
from tenacity import retry, stop_after_attempt, wait_exponential

from stateagent.frame_spec_writer import FrameSpecWriter
from stateagent.frame_selector import FrameSelector
from stateagent.edit_verifier import EditVerifier
from stateagent.memory import MemoryBankManager
from stateagent.models import (
    EntityMemory,
    EntityType,
    MemoryBank,
    PipelineInput,
    ShotOutput,
    VerificationResult,
    VisibilityState,
)
from stateagent.state_predictor import StatePredictor
from stateagent.verifier import Verifier
from stateagent.video_observer import VideoObserver
from generators import text2video, kf2v, image_generator
from generators.video_utils import extract_keyframes

load_dotenv()

console = Console()


# ---------------------------------------------------------------------------
# Logging & Config
# ---------------------------------------------------------------------------

def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )


def load_config(config_path: str = DEFAULT_CONFIG) -> dict:
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    # Environment variables override config values
    env_overrides = {
        "VLM_MODEL": "vlm_model",
        "IMAGE_MODEL": "image_model",
        "TEXT2VIDEO_MODEL": "text2video_model",
        "KF2V_MODEL": "kf2v_model",
        "I2V_MODEL": "i2v_model",
        "VIDEO_SIZE": "video_size",
    }
    for env_key, config_key in env_overrides.items():
        if os.environ.get(env_key):
            config[config_key] = os.environ[env_key]
    # Resolve relative output_dir against project root, not the current
    # working directory, so outputs land in the same place regardless of cwd.
    output_dir = config.get("output_dir", "outputs")
    if not os.path.isabs(output_dir):
        config["output_dir"] = os.path.join(PROJECT_ROOT, output_dir)
    return config


def create_llm_client(config: dict) -> OpenAI | None:
    # VLM_KEY takes priority, fall back to DASHSCOPE_API_KEY
    api_key = os.environ.get("VLM_KEY") or os.environ.get("DASHSCOPE_API_KEY", "")
    base_url = os.environ.get(
        "VLM_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    if not api_key:
        logging.getLogger(__name__).warning(
            "DASHSCOPE_API_KEY not set — LLM modules disabled"
        )
        return None
    return OpenAI(api_key=api_key, base_url=base_url)


# ---------------------------------------------------------------------------
# VLM entity extraction (for Shot 1)
# ---------------------------------------------------------------------------

EXTRACT_SYSTEM_PROMPT = """You are an entity extractor. Extract all entities from the video shot description.
Only extract entity information, do not predict states."""

EXTRACT_USER_TEMPLATE = """Extract all entities from the following video description:

"{prompt}"

Return JSON:
{{
  "entities": [
    {{
      "entity_id": "short descriptive ID, e.g. person_1, marble_1, box_1",
      "type": "person|object|container|clothing|unknown",
      "attributes": {{"color": "...", "material": "...", "shape": "...", "size": "..."}}
    }}
  ]
}}

Rules:
- Use {{type}}_{{number}} format for entity_id
- Include all mentioned entities, including people
- Only fill in attributes that are explicitly mentioned"""


def _strip_think_tags(content: str) -> str:
    """Strip <think>...</think> tags from model output (qwen3.5-plus thinking mode)."""
    return re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL | re.IGNORECASE).strip()


def _parse_json_response(content: str) -> dict:
    """Parse JSON from LLM response text, handling various formats."""
    # Strip thinking tags first (qwen3.5-plus may include <think>...</think>)
    content = _strip_think_tags(content)

    # Try direct JSON parse
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown code block
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

    raise ValueError(f"Cannot parse JSON from response: {content[:200]}")


def vlm_extract_entities(llm_client: OpenAI, model: str, prompt: str) -> list[EntityMemory]:
    """Extract entities from text prompt using VLM (for Shot 1 initialization)."""
    try:
        entities = _vlm_extract_entities_with_retry(llm_client, model, prompt)
        console.print(f"     Extracted {len(entities)} entities: "
                      f"{[e.entity_id for e in entities]}")
        if not entities:
            console.print("     [bold red]WARNING: No entities extracted from prompt![/bold red]")
        return entities
    except Exception as e:
        logging.getLogger(__name__).error(f"Entity extraction failed after retries: {e}")
        console.print(f"     [bold red]ERROR: Entity extraction failed: {e}[/bold red]")
        return []


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=30))
def _vlm_extract_entities_with_retry(llm_client: OpenAI, model: str, prompt: str) -> list[EntityMemory]:
    """Internal: call VLM with retry to extract entities.

    Uses raw requests instead of OpenAI SDK to avoid compatibility issues
    with thinking models (qwen3.5-plus) that return non-standard response formats.
    """
    import requests as req
    from generators.dashscope_task import DASHSCOPE_BASE

    logger = logging.getLogger(__name__)

    api_key = os.environ.get("VLM_KEY") or os.environ.get("DASHSCOPE_API_KEY", "")
    base_url = os.environ.get("VLM_URL", f"{DASHSCOPE_BASE}/compatible-mode/v1")
    url = f"{base_url}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": EXTRACT_SYSTEM_PROMPT},
            {"role": "user", "content": EXTRACT_USER_TEMPLATE.format(prompt=prompt)},
        ],
        "temperature": 0.1,
        "enable_thinking": False,
    }

    resp = req.post(url, json=payload, headers=headers, timeout=120)
    if resp.status_code != 200:
        raise RuntimeError(f"VLM API returned {resp.status_code}: {resp.text[:200]}")

    result = resp.json()

    # Extract content from response — handle both standard and thinking model formats
    choices = result.get("choices", [])
    if not choices:
        raise ValueError(f"No choices in VLM response")

    message = choices[0].get("message", {})
    content = message.get("content") or message.get("reasoning_content") or ""
    if not content:
        raise ValueError(f"Empty content in VLM response")

    data = _parse_json_response(content)

    entities = []
    for e in data.get("entities", []):
        if not isinstance(e, dict):
            logger.warning(f"Skipping non-dict entity: {e}")
            continue

        # entity_id is required — skip if missing
        eid = e.get("entity_id")
        if not eid:
            logger.warning(f"Skipping entity with missing entity_id: {e}")
            continue

        # type must be a valid EntityType — fall back to UNKNOWN
        raw_type = e.get("type") or "unknown"
        try:
            entity_type = EntityType(raw_type)
        except ValueError:
            logger.warning(f"Unknown entity type '{raw_type}' for {eid}, using UNKNOWN")
            entity_type = EntityType.UNKNOWN

        entities.append(EntityMemory(
            entity_id=eid,
            type=entity_type,
            attributes=e.get("attributes") or {},
            visibility=VisibilityState.VISIBLE,
            source_shot=1,
        ))

    return entities


# ---------------------------------------------------------------------------
# Pipeline core
# ---------------------------------------------------------------------------

def run_pipeline(
    input_path: str,
    config_path: str = DEFAULT_CONFIG,
    existing_video: str | None = None,
    existing_endframe: str | None = None,
) -> list[ShotOutput]:
    """Run the full StateAgent pipeline."""
    config = load_config(config_path)
    llm_client = create_llm_client(config)
    if llm_client is None:
        raise RuntimeError(
            "DASHSCOPE_API_KEY not set. "
            "Please set it as an environment variable: export DASHSCOPE_API_KEY=sk-xxx"
        )
    model = config.get("vlm_model", "qwen3.5-plus")

    with open(input_path, encoding="utf-8") as f:
        pipeline_input = PipelineInput.model_validate(json.load(f))

    if pipeline_input.previous_end_frame and not os.path.isabs(pipeline_input.previous_end_frame):
        input_dir = os.path.dirname(os.path.abspath(input_path))
        pipeline_input.previous_end_frame = os.path.join(input_dir, pipeline_input.previous_end_frame)

    console.print(Panel(
        f"[bold]Sample: {pipeline_input.sample_id}[/bold]\n"
        f"Shots: {len(pipeline_input.prompts)}",
        title="StateAgent Pipeline",
    ))

    # Initialize modules
    manager = MemoryBankManager()
    state_predictor = StatePredictor(llm_client, model)
    video_observer = VideoObserver(llm_client, model)
    frame_writer = FrameSpecWriter(llm_client, model)
    verifier = Verifier(config, llm_client)
    frame_selector = FrameSelector(llm_client, model)
    edit_verifier = EditVerifier(llm_client, model)

    baseline_mode = config.get("baseline_mode", "full")
    output_dir = os.path.join(config.get("output_dir", "outputs"), pipeline_input.sample_id)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    frames_dir = os.path.join(output_dir, "frames")
    Path(frames_dir).mkdir(parents=True, exist_ok=True)

    results: list[ShotOutput] = []
    previous_frame = pipeline_input.previous_end_frame
    previous_video_path = None

    for t, prompt in enumerate(pipeline_input.prompts):
        shot_id = t + 1
        console.print(f"\n[bold cyan]━━━ Shot {shot_id}/{len(pipeline_input.prompts)} ━━━[/bold cyan]")
        console.print(f"[dim]{prompt}[/dim]\n")

        if shot_id == 1:
            shot_output = _process_first_shot(
                prompt=prompt,
                manager=manager,
                llm_client=llm_client,
                model=model,
                video_observer=video_observer,
                output_dir=output_dir,
                frames_dir=frames_dir,
                config=config,
                existing_video=existing_video,
                existing_endframe=existing_endframe,
            )
        else:
            shot_output = _process_subsequent_shot(
                shot_id=shot_id,
                prompt=prompt,
                prev_prompt=pipeline_input.prompts[t - 1],
                manager=manager,
                state_predictor=state_predictor,
                video_observer=video_observer,
                frame_writer=frame_writer,
                verifier=verifier,
                frame_selector=frame_selector,
                edit_verifier=edit_verifier,
                previous_frame=previous_frame,
                previous_video_path=previous_video_path,
                baseline_mode=baseline_mode,
                config=config,
                output_dir=output_dir,
                frames_dir=frames_dir,
            )

        results.append(shot_output)
        previous_frame = shot_output.generated_end_frame
        previous_video_path = shot_output.generated_video
        _print_shot_summary(shot_output)

    # Save results
    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump([r.model_dump() for r in results], f, indent=2, default=str)

    # Save final memory state
    memory_path = os.path.join(output_dir, "final_memory.json")
    with open(memory_path, "w", encoding="utf-8") as f:
        json.dump(manager.get_memory().model_dump(), f, indent=2, default=str)

    console.print(f"\n[bold green]Results saved to: {output_dir}[/bold green]")
    return results


# ---------------------------------------------------------------------------
# Shot 1: Direct flow
# ---------------------------------------------------------------------------

def _process_first_shot(
    prompt: str,
    manager: MemoryBankManager,
    llm_client: OpenAI,
    model: str,
    video_observer: VideoObserver,
    output_dir: str,
    frames_dir: str,
    config: dict,
    existing_video: str | None = None,
    existing_endframe: str | None = None,
) -> ShotOutput:
    """Shot 1: VLM extract entities → text-to-video → select best frame per entity.

    If existing_video is provided, skips T2V generation and uses the existing
    video/endframe instead (continuation mode).
    """

    # Step 1: VLM extract entities from text
    console.print("  [yellow]1. Extracting entities from text[/yellow]")
    entities = vlm_extract_entities(llm_client, model, prompt)
    if not entities:
        console.print("  [bold red]⚠ WARNING: No entities extracted! Memory and appearance tracking will be empty for Shot 1.[/bold red]")
    for entity in entities:
        manager.update_entity(entity)

    # Step 2: Generate video or use existing
    if existing_video:
        console.print("  [yellow]2. Using existing video (skip T2V)[/yellow]")
        video_path = existing_video
        if existing_endframe:
            end_frame_path = existing_endframe
        else:
            from generators.video_utils import extract_ending_frame
            end_frame_path = os.path.join(output_dir, "01_endframe.png")
            extract_ending_frame(video_path, end_frame_path)
        console.print(f"     video: {video_path}")
        console.print(f"     endframe: {end_frame_path}")
    else:
        console.print("  [yellow]2. Generating video (text-to-video)[/yellow]")
        video_result = text2video.generate_video(
            prompt=prompt,
            output_dir=output_dir,
            shot_id=1,
            model=config.get("text2video_model", "wan2.2-t2v-plus"),
            size=config.get("video_size", "832*480"),
        )
        video_path = video_result["video_path"]
        end_frame_path = video_result["end_frame_path"]

    # Step 3: Extract keyframes and select best frame per entity
    console.print("  [yellow]3. Extracting keyframes and selecting entity appearances[/yellow]")
    _select_best_frames(
        manager=manager,
        video_observer=video_observer,
        video_path=video_path,
        frames_dir=frames_dir,
        shot_id=1,
    )

    return ShotOutput(
        shot_id=1,
        prompt=prompt,
        memory_snapshot=manager.get_memory(),
        image_prompt=prompt,
        reference_images=[],
        generated_end_frame=end_frame_path,
        generated_video=video_path,
        verification=VerificationResult(),
    )


# ---------------------------------------------------------------------------
# Shot 2+: Full closed-loop flow
# ---------------------------------------------------------------------------

def _process_subsequent_shot(
    shot_id: int,
    prompt: str,
    prev_prompt: str,
    manager: MemoryBankManager,
    state_predictor: StatePredictor,
    video_observer: VideoObserver,
    frame_writer: FrameSpecWriter,
    verifier: Verifier,
    frame_selector: FrameSelector,
    edit_verifier: EditVerifier,
    previous_frame: str,
    previous_video_path: str | None,
    baseline_mode: str,
    config: dict,
    output_dir: str,
    frames_dir: str,
) -> ShotOutput:
    """Shot 2+: observe → predict → select frame → generate with refs → verify → Keyframe-to-Video."""

    # Step 1: Video observation (VLM looks at previous ending frame)
    console.print("  [yellow]1. Video observation (VLM observes previous frame)[/yellow]")
    observed_memory = video_observer.observe(
        video_frame_path=previous_frame,
        previous_memory=manager.get_memory(),
        prompt=prev_prompt,
    )
    manager.set_memory(observed_memory)

    _print_memory_summary(observed_memory, "Observed state")

    # Step 2: VLM prediction (predict ending state)
    console.print("  [yellow]2. VLM state prediction[/yellow]")
    prediction = state_predictor.predict(manager.get_memory(), prompt)

    # Update memory with predicted state (but keep appearance images)
    _merge_predicted_state(manager, prediction.predicted_memory)

    becoming_visible = prediction.entities_becoming_visible
    console.print(f"     Entities becoming visible: {becoming_visible}")
    console.print(f"     Description: {prediction.description}")

    _print_memory_summary(prediction.predicted_memory, "Predicted state")

    # Step 3: Select best base frame from video history (VLM)
    use_frame_selector = (
        config.get("frame_selector", {}).get("enabled", True)
        and previous_video_path
    )
    if use_frame_selector:
        console.print("  [yellow]3. Selecting best base frame (VLM)[/yellow]")
        num_candidates = config.get("frame_selector", {}).get("num_candidates", 6)
        selection = frame_selector.select_base_frame(
            video_path=previous_video_path,
            predicted_memory=prediction.predicted_memory,
            prompt=prompt,
            frames_dir=os.path.join(frames_dir, f"shot{shot_id}_select"),
            num_candidates=num_candidates,
        )
        console.print(f"     Selected: {selection.selected_frame_path}")
        console.print(f"     Score: {selection.satisfaction_score}/10, needs_editing: {selection.needs_editing}")
        console.print(f"     Reason: {selection.reasoning}")
        base_frame = selection.selected_frame_path
    else:
        console.print("  [yellow]3. Using last frame as base (frame selector disabled)[/yellow]")
        base_frame = previous_frame
        selection = None

    # Steps 4-6: Conditional image editing
    if selection and not selection.needs_editing:
        # Selected frame already satisfies future state — skip editing
        console.print("  [green]4-6. Skipping image edit (frame already satisfies future state)[/green]")
        end_frame_path = selection.selected_frame_path
        image_prompt = "(no editing needed)"
        ref_images = []
    else:
        # Need editing: collect refs → generate prompt → generate image → verify loop
        console.print("  [yellow]4. Collecting reference images[/yellow]")
        ref_images = _collect_reference_images(manager, becoming_visible, base_frame)
        console.print(f"     {len(ref_images)} reference images collected")
        for i, img in enumerate(ref_images):
            label = "previous_frame" if i == len(ref_images) - 1 else f"entity_ref_{i}"
            console.print(f"       [{label}]: {img}")

        console.print("  [yellow]5. Generating image prompt[/yellow]")
        ref_image_dict = {}
        for eid in becoming_visible:
            entity = manager.get_entity(eid)
            if entity and entity.appearance_image:
                ref_image_dict[eid] = entity.appearance_image

        image_prompt = frame_writer.write(
            prompt=prompt,
            previous_memory=observed_memory,
            predicted_memory=prediction.predicted_memory,
            current_frame_path=base_frame,
            reference_images=ref_image_dict if ref_image_dict else None,
            entities_becoming_visible=becoming_visible,
        )

        # Step 6: Generate + verify loop
        use_edit_verifier = config.get("edit_verifier", {}).get("enabled", True)
        max_retries = config.get("edit_verifier", {}).get("max_retries", 2) if use_edit_verifier else 0
        end_frame_filename = f"shot{shot_id}_endframe.png"
        end_frame_path = os.path.join(output_dir, end_frame_filename)

        for attempt in range(max_retries + 1):
            console.print(f"  [yellow]6. Generating ending frame (attempt {attempt + 1})[/yellow]")
            image_generator.generate_image(
                prompt=image_prompt,
                reference_images=ref_images,
                output_path=end_frame_path,
                model=config.get("image_model", "wan2.7-image"),
                size=config.get("image_size", "1K"),
                target_size=config.get("image_target_size", "832*480"),
            )

            if not use_edit_verifier:
                break

            verify_result = edit_verifier.verify_edit(
                edited_frame_path=end_frame_path,
                base_frame_path=base_frame,
                predicted_memory=prediction.predicted_memory,
                prompt=prompt,
            )
            if verify_result.passed:
                console.print(f"     Edit verified (attempt {attempt + 1})")
                break
            else:
                console.print(f"     Verification failed: {verify_result.feedback}")
                if attempt < max_retries:
                    image_prompt = frame_writer.refine(
                        original_prompt=image_prompt,
                        feedback=verify_result.feedback,
                        base_frame_path=base_frame,
                        predicted_memory=prediction.predicted_memory,
                    )
                    console.print(f"     Refined prompt: {image_prompt[:80]}...")

    # Step 7: Generate video (Keyframe-to-Video) — first_frame stays as previous_frame for continuity
    console.print("  [yellow]7. Generating video (Keyframe-to-Video)[/yellow]")
    video_result = kf2v.generate_video(
        first_frame_path=previous_frame,
        last_frame_path=end_frame_path,
        prompt=prompt,
        output_dir=output_dir,
        shot_id=shot_id,
        model=config.get("kf2v_model", "wan2.2-kf2v-flash"),
        resolution="480P",
    )

    # Step 8: Update entity appearances from generated video
    console.print("  [yellow]8. Updating entity appearances from video[/yellow]")
    video_path = video_result["video_path"]
    _select_best_frames(
        manager=manager,
        video_observer=video_observer,
        video_path=video_path,
        frames_dir=frames_dir,
        shot_id=shot_id,
    )

    # Step 9: Verification (optional, only in full mode)
    verification = VerificationResult()
    if baseline_mode == "full":
        console.print("  [yellow]9. Verification[/yellow]")
        verification = verifier.verify(
            end_frame_path=end_frame_path,
            predicted_memory=prediction.predicted_memory,
            prompt=prompt,
        )
        _print_verification(verification)

    return ShotOutput(
        shot_id=shot_id,
        prompt=prompt,
        memory_snapshot=manager.get_memory(),
        image_prompt=image_prompt,
        reference_images=ref_images,
        generated_end_frame=end_frame_path,
        generated_video=video_result["video_path"],
        verification=verification,
    )


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _select_best_frames(
    manager: MemoryBankManager,
    video_observer: VideoObserver,
    video_path: str,
    frames_dir: str,
    shot_id: int,
) -> None:
    """Extract keyframes from video and select best frame per entity.

    For each entity, finds the keyframe where it's most clearly visible
    and stores that frame as the entity's appearance_image.
    """
    entities = list(manager.get_all_entities().values())
    if not entities:
        return

    entity_ids = [e.entity_id for e in entities]
    console.print(f"     Analyzing {len(entity_ids)} entities across keyframes")

    # Extract keyframes from video
    keyframes = extract_keyframes(
        video_path=video_path,
        output_dir=os.path.join(frames_dir, f"shot{shot_id}"),
        num_frames=6,
    )

    if not keyframes:
        console.print("     [red]No keyframes extracted, skipping appearance update[/red]")
        return

    # Analyze visibility in each keyframe
    best_frame_per_entity: dict[str, str] = {}
    for frame_info in keyframes:
        frame_path = frame_info["path"]
        timestamp = frame_info["timestamp"]

        visibility = video_observer.analyze_frame_visibility(frame_path, entity_ids)

        for eid, is_visible in visibility.items():
            if is_visible and eid not in best_frame_per_entity:
                # First frame where this entity is clearly visible
                best_frame_per_entity[eid] = frame_path

        console.print(
            f"     [dim]{timestamp:.1f}s: "
            f"{sum(1 for v in visibility.values() if v)}/{len(entity_ids)} visible[/dim]"
        )

    # Update entity appearance images
    updated_count = 0
    for eid, frame_path in best_frame_per_entity.items():
        entity = manager.get_entity(eid)
        if entity:
            entity.appearance_image = frame_path
            updated_count += 1

    console.print(
        f"     [cyan]Selected best frames for {updated_count}/{len(entity_ids)} entities[/cyan]"
    )


def _merge_predicted_state(manager: MemoryBankManager, predicted: MemoryBank) -> None:
    """Merge predicted state into manager, preserving appearance images."""
    for eid, predicted_entity in predicted.entities.items():
        current = manager.get_entity(eid)
        if current:
            # Update state fields, keep appearance
            current.visibility = predicted_entity.visibility
            current.current_location = predicted_entity.current_location
            current.open_state = predicted_entity.open_state
            current.relations = predicted_entity.relations
            current.attributes.update(predicted_entity.attributes)
        else:
            # New entity from prediction
            manager.update_entity(predicted_entity)


def _collect_reference_images(
    manager: MemoryBankManager,
    becoming_visible: list[str],
    previous_frame: str,
) -> list[str]:
    """Collect reference images for ending frame generation.

    Returns ordered list with duplicates removed:
      [entity_crop_1, ..., entity_crop_N, previous_frame]
    The last item is always previous_frame (maintains resolution).
    """
    ref_images = []
    seen = set()

    # Entity crops for entities becoming visible
    for eid in becoming_visible:
        entity = manager.get_entity(eid)
        if entity and entity.appearance_image:
            if entity.appearance_image not in seen:
                ref_images.append(entity.appearance_image)
                seen.add(entity.appearance_image)

    # Last reference: previous ending frame (maintains resolution)
    if previous_frame and previous_frame not in seen:
        ref_images.append(previous_frame)

    return ref_images


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _print_memory_summary(memory: MemoryBank, title: str = "State") -> None:
    for eid, entity in memory.entities.items():
        attrs = ", ".join(f"{k}={v}" for k, v in entity.attributes.items())
        has_img = "🖼" if entity.appearance_image else " "
        console.print(
            f"     [green]{eid}[/green]{has_img}: [{attrs}] "
            f"vis={entity.visibility.value} loc={entity.current_location or 'none'}"
        )
    for eid, entity in memory.entities.items():
        for r in entity.relations:
            if r.value and r.subject == eid:
                console.print(f"     [blue]{r.type}[/blue]({r.subject}, {r.object})")


def _print_shot_summary(shot_output: ShotOutput) -> None:
    table = Table(title=f"Shot {shot_output.shot_id} Summary", show_header=True)
    table.add_column("Field", style="bold")
    table.add_column("Value")

    table.add_row("Prompt", shot_output.prompt[:80])
    table.add_row("Entities", ", ".join(shot_output.memory_snapshot.entities.keys()))
    table.add_row("Ref Images", str(len(shot_output.reference_images)))
    table.add_row("End Frame", shot_output.generated_end_frame or "(none)")
    table.add_row("Video", shot_output.generated_video or "(none)")

    v = shot_output.verification
    checks = f"SCS={v.scs.value} ID={v.identity.value} VIS={v.visibility.value}"
    table.add_row("Verification", checks)

    console.print(table)


def _print_verification(result: VerificationResult) -> None:
    for check_name in ["scs", "identity", "visibility"]:
        value = getattr(result, check_name)
        color = {"pass": "green", "fail": "red", "partial": "yellow", "skip": "dim"}.get(
            value.value, "white"
        )
        console.print(f"     [{color}]{check_name.upper()}: {value.value}[/{color}]")
    if result.failure_reason:
        console.print(f"     [red]Failure: {result.failure_reason}[/red]")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="StateAgent Pipeline"
    )
    parser.add_argument("input", help="Path to input JSON file")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--existing-video", default=None,
                        help="Path to existing Shot 1 video (skip T2V generation)")
    parser.add_argument("--existing-endframe", default=None,
                        help="Path to existing Shot 1 endframe (auto-extracted if omitted)")
    args = parser.parse_args()
    setup_logging(args.log_level)
    run_pipeline(args.input, args.config,
                 existing_video=args.existing_video,
                 existing_endframe=args.existing_endframe)


if __name__ == "__main__":
    main()
