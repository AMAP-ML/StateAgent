"""StateAgent runner — runs StateAgent pipeline on StateBench items.

Reads pre-generated videos from statebench/data/ directory, then uses StateAgent
to generate the final reveal shot and concatenate all videos.

Usage:
    python run_stateagent.py red_ball_into_blue_box   # single item (default)
    python run_stateagent.py --all                    # all items
    python run_stateagent.py --all --skip-existing
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
# Add project root so `stateagent` and `generators` are importable
# regardless of the working directory the script is launched from.
sys.path.append(PROJECT_ROOT)

from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console
from rich.logging import RichHandler

from stateagent.edit_verifier import EditVerifier
from stateagent.frame_selector import FrameSelector
from stateagent.frame_spec_writer import FrameSpecWriter
from stateagent.memory import MemoryBankManager
from stateagent.models import (
    ShotOutput,
    VerificationResult,
)
from stateagent.state_predictor import StatePredictor
from stateagent.verifier import Verifier
from stateagent.video_observer import VideoObserver
from generators import kf2v, image_generator

from run_pipeline import (
    _collect_reference_images,
    _merge_predicted_state,
    _print_memory_summary,
    _print_shot_summary,
    _print_verification,
    _select_best_frames,
    create_llm_client,
    load_config,
    setup_logging,
    vlm_extract_entities,
)

load_dotenv()
console = Console()
logger = logging.getLogger(__name__)

STATEBENCH_DIR = os.path.join(PROJECT_ROOT, "statebench")
TASKS_JSON = os.path.join(STATEBENCH_DIR, "metadata", "statebench.json")
VIDEO_DIR = os.path.join(STATEBENCH_DIR, "data")
DEFAULT_CONFIG = os.path.join(PROJECT_ROOT, "configs", "default.yaml")


def load_items(tasks_file: str, ids: list[str] | None = None) -> list[dict]:
    with open(tasks_file, encoding="utf-8") as f:
        items = json.load(f)
    if ids:
        items = [it for it in items if it["id"] in ids]
    return items


def get_pre_reveal_paths(item_id: str, num_pre_shots: int, video_base: str) -> list[dict]:
    """Get paths to pre-generated videos and endframes."""
    item_dir = os.path.join(video_base, item_id)
    paths = []
    for i in range(1, num_pre_shots + 1):
        video = os.path.join(item_dir, f"shot{i}.mp4")
        endframe = os.path.join(item_dir, f"shot{i}_endframe.png")
        if not os.path.exists(video):
            raise FileNotFoundError(f"Missing pre-reveal video: {video}")
        if not os.path.exists(endframe):
            raise FileNotFoundError(f"Missing pre-reveal endframe: {endframe}")
        paths.append({"shot": i, "video": video, "endframe": endframe})
    return paths


def concat_videos(video_paths: list[str], output_path: str) -> None:
    """Concatenate multiple videos, trimming duplicate boundary frames."""
    n = len(video_paths)
    if n == 1:
        shutil.copy2(video_paths[0], output_path)
        return

    inputs = []
    for vp in video_paths:
        inputs.extend(["-i", vp])

    filter_parts = []
    concat_inputs = ""
    for i in range(n):
        if i == 0:
            filter_parts.append(
                f"[0:v]scale=832:480,setsar=1,fps=24,setpts=PTS-STARTPTS[v{i}]"
            )
        else:
            filter_parts.append(
                f"[{i}:v]select='not(eq(n,0))',scale=832:480,setsar=1,"
                f"fps=24,setpts=PTS-STARTPTS[v{i}]"
            )
        concat_inputs += f"[v{i}]"

    filter_parts.append(f"{concat_inputs}concat=n={n}:v=1:a=0[outv]")
    filter_complex = ";".join(filter_parts)

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        *inputs,
        "-filter_complex", filter_complex,
        "-map", "[outv]",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        output_path,
    ]
    subprocess.run(cmd, check=True)


def run_item(
    item: dict,
    config: dict,
    llm_client: OpenAI,
    model: str,
    output_base: str,
    video_base: str,
) -> None:
    """Run StateAgent on one StateBench item."""
    item_id = item["id"]
    shots = item["shots"]
    num_shots = len(shots)
    pre_reveal_shots = shots[:-1]
    reveal_shot = shots[-1]
    num_pre = len(pre_reveal_shots)

    output_dir = os.path.join(output_base, item_id)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    frames_dir = os.path.join(output_dir, "frames")
    Path(frames_dir).mkdir(parents=True, exist_ok=True)

    # Get pre-reveal video paths
    pre_paths = get_pre_reveal_paths(item_id, num_pre, video_base)

    console.print(f"  Pre-reveal shots: {num_pre}, Reveal: shot {reveal_shot['shot']}")

    # Initialize modules
    manager = MemoryBankManager()
    video_observer = VideoObserver(llm_client, model)
    state_predictor = StatePredictor(llm_client, model)
    frame_writer = FrameSpecWriter(llm_client, model)
    verifier = Verifier(config, llm_client)
    frame_selector_inst = FrameSelector(llm_client, model)
    edit_verifier_inst = EditVerifier(llm_client, model)

    baseline_mode = config.get("baseline_mode", "full")
    all_video_paths = []

    # ── Process pre-reveal shots (existing videos) ──
    for idx, (shot_data, pre_path) in enumerate(zip(pre_reveal_shots, pre_paths)):
        shot_num = shot_data["shot"]
        prompt = shot_data["prompt"]
        video_path = pre_path["video"]
        endframe_path = pre_path["endframe"]

        console.print(f"\n  [yellow]Pre-reveal Shot {shot_num} ({shot_data['role']})[/yellow]")
        console.print(f"    Video: {video_path}")

        # Extract entities from text (first shot only)
        if idx == 0:
            console.print(f"    Extracting entities from text...")
            entities = vlm_extract_entities(llm_client, model, prompt)
            for entity in entities:
                manager.update_entity(entity)

        # Observe endframe
        console.print(f"    Observing endframe...")
        prev_memory = manager.get_memory() if idx > 0 else manager.get_memory()
        observed_memory = video_observer.observe(
            video_frame_path=endframe_path,
            previous_memory=prev_memory,
            prompt=prompt,
        )
        manager.set_memory(observed_memory)
        _print_memory_summary(observed_memory, f"Shot {shot_num} observed")

        # Select best appearance frames from video
        console.print(f"    Selecting entity appearances...")
        _select_best_frames(
            manager=manager,
            video_observer=video_observer,
            video_path=video_path,
            frames_dir=frames_dir,
            shot_id=shot_num,
        )

        # Copy video to output dir
        dst_video = os.path.join(output_dir, f"shot{shot_num}.mp4")
        if os.path.abspath(video_path) != os.path.abspath(dst_video):
            shutil.copy2(video_path, dst_video)
        all_video_paths.append(dst_video)

        prev_endframe = endframe_path
        prev_prompt = prompt

    # ── Generate reveal shot using StateAgent ──
    reveal_shot_num = reveal_shot["shot"]
    reveal_prompt = reveal_shot["prompt"]
    previous_video_path = pre_paths[-1]["video"]

    console.print(f"\n  [bold cyan]Reveal Shot {reveal_shot_num} ({reveal_shot['role']})[/bold cyan]")
    console.print(f"    {reveal_prompt[:80]}...")

    # Step 1: Observe
    console.print("    [yellow]1. Observe[/yellow]")
    observed_memory = video_observer.observe(
        video_frame_path=prev_endframe,
        previous_memory=manager.get_memory(),
        prompt=prev_prompt,
    )
    manager.set_memory(observed_memory)
    _print_memory_summary(observed_memory, "Observed state")

    # Step 2: Predict
    console.print("    [yellow]2. Predict[/yellow]")
    prediction = state_predictor.predict(manager.get_memory(), reveal_prompt)
    _merge_predicted_state(manager, prediction.predicted_memory)
    becoming_visible = prediction.entities_becoming_visible
    console.print(f"    Becoming visible: {becoming_visible}")
    _print_memory_summary(prediction.predicted_memory, "Predicted state")

    # Step 3: Select base frame from previous video
    use_frame_selector = config.get("frame_selector", {}).get("enabled", True)
    if use_frame_selector:
        console.print("    [yellow]3. Select base frame (VLM)[/yellow]")
        num_candidates = config.get("frame_selector", {}).get("num_candidates", 6)
        selection = frame_selector_inst.select_base_frame(
            video_path=previous_video_path,
            predicted_memory=prediction.predicted_memory,
            prompt=reveal_prompt,
            frames_dir=os.path.join(frames_dir, f"shot{reveal_shot_num}_select"),
            num_candidates=num_candidates,
        )
        console.print(f"    Score: {selection.satisfaction_score}/10, needs_editing: {selection.needs_editing}")
        console.print(f"    Reason: {selection.reasoning}")
        base_frame = selection.selected_frame_path
    else:
        console.print("    [yellow]3. Using last frame as base[/yellow]")
        base_frame = prev_endframe
        selection = None

    # Steps 4-6: Conditional image editing
    if selection and not selection.needs_editing:
        console.print("    [green]4-6. Skipping image edit (frame satisfies future state)[/green]")
        end_frame_path = selection.selected_frame_path
        image_prompt = "(no editing needed)"
        ref_images = []
    else:
        console.print("    [yellow]4. Collecting references[/yellow]")
        ref_images = _collect_reference_images(manager, becoming_visible, base_frame)
        console.print(f"    {len(ref_images)} references")

        console.print("    [yellow]5. Image prompt[/yellow]")
        ref_image_dict = {}
        for eid in becoming_visible:
            entity = manager.get_entity(eid)
            if entity and entity.appearance_image:
                ref_image_dict[eid] = entity.appearance_image

        image_prompt = frame_writer.write(
            prompt=reveal_prompt,
            previous_memory=observed_memory,
            predicted_memory=prediction.predicted_memory,
            current_frame_path=base_frame,
            reference_images=ref_image_dict if ref_image_dict else None,
            entities_becoming_visible=becoming_visible,
        )

        # Step 6: Generate + verify loop
        use_edit_verifier = config.get("edit_verifier", {}).get("enabled", True)
        max_retries = config.get("edit_verifier", {}).get("max_retries", 2) if use_edit_verifier else 0
        end_frame_path = os.path.join(output_dir, f"shot{reveal_shot_num}_endframe.png")

        for attempt in range(max_retries + 1):
            console.print(f"    [yellow]6. Generate endframe (attempt {attempt + 1})[/yellow]")
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

            verify_result = edit_verifier_inst.verify_edit(
                edited_frame_path=end_frame_path,
                base_frame_path=base_frame,
                predicted_memory=prediction.predicted_memory,
                prompt=reveal_prompt,
            )
            if verify_result.passed:
                console.print(f"    Edit verified (attempt {attempt + 1})")
                break
            else:
                console.print(f"    Failed: {verify_result.feedback}")
                if attempt < max_retries:
                    image_prompt = frame_writer.refine(
                        original_prompt=image_prompt,
                        feedback=verify_result.feedback,
                        base_frame_path=base_frame,
                        predicted_memory=prediction.predicted_memory,
                    )

    # Step 7: Keyframe-to-Video
    console.print("    [yellow]7. Keyframe-to-Video[/yellow]")
    video_result = kf2v.generate_video(
        first_frame_path=prev_endframe,
        last_frame_path=end_frame_path,
        prompt=reveal_prompt,
        output_dir=output_dir,
        shot_id=reveal_shot_num,
        model=config.get("kf2v_model", "wan2.2-kf2v-flash"),
        resolution="480P",
    )
    reveal_video = video_result["video_path"]
    all_video_paths.append(reveal_video)

    # Step 8: Update appearances
    console.print("    [yellow]8. Update appearances[/yellow]")
    _select_best_frames(
        manager=manager,
        video_observer=video_observer,
        video_path=reveal_video,
        frames_dir=frames_dir,
        shot_id=reveal_shot_num,
    )

    # Step 9: Verification
    verification = VerificationResult()
    if baseline_mode == "full":
        console.print("    [yellow]9. Verification[/yellow]")
        verification = verifier.verify(
            end_frame_path=end_frame_path,
            predicted_memory=prediction.predicted_memory,
            prompt=reveal_prompt,
        )
        _print_verification(verification)

    # Save results
    reveal_output = ShotOutput(
        shot_id=reveal_shot_num,
        prompt=reveal_prompt,
        memory_snapshot=manager.get_memory(),
        image_prompt=image_prompt,
        reference_images=ref_images,
        generated_end_frame=end_frame_path,
        generated_video=reveal_video,
        verification=verification,
    )
    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(reveal_output.model_dump(), f, indent=2, default=str)

    # Concatenate all shots
    console.print("    [yellow]Concatenating all shots[/yellow]")
    combined_path = os.path.join(output_dir, "combined.mp4")
    concat_videos(all_video_paths, combined_path)
    console.print(f"    → {combined_path}")

    # Save metadata
    meta = {
        "id": item_id,
        "difficulty": item.get("difficulty", ""),
        "expected_state": item.get("expected_state", ""),
        "checklist": item.get("checklist", {}),
        "videos": all_video_paths,
        "combined": combined_path,
    }
    with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    _print_shot_summary(reveal_output)


def main():
    parser = argparse.ArgumentParser(description="StateBench StateAgent runner")
    parser.add_argument("ids", nargs="*", default=[], help="Task IDs (e.g. put_ball_on_table, pick_up_cup, etc.)")
    parser.add_argument("--all", action="store_true", help="Run all StateBench items")
    parser.add_argument("--tasks", default=TASKS_JSON,
                        help="StateBench tasks JSON file (default: statebench.json)")
    parser.add_argument("--video-dir", default=None,
                        help="Pre-generated video directory (auto: statebench/data/)")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", default=None,
                        help="Output base dir (default: outputs/stateagent/)")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--first", type=int, default=None,
                        help="Take only first N items per difficulty")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)
    config = load_config(args.config)

    llm_client = create_llm_client(config)
    if llm_client is None:
        raise RuntimeError("DASHSCOPE_API_KEY not set")
    model = config.get("vlm_model", "qwen3.5-plus")

    ids = args.ids if args.ids else None
    if args.all:
        ids = None
    items = load_items(args.tasks, ids)

    if args.first:
        from collections import Counter
        counts = Counter()
        filtered = []
        for it in items:
            key = it["difficulty"]
            if counts[key] < args.first:
                filtered.append(it)
                counts[key] += 1
        items = filtered

    if not items:
        console.print("[red]No items to process[/red]")
        return

    video_base = args.video_dir or VIDEO_DIR
    output_base = args.output_dir or os.path.join(
        config.get("output_dir", "outputs"), "stateagent"
    )
    console.print(f"StateBench runner: {len(items)} items")
    console.print(f"  Tasks: {args.tasks}")
    console.print(f"  Videos: {video_base}")
    console.print(f"  Output: {output_base}")

    ok = 0
    for i, item in enumerate(items, 1):
        item_id = item["id"]
        console.print(f"\n[bold]━━━ [{i}/{len(items)}] {item_id} ({item['difficulty']}) ━━━[/bold]")

        if args.skip_existing:
            combined = os.path.join(output_base, item_id, "combined.mp4")
            if os.path.exists(combined):
                console.print(f"  [dim]Already exists, skipping[/dim]")
                ok += 1
                continue

        try:
            run_item(item, config, llm_client, model, output_base, video_base)
            ok += 1
        except Exception as e:
            logger.error(f"[{item_id}] FAILED: {e}", exc_info=True)
            console.print(f"  [bold red]FAILED: {e}[/bold red]")

    console.print(f"\n[bold]Done: {ok}/{len(items)} succeeded[/bold]")


if __name__ == "__main__":
    main()
