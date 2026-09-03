"""StateBench evaluation — VLM-based checklist scoring.

Evaluates reveal shot videos against checklist questions.
Computes three metrics:
  - SES (State Equivalence Score): whether the reveal achieves the expected state
  - SCS (State Correctness Score): correctness of tracked entity states (conditioned on SES=1)
  - HR  (Hallucination Rate): fraction of hallucinated entities (conditioned on SES=1)

Usage:
    python evaluate.py                                       # evaluate all StateAgent outputs
    python evaluate.py --video-dir outputs/mymethod          # evaluate another method
    python evaluate.py --ids red_ball_into_blue_box          # specific items
    python evaluate.py --first 2                              # first 2 per difficulty
    python evaluate.py --resume                               # skip already evaluated items
    python evaluate.py --summary-only                         # print metrics from existing results
"""

import argparse
import base64
import json
import logging
import os
import re
import time
from collections import Counter, defaultdict

import requests
try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*args, **kwargs):
        return False

load_dotenv()

logging.basicConfig(level="INFO", format="%(asctime)s %(message)s", datefmt="[%X]")
logger = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, "..", "..")
TASKS_JSON = os.path.join(SCRIPT_DIR, "..", "metadata", "statebench.json")
DEFAULT_VIDEO_DIR = os.path.join(PROJECT_ROOT, "outputs", "stateagent")
EVAL_DIR = os.path.join(PROJECT_ROOT, "eval_results")

DIFFICULTY_LEVELS = ["past_visible", "occluded_process", "complex_transition"]

EVAL_SYSTEM_PROMPT = """You are a video state evaluator. Your task is to watch a video and answer each question with yes or no.

Rules:
- Carefully observe the video content and answer based on what you actually see
- Each question must be answered yes or no, no ambiguity
- Provide a brief reason (one sentence)
- Return only JSON, no additional text"""


# ── Prompt building ──

def _question_text(question: str | dict) -> str:
    if isinstance(question, dict):
        return question.get("question", "")
    return str(question)


def _question_refs(question: str | dict) -> list[str]:
    if isinstance(question, dict):
        return question.get("reference_frame_ids", [])
    return []


def _reference_frame_map(item: dict) -> dict:
    return {ref.get("id"): ref for ref in item.get("reference_frames", [])}


def _build_eval_prompt(item: dict, num_shots: int) -> tuple[dict, str]:
    """Build the checklist evaluation prompt for one item.

    Returns (question_map, user_text) where question_map maps Q-id to metadata.
    """
    reveal_prompt = item["shots"][-1]["prompt"]
    expected_state = item.get("expected_state", "")
    checklist = item.get("checklist", {})

    question_map = {}
    ref_map = _reference_frame_map(item)
    lines = [
        f"## Video Description",
        f'"{reveal_prompt}"',
        "",
    ]

    if expected_state:
        lines.extend([
            f"## Expected Final State",
            f'"{expected_state}"',
            "",
        ])

    if ref_map:
        lines.extend([
            "## Historical reference frames",
            "These frames are from the pre-reveal history and must be used to judge object/person/scene/container consistency.",
        ])
        for ref_id, ref in ref_map.items():
            purpose = ref.get("purpose", "")
            lines.append(f"- {ref_id}: {purpose}")
        lines.append("")

    lines.append("## Evaluation Questions")
    lines.append("")

    q_num = 1

    # reveal_achieved
    reveal_q = checklist.get("reveal_achieved", "")
    reveal_text = _question_text(reveal_q)
    if reveal_text:
        qid = f"Q{q_num}"
        question_map[qid] = {
            "category": "reveal_achieved",
            "reference_frame_ids": _question_refs(reveal_q),
        }
        lines.append(f"### reveal_achieved")
        lines.append(f"{qid}: {reveal_text}")
        lines.append("")
        q_num += 1

    # state_correct
    state_qs = checklist.get("state_correct", [])
    if state_qs:
        lines.append("### state_correct")
        for q in state_qs:
            q_text = _question_text(q)
            if not q_text:
                continue
            qid = f"Q{q_num}"
            q_refs = _question_refs(q)
            question_map[qid] = {
                "category": "state_correct",
                "reference_frame_ids": q_refs,
            }
            ref_hint = f" [refs: {', '.join(q_refs)}]" if q_refs else ""
            lines.append(f"{qid}: {q_text}{ref_hint}")
            q_num += 1
        lines.append("")

    # no_violation
    violation_qs = checklist.get("no_violation", [])
    if violation_qs:
        lines.append("### no_violation")
        for q in violation_qs:
            q_text = _question_text(q)
            if not q_text:
                continue
            qid = f"Q{q_num}"
            q_refs = _question_refs(q)
            question_map[qid] = {
                "category": "no_violation",
                "reference_frame_ids": q_refs,
            }
            ref_hint = f" [refs: {', '.join(q_refs)}]" if q_refs else ""
            lines.append(f"{qid}: {q_text}{ref_hint}")
            q_num += 1
        lines.append("")

    # Build answer template hint
    answer_lines = []
    for qid in question_map:
        answer_lines.append(f'    "{qid}": {{"answer": "yes/no", "reason": "brief reason"}}')
    answer_template = ",\n".join(answer_lines)

    lines.extend([
        "Please answer each question with yes or no, and provide a brief reason.",
        "Return JSON:",
        "{",
        '  "answers": {',
        answer_template,
        "  }",
        "}",
    ])

    user_text = "\n".join(lines)
    return question_map, user_text


# ── Media encoding ──

def _video_to_data_uri(path: str) -> str:
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:video/mp4;base64,{encoded}"


def _image_to_data_uri(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    mime = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}.get(ext, "image/png")
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{encoded}"


# ── Reference frame handling ──

def _resolve_reference_frame_path(ref: dict) -> str | None:
    path = ref.get("path")
    if not path:
        return None
    if os.path.isabs(path):
        return path
    return os.path.join(SCRIPT_DIR, "..", path)


def _used_reference_frame_ids(item: dict) -> set[str]:
    used = set()
    checklist = item.get("checklist", {})
    questions = []
    reveal = checklist.get("reveal_achieved")
    if reveal:
        questions.append(reveal)
    questions.extend(checklist.get("state_correct", []))
    questions.extend(checklist.get("no_violation", []))
    for question in questions:
        if isinstance(question, dict):
            used.update(question.get("reference_frame_ids", []))
    return used


def _reference_frame_content(item: dict) -> list[dict]:
    content = []
    used_ref_ids = _used_reference_frame_ids(item)
    if not used_ref_ids:
        return content

    for ref in item.get("reference_frames", []):
        ref_id = ref.get("id", "reference_frame")
        if ref_id not in used_ref_ids:
            continue
        path = _resolve_reference_frame_path(ref)
        if not path or not os.path.exists(path):
            logger.warning(f"Reference frame missing: {ref_id} ({path})")
            continue
        content.append({"type": "text", "text": f"Reference frame {ref_id}"})
        content.append({"type": "image_url", "image_url": {"url": _image_to_data_uri(path)}})
    return content


# ── VLM API ──

def _get_api_config() -> tuple[str, str, str]:
    api_key = os.environ.get("VLM_KEY") or os.environ.get("DASHSCOPE_API_KEY", "")
    base_url = os.environ.get(
        "VLM_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    model = os.environ.get("VLM_MODEL", "qwen3.5-plus")
    if not api_key:
        raise ValueError("Set VLM_KEY or DASHSCOPE_API_KEY")
    return api_key, base_url, model


def _parse_json_response(content: str) -> dict:
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", content, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    m = re.search(r"\{.*\}", content, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    raise ValueError(f"Cannot parse JSON: {content[:200]}")


def evaluate_item(
    item: dict,
    video_path: str,
    api_key: str,
    base_url: str,
    model: str,
    max_retries: int = 2,
) -> dict:
    """Evaluate one item's reveal video against its checklist."""
    num_shots = len(item["shots"])
    question_map, user_text = _build_eval_prompt(item, num_shots)
    video_uri = _video_to_data_uri(video_path)

    user_content = [{"type": "text", "text": user_text}]
    user_content.extend(_reference_frame_content(item))
    user_content.append({"type": "video_url", "video_url": {"url": video_uri}})

    messages = [
        {"role": "system", "content": EVAL_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    url = f"{base_url}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.1,
        "enable_thinking": False,
    }

    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=180)
            if resp.status_code != 200:
                raise RuntimeError(f"API {resp.status_code}: {resp.text[:200]}")
            data = resp.json()
            content = data["choices"][0]["message"]["content"] or ""
            result = _parse_json_response(content)
            return _score_answers(result, question_map)
        except Exception as e:
            if attempt < max_retries:
                wait = 5 * attempt
                logger.warning(f"Attempt {attempt} failed: {e}, retry in {wait}s")
                time.sleep(wait)
            else:
                logger.error(f"Evaluation failed: {e}")
                return {"error": str(e), "ses": None, "scs": None}


def _score_answers(result: dict, question_map: dict) -> dict:
    """Score parsed answers into SES and SCS."""
    answers = result.get("answers", {})

    ses_pass = None
    state_correct_pass = 0
    state_correct_total = 0
    no_violation_pass = 0
    no_violation_total = 0
    details = {}

    for qid, meta in question_map.items():
        if isinstance(meta, dict):
            category = meta.get("category", "")
            ref_ids = meta.get("reference_frame_ids", [])
        else:
            category = meta
            ref_ids = []

        ans_data = answers.get(qid, {})
        ans = ans_data.get("answer", "").lower().strip()
        reason = ans_data.get("reason", "")
        is_yes = ans in ("yes", "\u662f")
        details[qid] = {
            "category": category,
            "reference_frame_ids": ref_ids,
            "answer": ans,
            "reason": reason,
            "pass": is_yes,
        }

        if category == "reveal_achieved":
            ses_pass = is_yes
        elif category == "state_correct":
            state_correct_total += 1
            if is_yes:
                state_correct_pass += 1
        elif category == "no_violation":
            no_violation_total += 1
            if is_yes:
                no_violation_pass += 1

    sc_total = state_correct_total + no_violation_total
    sc_pass = state_correct_pass + no_violation_pass
    scs = sc_pass / sc_total if sc_total > 0 else None

    return {
        "ses": ses_pass,
        "scs": scs,
        "state_correct_score": state_correct_pass / state_correct_total if state_correct_total else None,
        "no_violation_score": no_violation_pass / no_violation_total if no_violation_total else None,
        "details": details,
    }


def _find_reveal_video(item_dir: str, num_shots: int) -> str | None:
    """Find the reveal (last shot) video file."""
    candidates = [
        f"{num_shots:02d}.mp4",
        f"shot{num_shots}.mp4",
        f"shot{num_shots:02d}.mp4",
        "02.mp4",
        "shot2.mp4",
        "shot02.mp4",
    ]
    for fmt in candidates:
        path = os.path.join(item_dir, fmt)
        if os.path.exists(path):
            return path
    return None


# ── Metrics ──

def _compute_metrics(results: list[dict]) -> dict:
    """Compute aggregate metrics from per-item results."""
    valid = [r for r in results if not r.get("error") and r.get("ses") is not None]
    if not valid:
        return {"n": 0}

    ses_vals = [r["ses"] for r in valid]
    conditioned = [r for r in valid if r["ses"]]

    scs_cond = [r["scs"] for r in conditioned if r.get("scs") is not None]
    sc_cond = [r["state_correct_score"] for r in conditioned if r.get("state_correct_score") is not None]
    nv_cond = [r["no_violation_score"] for r in conditioned if r.get("no_violation_score") is not None]

    metrics = {
        "n": len(valid),
        "n_conditioned": len(conditioned),
        "ses": sum(ses_vals) / len(ses_vals) if ses_vals else 0,
        "scs": sum(scs_cond) / len(scs_cond) if scs_cond else 0,
        "state_correct": sum(sc_cond) / len(sc_cond) if sc_cond else 0,
        "hallucination_rate": 1.0 - (sum(nv_cond) / len(nv_cond)) if nv_cond else 0,
    }

    by_diff = defaultdict(list)
    for r in valid:
        by_diff[r.get("difficulty", "unknown")].append(r)

    diff_metrics = {}
    for diff, diff_results in sorted(by_diff.items()):
        diff_ses = [r["ses"] for r in diff_results]
        diff_cond = [r for r in diff_results if r["ses"]]
        diff_scs = [r["scs"] for r in diff_cond if r.get("scs") is not None]
        diff_sc = [r["state_correct_score"] for r in diff_cond if r.get("state_correct_score") is not None]
        diff_nv = [r["no_violation_score"] for r in diff_cond if r.get("no_violation_score") is not None]
        diff_metrics[diff] = {
            "n": len(diff_results),
            "n_cond": len(diff_cond),
            "ses": sum(diff_ses) / len(diff_ses) if diff_ses else 0,
            "scs": sum(diff_scs) / len(diff_scs) if diff_scs else 0,
            "state_correct": sum(diff_sc) / len(diff_sc) if diff_sc else 0,
            "hallucination_rate": 1.0 - (sum(diff_nv) / len(diff_nv)) if diff_nv else 0,
        }
    metrics["by_difficulty"] = diff_metrics

    return metrics


def _print_metrics(metrics: dict):
    """Print metrics table to console."""
    if metrics.get("n", 0) == 0:
        print("  No valid results to compute metrics.")
        return

    print(f"\n{'━' * 80}")
    print(f"  METRICS (SCS / StateCorrect / Hallucination conditioned on SES=1)")
    print(f"{'━' * 80}")
    print(f"  {'Difficulty':<22} {'N':>4} {'SES':>7} {'SCS':>7} {'StateCorr':>10} {'Halluc':>8}")
    print(f"  {'─' * 76}")

    diff_metrics = metrics.get("by_difficulty", {})
    for level in DIFFICULTY_LEVELS:
        m = diff_metrics.get(level)
        if m:
            print(f"  {level:<20} {m['n']:>4} {m['ses']:>6.1%} {m['scs']:>6.1%} "
                  f"{m['state_correct']:>9.1%} {m['hallucination_rate']:>7.1%}")
        else:
            print(f"  {level:<20} {0:>4} {'—':>6} {'—':>6} {'—':>9} {'—':>7}")

    print(f"  {'─' * 76}")
    print(f"  {'Overall':<20} {metrics['n']:>4} {metrics['ses']:>6.1%} {metrics['scs']:>6.1%} "
          f"{metrics['state_correct']:>9.1%} {metrics['hallucination_rate']:>7.1%}")
    print(f"{'━' * 80}")


# ── Result persistence ──

def _load_existing(output_path: str) -> dict:
    if os.path.exists(output_path):
        with open(output_path, encoding="utf-8") as f:
            results = json.load(f)
        return {r["id"]: r for r in results if not r.get("error")}
    return {}


def _save_results(output_path: str, results_dict: dict):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(list(results_dict.values()), f, indent=2, ensure_ascii=False)


# ── Main ──

def main():
    parser = argparse.ArgumentParser(description="StateBench evaluation")
    parser.add_argument("--tasks", default=TASKS_JSON)
    parser.add_argument("--video-dir", default=None,
                        help=f"Video output directory (default: {DEFAULT_VIDEO_DIR})")
    parser.add_argument("--ids", nargs="*", default=None, help="Specific task IDs")
    parser.add_argument("--first", type=int, default=None,
                        help="First N items per difficulty level")
    parser.add_argument("--resume", action="store_true", default=False,
                        help="Skip items already in the output file")
    parser.add_argument("--summary-only", action="store_true", default=False,
                        help="Print metrics from existing results, no new evaluation")
    parser.add_argument("--output", default=None,
                        help=f"Output JSON path (default: {EVAL_DIR}/eval.json)")
    args = parser.parse_args()

    if args.video_dir is None:
        args.video_dir = DEFAULT_VIDEO_DIR
    if not os.path.isabs(args.video_dir):
        # Resolve relative paths against the project root, not the script
        # directory, so `--video-dir outputs/stateagent` works from anywhere.
        args.video_dir = os.path.normpath(os.path.join(PROJECT_ROOT, args.video_dir))

    output_path = args.output or os.path.join(EVAL_DIR, "eval.json")

    # Load tasks
    with open(args.tasks, encoding="utf-8") as f:
        items = json.load(f)

    if args.ids:
        items = [it for it in items if it["id"] in args.ids]

    if args.first:
        counts = Counter()
        filtered = []
        for it in items:
            if counts[it["difficulty"]] < args.first:
                filtered.append(it)
                counts[it["difficulty"]] += 1
        items = filtered

    # Summary-only: load existing results and print
    if args.summary_only:
        existing = _load_existing(output_path)
        if not existing:
            print(f"No results found: {output_path}")
            return
        print(f"Results: {output_path} ({len(existing)} items)")
        metrics = _compute_metrics(list(existing.values()))
        _print_metrics(metrics)
        return

    api_key, base_url, model = _get_api_config()
    print(f"Model: {model}")
    print(f"Video dir: {args.video_dir}")
    print(f"Output: {output_path}\n")

    # Resume: load existing results
    existing = _load_existing(output_path) if args.resume else {}
    if existing:
        print(f"Resuming: {len(existing)} items already evaluated\n")

    all_results = dict(existing)
    evaluated = 0
    failed = 0

    for i, item in enumerate(items):
        item_id = item["id"]

        if args.resume and item_id in all_results:
            continue

        num_shots = len(item["shots"])
        item_dir = os.path.join(args.video_dir, item_id)
        video_path = _find_reveal_video(item_dir, num_shots)

        if not video_path:
            print(f"  [{i+1}/{len(items)}] {item_id}: video missing, skip")
            continue

        video_size_mb = os.path.getsize(video_path) / (1024 * 1024)
        print(f"  [{i+1}/{len(items)}] {item_id} ({item['difficulty']}): "
              f"{os.path.basename(video_path)} ({video_size_mb:.1f}MB)")

        try:
            result = evaluate_item(item, video_path, api_key, base_url, model)
            result["id"] = item_id
            result["difficulty"] = item["difficulty"]
            result["video"] = video_path
            all_results[item_id] = result
            evaluated += 1

            ses_str = "Y" if result.get("ses") else "N"
            scs_str = f"{result.get('scs', 0):.0%}" if result.get("scs") is not None else "-"
            print(f"    SES: {ses_str}  SCS: {scs_str}")
        except Exception as e:
            logger.error(f"{item_id} failed: {e}")
            print(f"    FAILED: {e}")
            failed += 1

        if evaluated % 10 == 0 and evaluated > 0:
            _save_results(output_path, all_results)

    _save_results(output_path, all_results)
    print(f"\nDone: {evaluated} evaluated, {failed} failed, {len(all_results)} total")

    # Print metrics
    metrics = _compute_metrics(list(all_results.values()))
    _print_metrics(metrics)
    print(f"\nResults: {output_path}")


if __name__ == "__main__":
    main()
