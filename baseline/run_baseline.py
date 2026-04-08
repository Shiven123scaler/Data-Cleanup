#!/usr/bin/env python3
"""
Baseline Agent — Data Cleaning Environment

A deterministic, rule-based baseline that runs all three task difficulty
levels (easy, medium, hard) and prints the resulting scores.

Usage:
    # Direct Python API (no server needed)
    python baseline/run_baseline.py

    # Against a running FastAPI server
    python baseline/run_baseline.py --api http://localhost:8000

    # Run a specific task only
    python baseline/run_baseline.py --task easy

    # Multiple runs for reproducibility check
    python baseline/run_baseline.py --runs 3
"""

from __future__ import annotations

import argparse
import json
import sys
import os

# Ensure project root is on the path so `app.*` imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.env import DataCleaningEnv
from app.models import Action


# ── Rule-based strategies ────────────────────────────────────────────────

def strategy_easy() -> list[Action]:
    """Easy task: just drop all rows containing nulls."""
    return [
        Action(operation="drop_nulls", column=None, params={}),
    ]


def strategy_medium() -> list[Action]:
    """Medium task: normalise dates, then drop duplicates."""
    return [
        Action(operation="normalize_dates", column="join_date", params={}),
        Action(operation="drop_duplicates", column=None, params={}),
    ]


def strategy_hard() -> list[Action]:
    """Hard task: validate emails, then try status + phone cleaning."""
    return [
        Action(operation="validate_emails", column="email", params={}),
        Action(operation="drop_nulls", column=None, params={}),
        Action(operation="drop_duplicates", column=None, params={}),
        Action(operation="lowercase_column", column="status", params={}),
    ]


STRATEGIES = {
    "easy": strategy_easy,
    "medium": strategy_medium,
    "hard": strategy_hard,
}


# ── Direct Python API runner ─────────────────────────────────────────────

def run_direct(task_id: str, verbose: bool = True) -> dict:
    """Run a single task using the Python API (no server needed)."""
    env = DataCleaningEnv()
    session_id, obs = env.reset(task_id=task_id)

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Task: {task_id.upper()}")
        print(f"  Session: {session_id}")
        print(f"  Initial shape: {obs.schema_info.shape}")
        print(f"  Nulls: {sum(obs.null_count.values())}")
        print(f"  Duplicates: {obs.duplicate_count}")
        print(f"{'='*60}")

    actions = STRATEGIES[task_id]()

    for i, action in enumerate(actions):
        response = env.step(session_id, action)
        reward_info = response.reward_info

        if verbose:
            print(f"  Step {i+1}: {action.operation}"
                  f"{'(' + action.column + ')' if action.column else ''}"
                  f"  →  score={reward_info.score:.4f}"
                  f"  reward={reward_info.reward:+.4f}"
                  f"  done={reward_info.done}")

        if reward_info.done:
            break

    # Get final state
    final_state = env.state(session_id)
    final_score = reward_info.score

    if verbose:
        print(f"  ────────────────────────────────────")
        print(f"  Final Score: {final_score:.4f}")
        print(f"  Final Shape: {final_state['df_shape']}")
        print(f"  Steps Used:  {final_state['step_count']}/{final_state['max_steps']}")

    return {
        "task_id": task_id,
        "final_score": final_score,
        "steps_used": final_state["step_count"],
        "max_steps": final_state["max_steps"],
        "done": final_state["done"],
    }


# ── HTTP API runner ──────────────────────────────────────────────────────

def run_api(task_id: str, base_url: str, verbose: bool = True) -> dict:
    """Run a single task against a running FastAPI server."""
    try:
        import requests
    except ImportError:
        print("ERROR: `requests` package required for --api mode. "
              "Install with: pip install requests")
        sys.exit(1)

    # Reset
    resp = requests.post(f"{base_url}/reset", json={"task_id": task_id})
    resp.raise_for_status()
    reset_data = resp.json()
    obs = reset_data["observation"]

    # Extract session_id from the message
    # Message format: "Episode started. task='easy', session='UUID'"
    msg = reset_data.get("message", "")
    session_id = msg.split("session='")[-1].rstrip("'")

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Task: {task_id.upper()} (API mode)")
        print(f"  Session: {session_id}")
        print(f"  Initial shape: {obs['schema_info']['shape']}")
        print(f"{'='*60}")

    actions = STRATEGIES[task_id]()
    reward_info = None

    for i, action in enumerate(actions):
        payload = {
            "session_id": session_id,
            "action": action.model_dump(),
        }
        resp = requests.post(f"{base_url}/step", json=payload)
        resp.raise_for_status()
        step_data = resp.json()
        reward_info = step_data["reward_info"]

        if verbose:
            print(f"  Step {i+1}: {action.operation}"
                  f"  →  score={reward_info['score']:.4f}"
                  f"  done={reward_info['done']}")

        if reward_info["done"]:
            break

    # Get final state
    resp = requests.get(f"{base_url}/state",
                        params={"session_id": session_id})
    resp.raise_for_status()
    final_state = resp.json()

    final_score = reward_info["score"] if reward_info else 0.0

    if verbose:
        print(f"  ────────────────────────────────────")
        print(f"  Final Score: {final_score:.4f}")
        print(f"  Steps Used:  {final_state['step_count']}/{final_state['max_steps']}")

    return {
        "task_id": task_id,
        "final_score": final_score,
        "steps_used": final_state["step_count"],
        "max_steps": final_state["max_steps"],
        "done": final_state["done"],
    }


# ── Grader validation (direct scoring with task-specific graders) ────────

def run_grader_check(task_id: str, verbose: bool = True) -> float:
    """
    Run the task-specific standalone grader against a baseline-cleaned
    DataFrame, independent of the environment's built-in _grade().
    """
    env = DataCleaningEnv()
    session_id, _ = env.reset(task_id=task_id)

    actions = STRATEGIES[task_id]()
    for action in actions:
        response = env.step(session_id, action)
        if response.reward_info.done:
            break

    # Get the agent's current DataFrame
    ep = env._sessions[session_id]
    agent_df = ep.df.copy()

    # Import the appropriate grader
    if task_id == "easy":
        from app.graders.grader_easy import grade
        score = grade(agent_df)
    elif task_id == "medium":
        from app.graders.grader_medium import grade
        score = grade(agent_df)
    elif task_id == "hard":
        from app.graders.grader_hard import grade
        score = grade(agent_df)
    else:
        score = 0.0

    if verbose:
        print(f"  Grader ({task_id}): {score:.4f}")

    return score


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run the rule-based baseline agent on the Data Cleaning environment."
    )
    parser.add_argument(
        "--task", choices=["easy", "medium", "hard", "all"],
        default="all",
        help="Which task to run (default: all)",
    )
    parser.add_argument(
        "--api", type=str, default=None,
        help="Base URL of running API server (e.g. http://localhost:8000). "
             "If omitted, runs directly via Python API.",
    )
    parser.add_argument(
        "--runs", type=int, default=1,
        help="Number of runs per task (for reproducibility testing).",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-step output.",
    )

    args = parser.parse_args()

    tasks = ["easy", "medium", "hard"] if args.task == "all" else [args.task]
    runner = run_api if args.api else run_direct
    verbose = not args.quiet

    print("\n" + "=" * 60)
    print("  🧹  Data Cleaning Baseline Agent")
    print("  Mode:", "API" if args.api else "Direct (Python API)")
    print("  Tasks:", ", ".join(tasks))
    print("  Runs per task:", args.runs)
    print("=" * 60)

    all_results = []

    for task_id in tasks:
        task_scores = []
        for run_i in range(args.runs):
            if args.runs > 1 and verbose:
                print(f"\n  --- Run {run_i + 1}/{args.runs} ---")

            if args.api:
                result = runner(task_id, args.api, verbose=verbose)
            else:
                result = runner(task_id, verbose=verbose)

            task_scores.append(result["final_score"])
            all_results.append(result)

        if args.runs > 1:
            avg = sum(task_scores) / len(task_scores)
            std = (sum((s - avg) ** 2 for s in task_scores) / len(task_scores)) ** 0.5
            print(f"\n  [{task_id.upper()}] Avg: {avg:.4f}  Std: {std:.4f}  "
                  f"(over {args.runs} runs)")

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  📊  SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Task':<10} {'Score':>8} {'Steps':>8} {'Done':>6}")
    print(f"  {'─'*36}")

    for r in all_results:
        print(f"  {r['task_id']:<10} {r['final_score']:>8.4f} "
              f"{r['steps_used']:>5}/{r['max_steps']:<3} "
              f"{'✅' if r['done'] else '❌':>4}")

    # ── Grader cross-check (only in direct mode) ─────────────────────
    if not args.api:
        print(f"\n{'='*60}")
        print("  🔍  Standalone Grader Cross-Check")
        print(f"{'='*60}")
        for task_id in tasks:
            run_grader_check(task_id, verbose=True)

    print()


if __name__ == "__main__":
    main()
