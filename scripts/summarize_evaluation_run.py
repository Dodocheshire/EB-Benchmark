"""Summarize all subsets of one evaluation, including retries and token usage."""
import argparse
import csv
import json
from pathlib import Path


def read_requests(path):
    """Ignore a final partial write when inspecting a live JSONL file."""
    lines = Path(path).read_text().splitlines()
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            if index != len(lines) - 1:
                raise


def reasoning_usage(requests):
    values = [((r.get("usage") or {}).get("completion_tokens_details") or {}).get("reasoning_tokens")
              for r in requests]
    return {"reasoning_tokens": sum(v for v in values if v is not None),
            "reasoning_usage_reported_requests": sum(v is not None for v in values)}


def summarize(root):
    root = Path(root)
    requests = []
    for path in sorted(root.rglob("model_requests.jsonl")):
        for record in read_requests(path):
            requests.append({**record, "subset": path.parent.name})
    rows = []
    for path in sorted(root.rglob("episode_*_res.json")):
        try:
            metrics = json.loads(path.read_text())
        except json.JSONDecodeError:
            # Evaluators write their result files at the end of each episode.
            continue
        episode = int(path.name.split("_")[1])
        subset = path.parent.parent.name
        calls = [r for r in requests if r["episode"] == episode and r["subset"] == subset]
        elapsed_seconds = metrics.get("episode_elapsed_seconds", 0)
        # Older Habitat results serialize a one-element tuple as a JSON list.
        if isinstance(elapsed_seconds, list) and len(elapsed_seconds) == 1:
            elapsed_seconds = elapsed_seconds[0]
        row = {
            "subset": subset, "episode": episode,
            "instruction": metrics.get("instruction", ""),
            "task_success": metrics.get("task_success", 0),
            "actions": metrics.get("num_steps", 0),
            "planner_steps": metrics.get("planner_steps", 0),
            "planner_output_error": metrics.get("planner_output_error", 0),
            "api_requests": len(calls),
            "empty_responses": sum(not (r.get("message") or {}).get("content") for r in calls),
            "usage_missing_requests": sum(r.get("usage") is None for r in calls),
            "elapsed_seconds": elapsed_seconds,
        }
        action_log = path.parent.parent / f"episode_{episode}.json"
        if action_log.exists():
            actions = list(read_requests(action_log))
            row["invalid_actions"] = sum(not a.get("last_action_success", True) for a in actions)
        else:
            row["invalid_actions"] = metrics.get("num_invalid_actions", sum(not value for value in metrics.get("action_success", [])))
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            row[key] = sum((r.get("usage") or {}).get(key, 0) or 0 for r in calls)
        row.update(reasoning_usage(calls))
        rows.append(row)
    success = sum(bool(row["task_success"]) for row in rows)
    summary = {
        "root": str(root), "completed_episodes": len(rows),
        "successful_episodes": success, "success_rate": success / len(rows) if rows else None,
        "actions": sum(row["actions"] for row in rows),
        "invalid_actions": sum(row["invalid_actions"] for row in rows),
        "planner_output_error": sum(row["planner_output_error"] for row in rows),
        "api_requests": len(requests),
        "empty_responses": sum(not (r.get("message") or {}).get("content") for r in requests),
        "usage_missing_requests": sum(r.get("usage") is None for r in requests),
        "episodes": rows,
    }
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        summary[key] = sum((r.get("usage") or {}).get(key, 0) or 0 for r in requests)
    summary.update(reasoning_usage(requests))
    summary["avg_tokens_per_completed_episode"] = (
        sum(r["total_tokens"] for r in rows) / len(rows) if rows else None)
    root.mkdir(parents=True, exist_ok=True)
    (root / "review_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    if rows:
        with (root / "review_episodes.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    args = parser.parse_args()
    print(json.dumps(summarize(args.root), ensure_ascii=False, indent=2))
