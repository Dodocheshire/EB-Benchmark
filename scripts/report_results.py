"""Regenerate EXPERIMENT_RESULT.md, data exports and comparison charts locally."""
import argparse
import csv
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MODELS = [
    ("qwen3.8-27b", "Qwen", "default_evaluations_20261002"),
    ("gpt-6.1-sol", "GPT", "codex_evaluations_20261002"),
    ("deepseek-flash", "DeepSeek", "deepseek_evaluations_20261002"),
]
FIELDS = ["reward", "avg_reward", "task_progress", "subgoal_reward", "num_steps", "planner_steps",
          "planner_output_error", "num_invalid_actions", "num_invalid_action_ratio", "episode_elapsed_seconds"]
METRICS = [
    ("success_rate", "Success rate", "Success rate (%)", 100),
    ("mean_task_progress", "Task progress", "Progress (%)", 100),
    ("mean_num_steps", "Executed actions", "Mean actions per episode", 1),
    ("mean_planner_steps", "Planning rounds", "Mean planner rounds per episode", 1),
    ("mean_episode_elapsed_seconds", "Episode duration", "Mean elapsed time (seconds)", 1),
    ("mean_tokens_per_completed_episode", "Token cost", "Mean tokens per completed episode", 1),
]


def mean(values):
    numbers = []
    for value in values:
        if isinstance(value, list) and len(value) == 1:
            value = value[0]
        if isinstance(value, (int, float)) and math.isfinite(value):
            numbers.append(value)
    return sum(numbers) / len(numbers) if numbers else None


def collect():
    rows = []
    for model, _, directory in MODELS:
        report = REPO / "running" / directory
        status = json.loads((report / "status.json").read_text())
        jobs = status if isinstance(status, list) else status["jobs"]
        for job in jobs:
            root = REPO / job["output"]
            episodes, completed, requests = [], set(), []
            for path in sorted(root.rglob("episode_*_res.json")):
                try:
                    episode = json.loads(path.read_text())
                except json.JSONDecodeError:
                    continue  # A live evaluator may still be writing its last file.
                episodes.append(episode)
                completed.add((path.parent.parent.name, int(path.name.split("_")[1])))
            for path in sorted(root.rglob("model_requests.jsonl")):
                for line in path.read_text().splitlines():
                    if not line.strip():
                        continue
                    try:
                        requests.append((path.parent.name, json.loads(line)))
                    except json.JSONDecodeError:
                        continue
            count = len(episodes)
            successes = sum(bool(episode.get("task_success")) for episode in episodes)
            row = {"model": model, "environment": job["environment"], "status": job["status"],
                   "completed_episodes": count, "successful_episodes": successes,
                   "success_rate": successes / count if count else None}
            row.update({"mean_" + field: mean(episode.get(field) for episode in episodes) for field in FIELDS})
            for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
                row[field] = sum((request.get("usage") or {}).get(field, 0) or 0 for _, request in requests) if requests else None
            reasoning = [((request.get("usage") or {}).get("completion_tokens_details") or {}).get("reasoning_tokens")
                         for _, request in requests]
            row["reasoning_tokens"] = sum(value for value in reasoning if value is not None) if any(value is not None for value in reasoning) else None
            tokens = sum((request.get("usage") or {}).get("total_tokens", 0) or 0 for subset, request in requests
                         if (subset, request.get("episode")) in completed)
            row["mean_tokens_per_completed_episode"] = tokens / count if count else None
            rows.append(row)
    return rows


def charts(rows, out, stamp):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False})
    colors = ["#0072B2", "#E69F00", "#009E73"]
    envs = ["eb-nav", "eb-alf", "eb-hab", "eb-man"]
    names = ["Navigation", "ALFRED", "Habitat", "Manipulation"]
    lookup = {(row["model"], row["environment"]): row for row in rows}

    def draw(ax, metric, title, ylabel, scale, small=False):
        positions, width, values = np.arange(len(envs)), .23, []
        for j, (model, _, _) in enumerate(MODELS):
            for k, env in enumerate(envs):
                row = lookup[(model, env)]
                if row[metric] is None:
                    continue
                value = row[metric] * scale
                values.append(value)
                x = positions[k] + (j - 1) * width
                ax.bar(x, value, width, color=colors[j], edgecolor="#333333", linewidth=.6)
        ax.set_xticks(positions, names, fontsize=9 if small else 11)
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
        ax.set_ylim(0, 112 if scale == 100 else (max(values, default=1) * 1.22 or 1))

    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=color) for color in colors]
    labels = ["Qwen", "GPT", "DeepSeek"]
    for metric, title, ylabel, scale in METRICS:
        fig, ax = plt.subplots(figsize=(11, 5.8))
        draw(ax, metric, title, ylabel, scale)
        fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(.5, .97), frameon=False)
        fig.text(.5, .015, stamp, ha="center", fontsize=9, color="#555555")
        fig.subplots_adjust(top=.82, bottom=.17, left=.11, right=.97)
        fig.savefig(out / f"{metric}.png", dpi=180)
        plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(21, 10.5))
    for ax, (metric, title, ylabel, scale) in zip(axes.flat, METRICS):
        draw(ax, metric, title, ylabel, scale, small=True)
    fig.suptitle("EmbodiedBench — Model Comparison", fontsize=20, fontweight="bold", y=.98)
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(.5, .95), frameon=False)
    fig.text(.5, .015, stamp,
             ha="center", fontsize=11, color="#555555")
    fig.subplots_adjust(top=.86, bottom=.11, hspace=.45, wspace=.30)
    fig.savefig(out / "benchmark_overview.png", dpi=180)
    plt.close(fig)


def document(rows, stamp):
    lines = [stamp, ""]
    columns = list(rows[0])
    lines.extend(["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"])
    for row in rows:
        cells = ["" if row[key] is None else (f"{row[key]:.3f}" if isinstance(row[key], float) else str(row[key])) for key in columns]
        lines.append("| " + " | ".join(cells) + " |")
    lines.extend(["", "![关键指标总览](docs/report/benchmark_overview.png)", ""])
    for metric, title, _, _ in METRICS:
        lines.extend([f"![{title}](docs/report/{metric}.png)", ""])
    return "\n".join(lines).rstrip() + "\n"


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S UTC+8")
    rows = collect()
    out = REPO / "docs/report"
    out.mkdir(parents=True, exist_ok=True)
    charts(rows, out, stamp)
    (out / "benchmark_results.json").write_text(json.dumps({"checked_at": stamp, "results": rows}, ensure_ascii=False, indent=2) + "\n")
    with (out / "benchmark_results.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (REPO / "EXPERIMENT_RESULT.md").write_text(document(rows, stamp))
    print(stamp)


if __name__ == "__main__":
    main()
