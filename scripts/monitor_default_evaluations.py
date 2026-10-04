"""Hourly checks of the four evaluation jobs; never sends model requests."""
import argparse
import ast
import csv
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import socket
import time

from summarize_evaluation_run import read_requests, summarize

REPO = Path(__file__).resolve().parents[1]
REPORT = REPO / "running/default_evaluations_20261002"


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)


def evaluator_pids(name, config_path=None):
    pids = []
    for process in Path("/proc").iterdir():
        if not process.name.isdigit():
            continue
        try:
            args = (process / "cmdline").read_bytes().split(b"\0")
            if args[1:3] == [b"-m", b"embodiedbench.main"] and b"--config-name" in args:
                index = args.index(b"--config-name")
                environment_matches = (f"benchmark={name}".encode() in args if config_path
                                       else args[index + 1].decode() == name)
                path_matches = config_path is None or str(config_path).encode() in args
                if environment_matches and path_matches:
                    pids.append(int(process.name))
        except (OSError, IndexError):
            continue
    return pids


def inspect_plan(content, environment):
    """Check the action representation consumed by the actual planners."""
    try:
        parsed = json.loads(content)
        plan = parsed["executable_plan"]
        if environment == "eb-man" and isinstance(plan, str):
            plan = ast.literal_eval(plan)
        if not isinstance(plan, list):
            return False, False, 0
        invalid = 0
        for item in plan:
            if environment == "eb-man":
                action = item.get("action") if isinstance(item, dict) else item
                if isinstance(action, str):
                    action = ast.literal_eval(action)
                if not isinstance(action, (list, tuple)) or len(action) != 7:
                    return False, False, 0
                if any(type(v) not in (int, float) for v in action):
                    return False, False, 0
                invalid += (any(not 0 <= v <= 100 for v in action[:3])
                            or any(not 0 <= v < 120 for v in action[3:6])
                            or action[6] not in (0, 1))
            else:
                if not isinstance(item, dict) or type(item.get("action_id")) is not int:
                    return False, False, 0
                invalid += item["action_id"] < 0 or (environment == "eb-nav" and item["action_id"] >= 8)
        return True, not plan, invalid
    except (ValueError, TypeError, KeyError, SyntaxError):
        return False, False, 0


def check_jobs(now=None):
    now = time.time() if now is None else now
    jobs = json.loads((REPORT / "status.json").read_text())
    manifest = json.loads((REPORT / "dataset_manifest.json").read_text())
    runner_script = {
        "default": "scripts/run_default_evaluations.py",
        "codex": "scripts/run_codex_evaluations.py",
        "deepseek": "scripts/run_deepseek_evaluations.py",
    }.get(REPORT.name.split("_evaluations_")[0])
    runner_pids = []
    if runner_script:
        for process in Path("/proc").iterdir():
            if not process.name.isdigit():
                continue
            try:
                args = (process / "cmdline").read_bytes().split(b"\0")
                if runner_script.encode() in args and b"-c" not in args:
                    runner_pids.append(int(process.name))
            except OSError:
                continue
    rows = []
    for job in jobs:
        name = job["environment"]
        root = REPO / job["output"]
        summary = summarize(root)
        alerts = []
        malformed_responses = 0
        empty_plans = 0
        reasoning_responses = 0
        slow_requests = 0
        truncated_responses = 0
        budget_exhausted_responses = 0
        refusal_responses = 0
        invalid_model_actions = 0
        for path in root.rglob("model_requests.jsonl"):
            for record in read_requests(path):
                message = record.get("message") or {}
                content = message.get("content")
                thinking_disabled = (record.get("request_options", {}).get("extra_body") or {}).get("chat_template_kwargs", {}).get("enable_thinking") is False
                reasoning_responses += bool(message.get("reasoning") or message.get("reasoning_content")) and thinking_disabled
                slow_requests += record.get("elapsed_seconds", 0) > 120
                truncated_responses += record.get("finish_reason") == "length"
                options = record.get("request_options") or {}
                budget = options.get("max_completion_tokens", options.get("max_tokens"))
                usage = record.get("usage") or {}
                budget_exhausted_responses += bool(budget and usage.get("completion_tokens", 0) >= budget)
                refusal_responses += bool(message.get("refusal")) or record.get("finish_reason") == "content_filter"
                if not content:
                    continue
                valid, empty, invalid = inspect_plan(content, name)
                malformed_responses += not valid
                empty_plans += empty
                invalid_model_actions += invalid
        log_path = REPORT / f"{name}.log"
        log = log_path.read_text(errors="replace") if log_path.exists() else ""
        error_counts = {
            "tracebacks": log.count("Traceback (most recent call last)"),
            "api_errors": sum(log.count(text) for text in (
                "APIConnectionError", "APITimeoutError", "RateLimitError", "AuthenticationError",
                "Connection error.", "Request timed out.",
            )),
            "unexpected_errors": log.count("An unexpected error occurred"),
            "random_action_fallbacks": sum(any(text in line.lower() for text in (
                "using random action", "random action due", "random action instead",
            )) for line in log.splitlines()),
        }
        activity_paths = [log_path] if log_path.exists() else []
        activity_paths.extend(root.rglob("model_requests.jsonl"))
        activity_paths.extend(root.rglob("episode_*_res.json"))
        last_activity = max((p.stat().st_mtime for p in activity_paths), default=job.get("started_at", now))
        pids = evaluator_pids(name, job.get("config_path"))
        expected = manifest[name]["expected_episodes"]
        if job["status"] in ("running", "queued") and runner_script and not runner_pids:
            alerts.append("评测队列调度进程已退出，后续环境可能无法启动")
        if job["status"] == "running":
            if not pids:
                alerts.append("评测进程不在运行，但队列状态仍为 running")
            if now - last_activity > 900:
                alerts.append("超过 15 分钟没有新的请求、任务结果或日志")
        if job["status"] in ("failed", "needs_resume_existing_results", "waiting_service"):
            alerts.append(f"队列状态异常：{job['status']}")
        if job["status"] == "finished" and summary["completed_episodes"] != expected:
            alerts.append(f"进程结束但任务数量不完整：{summary['completed_episodes']}/{expected}")
        if summary["empty_responses"]:
            alerts.append(f"累计空回复 {summary['empty_responses']} 次（包括重试）")
        if malformed_responses or summary["planner_output_error"]:
            alerts.append(f"模型 JSON/动作结构异常 {malformed_responses} 次，planner 错误 {summary['planner_output_error']} 次")
        if truncated_responses or budget_exhausted_responses:
            alerts.append(f"输出长度截断 {truncated_responses} 次，耗尽输出/思考 token 预算 {budget_exhausted_responses} 次")
        if invalid_model_actions:
            alerts.append(f"模型动作编号或 Manipulation 动作数值越界 {invalid_model_actions} 次")
        if empty_plans:
            alerts.append(f"模型返回空动作计划 {empty_plans} 次；检查是否提前结束或触发随机动作")
        if refusal_responses:
            alerts.append(f"模型拒答/内容过滤 {refusal_responses} 次")
        if reasoning_responses:
            alerts.append(f"配置已关闭思考，但仍有 {reasoning_responses} 次返回 reasoning")
        if any(error_counts.values()):
            alerts.append(f"日志异常计数：{error_counts}")
        if summary["usage_missing_requests"]:
            alerts.append(f"有 {summary['usage_missing_requests']} 次请求未返回 token 用量")
        started = job.get("started_at")
        finished = job.get("finished_at")
        wall_seconds = max(0, (finished or now) - started) if started else 0
        row = {
            "environment": name, "status": job["status"],
            "expected_episodes": expected, "completed_episodes": summary["completed_episodes"],
            "successful_episodes": summary["successful_episodes"], "success_rate": summary["success_rate"],
            "started_at": started, "finished_at": finished,
            "wall_seconds": wall_seconds,
            "episode_seconds": sum(r["elapsed_seconds"] for r in summary["episodes"]),
            "actions": summary["actions"], "invalid_actions": summary["invalid_actions"],
            "api_requests": summary["api_requests"],
            "prompt_tokens": summary["prompt_tokens"],
            "completion_tokens": summary["completion_tokens"], "total_tokens": summary["total_tokens"],
            "reasoning_tokens": summary["reasoning_tokens"],
            "reasoning_usage_reported_requests": summary["reasoning_usage_reported_requests"],
            "avg_tokens_per_completed_episode": summary["avg_tokens_per_completed_episode"],
            "empty_responses": summary["empty_responses"],
            "malformed_responses": malformed_responses, "empty_plans": empty_plans,
            "planner_output_error": summary["planner_output_error"],
            "slow_requests": slow_requests, "error_counts": error_counts,
            "truncated_responses": truncated_responses,
            "budget_exhausted_responses": budget_exhausted_responses,
            "refusal_responses": refusal_responses, "invalid_model_actions": invalid_model_actions,
            "seconds_since_activity": max(0, now - last_activity), "pids": pids, "alerts": alerts,
        }
        rows.append(row)
    try:
        with socket.create_connection(("127.0.0.1", 7897), timeout=2):
            proxy_listening = True
    except OSError:
        proxy_listening = False
    previous_path = REPORT / "latest_status.json"
    previous = json.loads(previous_path.read_text()) if previous_path.exists() else {}
    old_jobs = {r["environment"]: r for r in previous.get("evaluations", [])}
    for row in rows:
        old = old_jobs.get(row["environment"], {})
        row["episodes_since_previous_check"] = row["completed_episodes"] - old.get("completed_episodes", 0)
        row["tokens_since_previous_check"] = row["total_tokens"] - old.get("total_tokens", 0)
        if (row["status"] == "running" and old.get("status") == "running"
                and now - previous.get("checked_at", now) > 900
                and row["episodes_since_previous_check"] == 0):
            row["alerts"].append("两次检查之间没有新的任务完成，需检查是否反复重试或卡住")
    return {
        "checked_at": now, "checked_at_utc": datetime.fromtimestamp(now, timezone.utc).isoformat(),
        "proxy_listening": proxy_listening,
        "runner_pids": runner_pids,
        "evaluations": rows,
        "notes": [
            "token 为模型接口报告的 usage，包括空回复重试；网络失败且未返回 usage 的消耗无法确认。",
            "思考 token 已计入输出 token；未提供思考用量时显示为未报告，而不是零。",
            "wall_seconds 包括环境启动和请求等待；episode_seconds 为已完成任务的环境计时之和。",
            "碰撞或任务失败属于评测结果，不自动判定为程序异常。",
        ],
    }


def save_check(snapshot):
    atomic_json(REPORT / "latest_status.json", snapshot)
    with (REPORT / "hourly_checks.jsonl").open("a") as file:
        file.write(json.dumps(snapshot, ensure_ascii=False) + "\n")
    rows = snapshot["evaluations"]
    fields = ["environment", "status", "expected_episodes", "completed_episodes", "successful_episodes",
              "wall_seconds", "episode_seconds", "actions", "api_requests", "prompt_tokens",
              "completion_tokens", "total_tokens", "reasoning_tokens", "reasoning_usage_reported_requests",
              "avg_tokens_per_completed_episode", "empty_responses", "malformed_responses", "planner_output_error"]
    with (REPORT / "evaluation_totals.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    lines = [f"检查时间：{snapshot['checked_at_utc']}", "",
             "| 环境 | 状态 | 完成/总数 | 成功 | 已用小时 | 输入 token | 输出 token | 其中思考 token | 总 token | 动作次数 |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        reasoning = row['reasoning_tokens'] if row['reasoning_usage_reported_requests'] else '未报告'
        lines.append(f"| {row['environment']} | {row['status']} | {row['completed_episodes']}/{row['expected_episodes']} | {row['successful_episodes']} | {row['wall_seconds']/3600:.2f} | {row['prompt_tokens']} | {row['completion_tokens']} | {reasoning} | {row['total_tokens']} | {row['actions']} |")
    lines.extend(["", "检查发现：", ""])
    for row in rows:
        for alert in row["alerts"]:
            lines.append(f"- {row['environment']}: {alert}")
    if not any(row["alerts"] for row in rows):
        lines.append("本次未发现运行异常。")
    if not snapshot["proxy_listening"]:
        lines.append("- 本地 Mihomo 7897 端口没有监听。")
    lines.extend(["", *snapshot["notes"]])
    (REPORT / "latest_status.md").write_text("\n".join(lines) + "\n")


def main():
    global REPORT
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=int, default=3600)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--report", type=Path, default=REPORT)
    args = parser.parse_args()
    REPORT = args.report.resolve()
    if args.interval < 1:
        parser.error("interval must be positive")
    REPORT.mkdir(parents=True, exist_ok=True)
    with (REPORT / "monitor.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        schedule_path = REPORT / "monitor_schedule.json"
        previous = json.loads(schedule_path.read_text()) if schedule_path.exists() else {}
        metadata = {"pid": os.getpid(), "interval_seconds": args.interval, "started_at": time.time()}
        # Check immediately after restart, then keep the existing hourly cadence.
        scheduled_at = (previous.get("next_check_at", time.time() + args.interval)
                        if previous.get("interval_seconds") == args.interval else time.time() + args.interval)
        next_check = time.monotonic() + max(0, scheduled_at - time.time())
        while True:
            try:
                snapshot = check_jobs()
                save_check(snapshot)
                alerts = sum(len(row["alerts"]) for row in snapshot["evaluations"])
                print(snapshot["checked_at_utc"], "check completed; alerts:", alerts, flush=True)
            except Exception as exc:
                print(datetime.now(timezone.utc).isoformat(), "check failed:", type(exc).__name__, str(exc), flush=True)
                with (REPORT / "monitor_errors.jsonl").open("a") as file:
                    file.write(json.dumps({"time": time.time(), "error": type(exc).__name__, "message": str(exc)}) + "\n")
                if args.once:
                    raise
            if args.once:
                return
            while next_check <= time.monotonic():
                next_check += args.interval
            atomic_json(REPORT / "monitor_schedule.json", {
                **metadata, "last_check_at": time.time(),
                "next_check_at": time.time() + max(0, next_check - time.monotonic()),
            })
            time.sleep(max(0, next_check - time.monotonic()))


if __name__ == "__main__":
    main()
