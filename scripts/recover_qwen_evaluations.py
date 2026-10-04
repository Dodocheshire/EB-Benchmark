"""Check an unavailable Qwen provider hourly and resume the saved queue."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
import httpx
import yaml
import sys
import shlex

REPO = Path(__file__).resolve().parents[1]
REPORT = REPO / "running/default_evaluations_20261002"


def runner_live():
    for process in Path("/proc").iterdir():
        if not process.name.isdigit():
            continue
        try:
            args = (process / "cmdline").read_bytes().split(b"\0")
            if b"scripts/run_default_evaluations.py" in args and b"-c" not in args:
                return True
        except OSError:
            continue
    return False


def check_provider():
    config = yaml.safe_load((REPO / "embodiedbench/configs/qwen/benchmark/eb-man.yaml").read_text())
    credentials = {key: os.environ[key] for key in ("OPENAI_BASE_URL", "OPENAI_API_KEY")}
    try:
        with httpx.Client(timeout=30, trust_env=False) as client:
            response = client.post(credentials["OPENAI_BASE_URL"].rstrip("/") + "/chat/completions",
                headers={"Authorization": "Bearer " + credentials["OPENAI_API_KEY"]},
                json={"model": config["model_name"], "messages": [{"role": "user", "content": 'Return JSON: {"ok":true}'}],
                      "max_tokens": 32, "chat_template_kwargs": {"enable_thinking": False}})
            if response.status_code != 200:
                return False, {"status_code": response.status_code}
            data = response.json()
            content = (data.get("choices") or [{}])[0].get("message", {}).get("content")
            return bool(content), {"status_code": 200, "usage": data.get("usage")}
    except (httpx.HTTPError, ValueError, AttributeError) as exc:
        return False, {"error_type": type(exc).__name__}


def main():
    os.chdir(REPO)
    with (REPORT / "recovery.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        next_check = json.loads((REPORT / "monitor_schedule.json").read_text())["next_check_at"]
        while True:
            state = {"pid": os.getpid(), "interval_seconds": 3600, "next_check_at": next_check}
            (REPORT / "recovery_schedule.json").write_text(json.dumps(state, indent=2))
            time.sleep(max(0, next_check - time.time()))
            jobs = json.loads((REPORT / "status.json").read_text())
            if all(job["status"] == "finished" for job in jobs):
                print("Qwen queue completed; recovery stopped", flush=True)
                return
            if runner_live():
                result = {"runner_live": True}
            else:
                ready, result = check_provider()
                if ready:
                    subprocess.run(["tmux", "kill-session", "-t", "embench-full"], capture_output=True)
                    command = (f"cd {shlex.quote(str(REPO))} && {shlex.quote(sys.executable)} -u "
                               f"scripts/run_default_evaluations.py --proxy {shlex.quote(os.getenv('https_proxy', os.getenv('HTTPS_PROXY', '')))} --resume "
                               f">> {REPORT}/runner.log 2>&1")
                    subprocess.run(["tmux", "new-session", "-d", "-s", "embench-full", command], check=True)
                    result["queue_restarted"] = True
            result["checked_at"] = time.time()
            with (REPORT / "recovery_checks.jsonl").open("a") as file:
                file.write(json.dumps(result) + "\n")
            print(json.dumps(result), flush=True)
            next_check += 3600
            while next_check <= time.time():
                next_check += 3600


if __name__ == "__main__":
    main()
