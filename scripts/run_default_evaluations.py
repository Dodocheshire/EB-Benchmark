"""Run the four default evaluations sequentially from a detached tmux session."""
import argparse
import json
import shlex
import os
from pathlib import Path
import subprocess
import time
from summarize_evaluation_run import summarize
from runtime_paths import CONDA, credentials, ensure_manifest

REPO = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO / "embodiedbench/configs/qwen"
CLIENT_ENV = credentials("qwen")
SPECS = [
    ("eb-nav", "embench_nav", "running/eb_nav/qwen3.8-27b_navigation_baseline"),
    ("eb-alf", "embench", "running/eb_alfred/qwen3.8-27b_baseline"),
    ("eb-hab", "embench", "running/eb_habitat/qwen3.8-27b_baseline"),
    ("eb-man", "embench_man", "running/eb_manipulation/qwen3.8-27b/baseline"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--proxy", required=True, help="Verified HTTP proxy independent of the user SSH connection")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    os.chdir(REPO)
    report = REPO / "running/default_evaluations_20261002"
    report.mkdir(parents=True, exist_ok=True)
    ensure_manifest(report)
    env = dict(os.environ, DISPLAY=":1", PYTHONUNBUFFERED="1", EB_DETECTION_DEVICE="cpu")
    for key in ("http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
        env[key] = args.proxy
    if args.resume:
        from resume_evaluation_queue import resume_existing_queue
        resume_existing_queue(report, env)
        return
    jobs = []
    def save():
        (report / "status.json").write_text(json.dumps(jobs, ensure_ascii=False, indent=2))
    for name, conda_env, output in SPECS:
        shell = (f"set -e; source {shlex.quote(CLIENT_ENV)}; exec python -m embodiedbench.main "
                 f"--config-path {shlex.quote(str(CONFIG_DIR))} --config-name config benchmark={name}")
        command = [CONDA, "run", "--no-capture-output", "-n", conda_env,
                   "bash", "-c", shell]
        jobs.append({"environment": name, "command": command, "output": output, "status": "queued",
                     "config_path": str(CONFIG_DIR)})
    save()
    for job in jobs:
        if list((REPO / job["output"]).rglob("episode_*_res.json")):
            job["status"] = "needs_resume_existing_results"
            save()
            continue
        job.update(status="running", started_at=time.time())
        save()
        print("Starting", job["environment"], flush=True)
        with (report / f"{job['environment']}.log").open("w") as log:
            result = subprocess.run(job["command"], env=env, stdout=log, stderr=subprocess.STDOUT)
        job.update(status="finished" if result.returncode == 0 else "failed", returncode=result.returncode, finished_at=time.time())
        job["summary"] = summarize(REPO / job["output"])
        save()
        print(job["environment"], job["status"], "episodes", job["summary"]["completed_episodes"], flush=True)


if __name__ == "__main__":
    main()
