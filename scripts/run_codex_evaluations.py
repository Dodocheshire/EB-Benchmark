"""Run GPT evaluations with external credentials and dedicated Hydra configs."""
import json
import shlex
import argparse
import os
from pathlib import Path
import subprocess
import time
from summarize_evaluation_run import summarize
from runtime_paths import CONDA, credentials, ensure_manifest

REPO = Path(__file__).resolve().parents[1]
REPORT = REPO / "running/codex_evaluations_20261002"
CONFIG_DIR = REPO / "embodiedbench/configs/gpt"
CLIENT_ENV = credentials("gpt")
SPECS = [
    ("eb-nav", "embench_nav", "running/eb_nav/gpt-6.1-sol_codex_baseline"),
    ("eb-alf", "embench", "running/eb_alfred/gpt-6.1-sol_codex_baseline"),
    ("eb-hab", "embench", "running/eb_habitat/gpt-6.1-sol_codex_baseline"),
    ("eb-man", "embench_man", "running/eb_manipulation/gpt-6.1-sol/codex_baseline"),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    os.chdir(REPO)
    REPORT.mkdir(parents=True, exist_ok=True)
    ensure_manifest(REPORT)
    env = dict(os.environ, DISPLAY=":1", PYTHONUNBUFFERED="1", EB_DETECTION_DEVICE="cpu")
    env["no_proxy"] = env["NO_PROXY"] = "localhost,127.0.0.1,::1,mirrors.zju.edu.cn,token.cogenivo.com,38.78.146.63"
    if args.resume:
        from resume_evaluation_queue import resume_existing_queue
        resume_existing_queue(REPORT, env)
        return
    jobs = []
    def save():
        temporary = REPORT / "status.json.tmp"
        temporary.write_text(json.dumps(jobs, ensure_ascii=False, indent=2))
        temporary.replace(REPORT / "status.json")
    for name, conda_env, output in SPECS:
        # Only the filename is persisted; credentials stay in the child's environment.
        shell = f"set -e; source {shlex.quote(CLIENT_ENV)}; exec python -m embodiedbench.main --config-path {shlex.quote(str(CONFIG_DIR))} --config-name config benchmark={name}"
        command = [CONDA, "run", "--no-capture-output", "-n", conda_env, "bash", "-c", shell]
        jobs.append({"environment": name, "model_name": "gpt-6.1-sol", "config_path": str(CONFIG_DIR),
                     "command": command, "output": output, "status": "queued", "effort": "medium (upstream default)"})
    save()
    for job in jobs:
        if list((REPO / job["output"]).rglob("episode_*_res.json")):
            job["status"] = "needs_resume_existing_results"
            save()
            continue
        job.update(status="running", started_at=time.time())
        save()
        print("Starting", job["environment"], flush=True)
        with (REPORT / f"{job['environment']}.log").open("w") as log:
            result = subprocess.run(job["command"], env=env, stdout=log, stderr=subprocess.STDOUT)
        job.update(status="finished" if result.returncode == 0 else "failed", returncode=result.returncode, finished_at=time.time())
        job["summary"] = summarize(REPO / job["output"])
        save()
        print(job["environment"], job["status"], "episodes", job["summary"]["completed_episodes"], flush=True)


if __name__ == "__main__":
    main()
