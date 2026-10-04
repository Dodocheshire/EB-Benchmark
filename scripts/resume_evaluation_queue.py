"""Continue persisted jobs, preserving results, logs, and original start times."""
import json
from pathlib import Path
import subprocess
import time
from summarize_evaluation_run import summarize
from runtime_paths import CONDA, credentials, ensure_manifest
import shlex


def resume_existing_queue(report, env):
    repo = Path(__file__).resolve().parents[1]
    report = Path(report)
    jobs = json.loads((report / "status.json").read_text())
    ensure_manifest(report)
    manifest = json.loads((report / "dataset_manifest.json").read_text())

    def save():
        temporary = report / "status.json.tmp"
        temporary.write_text(json.dumps(jobs, ensure_ascii=False, indent=2))
        temporary.replace(report / "status.json")

    for job in jobs:
        summary = summarize(repo / job["output"])
        if summary["completed_episodes"] == manifest[job["environment"]]["expected_episodes"]:
            job.update(status="finished", summary=summary)
            if not job.get("finished_at"):
                job["finished_at"] = max(p.stat().st_mtime for p in (repo / job["output"]).rglob("episode_*_res.json"))
            save()
            continue
        # Saved status files may have been copied from a different server.
        model = job.get("model_name", job["output"])
        provider = "qwen" if "qwen" in model.lower() else "deepseek" if "deepseek" in model.lower() else "gpt"
        conda_env = {"eb-nav": "embench_nav", "eb-alf": "embench", "eb-hab": "embench", "eb-man": "embench_man"}[job["environment"]]
        config_dir = repo / "embodiedbench/configs" / provider
        shell = f"set -e; source {shlex.quote(credentials(provider))}; exec python -m embodiedbench.main --config-path {shlex.quote(str(config_dir))} --config-name config benchmark={job['environment']}"
        command = [CONDA, "run", "--no-capture-output", "-n", conda_env, "bash", "-c", shell]
        if command != job.get("command"):
            job.setdefault("previous_command", job.get("command"))
            job["command"] = list(command)
            job["config_path"] = str(config_dir)
        if "bash" in command and "-c" in command:
            command[-1] += " ++resume=true"
        else:
            command.append("++resume=true")
        job.setdefault("started_at", time.time())
        job.setdefault("resumptions", []).append({"at": time.time(), "completed_episodes": summary["completed_episodes"]})
        job.update(status="running")
        job.pop("finished_at", None)
        job.pop("returncode", None)
        save()
        for attempt in range(3):
            with (report / f"{job['environment']}.log").open("a") as log:
                log.write(f"\nResuming saved evaluation; process attempt {attempt + 1}/3\n")
                log.flush()
                result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode == 0:
                break
            job.setdefault("process_failures", []).append({"at": time.time(), "returncode": result.returncode})
            save()
        job.update(status="finished" if result.returncode == 0 else "failed",
                   returncode=result.returncode, finished_at=time.time())
        job["summary"] = summarize(repo / job["output"])
        save()
