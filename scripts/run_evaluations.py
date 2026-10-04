"""One entry point for detached evaluation queues and individual reruns."""
import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys
from runtime_paths import CONDA, credentials

REPO = Path(__file__).resolve().parents[1]
PROVIDERS = {
    "qwen": ("run_default_evaluations.py", "default_evaluations_20261002", "embodiedbench-qwen", "embench-full"),
    "gpt": ("run_codex_evaluations.py", "codex_evaluations_20261002", "embodiedbench-codex", "embench-codex-full"),
    "deepseek": ("run_deepseek_evaluations.py", "deepseek_evaluations_20261002", "embodiedbench-deepseek", "embench-deepseek-full"),
}
CONDA_ENVS = {"eb-nav": "embench_nav", "eb-alf": "embench", "eb-hab": "embench", "eb-man": "embench_man"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=PROVIDERS, required=True)
    parser.add_argument("--env", choices=CONDA_ENVS, help="Evaluate one environment instead of resuming the saved four-environment queue")
    parser.add_argument("--exp-name", help="Separate output name for a single-environment rerun")
    parser.add_argument("--proxy", default=os.environ.get("https_proxy", os.environ.get("HTTPS_PROXY", "")))
    parser.add_argument("--foreground", action="store_true", help="Run synchronously; otherwise use detached tmux")
    parser.add_argument("--dry-run", action="store_true", help="Print the command without starting evaluations")
    args = parser.parse_args()
    if args.exp_name and not args.env:
        parser.error("--exp-name requires --env")
    if args.exp_name and (not args.exp_name.replace("-", "").replace("_", "").isalnum()):
        parser.error("--exp-name must contain only letters, numbers, underscores or hyphens")

    if not CONDA:
        parser.error("Conda not found; run bash scripts/setup_ubuntu.sh or activate Conda first")
    runner, report_dir, credential_dir, session = PROVIDERS[args.provider]
    if args.env:
        hydra = ["python", "-m", "embodiedbench.main", "--config-path", str(REPO / "embodiedbench/configs" / args.provider),
                 "--config-name", "config", "benchmark=" + args.env, "++resume=true"]
        if args.exp_name:
            hydra.append("exp_name=" + args.exp_name)
        client_env = credentials(args.provider)
        shell = "set -e; source " + shlex.quote(client_env) + "; exec " + shlex.join(hydra)
        command = [CONDA, "run", "--no-capture-output", "-n", CONDA_ENVS[args.env], "bash", "-c", shell]
        session = f"embench-{args.provider}-{args.env}-{args.exp_name or 'resume'}"
    else:
        command = [sys.executable, str(REPO / "scripts" / runner)]
        if args.provider == "qwen":
            command += ["--proxy", args.proxy]
        if (REPO / "running" / report_dir / "status.json").exists():
            command.append("--resume")
    if args.dry_run:
        print("Working directory:", REPO)
        print("Tmux session:", session)
        print(shlex.join(command))
        return

    # Prevent accidental duplicate queues, including runners started outside tmux.
    for process in Path("/proc").iterdir():
        if not process.name.isdigit():
            continue
        try:
            argv = (process / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        runner_active = any(item.endswith(("/" + runner).encode()) for item in argv)
        provider_active = b"embodiedbench.main" in argv and str(REPO / "embodiedbench/configs" / args.provider).encode() in argv
        if runner_active or provider_active:
            parser.exit(0, f"{args.provider} evaluation already running (PID {process.name}); kept the existing process.\n")
    if not args.foreground:
        if subprocess.run(["tmux", "has-session", "-t", session], capture_output=True).returncode == 0:
            parser.exit(0, f"Tmux session {session} already exists; kept it.\n")
        replay = [sys.executable, str(Path(__file__).resolve()), "--provider", args.provider, "--proxy", args.proxy, "--foreground"]
        if args.env:
            replay += ["--env", args.env]
        if args.exp_name:
            replay += ["--exp-name", args.exp_name]
        log_dir = REPO / "running" / report_dir
        log_dir.mkdir(parents=True, exist_ok=True)
        log = log_dir / ("entrypoint.log" if not args.env else f"entrypoint-{args.env}-{args.exp_name or 'resume'}.log")
        shell = "cd " + shlex.quote(str(REPO)) + " && " + shlex.join(replay) + " >> " + shlex.quote(str(log)) + " 2>&1"
        subprocess.run(["tmux", "new-session", "-d", "-s", session, shell], check=True)
        print(f"Started tmux session {session}; log: {log}")
        return
    env = dict(os.environ, DISPLAY=":1", PYTHONUNBUFFERED="1", EB_DETECTION_DEVICE="cpu")
    for key in ("http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
        env[key] = args.proxy
    direct = {"localhost", "127.0.0.1", "::1", "mirrors.zju.edu.cn", "token.cogenivo.com", "38.78.146.63"}
    direct.update(filter(None, os.environ.get("NO_PROXY", os.environ.get("no_proxy", "")).split(",")))
    env["no_proxy"] = env["NO_PROXY"] = ",".join(sorted(direct))
    result = subprocess.run(command, cwd=REPO, env=env)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
