[原作者仓库与 README](https://github.com/EmbodiedBench/EmbodiedBench#readme)

# Installation

```bash
git clone https://github.com/Dodocheshire/EB-Benchmark.git
cd EB-Benchmark
bash install.sh
```

The installer installs system OpenGL/Xorg/font/Git-LFS dependencies, creates `embench`, `embench_nav`, `embench_man`, installs pinned Habitat-Sim/Lab 0.3.0 and CoppeliaSim/PyRep 4.1, and downloads the datasets.

```bash
# Inspect every command without installing or downloading anything.
bash install.sh --dry-run
# Independent environment and data entry points:
bash scripts/setup_ubuntu.sh
bash scripts/download_datasets.sh
# Optional: --conda-root /path/to/conda; setup also accepts --skip-system.
# Download only one dataset:
bash scripts/download_datasets.sh --dataset manipulation
```

## Credentials and launching

```bash
mkdir -p ~/.config/embodiedbench-qwen ~/.config/embodiedbench-codex ~/.config/embodiedbench-deepseek
umask 077
cat > ~/.config/embodiedbench-qwen/client.env <<'EOF'
export OPENAI_BASE_URL="https://YOUR_QWEN_ENDPOINT/v1"
export OPENAI_API_KEY="YOUR_API_KEY"
EOF
cat > ~/.config/embodiedbench-codex/client.env <<'EOF'
export OPENAI_BASE_URL="https://YOUR_GPT_PROXY_ENDPOINT/v1"
export OPENAI_API_KEY="YOUR_API_KEY"
EOF
cat > ~/.config/embodiedbench-deepseek/client.env <<'EOF'
export DEEPSEEK_BASE_URL="https://api.deepseek.com"
export DEEPSEEK_API_KEY="YOUR_API_KEY"
EOF
chmod 700 ~/.config/embodiedbench-{qwen,codex,deepseek}
chmod 600 ~/.config/embodiedbench-{qwen,codex,deepseek}/client.env
# Edit the endpoint/key in these files before launching.

source ~/miniforge3/etc/profile.d/conda.sh  # use your existing Conda prefix if different
conda activate embench_nav
bash scripts/start_headless.sh 1
python scripts/run_evaluations.py --provider qwen
python scripts/run_evaluations.py --provider gpt
python scripts/run_evaluations.py --provider deepseek
```

To run a separate repeat experiment, use `--env eb-nav --exp-name reproduce_nav`; supported environments are `eb-nav`, `eb-alf`, `eb-hab`, `eb-man`.  See [EXPERIMENT_RESULT.md](EXPERIMENT_RESULT.md) for results, charts. Regenerate it with `python scripts/report_results.py` after evaluations.

### Continuing this server's unfinished experiments

Git contains code, configs and aggregate results, **not** raw evaluation logs or downloaded data. To resume rather than start from zero, copy the old repository's `running/` directory to the new repository's `running/` directory, including queue `status.json`, result JSONs and request logs. Result paths and persisted commands are rebased to the new checkout automatically. Keep this directory private because request logs contain complete prompts/images metadata. Download the pinned datasets, copy the external credentials, then run the same queue commands above. Completed episodes are skipped. Do not copy stale PID/schedule files as proof that monitors are alive; start new monitors on the new host:

```bash
mkdir -p running/default_evaluations_20261002
tmux new-session -d -s embench-monitor \
  "cd '$PWD' && python -u scripts/monitor_default_evaluations.py --report '$PWD/running/default_evaluations_20261002' --interval 3600 >> '$PWD/running/default_evaluations_20261002/monitor.log' 2>&1"
# GPT/DeepSeek: use codex_evaluations_20261002 / deepseek_evaluations_20261002
# and distinct tmux session names for their monitors.
```
