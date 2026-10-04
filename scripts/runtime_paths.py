"""Host-independent paths shared by launchers and queue recovery."""
import os
from pathlib import Path
import shutil
import json

CONDA = os.environ.get("CONDA_EXE") or shutil.which("conda")
if not CONDA:
    for prefix in (Path.home() / "miniforge3", Path("/opt/conda")):
        if (prefix / "bin/conda").exists():
            CONDA = str(prefix / "bin/conda")
            break
CONFIG_HOME = Path(os.environ.get("EB_CLIENT_CONFIG_DIR", os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))))


def credentials(provider):
    directory = {"qwen": "embodiedbench-qwen", "gpt": "embodiedbench-codex", "deepseek": "embodiedbench-deepseek"}[provider]
    return str(CONFIG_HOME / directory / "client.env")


def ensure_manifest(report):
    path = Path(report) / "dataset_manifest.json"
    if not path.exists():
        path.write_text(json.dumps({env: {"expected_episodes": count} for env, count in
                                   [("eb-nav", 300), ("eb-alf", 300), ("eb-hab", 300), ("eb-man", 228)]}, indent=2))
