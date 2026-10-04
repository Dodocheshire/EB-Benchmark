"""Connect evaluator episodes to model request logs without recording credentials."""
import os
import json
from omegaconf import OmegaConf


def configure_model_logging(model, config, log_path):
    os.makedirs(log_path, exist_ok=True)
    model.request_log_path = os.path.join(log_path, "model_requests.jsonl")
    options = config.get("api_extra_body", {})
    model.api_extra_body = (
        OmegaConf.to_container(options, resolve=True)
        if OmegaConf.is_config(options) else dict(options)
    )
    if getattr(model, "model_name", "").startswith(("gpt-5", "gpt-6")):
        # The base environment configs also contain Qwen-specific provider options.
        model.api_extra_body.pop("chat_template_kwargs", None)
    model.api_timeout = config.get("api_timeout", 120)
    model.reasoning_effort = config.get("reasoning_effort")
    model.api_max_tokens = config.get("api_max_tokens", getattr(model, "api_max_tokens", 4096))
    model.api_max_tokens_limit = config.get("api_max_tokens_limit", model.api_max_tokens * 4)
    base_url = config.get("api_base_url")
    if base_url and hasattr(model.model, "base_url"):
        model.model.base_url = base_url


def set_model_request_context(planner, env, images, instruction):
    planner.model.request_context = {
        "episode": env._current_episode_num,
        "env_step_before_request": env._current_step,
        "planner_step": planner.planner_steps + 1,
        "instruction": instruction,
        "image_path": images,
    }


def skip_completed_episode(env, config, habitat=False):
    """Resume from saved final results without overwriting completed episodes."""
    if not config.get("resume", False):
        return False
    episode = env._current_episode_num + 1
    selected = getattr(env, "selected_indexes", [])
    if selected:
        episode = selected[env._current_episode_num] + 1
    for suffix in ("final_res", "res"):
        path = os.path.join(env.log_path, "results", f"episode_{episode}_{suffix}.json")
        if not os.path.exists(path):
            continue
        try:
            with open(path) as file:
                result = json.load(file)
        except json.JSONDecodeError:
            continue
        if "task_success" not in result or "num_steps" not in result:
            continue
        if habitat:
            # Advance Habitat's own episode iterator as well as the wrapper.
            env.reset()
        else:
            env._current_episode_num += 1
        return True
    return False
