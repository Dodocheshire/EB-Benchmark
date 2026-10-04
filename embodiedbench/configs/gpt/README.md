GPT 模型评测配置
================

入口是 `config.yaml`，通过 `benchmark=eb-nav`、`eb-alf`、`eb-hab` 或 `eb-man` 选择四个环境。
每个环境的字段位于 `benchmark/`，评测参数与原来的 `configs/eb-*.yaml` 保持一致。
模型为 `gpt-6.1-sol`，实验名称为 `codex_baseline`。不发送 effort，使用上游反代默认的 medium。

API 地址和密钥只从项目外的 `/root/.config/embodiedbench-codex/client.env` 读取。
本目录不设置 Hydra 的 API 环境变量，不包含千问的关闭思考参数。

从仓库根目录运行 Navigation：

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate embench_nav
source /root/.config/embodiedbench-codex/client.env
cd /home/pengcheng/EM-Benchmark/EmbodiedBench

DISPLAY=:1 python -m embodiedbench.main \
  --config-path /home/pengcheng/EM-Benchmark/EmbodiedBench/embodiedbench/configs/gpt \
  --config-name config \
  benchmark=eb-nav
```

ALFRED 和 Habitat 使用 `embench`，Manipulation 使用 `embench_man`，同时替换 `benchmark=`。
`eval_sets=[]` 表示全部默认评测子集，`down_sample_ratio=1` 表示完整数据。
可以覆盖字段做小规模测试，例如 `'eval_sets=[base]' down_sample_ratio=0.01 exp_name=codex_debug`。

四组完整评测队列：

```bash
tmux new-session -d -s embench-codex-full \
  'cd /home/pengcheng/EM-Benchmark/EmbodiedBench && /opt/conda/envs/embench_nav/bin/python -u scripts/run_codex_evaluations.py > running/codex_evaluations_20261002/runner.log 2>&1'
```

队列状态和组别日志位于 `running/codex_evaluations_20261002/`。
各环境结果目录保存请求明细 `model_requests.jsonl`、任务结果和 token/动作统计。
token 数以反代返回的 `usage` 为准，包括有 usage 的重试请求。
Manipulation 的目标检测使用 CPU，以兼容当前 PyTorch 与 RTX 5090 的组合；目标框功能保持开启。
