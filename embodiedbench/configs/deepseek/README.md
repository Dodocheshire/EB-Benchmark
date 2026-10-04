DeepSeek 评测配置
================

入口是 `config.yaml`，通过 `benchmark=eb-nav`、`eb-alf`、`eb-hab` 或 `eb-man` 选择环境。
四个 YAML 位于 `benchmark/`，采样比例、子集、分辨率、示例数量等与原配置一致。

模型默认使用支持图片输入的 `deepseek-flash`，地址为 `https://api.deepseek.com`。
`reasoning_effort: high` 会作为请求字段发送。实验名称为 `deepseek_baseline`。
`api_max_tokens: 16384` 为思考及最终输出留出预算，`api_timeout: 600` 允许较长的思考请求。
DeepSeek 使用 JSON Object 输出模式；适配器将 benchmark 的动作 schema 加入提示词。

API Key 存放于项目外的 `/root/.config/embodiedbench-deepseek/client.env`，权限为 0600。
YAML 不保存密钥。

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate embench_nav
source /root/.config/embodiedbench-deepseek/client.env
cd /home/pengcheng/EM-Benchmark/EmbodiedBench

DISPLAY=:1 python -m embodiedbench.main \
  --config-path /home/pengcheng/EM-Benchmark/EmbodiedBench/embodiedbench/configs/deepseek \
  --config-name config \
  benchmark=eb-nav
```

ALFRED 和 Habitat 使用 `embench`，Manipulation 使用 `embench_man`，同时替换 `benchmark=`。
Manipulation 可以设置 `export EB_DETECTION_DEVICE=cpu` 适配当前 PyTorch/RTX 5090 组合。
调试可追加 `'eval_sets=[base]' down_sample_ratio=0.01 exp_name=deepseek_debug`。

执行时会记录模型回复、思考强度、token 用量和图片路径到各子集目录的 `model_requests.jsonl`。
完整评测默认覆盖全部子集：Navigation、ALFRED、Habitat 各 300 条，Manipulation 228 条。

官方说明：

- 图片输入：https://api-docs.deepseek.com/guides/vision/
- 请求参数及 JSON 输出：https://api-docs.deepseek.com/api/create-chat-completion/
