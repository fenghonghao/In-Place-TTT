# Changelog

## 1. 推理时 fast-weight 更新 clipping(论文附录 C.2)

为推理路径(`inference_model/hf_qwen3/`、`inference_model/hf_llama3/`)添加 per-chunk
fast-weight 增量的 Frobenius 范数裁剪,与论文附录 C.2 在 Qwen3-4B 评测中使用的
稳定化机制对齐。

- 新增 config 字段 `ttt_clip_tau`,默认 `1e-5`(论文报告值),设为 `<= 0` 可关闭。
  影响 `Qwen3Config` 与 `LlamaConfig`。
- 仅推理生效;训练侧 `hf_models/` 不变(保留论文 §3.4 的并行 prefix-sum 实现)。
- 暴露 `set_ttt_clip_hook(hook)`,便于观测裁剪频率与幅度。
- 旧 HF checkpoint 无需重训,modeling 侧 `getattr` 兜底向后兼容。

> 完整说明、使用方法、注意事项、实现要点见
> [docs/inference_fast_weight_clipping.md](docs/inference_fast_weight_clipping.md)。

## 2. 训练时选择性参数冻结(instruct 模型微调)

为训练侧(`tasks/train_torch.py`)新增基于参数名模式匹配的选择性冻结机制,
配合论文 §C.3 的近零初始化,用于在不破坏 instruct 模型对齐能力的前提下,
仅训练 TTT 模块(或 TTT + 同层 `down_proj`)。

- 新增顶层 YAML 段 `freeze`(`FreezeArguments`);`enable` 默认 `false`,
  未启用时训练行为与改动前一致。
- 新增示例配置 `configs/pretrain/qwen3_instruct_ttt.yaml`、
  `configs/pretrain/llama3_instruct_ttt.yaml`。
- 推理侧 `inference_model/` 不受影响;DCP / HF checkpoint 产物仍为完整模型。

> 完整说明、使用方法、注意事项、实现要点、验证与排错见
> [docs/training_selective_freeze.md](docs/training_selective_freeze.md)。

