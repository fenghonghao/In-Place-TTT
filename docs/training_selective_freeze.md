# 训练时选择性参数冻结

> 适用范围：仅训练侧（`tasks/train_torch.py` + `hf_models/`）。推理侧（`inference_model/`）不涉及。
> 相关入口代码：[tasks/train_torch.py:82-228](../tasks/train_torch.py#L82-L228)、[tasks/train_torch.py:378](../tasks/train_torch.py#L378)、[tasks/train_torch.py:422-430](../tasks/train_torch.py#L422-L430)
> 推荐 YAML：[configs/pretrain/qwen3_instruct_ttt.yaml](../configs/pretrain/qwen3_instruct_ttt.yaml)、[configs/pretrain/llama3_instruct_ttt.yaml](../configs/pretrain/llama3_instruct_ttt.yaml)

---

## 1. 改动说明

针对论文作者在 GitHub issue 中给出的"在已 instruction-tuned 的模型上**仅训练 TTT 模块**"用法，为训练侧新增基于参数名模式匹配的选择性冻结机制。

该机制配合论文 §C.3 描述的近零初始化（`ttt_conv` 零初始化 + `ttt_proj` 对角随机初始化，使 TTT 模块在 step 0 等价于原 `down_proj`），目标是：**在不破坏 instruct 模型对齐能力的前提下，渐进式地为其注入 TTT 长上下文能力**。

- **新增**：顶层 YAML 段 `freeze:`（对应数据类 `FreezeArguments`）。
- **新增**：示例 YAML `configs/pretrain/qwen3_instruct_ttt.yaml`、`configs/pretrain/llama3_instruct_ttt.yaml`。
- **不变**：所有现有训练配置；`freeze.enable` 默认 `false`，未启用时训练行为与改动前一致。

---

## 2. 使用方法

### 2.1 YAML 配置字段

```yaml
freeze:
  enable: false                  # 总开关。false 时本节完全 no-op
  trainable_patterns: []         # 全局子串匹配：任一 pattern in param_name → 保留可训练
  trainable_on_ttt_layers: []    # 仅在 config.ttt_layers 指定的层做子串匹配
  verbose: true                  # rank-0 打印统计与抽样名称
```

| 字段 | 默认 | 语义 |
|---|---|---|
| `enable` | `false` | 总开关；`false` 时该 pass 完全 no-op，行为等价于全参微调 |
| `trainable_patterns` | `[]` | 全局子串匹配：任一 pattern `in` 参数名则保留可训练（适合 `ttt_proj` / `ttt_conv` 这类**只存在于 TTT 层**的模块） |
| `trainable_on_ttt_layers` | `[]` | 仅在 `config.ttt_layers` 指定的层上做子串匹配（适合 `down_proj` 这类**每层都有**、但只想在 TTT 层训练的模块） |
| `verbose` | `true` | rank-0 打印 trainable / frozen 统计与抽样名称 |

匹配规则：
- 命中 `trainable_patterns` **或** 命中 `trainable_on_ttt_layers` 的参数 → `requires_grad = True`。
- 否则 → `requires_grad = False`。
- `trainable_on_ttt_layers` 的命中条件是「pattern 在参数名内」**且**「`.layers.{i}.` 在参数名内，其中 `i ∈ config.ttt_layers`」。

### 2.2 推荐配置（CHANGELOG §2 默认）

`configs/pretrain/qwen3_instruct_ttt.yaml` 默认：TTT 模块 + 仅 TTT 层的 `down_proj` 可训练。对 Qwen3-1.7B-Instruct 约 **0.7%** 参数量。

```yaml
freeze:
  enable: true
  trainable_patterns:
    - "ttt_proj"
    - "ttt_conv"
  trainable_on_ttt_layers:
    - "down_proj"
  verbose: true
```

启动命令（与现有 `train.sh` 完全兼容）：

```bash
bash train.sh tasks/train_torch.py configs/pretrain/qwen3_instruct_ttt.yaml \
  --model.model_path /path/to/Qwen3-1.7B-Instruct \
  --data.train_path /path/to/your_data \
  --train.output_dir /path/to/output \
  --train.lr 1.0e-4 \
  --train.wandb_project ttt --train.wandb_name qwen3-1.7b-instruct-ttt
```

### 2.3 其他 ablation 模板

仅改 `freeze` 段即可切换：

```yaml
# (a) 纯 TTT（最激进的 freeze；只学 fast-weight 路径上的参数）
freeze: { enable: true, trainable_patterns: ["ttt_proj", "ttt_conv"], trainable_on_ttt_layers: [] }

# (b) TTT + down_proj（推荐基线 = qwen3_instruct_ttt.yaml）
freeze: { enable: true, trainable_patterns: ["ttt_proj", "ttt_conv"], trainable_on_ttt_layers: ["down_proj"] }

# (c) TTT + down_proj + 同层 norm（给残差流多一点适应空间）
freeze:
  enable: true
  trainable_patterns: ["ttt_proj", "ttt_conv"]
  trainable_on_ttt_layers: ["down_proj", "input_layernorm", "post_attention_layernorm"]

# (d) 全参对照组（与 enable: false 完全等价）
freeze: { enable: false }
```

---

## 3. 注意事项

- **不可从 full-finetune 的 DCP checkpoint resume 进 freeze 模式** —— 旧 optimizer state 包含**全部参数**的 AdamW 动量/方差，与新的 trainable 子集对不上。新的 freeze run 必须**从 HF 权重起步**：`model.model_path` 指向 instruct 模型本身，**不要**设 `train.load_checkpoint_path`。
- **建议提高 LR** —— TTT-only 训练时梯度信号集中在 ~1% 的参数上，实践中常用 `1e-4 ~ 5e-4` 量级。默认 YAML 仍保留 `5e-6`（与 full-finetune 对齐），需要用户根据收敛情况自行调整。
- **作用域限定**：本机制仅修改训练侧 `tasks/train_torch.py`，**推理侧 `inference_model/` 无任何变化**，HF checkpoint 仍是完整模型（见 §4.3）。
- **`trainable_on_ttt_layers` 强依赖 `config.ttt_layers`**：如果模型 config 里没有 `ttt_layers`（例如关闭了 TTT），该字段不会命中任何参数。此时如果 `trainable_patterns` 也为空且 `enable=true`，会主动报错（见 §4.5）。
- **匹配采用朴素子串**：例如 `"down_proj"` 会同时命中 `mlp.down_proj.weight`，不会命中 `up_proj`。如果未来引入同名子串的新模块需要重新审视模式。

---

## 4. 实现要点

### 4.1 注入点

`_apply_freeze` 放在 `model_config = model.config` 之后、`build_parallelize_model`（FSDP2 wrap）之前（[tasks/train_torch.py:378](../tasks/train_torch.py#L378)）。现成 yaml 都用 `init_device: meta`，这步操作的实际上是 meta tensor，但 `requires_grad` 只是个 Python 标记，跟参数有没有真实数据无关，cpu/cuda 上一样能设。

这个标记能撑过 FSDP2 wrap 和后续物化，是因为 PyTorch 的 `nn.Module._apply`（`model.float()` 和 `model.to_empty(...)` 都走这条）会用 `nn.Parameter(t, requires_grad=orig.requires_grad)` 把新 tensor 重新包成 Parameter，原标记会一路带下去。

### 4.2 优化器集成

VeOmni 的 `build_optimizer` 在 `optimizer.py` 里用 `p.requires_grad` 把不可训练参数过滤掉。被冻的参数因此不会进 AdamW，也不占 momentum / variance 的显存，优化器一侧不用碰。

### 4.3 checkpoint 与 HF 转换

中途保存把整个 `model` 模块直接传给 `Checkpointer.save`（[tasks/train_torch.py:584-595](../tasks/train_torch.py#L584-L595)），DCP 内部走 state_dict 时会把 frozen 权重一并写盘。训练收尾的 `save_hf_weights: true` 分支（[tasks/train_torch.py:626-633](../tasks/train_torch.py#L626-L633)）先 `ckpt_to_state_dict` 从 DCP 把权重还原回来，再写 safetensors，全程不看 `requires_grad`；`scripts/merge_dcp_to_hf.py` 单独跑也一样。所以产出的 HF checkpoint 还是完整模型，推理侧不用感知哪些权重训练时是 frozen 的。

### 4.4 wandb 集成

`wandb.init(config=...)` 额外注入两类字段（[tasks/train_torch.py:422-430](../tasks/train_torch.py#L422-L430)）：

- `freeze.<yaml-字段>`：YAML 原值。
  > 注意：用 `freeze.` 前缀是为了避免与 `TrainingArguments` 的同名字段冲突（例如 `enable`）。
- `_apply_freeze` 返回的派生统计：`freeze.applied`、`freeze.trainable_param_tensors`、`freeze.total_param_tensors`、`freeze.trainable_elems`、`freeze.total_elems`、`freeze.trainable_ratio`。

这些字段便于跨 run ablation 对比（按 `freeze.trainable_ratio` 排序、按 `freeze.applied` 分组等）。

### 4.5 安全护栏

- `enable=true` + `ttt_mode=false` —— rank-0 打 `[freeze][WARN]` 后跳过 `_apply_freeze`，fallback 到全参微调（不抛错）。wandb 中 `freeze.enable=true` 但 `freeze.applied=false` 即此场景的指纹。
- `enable=true` 且最终 0 trainable（pattern 拼错或两个列表都空）—— 抛 `RuntimeError`，错误信息打印两个 patterns 列表与 `config.ttt_layers`。

### 4.6 向后兼容

`freeze.enable` 默认 `false`；`_apply_freeze` 直接 early-return，返回「全员可训练」的统计 dict。不改 YAML 或显式 `--freeze.enable false` 跑，训练行为（loss、权重、梯度）跟改动前一字不差。差别只体现在 wandb 上——会多出一整组 `freeze.*` 字段（`applied=false`、`trainable_ratio=1.0` 等），schema 跟启用 freeze 的 run 完全对齐，做 ablation 时能直接放一起比。

---

## 5. 验证和排错

### 5.1 启动锚点

`enable=true` 且 `verbose=true` 时，rank-0 stdout 会打印（格式见 [tasks/train_torch.py:205-211](../tasks/train_torch.py#L205-L211)）：

```
[freeze] trainable param tensors: <n>/<N> (<X>M / <Y>M = <Z>%)
[freeze] first 8 trainable: [...]
[freeze] first 4 frozen:    [...]
```

对照你的 freeze 配置确认：
- 比例符合预期（粗算：trainable 参数量 / 总参数量）。
- `first 8 trainable` 里的 `down_proj` 层号都在 `config.ttt_layers` 内。
- `first 4 frozen` 里不出现 `ttt_proj` / `ttt_conv`。

### 5.2 冒烟（推荐先跑一次）

```bash
bash train.sh tasks/train_torch.py configs/pretrain/qwen3_instruct_ttt.yaml \
  --model.model_path /path/to/Qwen3-1.7B-Instruct \
  --data.train_path /path/to/your_data \
  --train.output_dir /tmp/ttt_freeze_smoke \
  --train.max_steps 1 \
  --train.use_wandb false
```

看到 `[freeze]` 三行 + 单步 loss 打印即通过。若 0 trainable 会被 §4.5 的护栏直接拦下。
