# 推理时 fast-weight 更新的 clipping

> 仅推理侧（`inference_model/hf_qwen3/`、`inference_model/hf_llama3/`）。论文附录 C.2，Qwen3-4B 评测中 `τ = 1e-5`。

## 1. 改动说明

逐 chunk 算出 `dw = η · contract(h, t, ttt_proj)` 后，若 `‖dw‖_F > τ`，按 `τ / ‖dw‖_F` 等比缩放再累加到 `current_w`。目的是压住单个 chunk 偶发的大增量，避免后续 chunk 被链式带偏。

- 新增 `Qwen3Config` / `LlamaConfig` 字段 `ttt_clip_tau`，默认 `1e-5`，`<= 0` 关闭。
- 训练侧 `hf_models/` 不读这个字段，旧 HF checkpoint 不需要重训。

## 2. 使用方法

```python
import inference_model
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(path)                    # 默认 τ=1e-5
model = AutoModelForCausalLM.from_pretrained(path, ttt_clip_tau=-1)   # 关闭
model = AutoModelForCausalLM.from_pretrained(path, ttt_clip_tau=5e-6) # 改 τ
```

观测 hook：`set_ttt_clip_hook(hook)` 在每次实际裁剪时调用 `hook(event_dict)`，dict 字段：

| key | 含义 |
|---|---|
| `layer_idx` | 触发裁剪的 decoder 层索引 |
| `chunk_idx` | 当前 chunk 在序列中的序号 |
| `chunk_start` / `chunk_end` | 该 chunk 覆盖的 token 区间 |
| `chunk_num` / `seq_len` | 当前 forward 的总 chunk 数与序列长度 |
| `pre_clip_norm` | 裁剪前的 `‖dw‖_F`（Python float） |
| `clip_tau` | 当前阈值 τ |
| `clip_scale` | 实际施加的缩放系数 `τ / ‖dw‖_F`（Python float） |
| `dtype` / `device` | `dw` 的 dtype 与 device |

```python
events = []
model.set_ttt_clip_hook(lambda ev: events.append(ev))
# ... forward / generate ...
model.set_ttt_clip_hook(None)
```

## 3. 注意事项

- **Hook 会触发 CUDA→host 同步** —— `pre_clip_norm` / `clip_scale` 用 `float(tensor.detach().cpu())` 拿到 Python 值，每次触发都同步一次。生产推理务必保持 `set_ttt_clip_hook(None)`，仅在调试 / 采样统计时开。未触发的 chunk 不会进 hook 调用（`if self.ttt_clip_hook is not None` 在外层），所以不挂 hook 时**零额外开销**。
- **τ 与其他 TTT 超参耦合** —— 论文报告值 `1e-5` 是在 `ttt_lr=3`、`ttt_chunk=4096` 设定下选的。若显著调大 `ttt_lr` 或 `ttt_chunk`，单步 `dw` 的典型范数会随之变化，建议先用 hook 采样一组 `pre_clip_norm` 看分布再调 τ，盲选可能导致几乎所有 chunk 都被压平（τ 太小）或裁剪完全失效（τ 太大）。
- **仅推理生效，训练侧不读** —— 训练阶段 `hf_models/` 不读 `ttt_clip_tau`，所以不会影响训练 loss / 梯度。若希望训练阶段也做类似稳定化，需要重写 §4 中提到的 prefix-sum 实现，本次改动未做。
- **τ 约束的是单步增量，不是累积 W** —— clipping 作用在每个 chunk 的 `dw` 上。即使每个 chunk 都被压到 τ，长序列下 `current_w` 仍可能远超 τ，这是设计预期，符合论文公式语义。

## 4. 实现要点

- **位置**：[modeling_qwen3.py:154-176](../inference_model/hf_qwen3/modeling_qwen3.py#L154-L176) / [modeling_llama.py:241-263](../inference_model/hf_llama3/modeling_llama.py#L241-L263)，紧贴 `dw` 计算之后、`current_w += dw` 之前——τ 约束的是单步增量，不是累积 W。
- **fp32 范数**：bf16 尾数 7 位，`1e-5` 量级比较会丢精度。`dw.to(float32)` 算 norm / scale，再 cast 回原 dtype。
- **训练侧不动**：训练用 `cumsum` 做并行 prefix-sum（论文 §3.4），逐 chunk 条件 clip 会失去向量化收益，且阈值处梯度不连续。
- **向后兼容**：`getattr(config, "ttt_clip_tau", 1e-5)`，旧 `config.json` 没字段也能加载。
- **Qwen3 / Llama 对称**，扩展规则需同步两边：

| | Qwen3 | Llama |
|---|---|---|
| Config | [configuration_qwen3.py:190](../inference_model/hf_qwen3/configuration_qwen3.py#L190) | [configuration_llama.py:196](../inference_model/hf_llama3/configuration_llama.py#L196) |
| MLP clip 块 | [modeling_qwen3.py:154-176](../inference_model/hf_qwen3/modeling_qwen3.py#L154-L176) | [modeling_llama.py:241-263](../inference_model/hf_llama3/modeling_llama.py#L241-L263) |
| Model API | [modeling_qwen3.py:512-515](../inference_model/hf_qwen3/modeling_qwen3.py#L512-L515) | [modeling_llama.py:516-519](../inference_model/hf_llama3/modeling_llama.py#L516-L519) |
| ForCausalLM | [modeling_qwen3.py:621-622](../inference_model/hf_qwen3/modeling_qwen3.py#L621-L622) | [modeling_llama.py:612-613](../inference_model/hf_llama3/modeling_llama.py#L612-L613) |
