# Changelog

## 1. 推理时 fast-weight 更新 clipping(论文附录 C.2)

为推理路径(`inference_model/hf_qwen3/`、`inference_model/hf_llama3/`)添加 per-chunk
快速权重增量的 Frobenius 范数裁剪,与论文附录 C.2 中用于 Qwen3-4B 评测的稳定化机制对齐。

### 改动要点

- **新增 config 字段** `ttt_clip_tau`,默认 `1e-5`(论文报告值)。设为 `<= 0` 可关闭裁剪。
  影响 `Qwen3Config` 与 `LlamaConfig`。
- **裁剪规则**:每个 chunk 计算 `dw = η · contract(h, t, ttt_proj)` 后,若
  `‖dw‖_F > τ`,则 `dw ← τ · dw / ‖dw‖_F`,再累加到 `current_w`。
  范数计算在 fp32 下完成以避免 bf16 精度问题。
- **作用域**:仅推理。训练侧(`hf_models/`)不变,以免破坏 autograd 与 prefix-sum。
- **向后兼容**:`getattr(config, "ttt_clip_tau", 1e-5)` 兜底,旧 checkpoint 无需重训。

### Hook 用法(可选,用于观测裁剪频率/幅度)

`Qwen3ForCausalLM` 与 `LlamaForCausalLM` 暴露 `set_ttt_clip_hook(hook)`。每次发生
裁剪时,hook 会被以一个 dict 回调,字段如下:

| key | 含义 |
|---|---|
| `layer_idx` | 触发裁剪的 decoder 层索引 |
| `chunk_idx` | 当前 chunk 在序列中的序号 |
| `chunk_start` / `chunk_end` | 该 chunk 覆盖的 token 区间 |
| `chunk_num` / `seq_len` | 当前 forward 的总 chunk 数与序列长度 |
| `pre_clip_norm` | 裁剪前的 `‖dw‖_F`(Python float) |
| `clip_tau` | 当前阈值 τ |
| `clip_scale` | 实际施加的缩放系数 `τ / ‖dw‖_F` |
| `dtype` / `device` | `dw` 的 dtype 与 device |

示例:

```python
import inference_model  # 触发 AutoModel 注册
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("/path/to/hf_ckpt")

events = []
model.set_ttt_clip_hook(lambda ev: events.append(ev))

# ... 跑一次 generate / forward ...

model.set_ttt_clip_hook(None)        # 关闭 hook
print(f"clip triggered {len(events)} times")
```

> 注意:hook 内的 `pre_clip_norm` / `clip_scale` 会触发 CUDA→host 同步,启用后
> 推理吞吐会下降。仅建议在调试或采样统计时打开,生产推理保持 `None`。
