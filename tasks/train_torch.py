# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import math
import os
import ast
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import timedelta
from functools import partial
import inspect
from typing import Any, Dict, List

# Force use HuggingFace backend for custom TTT models
os.environ["MODELING_BACKEND"] = "hf"

import torch
import torch.distributed as dist
import wandb
from tqdm import trange

# Import custom TTT models (must be before veomni imports to override AutoModel registration)
import hf_models  # noqa: F401

from veomni.checkpoint import build_checkpointer, ckpt_to_state_dict
from veomni.data import (
    build_chat_template,
    build_dataloader,
    build_dataset,
)
try:
    from veomni.data.constants import IGNORE_INDEX
except ImportError:
    from veomni.utils.constants import IGNORE_INDEX
from veomni.data import data_transform as _data_transform
process_pretrain_example = getattr(_data_transform, "process_pretrain_example", None) or getattr(
    _data_transform, "process_plaintext_example", None
)
process_sft_example = getattr(_data_transform, "process_sft_example", None) or getattr(
    _data_transform, "process_conversation_example", None
)
process_pretokenized_example = getattr(_data_transform, "process_pretokenized_example", None)
if process_pretrain_example is None or process_sft_example is None:
    raise ImportError("Installed veomni package does not provide required text data transform functions.")
from veomni.distributed.clip_grad_norm import veomni_clip_grad_norm
from veomni.distributed.offloading import build_activation_offloading_context
from veomni.distributed.parallel_state import get_parallel_state, init_parallel_state
from veomni.distributed.torch_parallelize import build_parallelize_model
from veomni.models import build_foundation_model, build_tokenizer, save_model_assets, save_model_weights
from veomni.optim import build_lr_scheduler, build_optimizer
from veomni.utils import helper
try:
    from veomni.utils.arguments import DataArguments, ModelArguments, TrainingArguments, parse_args, save_args
except ImportError:
    from veomni.arguments import DataArguments, ModelArguments, TrainingArguments, parse_args, save_args
from veomni.utils.device import (
    get_device_type,
    get_dist_comm_backend,
    get_torch_device,
    is_nccl_backend,
    synchronize,
)
from veomni.utils.dist_utils import all_reduce


logger = helper.create_logger(__name__)


@dataclass
class FreezeArguments:
    """
    支持在微调指令模型时进行选择性参数冻结（适用于仅微调 TTT 或 TTT+down_proj 的场景）。

    可通过 YAML 配置文件（顶层的 `freeze:` 字段）或命令行参数（`--freeze.enable true`）来启用。
    当 `enable=False` 时，不执行任何参数冻结操作（no-op），此时的训练行为与标准的全参数微调流程完全一致。
    """

    enable: bool = False
    # 对 `named_parameters()` 返回的参数名进行子串匹配。
    # 如果参数名中包含这里的任意一个模式（pattern），该参数将保持可训练。
    # 如果列表为空且 enable=True，则会冻结所有参数，并抛出异常（见 `_apply_freeze`）。
    trainable_patterns: List[str] = field(default_factory=list)
    # 仅当参数的完整名称中包含 `.layers.<i>.`（其中 `i` 属于 `config.ttt_layers`）时，才会匹配此处的模式。
    # 适用于那些每一层都存在、但只应在启用了 TTT 的层上进行训练的逐层模块（例如 `down_proj`）。
    trainable_on_ttt_layers: List[str] = field(default_factory=list)
    # 在 rank-0 上打印统计信息，并抽样展示部分可训练/已冻结的参数名称，以便进行正确性检查（sanity check）。
    verbose: bool = True


@dataclass
class Arguments:
    model: "ModelArguments" = field(default_factory=ModelArguments)
    data: "DataArguments" = field(default_factory=DataArguments)
    train: "TrainingArguments" = field(default_factory=TrainingArguments)
    freeze: FreezeArguments = field(default_factory=FreezeArguments)


def _filter_kwargs_for_callable(fn, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    sig = inspect.signature(fn)
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in sig.parameters.values()):
        return kwargs
    return {k: v for k, v in kwargs.items() if k in sig.parameters}


def _pop_dict_cli_arg(arg_name: str) -> Dict[str, Any] | None:
    if arg_name not in sys.argv:
        return None
    idx = sys.argv.index(arg_name)
    if idx + 1 >= len(sys.argv):
        raise ValueError(f"{arg_name} expects a value.")
    raw = sys.argv[idx + 1]
    del sys.argv[idx : idx + 2]
    parsed = None
    for fn in (json.loads, ast.literal_eval):
        try:
            candidate = fn(raw)
            if isinstance(candidate, dict):
                parsed = candidate
                break
        except Exception:
            continue
    if parsed is None:
        raise ValueError(f"{arg_name} expects a dict-like string, got: {raw}")
    return parsed


def _apply_freeze(model, freeze_cfg: "FreezeArguments", model_config) -> Dict[str, Any]:
    """
    根据冻结配置设置参数的 requires_grad 属性。

    必须在 `build_parallelize_model`（FSDP2 封装）之前和 `build_optimizer` 之前调用。
    对于 meta tensor 也是安全的，因为 requires_grad 只是一个标记，不依赖于具体数据；
    该标记可以在 PyTorch 的 `_apply` 路径（`model.float()` 和 `model.to_empty(...)` 都会用到）中保留，
    因为 `nn.Module._apply` 会通过 `nn.Parameter(t, requires_grad=orig.requires_grad)` 重新构建 Parameter 对象。

    返回一个可以直接合并到 wandb config 中的统计信息字典。
    无论如何都会返回（即使 enable=False 也会返回默认统计），因此调用层不需要写条件分支。

    No-op 路径（返回的 stats 中 `freeze.applied=False`）有两条：
      1. `enable=False` —— 静默 no-op。
      2. `enable=True` 但 `model_config.ttt_mode=False` —— rank-0 打 warning 后 fallback
         到全参微调，避免在没有 TTT 模块的模型上误把 down_proj 当成 TTT 目标训练。
    """
    n_total_elems_all = sum(p.numel() for p in model.parameters())
    n_param_tensors_all = sum(1 for _ in model.parameters())
    no_op_stats = {
        "freeze.applied": False,
        "freeze.trainable_param_tensors": n_param_tensors_all,
        "freeze.total_param_tensors": n_param_tensors_all,
        "freeze.trainable_elems": n_total_elems_all,
        "freeze.total_elems": n_total_elems_all,
        "freeze.trainable_ratio": 1.0,
    }
    if not freeze_cfg.enable:
        return no_op_stats

    if not bool(getattr(model_config, "ttt_mode", False)):
        logger.info_rank0(
            "[freeze][WARN] freeze.enable=True but model_config.ttt_mode=False. "
            "Selective freeze targets TTT modules (ttt_proj/ttt_conv) which do not "
            "exist when ttt_mode is off; trainable_on_ttt_layers would also match "
            "non-TTT down_proj layers. Skipping _apply_freeze and falling back to "
            "full fine-tuning. Set freeze.enable=false in your YAML to suppress."
        )
        return no_op_stats

    ttt_layer_ids = set(getattr(model_config, "ttt_layers", []) or [])
    ttt_layer_tokens = tuple(f".layers.{i}." for i in ttt_layer_ids)

    n_total, n_trainable = 0, 0
    trainable_names: List[str] = []
    frozen_names: List[str] = []
    for name, param in model.named_parameters():
        n_total += 1
        hit_global = any(pat in name for pat in freeze_cfg.trainable_patterns)
        hit_layer = bool(ttt_layer_tokens) and (
            any(pat in name for pat in freeze_cfg.trainable_on_ttt_layers)
            and any(tok in name for tok in ttt_layer_tokens)
        )
        if hit_global or hit_layer:
            param.requires_grad = True
            n_trainable += 1
            trainable_names.append(name)
        else:
            param.requires_grad = False
            frozen_names.append(name)

    n_trainable_elems = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total_elems = sum(p.numel() for p in model.parameters())
    ratio = n_trainable_elems / max(1, n_total_elems)

    if freeze_cfg.verbose:
        logger.info_rank0(
            f"[freeze] trainable param tensors: {n_trainable}/{n_total} "
            f"({n_trainable_elems / 1e6:.2f}M / {n_total_elems / 1e6:.2f}M = {100 * ratio:.3f}%)"
        )
        logger.info_rank0(f"[freeze] first 8 trainable: {trainable_names[:8]}")
        logger.info_rank0(f"[freeze] first 4 frozen:    {frozen_names[:4]}")

    if n_trainable == 0:
        raise RuntimeError(
            "[freeze] No trainable params after applying freeze rules. Check "
            f"trainable_patterns={freeze_cfg.trainable_patterns} and "
            f"trainable_on_ttt_layers={freeze_cfg.trainable_on_ttt_layers} "
            f"vs ttt_layers={sorted(ttt_layer_ids)}."
        )

    return {
        "freeze.applied": True,
        "freeze.trainable_param_tensors": n_trainable,
        "freeze.total_param_tensors": n_total,
        "freeze.trainable_elems": n_trainable_elems,
        "freeze.total_elems": n_total_elems,
        "freeze.trainable_ratio": ratio,
    }


def _compute_train_steps_compat(args, dataset_length):
    if hasattr(args.train, "compute_train_steps"):
        args.train.compute_train_steps(args.data.max_seq_len, args.data.train_size, dataset_length)
        return args.train.train_steps

    if args.data.datasets_type == "mapping" and dataset_length is not None:
        train_steps = math.floor(dataset_length / args.train.dataloader_batch_size)
    else:
        if getattr(args.train, "dyn_bsz", True):
            train_size = int(args.data.train_size * (1 + args.train.bsz_warmup_ratio / 2))
            train_steps = math.ceil(train_size / (args.train.global_batch_size * args.data.max_seq_len))
        else:
            train_sample = getattr(args.data, "train_sample", 10_000)
            train_steps = math.ceil(train_sample / args.train.dataloader_batch_size)

    if getattr(args.train, "max_steps", None) is not None and train_steps >= args.train.max_steps:
        return args.train.max_steps

    return train_steps


def _build_dataloader_compat(args, train_dataset, train_steps):
    dataloader_kwargs = dict(
        dataset=train_dataset,
        micro_batch_size=args.train.micro_batch_size,
        global_batch_size=args.train.global_batch_size,
        dataloader_batch_size=args.train.dataloader_batch_size,
        seed=args.train.seed,
        max_seq_len=args.data.max_seq_len,
        train_steps=train_steps,
        bsz_warmup_ratio=args.train.bsz_warmup_ratio,
        bsz_warmup_init_mbtoken=args.train.bsz_warmup_init_mbtoken,
        num_workers=args.data.num_workers,
        drop_last=args.data.drop_last,
        pin_memory=args.data.pin_memory,
        prefetch_factor=args.data.prefetch_factor,
        # Compatibility across veomni variants:
        rmpad=getattr(args.train, "rmpad", False),
        rmpad_with_pos_ids=getattr(args.train, "rmpad_with_pos_ids", False),
        dyn_bsz_margin=getattr(args.train, "dyn_bsz_margin", 0),
        dyn_bsz_buffer_size=getattr(args.data, "dyn_bsz_buffer_size", getattr(args.train, "dyn_bsz_buffer_size", 500)),
        dyn_bsz=getattr(args.train, "dyn_bsz", getattr(args.train, "rmpad", True)),
    )
    try:
        from veomni.data.data_loader import DATALOADER_REGISTRY

        builder = DATALOADER_REGISTRY[args.data.dataloader_type]
        dataloader_kwargs = _filter_kwargs_for_callable(builder, dataloader_kwargs)
    except Exception:
        pass
    return build_dataloader(dataloader_type=args.data.dataloader_type, **dataloader_kwargs)


def main():
    foundation_override = _pop_dict_cli_arg("--model.foundation")

    nccl_timeout = os.getenv("NCCL_TIMEOUT", None)
    pg_nccl_timeout = None
    if nccl_timeout is not None and is_nccl_backend():
        pg_nccl_timeout = timedelta(seconds=int(nccl_timeout))
    logger.info(f"Process_group timeout: {nccl_timeout}")
    dist.init_process_group(backend=get_dist_comm_backend(), timeout=pg_nccl_timeout)

    args = parse_args(Arguments)
    if foundation_override is not None:
        if args.model.foundation is None:
            args.model.foundation = {}
        args.model.foundation.update(foundation_override)
    logger.info(f"Process rank: {args.train.global_rank}, world size: {args.train.world_size}")
    logger.info_rank0(json.dumps(asdict(args), indent=2))
    get_torch_device().set_device(f"{get_device_type()}:{args.train.local_rank}")
    helper.set_seed(args.train.seed, args.train.enable_full_determinism)
    if args.train.local_rank == 0:
        helper.enable_third_party_logging()

    if args.train.global_rank == 0:
        save_args(args, args.train.output_dir)

    Checkpointer = build_checkpointer(dist_backend=args.train.data_parallel_mode, ckpt_manager=args.train.ckpt_manager)

    init_parallel_state(
        dp_size=args.train.data_parallel_size,
        dp_replicate_size=args.train.data_parallel_replicate_size,
        dp_shard_size=args.train.data_parallel_shard_size,
        tp_size=args.train.tensor_parallel_size,
        ep_size=args.train.expert_parallel_size,
        pp_size=args.train.pipeline_parallel_size,
        cp_size=args.train.context_parallel_size,
        ulysses_size=args.train.ulysses_parallel_size,
        dp_mode=args.train.data_parallel_mode,
    )

    logger.info_rank0("Prepare data")
    tokenizer = build_tokenizer(args.model.tokenizer_path)
    if args.data.data_type == "plaintext":
        transform = partial(
            process_pretrain_example,
            tokenizer=tokenizer,
            max_seq_len=args.data.max_seq_len,
            text_keys=args.data.text_keys,
        )
    elif args.data.data_type == "conversation":
        chat_template = build_chat_template(args.data.chat_template, tokenizer)
        transform = partial(
            process_sft_example,
            chat_template=chat_template,
            max_seq_len=args.data.max_seq_len,
            text_keys=args.data.text_keys,
        )
    elif args.data.data_type == "pretokenized":
        if process_pretokenized_example is None:
            raise NotImplementedError("Installed veomni package does not provide `process_pretokenized_example`.")
        transform = partial(
            process_pretokenized_example,
            input_ids_key=args.data.text_keys,  # text_keys is used as input_ids_key for pretokenized
        )
    else:
        raise NotImplementedError(f"Unsupported data type: {args.data.data_type}.")

    train_dataset = build_dataset(
        dataset_name=args.data.dataset_name,
        transform=transform,
        dataloader_batch_size=args.train.dataloader_batch_size,
        seed=args.train.seed,
        **asdict(args.data),
    )
    dataset_length = None if not hasattr(train_dataset, "__len__") else len(train_dataset)
    if args.data.datasets_type == "mapping":
        dataset_length = dataset_length / args.train.data_parallel_size
    train_steps = _compute_train_steps_compat(args, dataset_length)
    train_dataloader = _build_dataloader_compat(args, train_dataset, train_steps)

    logger.info_rank0("Prepare model")
    model = build_foundation_model(
        config_path=args.model.config_path,
        weights_path=args.model.model_path,
        torch_dtype="float32" if args.train.enable_mixed_precision else "bfloat16",
        attn_implementation=args.model.attn_implementation,
        moe_implementation=args.model.moe_implementation,
        init_device=args.train.init_device,
        config_kwargs=args.model.foundation,
    )
    model_config = model.config
    helper.print_device_mem_info("VRAM usage after building model")

    # 针对指令模型的仅 TTT 或 TTT+down_proj 微调，执行选择性参数冻结。
    # 该操作在 FSDP2 封装前的 meta tensor 上运行；build_optimizer 会自动跳过已冻结的参数。
    freeze_stats = _apply_freeze(model, args.freeze, model_config)

    get_optimizer_pre_hook = getattr(model, "get_optimizer_pre_hook", None)
    model = build_parallelize_model(
        model,
        init_device=args.train.init_device,
        weights_path=args.model.model_path,
        enable_full_shard=args.train.enable_full_shard,
        enable_mixed_precision=args.train.enable_mixed_precision,
        enable_gradient_checkpointing=args.train.enable_gradient_checkpointing,
        enable_fsdp_offload=args.train.enable_fsdp_offload,
        basic_modules=model._no_split_modules + args.model.basic_modules,
        enable_reentrant=args.train.enable_reentrant,
        enable_forward_prefetch=args.train.enable_forward_prefetch,
    )

    optimizer = build_optimizer(
        model,
        lr=args.train.lr,
        weight_decay=args.train.weight_decay,
        fused=True,
        optimizer_type=args.train.optimizer,
    )
    if get_optimizer_pre_hook is not None:
        optimizer_pre_hook = get_optimizer_pre_hook(model, model_config, args.train.data_parallel_mode)
        optimizer.register_step_pre_hook(optimizer_pre_hook)

    lr_scheduler = build_lr_scheduler(
        optimizer,
        train_steps=train_steps * args.train.num_train_epochs,
        lr=args.train.lr,
        lr_min=args.train.lr_min,
        lr_decay_style=args.train.lr_decay_style,
        lr_decay_ratio=args.train.lr_decay_ratio,
        lr_warmup_ratio=args.train.lr_warmup_ratio,
        lr_start=args.train.lr_start,
    )

    if args.train.global_rank == 0:
        if args.train.use_wandb:
            wandb.init(
                project=args.train.wandb_project,
                name=args.train.wandb_name,
                settings=wandb.Settings(console="off"),
                config={
                    **vars(args.model),
                    **vars(args.data),
                    **vars(args.train),
                    # 使用 freeze.* 前缀，避免与 TrainingArguments 中的键发生冲突（例如 `enable` 字段）。
                    **{f"freeze.{k}": v for k, v in vars(args.freeze).items()},
                    # 包含从 _apply_freeze 函数获取的推导统计信息（如 freeze.applied、freeze.trainable_ratio 等）。
                    **freeze_stats,
                },
            )

        # save model_assets before training
        if args.data.data_type in ["plaintext", "pretokenized"]:
            model_assets = [model_config, tokenizer]
        else:
            model_assets = [model_config, chat_template]
        save_model_assets(args.train.model_assets_dir, model_assets)

    if args.train.profile_this_rank:
        profiler = helper.create_profiler(
            start_step=args.train.profile_start_step,
            end_step=args.train.profile_end_step,
            trace_dir=args.train.profile_trace_dir,
            record_shapes=args.train.profile_record_shapes,
            profile_memory=args.train.profile_profile_memory,
            with_stack=args.train.profile_with_stack,
            global_rank=args.train.global_rank,
        )
        profiler.start()

    start_epoch, start_step, global_step = 0, 0, 0
    save_checkpoint_path = None
    environ_meter_kwargs = dict(
        config=model_config,
        global_batch_size=args.train.global_batch_size,
        rmpad=getattr(args.train, "rmpad", False),
        rmpad_with_pos_ids=getattr(args.train, "rmpad_with_pos_ids", False),
        empty_cache_steps=args.train.empty_cache_steps,
        enable_multisource=args.data.enable_multisource,
        dataloader=train_dataloader,
        data_path=args.data.train_path,
        gc_steps=getattr(args.train, "gc_steps", 0),
    )
    environ_meter = helper.EnvironMeter(**_filter_kwargs_for_callable(helper.EnvironMeter, environ_meter_kwargs))

    if args.train.load_checkpoint_path:
        state = {"model": model, "optimizer": optimizer, "extra_state": {}}  # cannot be None
        Checkpointer.load(args.train.load_checkpoint_path, state)
        global_step = state["extra_state"]["global_step"]
        start_epoch = global_step // train_steps
        start_step = global_step % train_steps
        lr_scheduler.load_state_dict(state["extra_state"]["lr_scheduler"])
        train_dataloader.load_state_dict(state["extra_state"]["train_dataloader"])
        environ_meter.load_state_dict(state["extra_state"]["environ_meter"])
        torch.set_rng_state(state["extra_state"]["torch_rng_state"])
        if start_step == 0:  # resume at the end of epoch
            iter(train_dataloader)  # clear resume state and prefetch data

        dist.barrier()
        logger.info_rank0(f"Load distributed checkpoint from {args.train.load_checkpoint_path} successfully!")

    helper.empty_cache()
    model_fwd_context, model_bwd_context = build_activation_offloading_context(
        args.train.enable_activation_offload, args.train.enable_gradient_checkpointing, args.train.activation_gpu_limit
    )
    model.train()
    logger.info(
        f"rank{args.train.local_rank} Start training, train_steps: {train_steps}, epochs: {args.train.num_train_epochs}"
    )
    for epoch in range(start_epoch, args.train.num_train_epochs):
        if hasattr(train_dataloader, "set_epoch"):
            train_dataloader.set_epoch(epoch)

        data_loader_tqdm = trange(
            train_steps,
            desc=f"Epoch {epoch + 1}/{args.train.num_train_epochs}",
            total=train_steps,
            initial=start_step,
            disable=args.train.local_rank != 0,
        )
        data_iterator = iter(train_dataloader)
        for _ in range(start_step, train_steps):
            global_step += 1

            try:
                micro_batches: List[Dict[str, Any]] = next(data_iterator)
            except StopIteration:
                logger.info(f"epoch:{epoch} Dataloader finished with drop_last {args.data.drop_last}")
                break

            if global_step == 1:
                helper.print_example(example=micro_batches[0], rank=args.train.local_rank)

            total_loss = 0
            synchronize()
            start_time = time.time()

            length_in_batch = torch.tensor(0, dtype=torch.int32, device=get_device_type())
            for micro_batch in micro_batches:
                length_in_batch += torch.sum(micro_batch["labels"] != IGNORE_INDEX)
            length_in_batch = all_reduce(length_in_batch, op="sum", group=get_parallel_state().fsdp_group)

            for micro_batch in micro_batches:
                environ_meter.add(micro_batch)
                if args.data.enable_multisource:
                    micro_batch.pop("ds_idx", None)
                    micro_batch.pop("cur_token_num", None)
                    micro_batch.pop("source_name", None)

                micro_batch = {
                    k: v.to(get_device_type(), non_blocking=True) if isinstance(v, torch.Tensor) else v
                    for k, v in micro_batch.items()
                }
                with model_fwd_context:
                    model_outputs = model(**micro_batch, use_cache=False)

                length_in_micro_batch = torch.sum(micro_batch["labels"] != IGNORE_INDEX)
                loss: "torch.Tensor" = (
                    model_outputs.loss * length_in_micro_batch / length_in_batch * get_parallel_state().dp_size
                )

                with model_bwd_context:
                    loss.backward()

                total_loss += loss.item()
                del micro_batch

            grad_norm = veomni_clip_grad_norm(model, args.train.max_grad_norm)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            if hasattr(grad_norm, "full_tensor"):
                grad_norm = grad_norm.full_tensor().item()

            # collect mean loss across data parallel group
            total_loss, grad_norm = all_reduce((total_loss, grad_norm), group=get_parallel_state().fsdp_group)
            synchronize()
            delta_time = time.time() - start_time
            lr = max(lr_scheduler.get_last_lr())
            train_metrics = environ_meter.step(delta_time, global_step=global_step)

            data_loader_tqdm.set_postfix_str(
                f"loss: {total_loss:.4f}, grad_norm: {grad_norm:.4f}, lr: {lr:.2e}", refresh=False
            )
            data_loader_tqdm.update()

            if args.train.global_rank == 0:
                if args.train.use_wandb:
                    train_metrics.update(
                        {"training/loss": total_loss, "training/grad_norm": grad_norm, "training/lr": lr}
                    )
                    wandb.log(train_metrics, step=global_step)

            if args.train.profile_this_rank and global_step <= args.train.profile_end_step:
                profiler.step()
                if global_step == args.train.profile_end_step:
                    profiler.stop()

            if args.train.save_steps and global_step % args.train.save_steps == 0:
                helper.empty_cache()
                save_checkpoint_path = os.path.join(args.train.save_checkpoint_path, f"global_step_{global_step}")
                state = {
                    "model": model,
                    "optimizer": optimizer,
                    "extra_state": {
                        "global_step": global_step,
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "train_dataloader": train_dataloader.state_dict(),
                        "environ_meter": environ_meter.state_dict(),
                        "torch_rng_state": torch.get_rng_state(),
                    },
                }
                Checkpointer.save(args.train.save_checkpoint_path, state, global_steps=global_step)

                dist.barrier()
                logger.info_rank0(f"Distributed checkpoint saved at {save_checkpoint_path} successfully!")

        data_loader_tqdm.close()
        start_step = 0
        helper.print_device_mem_info(f"VRAM usage after epoch {epoch + 1}")
        if args.train.save_epochs and (epoch + 1) % args.train.save_epochs == 0:
            helper.empty_cache()
            save_checkpoint_path = os.path.join(args.train.save_checkpoint_path, f"global_step_{global_step}")
            state = {
                "model": model,
                "optimizer": optimizer,
                "extra_state": {
                    "global_step": global_step,
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "train_dataloader": train_dataloader.state_dict(),
                    "environ_meter": environ_meter.state_dict(),
                    "torch_rng_state": torch.get_rng_state(),
                },
            }
            Checkpointer.save(args.train.save_checkpoint_path, state, global_steps=global_step)
            dist.barrier()
            logger.info_rank0(f"Distributed checkpoint saved at {save_checkpoint_path} successfully!")

    synchronize()
    # release memory
    del optimizer, lr_scheduler
    helper.empty_cache()
    # save model in huggingface's format
    if args.train.global_rank == 0 and args.train.save_hf_weights and save_checkpoint_path is not None:
        hf_weights_path = os.path.join(save_checkpoint_path, "hf_ckpt")
        model_state_dict = ckpt_to_state_dict(
            save_checkpoint_path=save_checkpoint_path,
            ckpt_manager=args.train.ckpt_manager,
        )
        save_model_weights(hf_weights_path, model_state_dict, model_assets=model_assets)
        logger.info_rank0(f"Huggingface checkpoint saved at {hf_weights_path} successfully!")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
