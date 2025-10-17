import os
import time
import math
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional

import wandb
import random
import torch
import torch.distributed as dist
from tqdm import trange
from safetensors.torch import load_file

from veomni.checkpoint import build_checkpointer, ckpt_to_state_dict
from veomni.data.diffusion.data_loader import build_dit_dataloader
from veomni.data.diffusion.dataset import build_text_dataset
from veomni.distributed.offloading import build_activation_offloading_context
from veomni.distributed.parallel_state import get_parallel_state, init_parallel_state
from veomni.distributed.torch_parallelize import build_parallelize_model
from veomni.models import save_model_assets

from veomni.optim import build_lr_scheduler, build_optimizer
from veomni.utils import helper
from veomni.utils.arguments import DataArguments, ModelArguments, TrainingArguments, parse_args, save_args
from veomni.utils.device import (
    get_device_type,
    get_nccl_backend,
    get_torch_device,
    synchronize,
)
from veomni.utils.dist_utils import all_reduce
from veomni.utils.dit_utils import EnvironMeter, save_model_weights
from veomni.utils.lora_utils import add_lora_to_model, freeze_parameters
from veomni.utils.recompute_utils import convert_ops_to_objects

# qwenimage from diffusers
from diffusers import QwenImagePipeline, QwenImageTransformer2DModel, FlowMatchEulerDiscreteScheduler
from transformers import Qwen2Tokenizer, Qwen2_5_VLForConditionalGeneration

logger = helper.create_logger(__name__)


@dataclass
class MyDataArguments(DataArguments):
    text_path: str = field(
        default=None,
        metadata={"help": "Text path."},
    )
    datasets_repeat: int = field(
        default=1,
        metadata={"help": "The number of times to repeat the datasets."},
    )


@dataclass
class MyModelArguments(ModelArguments):
    tokenizer_path: str = field(
        default=None,
        metadata={"help": "Path to the tokenizer."},
    )
    pretrained_text_encoder_path: str = field(
        default=None,
        metadata={"help": "Path to the pretrained text encoder."},
    )
    lora_rank: int = field(
        default=4,
        metadata={"help": "The dimension of the LoRA update matrices."},
    )
    lora_alpha: float = field(
        default=4.0,
        metadata={"help": "The weight of the LoRA update matrices."},
    )
    lora_target_modules: str = field(
        default="q,k,v,o,ffn.0,ffn.2",
        metadata={"help": "Modules to train with LoRA (must be in lora_target_modules_support)."},
    )
    lora_target_modules_support: str = field(
        default="q,k,v,o,ffn.0,ffn.2",
        metadata={"help": "All modules supported by the model for LoRA training."},
    )
    init_lora_weights: Optional[Literal["kaiming", "full"]] = field(
        default="kaiming",
        metadata={"help": "Initialization method for LoRA weights."},
    )
    pretrained_lora_path: str = field(
        default=None,
        metadata={"help": "Pretrained LoRA path. Required if the training is resumed."},
    )


@dataclass
class MyTrainingArguments(TrainingArguments):
    save_initial_model: bool = field(
        default=False,
        metadata={"help": "Whether or not to save the initial model."},
    )
    ops_to_save: List[str] = field(
        default_factory=list,
        metadata={"help": "Ops to save."},
    )
    train_architecture: Literal["lora", "full"] = field(
        default="full",
        metadata={"help": "Model structure to train. LoRA training or full training."},
    )


@dataclass
class Arguments:
    model: MyModelArguments = field(default_factory=MyModelArguments)
    data: MyDataArguments = field(default_factory=MyDataArguments)
    train: MyTrainingArguments = field(default_factory=MyTrainingArguments)


def get_param_groups(model: torch.nn.Module, default_lr: float):
    params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        params.append(param)
    return [
        {"params": params, "lr": default_lr},
    ]


def load_dit_model(dtype: torch.dtype = torch.bfloat16) -> QwenImageTransformer2DModel:
    config = {
        "patch_size": 2,
        "in_channels": 64,
        "out_channels": 16,
        "num_layers": 60,
        "attention_head_dim": 128,
        "num_attention_heads": 24,
        "joint_attention_dim": 3584,
        "guidance_embeds": False,
        "axes_dims_rope": [
            16,
            56,
            56
        ],
        # "pooled_projection_dim": 768 # useless
    }
    model = QwenImageTransformer2DModel(**config)
    model = model.to(dtype=dtype)
    return model


def load_safetensors(path_or_dir: str) -> Dict:
    path_or_dir = Path(path_or_dir)
    state_dict = {}
    if path_or_dir.is_dir():
        for p in path_or_dir.glob('*.safetensors'):
            state_dict.update(load_file(p))
    elif path_or_dir.suffix == '.safetensors':
        state_dict.update(load_file(path_or_dir))
    else:
        raise ValueError(f'{path_or_dir=} is not supported!')
    return state_dict


def get_random_step_id(num_steps: int) -> int:
    group = get_parallel_state()
    rank = dist.get_rank(group.fsdp_group) if dist.is_initialized() else 0

    if rank == 0:
        indices = torch.randint(
            low=0,
            high=num_steps,
            size=(1,),
            dtype=torch.long,
            device=get_torch_device(),
        )
    else:
        indices = torch.empty(
            (1,),
            dtype=torch.long,
            device=get_torch_device(),
        )

    if dist.is_initialized():
        dist.broadcast(indices, src=0, group=group.fsdp_group)
    return indices[0].item()


def main():
    args = parse_args(Arguments)
    get_torch_device().set_device(f"{get_device_type()}:{args.train.local_rank}")
    dist.init_process_group(backend=get_nccl_backend())
    helper.set_seed(args.train.seed, args.train.enable_full_determinism)
    if args.train.global_rank == 0:
        save_args(args, args.train.output_dir)

    Checkpointer = build_checkpointer(
        dist_backend=args.train.data_parallel_mode,
        ckpt_manager=args.train.ckpt_manager,
    )

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
    logger.info_rank0(
        f"Parallel state: dp:{args.train.data_parallel_mode}, tp:{args.train.tensor_parallel_size}, ep:{args.train.expert_parallel_size}, pp:{args.train.pipeline_parallel_size}, cp:{args.train.context_parallel_size}, ulysses:{args.train.ulysses_parallel_size}"
    )

    train_dataset = build_text_dataset(
        text_path=args.data.text_path,
        datasets_repeat=args.data.datasets_repeat,
    )

    args.train.compute_train_steps(
        args.data.max_seq_len,
        args.data.train_size,
        len(train_dataset) // args.train.data_parallel_size,
    )

    train_dataloader = build_dit_dataloader(
        dataset=train_dataset,
        micro_batch_size=args.train.micro_batch_size,
        global_batch_size=args.train.global_batch_size,
        dataloader_batch_size=args.train.dataloader_batch_size,
        seed=args.train.seed,
        train_steps=args.train.train_steps,
        num_workers=args.data.num_workers,
        drop_last=args.data.drop_last,
        pin_memory=args.data.pin_memory,
        prefetch_factor=args.data.prefetch_factor,
    )
    # build foundation model
    generator: QwenImageTransformer2DModel = load_dit_model(torch.bfloat16)
    fake_score: QwenImageTransformer2DModel = load_dit_model(torch.bfloat16)
    real_score: QwenImageTransformer2DModel = load_dit_model(torch.bfloat16)

    state_dict = load_safetensors(args.model.model_path)

    generator.load_state_dict(state_dict, strict=True)
    fake_score.load_state_dict(state_dict, strict=True)
    real_score.load_state_dict(state_dict, strict=True)

    generator.micro_batch_size = args.train.micro_batch_size
    generator.requires_grad_(True)

    fake_score.micro_batch_size = args.train.micro_batch_size
    fake_score.requires_grad_(True)

    real_score.eval()
    real_score.requires_grad_(False)

    tokenizer = Qwen2Tokenizer.from_pretrained(
        args.model.model_path,
        subfolder='tokenizer',
    )

    text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model.model_path,
        subfolder='text_encoder',
        dtype=torch.bfloat16,
    )
    text_encoder.eval()
    text_encoder.requires_grad_(False)
    text_encoder.to(device=get_torch_device())

    pipeline: QwenImagePipeline = QwenImagePipeline.from_pretrained(
        args.model.model_path,
        scheduler=None,
        vae=None,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        transformer=None,
    )

    helper.print_device_mem_info("VRAM usage after building model")

    if args.train.train_architecture == "lora":
        logger.info_rank0("train_architecture is lora")
        _use_orig_params = True
        freeze_parameters(generator)
        freeze_parameters(fake_score)
        add_lora_to_model(
            generator,
            lora_rank=args.model.lora_rank,
            lora_alpha=args.model.lora_alpha,
            lora_target_modules=args.model.lora_target_modules,
            init_lora_weights=args.model.init_lora_weights,
            pretrained_lora_path=args.model.pretrained_lora_path,
            lora_target_modules_support=args.model.lora_target_modules_support.split(","),
        )
        add_lora_to_model(
            fake_score,
            lora_rank=args.model.lora_rank,
            lora_alpha=args.model.lora_alpha,
            lora_target_modules=args.model.lora_target_modules,
            init_lora_weights=args.model.init_lora_weights,
            pretrained_lora_path=args.model.pretrained_lora_path,
            lora_target_modules_support=args.model.lora_target_modules_support.split(","),
        )
    else:
        logger.info_rank0("train_architecture is full")
        _use_orig_params = False

    logger.info_rank0(f"generator: {generator}")

    if args.train.save_initial_model:
        if args.train.global_rank == 0:
            generator_state_dict = generator.state_dict()
            generator_state_dict = {k: v for k, v in state_dict.items() if "lora" in k}
            save_model_weights(args.train.output_dir, generator_state_dict, model_assets=[generator.config])

        dist.barrier()
        return

    ops_to_save = convert_ops_to_objects(args.train.ops_to_save)

    def _build_parallelize_model(model):
        model = build_parallelize_model(
            model,
            enable_full_shard=args.train.enable_full_shard,
            enable_mixed_precision=args.train.enable_mixed_precision,
            enable_gradient_checkpointing=args.train.enable_gradient_checkpointing,
            init_device=args.train.init_device,
            enable_fsdp_offload=args.train.enable_fsdp_offload,
            basic_modules=model._no_split_modules,
            enable_reentrant=args.train.enable_reentrant,
            enable_forward_prefetch=args.train.enable_forward_prefetch,
            use_orig_params=_use_orig_params,
            ops_to_save=ops_to_save,
        )
        return model

    generator = _build_parallelize_model(generator)
    fake_score = _build_parallelize_model(fake_score)

    def _build_optimizer(model):
        model = build_optimizer(
            model,
            lr=args.train.lr,
            weight_decay=args.train.weight_decay,
            fused=True,
            optimizer_type=args.train.optimizer,
            param_groups=get_param_groups(model, args.train.lr),
        )
        return model

    generator_optimizer = _build_optimizer(generator)
    fake_score_optimizer = _build_optimizer(fake_score)

    if args.train.global_rank == 0:
        if args.train.use_wandb:
            wandb.init(
                project=args.train.wandb_project,
                name=args.train.wandb_name,
                config={**vars(args.model), **vars(args.data), **vars(args.train)},  # flatten dict
            )

        if args.train.enable_profiling:
            profiler = helper.create_profiler(
                start_step=args.train.profile_start_step,
                end_step=args.train.profile_end_step,
                trace_dir=args.train.profile_trace_dir,
                record_shapes=args.train.profile_record_shapes,
                profile_memory=args.train.profile_profile_memory,
                with_stack=args.train.profile_with_stack,
                global_rank=get_parallel_state().global_rank,
            )
            profiler.start()

        save_model_assets(args.train.model_assets_dir, [generator.config])

    shift = 3.0
    flow_scheduler: FlowMatchEulerDiscreteScheduler = FlowMatchEulerDiscreteScheduler.from_config(
        {
            "base_image_seq_len": 256,
            "base_shift": math.log(shift),  # We use shift=3 in distillation
            "invert_sigmas": False,
            "max_image_seq_len": 8192,
            "max_shift": math.log(shift),  # We use shift=3 in distillation
            "num_train_timesteps": 1000,
            "shift": 1.0,
            "shift_terminal": None,  # set shift_terminal to None
            "stochastic_sampling": False,
            "time_shift_type": "exponential",
            "use_beta_sigmas": False,
            "use_dynamic_shifting": True,
            "use_exponential_sigmas": False,
            "use_karras_sigmas": False,
        }
    )
    flow_scheduler.set_timesteps(num_inference_steps=1000)
    flow_scheduler.timesteps = flow_scheduler.timesteps.to(device=get_torch_device())
    flow_scheduler.sigmas = flow_scheduler.sigmas.to(device=get_torch_device())

    def add_noise(
            original_samples: torch.Tensor,
            noise: torch.Tensor,
            timestep: torch.Tensor,
    ):
        timesteps = flow_scheduler.timesteps
        sigmas = flow_scheduler.sigmas

        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(),
            dim=1,
        )
        sigma = sigmas[timestep_id]
        sample = (1 - sigma) * original_samples + sigma * noise
        return sample.type_as(noise)

    def convert_x0_to_flow_pred(x0_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        original_dtype = x0_pred.dtype
        device = x0_pred.device

        x0_pred = x0_pred.to(dtype=torch.float32, device=device)
        xt = xt.to(dtype=torch.float32, device=device)
        sigmas = flow_scheduler.sigmas.to(dtype=torch.float32, device=device)
        timesteps = flow_scheduler.timesteps.to(dtype=torch.float32, device=device)

        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(),
            dim=1,
        )
        sigma_t = sigmas[timestep_id]
        flow_pred = (xt - x0_pred) / sigma_t
        return flow_pred.to(original_dtype)

    denoising_step_list = torch.tensor([1000, 750, 500, 250], dtype=torch.int64)

    total_train_steps = args.train.train_steps * args.train.num_train_epochs

    def _build_lr_scheduler(optimizer):
        optimizer = build_lr_scheduler(
            optimizer,
            train_steps=total_train_steps,
            lr=args.train.lr,
            lr_min=args.train.lr_min,
            lr_decay_style=args.train.lr_decay_style,
            lr_decay_ratio=args.train.lr_decay_ratio,
            lr_warmup_ratio=args.train.lr_warmup_ratio,
            lr_start=args.train.lr_start,
        )
        return optimizer

    generator_lr_scheduler = _build_lr_scheduler(generator_optimizer)
    fake_score_lr_scheduler = _build_lr_scheduler(fake_score_optimizer)

    start_epoch, start_step, global_step = 0, 0, 0
    save_checkpoint_path = None
    environ_meter = EnvironMeter(
        config=[generator.config],
        global_batch_size=args.train.global_batch_size,
        empty_cache_steps=args.train.empty_cache_steps,
    )

    if args.train.load_checkpoint_path:
        generator_state = {"model": generator, "optimizer": generator_optimizer, "extra_state": {}}  # cannot be None
        fake_score_state = {"model": fake_score, "optimizer": fake_score_optimizer, "extra_state": {}}  # cannot be None
        Checkpointer.load(os.path.join(args.train.load_checkpoint_path, 'generator'), generator_state)
        Checkpointer.load(os.path.join(args.train.load_checkpoint_path, 'fake_score'), fake_score_state)

        global_step = fake_score_state["extra_state"]["global_step"]
        start_epoch = global_step // args.train.train_steps
        start_step = global_step % args.train.train_steps

        generator_lr_scheduler.load_state_dict(generator_state["extra_state"]["lr_scheduler"])
        fake_score_lr_scheduler.load_state_dict(fake_score_state["extra_state"]["lr_scheduler"])

        train_dataloader.load_state_dict(fake_score_lr_scheduler["extra_state"]["train_dataloader"])
        environ_meter.load_state_dict(fake_score_lr_scheduler["extra_state"]["environ_meter"])
        torch.set_rng_state(fake_score_lr_scheduler["extra_state"]["torch_rng_state"])

        if start_step == 0:  # resume at the end of epoch
            iter(train_dataloader)  # clear resume state and prefetch data

        dist.barrier()
        logger.info_rank0(f"Load distributed checkpoint from {args.train.load_checkpoint_path} successfully!")

    helper.empty_cache()

    model_fwd_context, model_bwd_context = build_activation_offloading_context(
        args.train.enable_activation_offload,
        args.train.enable_gradient_checkpointing,
        args.train.activation_gpu_limit,
    )

    shapes = [
        (1328, 1328),
        (1664, 928),
        (928, 1664),
        (1472, 1140),
        (1140, 1472),
        (1584, 1056),
        (1056, 1584),
    ]
    num_channels_latents = 16
    train_generator_step = 5
    dmd_distill_step = 4

    fake_guidance_scale = 0.0
    real_guidance_scale = 4.0

    def gen_noise(batch_size: int, height: int, width: int, generator: torch.Generator | None = None) -> torch.Tensor:
        noise = torch.randn(
            (batch_size, 1, num_channels_latents, height // 8, width // 8),
            device=get_torch_device(),
            dtype=torch.bfloat16,
            generator=generator,
        )
        noise = pipeline._pack_latents(
            noise,
            batch_size=batch_size,
            num_channels_latents=num_channels_latents,
            height=height // 8,
            width=width // 8,
        )
        return noise

    with torch.inference_mode():
        negative_prompt_embeds, negative_prompt_embeds_mask = pipeline.encode_prompt(
            prompt=[''],
            prompt_embeds=None,
            prompt_embeds_mask=None,
            device=get_torch_device(),
            num_images_per_prompt=1,
            max_sequence_length=512,
        )

    neg_txt_seq_lens = negative_prompt_embeds_mask.sum(dim=1).tolist()

    helper.empty_cache()
    generator.train()
    fake_score.train()
    logger.info(
        f"rank{args.train.local_rank} Start training, train_steps: {args.train.train_steps}, epochs: {args.train.num_train_epochs}"
    )

    for epoch in range(start_epoch, args.train.num_train_epochs):
        if hasattr(train_dataloader, "set_epoch"):
            train_dataloader.set_epoch(epoch)

        data_loader_tqdm = trange(
            args.train.train_steps,
            desc=f"Epoch {epoch + 1}/{args.train.num_train_epochs}",
            total=args.train.train_steps,
            initial=start_step,
            disable=args.train.local_rank != 0,
        )
        epoch_start_time = time.time()
        data_iterator = iter(train_dataloader)

        epoch_loss = 0
        for _ in range(args.train.train_steps):
            train_generator = global_step % train_generator_step == 0
            global_step += 1
            synchronize()
            generator_loss = 0
            fake_score_loss = 0
            generator_grad_norm = 0
            fake_score_grad_norm = 0

            start_time = time.time()

            if train_generator:
                generator_optimizer.zero_grad()
                try:
                    micro_batches: List[Dict[str, str]] = next(data_iterator)
                except StopIteration:
                    logger.info(f"epoch:{epoch} Dataloader finished with drop_last {args.data.drop_last}")
                    break

                # Data
                text = [batch["text"] for batch in micro_batches]
                width, height = random.sample(shapes, k=1)[0]
                batch_size = len(text)

                with torch.inference_mode():
                    prompt_embeds, prompt_embeds_mask = pipeline.encode_prompt(
                        prompt=text,
                        prompt_embeds=None,
                        prompt_embeds_mask=None,
                        device=get_torch_device(),
                        num_images_per_prompt=1,
                        max_sequence_length=512,
                    )
                img_shapes = [1, height // 16, width // 16] * batch_size
                txt_seq_lens = prompt_embeds_mask.sum(dim=1).tolist()
                exit_step = get_random_step_id(dmd_distill_step)

                noisy_image = gen_noise(batch_size, height, width)

                for index, cur_timestep in enumerate(denoising_step_list):
                    timestep = torch.full(
                        (batch_size,),
                        fill_value=float(cur_timestep.item()),
                        dtype=torch.float32,
                        device=get_torch_device(),
                    )
                    if index != exit_step:
                        # disable grad
                        with torch.inference_mode():
                            denoised_pred = generator(
                                hidden_states=noisy_image,
                                encoder_hidden_states=prompt_embeds,
                                encoder_hidden_states_mask=prompt_embeds_mask,
                                timestep=timestep,
                                img_shapes=img_shapes,
                                txt_seq_lens=txt_seq_lens,
                                return_dict=False,
                            )[0]
                            next_timestep = torch.full(
                                (batch_size,),
                                fill_value=float(denoising_step_list[index + 1].item()),
                                dtype=torch.float32,
                                device=get_torch_device(),
                            )
                            noisy_image = add_noise(
                                denoised_pred,
                                gen_noise(batch_size, height, width),
                                next_timestep,
                            )
                    else:
                        # enable grad
                        with model_fwd_context:
                            denoised_pred = generator(
                                hidden_states=noisy_image,
                                encoder_hidden_states=prompt_embeds,
                                encoder_hidden_states_mask=prompt_embeds_mask,
                                timestep=timestep,
                                img_shapes=img_shapes,
                                txt_seq_lens=txt_seq_lens,
                                return_dict=False,
                            )[0]
                            break

                # denoised_pred, denoised_timestep_to, denoised_timestep_from
                pred_image = denoised_pred

                # disable grad
                with torch.inference_mode():
                    timestep = torch.randint(
                        0, 1000,
                        (batch_size,),
                        device=get_torch_device(),
                        dtype=torch.int64,
                    ).to(dtype=torch.float32)
                    if shift > 1:
                        sigma = timestep / 1000
                        shift_sigma = shift * sigma / (1 + (shift - 1) * sigma)
                        timestep = shift_sigma * 1000

                    timestep.clamp_(20, 980)  # [0.02-0.98]

                    noisy_image = add_noise(
                        denoised_pred,
                        gen_noise(batch_size, height, width),
                        timestep,
                    ).detach()

                    pred_fake_image_cond = fake_score(
                        hidden_states=noisy_image,
                        encoder_hidden_states=prompt_embeds,
                        encoder_hidden_states_mask=prompt_embeds_mask,
                        timestep=timestep,
                        img_shapes=img_shapes,
                        txt_seq_lens=txt_seq_lens,
                        return_dict=False,
                    )[0]

                    if fake_guidance_scale != 0.0:

                        pred_fake_image_uncond = fake_score(
                            hidden_states=noisy_image,
                            encoder_hidden_states=negative_prompt_embeds,
                            encoder_hidden_states_mask=negative_prompt_embeds_mask,
                            timestep=timestep,
                            img_shapes=img_shapes,
                            txt_seq_lens=neg_txt_seq_lens,
                            return_dict=False,
                        )[0]

                        comb_pred = pred_fake_image_uncond + fake_guidance_scale * (
                                pred_fake_image_cond - pred_fake_image_uncond)
                        cond_norm = torch.norm(pred_fake_image_cond, dim=-1, keepdim=True)
                        noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
                        pred_fake_image = comb_pred * (cond_norm / noise_norm)
                    else:
                        pred_fake_image = pred_fake_image_cond

                    pred_real_image_cond = real_score(
                        hidden_states=noisy_image,
                        encoder_hidden_states=prompt_embeds,
                        encoder_hidden_states_mask=prompt_embeds_mask,
                        timestep=timestep,
                        img_shapes=img_shapes,
                        txt_seq_lens=txt_seq_lens,
                        return_dict=False,
                    )[0]

                    pred_real_image_uncond = real_score(
                        hidden_states=noisy_image,
                        encoder_hidden_states=negative_prompt_embeds,
                        encoder_hidden_states_mask=negative_prompt_embeds_mask,
                        timestep=timestep,
                        img_shapes=img_shapes,
                        txt_seq_lens=neg_txt_seq_lens,
                        return_dict=False,
                    )[0]

                    comb_pred = pred_real_image_uncond + real_guidance_scale * (
                            pred_real_image_cond - pred_real_image_uncond)
                    cond_norm = torch.norm(pred_real_image_cond, dim=-1, keepdim=True)
                    noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
                    pred_real_image = comb_pred * (cond_norm / noise_norm)

                    grad = (pred_fake_image - pred_real_image)

                    p_real = (pred_image - pred_real_image)
                    normalizer = torch.abs(p_real).mean(dim=[1, 2], keepdim=True)
                    grad = grad / normalizer

                    # dmd_train_gradient_norm = torch.mean(torch.abs(grad)).detach()

                with model_fwd_context:
                    dmd_loss = 0.5 * torch.nn.functional.mse_loss(
                        pred_image.float(),  # enable grad
                        (pred_image.float() - grad.float()).detach(),  # disable grad
                        reduction='mean',
                    )

                with model_bwd_context:
                    dmd_loss.backward()

                generator_loss = dmd_loss.item()
                if args.train.data_parallel_mode == "fsdp1":
                    grad_norm = generator.clip_grad_norm_(args.train.max_grad_norm).item()
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(generator.parameters(), args.train.max_grad_norm,
                                                               foreach=True)

                if hasattr(grad_norm, "full_tensor"):
                    grad_norm = grad_norm.full_tensor().item()
                generator_grad_norm = grad_norm
                generator_optimizer.step()

            try:
                micro_batches: List[Dict[str, str]] = next(data_iterator)
            except StopIteration:
                logger.info(f"epoch:{epoch} Dataloader finished with drop_last {args.data.drop_last}")
                break

            fake_score_optimizer.zero_grad()

            # Data
            text = [batch["text"] for batch in micro_batches]
            width, height = random.sample(shapes, k=1)[0]
            batch_size = len(text)

            with torch.inference_mode():
                prompt_embeds, prompt_embeds_mask = pipeline.encode_prompt(
                    prompt=text,
                    prompt_embeds=None,
                    prompt_embeds_mask=None,
                    device=get_torch_device(),
                    num_images_per_prompt=1,
                    max_sequence_length=512,
                )
                img_shapes = [1, height // 16, width // 16] * batch_size
                txt_seq_lens = prompt_embeds_mask.sum(dim=1).tolist()
                exit_step = get_random_step_id(dmd_distill_step)

                noisy_image = gen_noise(batch_size, height, width)

                for index, cur_timestep in enumerate(denoising_step_list):
                    timestep = torch.full(
                        (batch_size,),
                        fill_value=float(cur_timestep.item()),
                        dtype=torch.float32,
                        device=get_torch_device(),
                    )
                    if index != exit_step:
                        # disable grad
                        denoised_pred = generator(
                            hidden_states=noisy_image,
                            encoder_hidden_states=prompt_embeds,
                            encoder_hidden_states_mask=prompt_embeds_mask,
                            timestep=timestep,
                            img_shapes=img_shapes,
                            txt_seq_lens=txt_seq_lens,
                            return_dict=False,
                        )[0]
                        next_timestep = torch.full(
                            (batch_size,),
                            fill_value=float(denoising_step_list[index + 1].item()),
                            dtype=torch.float32,
                            device=get_torch_device(),
                        )
                        noisy_image = add_noise(
                            denoised_pred,
                            gen_noise(batch_size, height, width),
                            next_timestep,
                        )
                    else:
                        # disable grad
                        denoised_pred = generator(
                            hidden_states=noisy_image,
                            encoder_hidden_states=prompt_embeds,
                            encoder_hidden_states_mask=prompt_embeds_mask,
                            timestep=timestep,
                            img_shapes=img_shapes,
                            txt_seq_lens=txt_seq_lens,
                            return_dict=False,
                        )[0]
                        break

            # enable grad
            # denoised_pred, denoised_timestep_to, denoised_timestep_from
            generate_image = denoised_pred

            timestep = torch.randint(
                0, 1000,
                (batch_size,),
                device=get_torch_device(),
                dtype=torch.int64,
            ).to(dtype=torch.float32)

            if shift > 1:
                sigma = timestep / 1000
                shift_sigma = shift * sigma / (1 + (shift - 1) * sigma)
                timestep = shift_sigma * 1000
            timestep.clamp_(20, 980)  # [0.02-0.98]

            noisy = gen_noise(batch_size, height, width)

            generate_noisy_image = add_noise(
                generate_image,
                noisy,
                timestep,
            )

            with model_fwd_context:
                pred_fake_image = fake_score(
                    hidden_states=generate_noisy_image,
                    encoder_hidden_states=prompt_embeds,
                    encoder_hidden_states_mask=prompt_embeds_mask,
                    timestep=timestep,
                    img_shapes=img_shapes,
                    txt_seq_lens=txt_seq_lens,
                    return_dict=False,
                )[0]

                flow_pred = convert_x0_to_flow_pred(
                    pred_fake_image,
                    generate_noisy_image,
                    timestep,
                )

                critic_loss = torch.mean(
                    (flow_pred - (noisy - generate_image)) ** 2
                )

            with model_bwd_context:
                critic_loss.backward()

            if args.train.data_parallel_mode == "fsdp1":
                grad_norm = fake_score.clip_grad_norm_(args.train.max_grad_norm).item()
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(fake_score.parameters(), args.train.max_grad_norm,
                                                           foreach=True)

            if hasattr(grad_norm, "full_tensor"):
                grad_norm = grad_norm.full_tensor().item()

            if hasattr(grad_norm, "full_tensor"):
                grad_norm = grad_norm.full_tensor().item()
            fake_score_grad_norm = grad_norm

            fake_score_optimizer.step()
            fake_score_loss = critic_loss.item()
            del micro_batches

            generator_loss, generator_grad_norm, fake_score_loss, fake_score_grad_norm = all_reduce(
                [generator_loss, generator_grad_norm, fake_score_loss, fake_score_grad_norm],
                group=get_parallel_state().fsdp_group,
            )

            total_loss = generator_loss + fake_score_loss
            total_grad_norm = generator_grad_norm + fake_score_grad_norm
            epoch_loss += total_loss
            synchronize()
            delta_time = time.time() - start_time
            generator_lr = max(generator_lr_scheduler.get_last_lr())
            fake_score_lr = max(fake_score_lr_scheduler.get_last_lr())
            train_metrics = environ_meter.step(delta_time, global_step=global_step)

            data_loader_tqdm.set_postfix_str(
                f"loss: {total_loss:.4f}, grad_norm: {total_grad_norm:.2f}, gen_lr: {generator_lr:.2e}, fake_lr: {fake_score_lr}, step_time: {delta_time:.2f}s"
            )
            data_loader_tqdm.update()

            if args.train.global_rank == 0:
                if args.train.use_wandb:
                    train_metrics.update(
                        {"training/loss": total_loss, "training/grad_norm": total_grad_norm,
                         "training/gen_lr": generator_lr, "training/fake_lr": fake_score_lr}
                    )
                    wandb.log(train_metrics, step=global_step)

                if args.train.enable_profiling and global_step <= args.train.profile_end_step:
                    profiler.step()
                    if global_step == args.train.profile_end_step:
                        profiler.stop()
            if args.train.save_steps and global_step % args.train.save_steps == 0:
                helper.empty_cache()
                generator_state = {
                    "model": generator,
                    "optimizer": generator_optimizer,
                    "extra_state": {
                        "global_step": global_step,
                        "lr_scheduler": generator_lr_scheduler.state_dict(),
                        "train_dataloader": train_dataloader.state_dict(),
                        "environ_meter": environ_meter.state_dict(),
                        "torch_rng_state": torch.get_rng_state(),
                    },
                }
                fake_score_state = {
                    "model": fake_score,
                    "optimizer": fake_score_optimizer,
                    "extra_state": {
                        "global_step": global_step,
                        "lr_scheduler": fake_score_lr_scheduler.state_dict(),
                        "train_dataloader": train_dataloader.state_dict(),
                        "environ_meter": environ_meter.state_dict(),
                        "torch_rng_state": torch.get_rng_state(),
                    },
                }
                generator_save_checkpoint_path = os.path.join(args.train.save_checkpoint_path, "generator",
                                                              f"global_step_{global_step}")
                fake_score_save_checkpoint_path = os.path.join(args.train.save_checkpoint_path, "fake_score",
                                                               f"global_step_{global_step}")
                Checkpointer.save(os.path.join(args.train.save_checkpoint_path, "generator"), generator_state,
                                  global_steps=global_step)
                Checkpointer.save(os.path.join(args.train.save_checkpoint_path, "fake_score"), fake_score_state,
                                  global_steps=global_step)
                if args.train.global_rank == 0:
                    save_hf_weights(args, generator_save_checkpoint_path, [generator.config])
                    save_hf_weights(args, fake_score_save_checkpoint_path, [fake_score.config])

        data_loader_tqdm.close()
        epoch_time = time.time() - epoch_start_time
        start_step = 0
        helper.print_device_mem_info(f"VRAM usage after epoch {epoch + 1}")
        if args.train.global_rank == 0:
            logger.info_rank0(
                f"Epoch {epoch + 1} completed, epoch_time={epoch_time:.4f}s, epoch_loss={epoch_loss / args.train.train_steps:.4f}"
            )
        if args.train.global_rank == 0:
            if args.train.use_wandb:
                wandb.log({"training/loss_per_epoch": epoch_loss / args.train.train_steps}, step=global_step)
        if args.train.save_epochs and (epoch + 1) % args.train.save_epochs == 0:
            helper.empty_cache()
            generator_state = {
                "model": generator,
                "optimizer": generator_optimizer,
                "extra_state": {
                    "global_step": global_step,
                    "lr_scheduler": generator_lr_scheduler.state_dict(),
                    "train_dataloader": train_dataloader.state_dict(),
                    "environ_meter": environ_meter.state_dict(),
                    "torch_rng_state": torch.get_rng_state(),
                },
            }
            fake_score_state = {
                "model": fake_score,
                "optimizer": fake_score_optimizer,
                "extra_state": {
                    "global_step": global_step,
                    "lr_scheduler": fake_score_lr_scheduler.state_dict(),
                    "train_dataloader": train_dataloader.state_dict(),
                    "environ_meter": environ_meter.state_dict(),
                    "torch_rng_state": torch.get_rng_state(),
                },
            }

            generator_save_checkpoint_path = os.path.join(args.train.save_checkpoint_path, "generator",
                                                          f"global_step_{global_step}")
            fake_score_save_checkpoint_path = os.path.join(args.train.save_checkpoint_path, "fake_score",
                                                           f"global_step_{global_step}")
            Checkpointer.save(os.path.join(args.train.save_checkpoint_path, "generator"), generator_state,
                              global_steps=global_step)
            Checkpointer.save(os.path.join(args.train.save_checkpoint_path, "fake_score"), fake_score_state,
                              global_steps=global_step)
            if args.train.global_rank == 0:
                save_hf_weights(args, generator_save_checkpoint_path, [generator.config])
                save_hf_weights(args, fake_score_save_checkpoint_path, [fake_score.config])

    synchronize()
    # release memory
    del generator_optimizer, generator_lr_scheduler, fake_score_optimizer, fake_score_lr_scheduler
    helper.empty_cache()

    dist.barrier()
    dist.destroy_process_group()


def save_hf_weights(args, save_checkpoint_path, model_assets):
    hf_weights_path = os.path.join(save_checkpoint_path, "hf_ckpt")
    model_state_dict = ckpt_to_state_dict(
        save_checkpoint_path=save_checkpoint_path,
        output_dir=args.train.output_dir,
        ckpt_manager=args.train.ckpt_manager,
    )
    if args.train.train_architecture == "lora":
        model_state_dict = {k: v for k, v in model_state_dict.items() if "lora" in k}
    save_model_weights(
        hf_weights_path,
        model_state_dict,
        model_assets=model_assets,
    )
    logger.info_rank0(f"Huggingface checkpoint saved at {hf_weights_path} successfully!")


if __name__ == "__main__":
    main()
