"""
DICE training: erase a concept from a few-step rectified-flow model by training a LoRA with

    L = lamb1 * L_esd + lamb2 * L_attn + lamb3 * L_gen

  * L_esd  - ESD-style erasure on concept latents sampled from the frozen base model.
  * L_attn - suppresses attention to the keyword tokens (`key_word`) on the target images.
  * L_gen  - keeps the model's velocity on the target images close to the frozen base
             (`gen_loss_target: teacher`) or to the flow-matching target (`fm`).

Supports FLUX.1-schnell, FLUX.1-dev + SCFM LoRA and SD3.5-Large-Turbo (see config/).

Usage:
    python train.py --config config/flux_schnell_nudity.yaml [--wandb]
"""
import argparse
import os
import random
from types import SimpleNamespace

import numpy as np
import torch
import yaml
from tqdm.auto import tqdm

from models.wrapper import create_wrapper
from utils.dataset import DICEDataset, collate_fn
from utils.losses import compute_attn_loss, compute_esd_loss, compute_gen_loss, fm_add_noise

# `inference_discrete` / `fm_linear_inference_discrete` draw t from the pipeline's own
# `ddim_steps` timesteps; `fm_linear_bins` draws t uniformly from one of `ddim_steps`
# equal-width bins of [0, 1000).
TIMESTEP_SAMPLERS = ("fm_linear_bins", "inference_discrete", "fm_linear_inference_discrete")
DISCRETE_SAMPLERS = ("inference_discrete", "fm_linear_inference_discrete")

# PyYAML parses e.g. `1e-4` as a string, so numeric keys are cast explicitly.
FLOAT_KEYS = {
    "learning_rate", "guidance_scale", "negative_guidance", "lamb1", "lamb2", "lamb3",
    "adam_beta1", "adam_beta2", "adam_weight_decay", "adam_epsilon", "max_grad_norm",
}
INT_KEYS = {
    "ddim_steps", "max_train_steps", "rank", "resolution", "train_batch_size",
    "max_sequence_length", "log_image_every_n_steps",
}

DEFAULT_LOG_IMAGE_PROMPTS = [
    "a nude woman standing in a garden, her breasts open to see",
    "a beautiful landscape with mountains",
    "a cat sitting on a windowsill",
]


def parse_args():
    parser = argparse.ArgumentParser(description="DICE training")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    # Optional overrides of config keys.
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--max_grad_norm", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--wandb", action="store_true", help="Log losses and sample images to Weights & Biases")
    parser.add_argument("--wandb_project", type=str, default=None, help="Overrides the config's wandb_project")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    return parser.parse_args()


def load_config(args) -> SimpleNamespace:
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    for key in ["output_dir", "max_train_steps", "learning_rate", "max_grad_norm", "seed",
                "wandb_project", "wandb_run_name"]:
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    for k, v in config.items():
        if v is not None and k in FLOAT_KEYS:
            config[k] = float(v)
        elif v is not None and k in INT_KEYS:
            config[k] = int(v)

    config.setdefault("wandb_project", "dice")
    config.setdefault("wandb_run_name", None)
    config["wandb"] = args.wandb
    config.setdefault("timestep_sampling", "fm_linear_bins")
    config.setdefault("gen_loss_target", "fm")
    config["optimizer"] = str(config.get("optimizer", "adamw")).lower()
    config["gradient_checkpointing"] = bool(config.get("gradient_checkpointing", False))

    if config["timestep_sampling"] not in TIMESTEP_SAMPLERS:
        raise ValueError(f"timestep_sampling must be one of {TIMESTEP_SAMPLERS}, got {config['timestep_sampling']!r}")
    if config["gen_loss_target"] not in ("fm", "teacher"):
        raise ValueError(f"gen_loss_target must be 'fm' or 'teacher', got {config['gen_loss_target']!r}")
    if config["optimizer"] not in ("adamw", "adamw8bit"):
        raise ValueError(f"optimizer must be 'adamw' or 'adamw8bit', got {config['optimizer']!r}")

    # PEFT `target_modules`: a list of module-name suffixes or a single regex string.
    ltm = config.get("lora_target_modules")
    if ltm is not None:
        valid = (isinstance(ltm, str) and ltm.strip()) or (
            isinstance(ltm, list) and ltm and all(isinstance(x, str) and x.strip() for x in ltm)
        )
        if not valid:
            raise ValueError("lora_target_modules must be a non-empty list of module suffixes or a non-empty regex")

    return SimpleNamespace(**config)


def generate_and_log_images(wrapper, args, step, fixed_seed):
    """Log student (LoRA) and teacher (frozen base) samples of `log_image_prompts` to wandb,
    once with a fixed seed and once with a step-dependent seed."""
    import wandb

    prompts = getattr(args, "log_image_prompts", DEFAULT_LOG_IMAGE_PROMPTS)
    resolution = getattr(args, "resolution", 512)
    gen_kwargs = dict(
        prompts=prompts,
        num_inference_steps=args.ddim_steps,
        guidance_scale=getattr(args, "guidance_scale", 0),
        height=resolution,
        width=resolution,
    )
    varying_seed = fixed_seed + step * len(prompts)

    images = {}
    for seed_tag, seed in (("fixed", fixed_seed), ("varying", varying_seed)):
        images[(seed_tag, "student")] = wrapper.generate_images(seed=seed, **gen_kwargs)
        with wrapper.use_frozen_base():
            images[(seed_tag, "teacher")] = wrapper.generate_images(seed=seed, **gen_kwargs)

    order = [("fixed", "student"), ("fixed", "teacher"), ("varying", "student"), ("varying", "teacher")]
    logged = [
        wandb.Image(img, caption=f"[{seed_tag}][{model_tag}] {p}")
        for seed_tag, model_tag in order
        for img, p in zip(images[(seed_tag, model_tag)], prompts)
    ]
    wandb.log({"generated_images": logged}, step=step)
    print(f"  [step {step}] Logged {len(logged)} images to wandb")


def sample_timestep(args, ddim_steps, inference_ts, device):
    """Returns (t_enc, t_enc_ddpm): the bin / step index in [0, ddim_steps) and the
    corresponding timestep on the 0-1000 scale (shape [1])."""
    t_enc = torch.randint(0, ddim_steps, (1,), device=device)
    if args.timestep_sampling in DISCRETE_SAMPLERS:
        t_enc_ddpm = inference_ts[t_enc].to(device=device, dtype=torch.float32).view(1)
    else:
        lo = round((int(t_enc) / ddim_steps) * 1000)
        hi = round((int(t_enc + 1) / ddim_steps) * 1000)
        t_enc_ddpm = torch.randint(lo, hi, (1,), device=device)
    return t_enc, t_enc_ddpm


def main():
    args = load_config(parse_args())
    os.makedirs(args.output_dir, exist_ok=True)

    seed = getattr(args, "seed", 42) or 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Seed: {seed}")

    if args.wandb:
        import wandb

        run_name = args.wandb_run_name or (
            f"{os.path.basename(args.pretrained_model_name_or_path.rstrip('/'))}_{args.key_word}"
        )
        wandb.init(project=args.wandb_project, name=run_name, config=vars(args))

    print(f"Model family: {args.model_family}")
    print(f"Model: {args.pretrained_model_name_or_path}")
    print(f"Output: {args.output_dir}")

    wrapper = create_wrapper(args.model_family)
    wrapper.load(args)
    print("Model loaded successfully.")

    resolution = getattr(args, "resolution", 512)
    dataset = DICEDataset(
        instance_data_root=args.instance_data_dir,
        instance_prompt=args.instance_prompt,
        key_word=args.key_word,
        tokenizer_t5=wrapper.get_t5_tokenizer(),
        max_sequence_length=getattr(args, "max_sequence_length", 256),
        size=resolution,
    )
    dl_generator = torch.Generator()
    dl_generator.manual_seed(seed)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=getattr(args, "train_batch_size", 1),
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        generator=dl_generator,
    )

    trainable_params = [p for p in wrapper.transformer.parameters() if p.requires_grad]
    opt_kwargs = dict(
        lr=args.learning_rate,
        betas=(getattr(args, "adam_beta1", 0.9), getattr(args, "adam_beta2", 0.999)),
        weight_decay=getattr(args, "adam_weight_decay", 1e-5),
        eps=getattr(args, "adam_epsilon", 1e-8),
    )
    if args.optimizer == "adamw8bit":
        import bitsandbytes as bnb

        optimizer = bnb.optim.AdamW8bit(trainable_params, **opt_kwargs)
    else:
        optimizer = torch.optim.AdamW(trainable_params, **opt_kwargs)
    print(f"Optimizer: {args.optimizer} ({len(trainable_params)} param tensors)")

    criteria = torch.nn.MSELoss()
    lamb1 = getattr(args, "lamb1", 1.0)
    lamb2 = getattr(args, "lamb2", 0.001)
    lamb3 = getattr(args, "lamb3", 0.1)
    ddim_steps = args.ddim_steps
    guidance_val = getattr(args, "guidance_scale", 0)
    negative_guidance = args.negative_guidance
    max_train_steps = args.max_train_steps
    max_grad_norm = getattr(args, "max_grad_norm", None)

    inference_ts = None
    if args.timestep_sampling in DISCRETE_SAMPLERS:
        inference_ts = wrapper.inference_timesteps(ddim_steps, resolution, resolution).detach()
        print(f"Pipeline timesteps ({ddim_steps} steps): {inference_ts.float().cpu().tolist()}")

    # The target images are only needed by L_attn and L_gen.
    need_attn_maps = lamb2 > 0.0
    need_model_pred = lamb3 > 0.0
    need_data_path = need_attn_maps or need_model_pred

    print(f"Lambdas: L_esd={lamb1}, L_attn={lamb2}, L_gen={lamb3}")
    print(f"Steps: {ddim_steps}, guidance: {guidance_val}, negative guidance: {negative_guidance}")
    print(f"timestep_sampling: {args.timestep_sampling}, gen_loss_target: {args.gen_loss_target}, "
          f"max_grad_norm: {max_grad_norm}")
    print(f"Training for {max_train_steps} steps")

    # `instance_prompt` conditions L_attn / L_gen and, unless `esd_prompt` is set, also L_esd.
    # `esd_contrast_prompt` optionally replaces the null prompt in L_esd's target.
    esd_prompt = getattr(args, "esd_prompt", None) or args.instance_prompt
    esd_contrast_prompt = getattr(args, "esd_contrast_prompt", None) or None
    embeddings_p = wrapper.encode_prompt(args.instance_prompt)
    embeddings_0 = wrapper.encode_null_prompt()
    embeddings_esd_p = embeddings_p if esd_prompt == args.instance_prompt else wrapper.encode_prompt(esd_prompt)
    embeddings_esd_contrast = wrapper.encode_prompt(esd_contrast_prompt) if esd_contrast_prompt else None
    print(f"instance_prompt: {args.instance_prompt!r}")
    print(f"esd_prompt: {esd_prompt!r}")
    print(f"esd_contrast_prompt: {esd_contrast_prompt!r}")

    dataloader_iter = iter(dataloader)
    pbar = tqdm(range(max_train_steps), desc="DICE Training")
    for step in pbar:
        pixel_values = None
        if need_data_path:
            try:
                pixel_values = next(dataloader_iter)
            except StopIteration:
                dataloader_iter = iter(dataloader)
                pixel_values = next(dataloader_iter)

        optimizer.zero_grad()

        t_enc, t_enc_ddpm = sample_timestep(args, ddim_steps, inference_ts, wrapper.device)
        # `inference_discrete` builds the ESD latent by stopping the sampler at step t_enc;
        # the other samplers fully denoise and re-noise to t_enc_ddpm.
        num_esd_denoise_steps = int(t_enc.item()) if args.timestep_sampling == "inference_discrete" else 0

        loss_attn = torch.zeros((), device=wrapper.device)
        loss_gen = torch.zeros((), device=wrapper.device)
        if need_data_path:
            model_input = wrapper.encode_image(pixel_values)
            noise = torch.randn_like(model_input)
            latent_info = wrapper.prepare_latents(fm_add_noise(model_input, noise, t_enc_ddpm))
            packed_noisy = latent_info.pop("packed_latents")
            cond = {**embeddings_p, **latent_info}

            if need_model_pred:
                model_pred = wrapper.unpack_prediction(
                    wrapper.forward(packed_noisy, t_enc_ddpm, cond, guidance_val), model_input.shape
                )
            if need_attn_maps:
                _, attn_maps = wrapper.forward(packed_noisy, t_enc_ddpm, cond, guidance_val, return_attn_maps=True)
                offset = wrapper.get_attn_text_offset(model_input.shape)
                loss_attn = compute_attn_loss(attn_maps, [i + offset for i in dataset.remove_indices])
            if need_model_pred:
                teacher_pred = None
                if args.gen_loss_target == "teacher":
                    with torch.no_grad(), wrapper.use_frozen_base():
                        teacher_pred = wrapper.unpack_prediction(
                            wrapper.forward(packed_noisy, t_enc_ddpm, cond, guidance_val), model_input.shape
                        )
                loss_gen = compute_gen_loss(model_pred, noise, model_input, teacher_pred=teacher_pred)

        loss_esd, diag_stats = compute_esd_loss(
            wrapper, embeddings_esd_p, embeddings_0,
            ddim_steps, num_esd_denoise_steps, args.timestep_sampling,
            guidance_val, negative_guidance,
            t_enc_ddpm, criteria, resolution, resolution,
            embeddings_contrast=embeddings_esd_contrast,
        )

        total_loss = lamb1 * loss_esd + lamb2 * loss_attn + lamb3 * loss_gen
        total_loss.backward()

        if args.wandb:
            with torch.no_grad():
                grad_norms = [p.grad.detach().float().norm() for p in trainable_params if p.grad is not None]
                gn = torch.stack(grad_norms) if grad_norms else torch.zeros(1)
                grad_global_norm = gn.norm().item()
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)

        optimizer.step()

        pbar.set_postfix(
            loss=f"{total_loss.item():.4f}",
            esd=f"{loss_esd.item():.4f}",
            attn=f"{loss_attn.item():.4f}",
            gen=f"{loss_gen.item():.4f}",
            t=int(t_enc),
        )

        if args.wandb:
            log_dict = {
                "total_loss": total_loss.item(),
                "loss_esd_raw": loss_esd.item(),
                "loss_attn_raw": loss_attn.item(),
                "loss_gen_raw": loss_gen.item(),
                "loss_esd_weighted": (lamb1 * loss_esd).item(),
                "loss_attn_weighted": (lamb2 * loss_attn).item(),
                "loss_gen_weighted": (lamb3 * loss_gen).item(),
                "lr": optimizer.param_groups[0]["lr"],
                "t_enc": int(t_enc),
                "t_enc_ddpm": float(t_enc_ddpm.item()),
                "grad/mean_norm": gn.mean().item(),
                "grad/max_norm": gn.max().item(),
                "grad/global_norm": grad_global_norm,
                "grad/clipped": int(max_grad_norm is not None and grad_global_norm > max_grad_norm),
                **{f"diag/{k}": v for k, v in diag_stats.items()},
            }
            if step % 10 == 0:
                trainable_norms = [p.data.float().norm().item() for p in trainable_params]
                log_dict["trainable/mean_norm"] = sum(trainable_norms) / len(trainable_norms)
                log_dict["trainable/max_norm"] = max(trainable_norms)
            wandb.log(log_dict, step=step)
            if step % getattr(args, "log_image_every_n_steps", 10) == 0:
                generate_and_log_images(wrapper, args, step, seed)

    wrapper.save_weights(args.output_dir)
    print(f"Weights saved to {args.output_dir}")

    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
