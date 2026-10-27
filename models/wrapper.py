"""
Model abstraction for DICE training. `FluxWrapper` and `SD3Wrapper` expose one interface
(prompt / image encoding, transformer forwards, ESD latent sampling, LoRA saving) so that
`train.py` and `utils/losses.py` are model-agnostic.

Conventions:
  * Timesteps are on the 0-1000 scale.
  * `model_input` is the VAE latent [B, C, H, W]. Flux packs it into a token sequence
    (`prepare_latents`) and predictions are unpacked back (`unpack_prediction`); for SD3
    both are the identity because the transformer patchifies internally.
  * `cond` dicts hold the prompt embeddings (`emb`, `pooled_emb`, and for Flux `text_ids`)
    plus, for Flux, the `latent_image_ids` of the current latent.
"""
import os
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from diffusers import FluxPipeline, SD3Transformer2DModel
from diffusers.pipelines.flux.pipeline_flux import calculate_shift, retrieve_timesteps
from diffusers.utils.torch_utils import randn_tensor
from peft import LoraConfig, get_peft_model
from peft.utils import get_peft_model_state_dict

from models.transformer_flux import FluxTransformer2DModel as FluxAttnMapTransformer
from models.transformer_sd3 import SD3Transformer2DModel as SD3AttnMapTransformer
from utils.lora_utils import create_zero_lora_parameters, disable_lora, enable_lora
from utils.pipeline import PIPELINES, load_pipeline

# Attention projections of the image and text streams (same names in Flux and SD3).
# Override with `lora_target_modules` in the config.
DEFAULT_LORA_TARGET_MODULES = [
    "to_q", "to_k", "to_v", "to_out.0",
    "add_q_proj", "add_k_proj", "add_v_proj", "to_add_out",
]

_PEFT_PREFIX = "base_model.model."


def _add_lora(transformer: nn.Module, rank: int, target_modules) -> nn.Module:
    lora_config = LoraConfig(r=rank, lora_alpha=rank, target_modules=target_modules, lora_dropout=0.0)
    transformer = get_peft_model(transformer, lora_config)
    trainable = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    total = sum(p.numel() for p in transformer.parameters())
    print(f"LoRA target_modules: {target_modules}")
    print(f"LoRA added: {trainable:,} trainable / {total:,} total parameters ({100 * trainable / total:.2f}%)")
    return transformer


class ModelWrapper(ABC):
    model_family: str
    # Modified transformer class that can return attention maps (see models/transformer_*.py).
    attn_map_transformer_cls: type
    t5_tokenizer_attr: str
    default_max_seq_len: int
    # dtype of the timestep passed to the transformer while sampling the ESD latent; mirrors
    # what each diffusers pipeline does at inference.
    esd_sample_timestep_dtype: torch.dtype

    def __init__(self):
        self.weight_dtype = torch.bfloat16
        self.device = torch.device("cuda")
        self.pipe = None
        self.vae = None
        self.transformer = None
        self.transformer_config = None
        self._zero_params: Optional[dict] = None
        self._max_seq_len = self.default_max_seq_len

    # ---- Setup ----

    def load(self, args) -> None:
        """Load the pipeline, swap in the attention-map transformer and attach trainable LoRA."""
        self._max_seq_len = getattr(args, "max_sequence_length", self.default_max_seq_len)
        pipe = load_pipeline(
            self.model_family,
            args.pretrained_model_name_or_path,
            getattr(args, "scfm_lora_path", None),
            getattr(args, "scfm_lora_weight_name", None),
            device=self.device,
        )
        for module in (pipe.vae, pipe.text_encoder, pipe.text_encoder_2, getattr(pipe, "text_encoder_3", None)):
            if module is not None:
                module.requires_grad_(False)
        self.vae = pipe.vae

        transformer = self._to_attn_map_transformer(pipe.transformer, self.device)
        pipe.transformer = None
        torch.cuda.empty_cache()

        # Checkpointing only affects the trained forwards; the frozen ones run under no_grad.
        if getattr(args, "gradient_checkpointing", False):
            transformer.gradient_checkpointing = True
            print("Gradient checkpointing enabled")
        self.transformer_config = transformer.config
        lora_targets = getattr(args, "lora_target_modules", None) or DEFAULT_LORA_TARGET_MODULES
        self.transformer = _add_lora(transformer, getattr(args, "rank", 8), lora_targets).to(self.device)
        self._zero_params = create_zero_lora_parameters(self.transformer)
        self.pipe = pipe

    def _to_attn_map_transformer(self, transformer: nn.Module, device) -> nn.Module:
        model = self.attn_map_transformer_cls.from_config(transformer.config)
        model.load_state_dict(transformer.state_dict(), strict=False)
        return model.to(device, dtype=self.weight_dtype)

    @contextmanager
    def use_frozen_base(self):
        """Run the enclosed code as if `self.transformer` were the frozen base model."""
        original_params = disable_lora(self.transformer, self._zero_params)
        try:
            yield
        finally:
            enable_lora(self.transformer, original_params)

    # ---- Encoding ----

    @abstractmethod
    def _encode_prompt(self, prompt: str) -> dict:
        ...

    @torch.no_grad()
    def encode_prompt(self, prompt: str) -> dict:
        return self._encode_prompt(prompt)

    def encode_null_prompt(self) -> dict:
        return self.encode_prompt("")

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        pixel_values = pixel_values.to(dtype=self.vae.dtype, device=self.device)
        model_input = self.vae.encode(pixel_values).latent_dist.sample()
        model_input = (model_input - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        return model_input.to(dtype=self.weight_dtype)

    def get_t5_tokenizer(self):
        return getattr(self.pipe, self.t5_tokenizer_attr)

    # ---- Latent layout ----

    @abstractmethod
    def prepare_latents(self, model_input: torch.Tensor) -> dict:
        """Returns {"packed_latents": ...} plus any latent-dependent conditioning."""

    def unpack_prediction(self, model_pred: torch.Tensor, ref_shape: tuple) -> torch.Tensor:
        return model_pred

    @abstractmethod
    def get_attn_text_offset(self, model_input_shape: tuple) -> int:
        """Position of the first T5 token in the joint attention sequence."""

    # ---- Transformer forwards ----

    @abstractmethod
    def _forward(self, transformer, latents, timestep, cond, guidance_val, return_attn_maps=False, device=None):
        """Returns the velocity prediction, or (prediction, attn_maps) if `return_attn_maps`."""

    def forward(self, latents, timestep, cond, guidance_val, return_attn_maps=False):
        """Trained-model forward on (packed) latents."""
        return self._forward(self.transformer, latents, timestep, cond, guidance_val, return_attn_maps)

    # ---- ESD ----

    @abstractmethod
    def inference_timesteps(self, num_steps: int, height: int, width: int) -> torch.Tensor:
        """Scheduler timesteps used by the diffusers pipeline for `num_steps` steps at this resolution.
        Also (re)initializes `self.pipe.scheduler` for stepping."""

    @abstractmethod
    def _init_esd_latents(self, height: int, width: int):
        """Initial noise for ESD sampling, and `extra` conditioning for that latent shape."""

    def latent_sample_esd(self, cond, ddim_steps, num_denoise_steps, guidance_val, height, width):
        """Sample from noise with the pipeline's own schedule, applying only the first
        `num_denoise_steps` of the `ddim_steps` updates (0 = pure noise).
        Returns (latents, extra) where `extra` is needed by `predict_noise_esd`."""
        latents, extra = self._init_esd_latents(height, width)
        cond = {**cond, **extra}
        sched = self.pipe.scheduler
        for t in self.inference_timesteps(ddim_steps, height, width)[:num_denoise_steps]:
            timestep = t.expand(latents.shape[0]).to(self.esd_sample_timestep_dtype)
            noise_pred = self._forward(self.transformer, latents, timestep, cond, guidance_val)
            latents = sched.step(noise_pred, t, latents, return_dict=False)[0]
        return latents, extra

    def predict_noise_esd(self, z, cond, timestep, guidance_val, extra) -> torch.Tensor:
        pred = self._forward(self.transformer, z, timestep, {**cond, **extra}, guidance_val)
        return self.unpack_prediction(pred, extra.get("latent_shape"))

    @property
    def has_esd_offload(self) -> bool:
        """True if a frozen copy of the base transformer lives on `esd_offload_device`
        and is used for the no-grad ESD velocities v_0 / v_p."""
        return False

    def predict_noise_esd_offload(self, z, cond, timestep, guidance_val, extra) -> torch.Tensor:
        raise NotImplementedError(f"{type(self).__name__} does not support `esd_offload_device`.")

    # ---- Inference / saving ----

    def generate_images(self, prompts: List[str], num_inference_steps: int, guidance_scale: float,
                        height: int, width: int, seed: int) -> list:
        """Generate one image per prompt (seed + i) with the current LoRA weights."""
        was_training = self.transformer.training
        self.transformer.eval()
        self.pipe.transformer = self.transformer
        images = []
        with torch.no_grad():
            for i, prompt in enumerate(prompts):
                images.append(self.pipe(
                    prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    height=height,
                    width=width,
                    max_sequence_length=self._max_seq_len,
                    generator=torch.Generator(self.device).manual_seed(seed + i),
                ).images[0])
        self.pipe.transformer = None
        if was_training:
            self.transformer.train()
        torch.cuda.empty_cache()
        return images

    def save_weights(self, output_dir: str) -> None:
        """Save the LoRA in diffusers format (`pytorch_lora_weights.safetensors`)."""
        os.makedirs(output_dir, exist_ok=True)
        state_dict = {
            k[len(_PEFT_PREFIX):] if k.startswith(_PEFT_PREFIX) else k: v
            for k, v in get_peft_model_state_dict(self.transformer).items()
        }
        PIPELINES[self.model_family].save_lora_weights(
            save_directory=output_dir, transformer_lora_layers=state_dict
        )


class FluxWrapper(ModelWrapper):
    model_family = "flux"
    attn_map_transformer_cls = FluxAttnMapTransformer
    t5_tokenizer_attr = "tokenizer_2"
    default_max_seq_len = 256
    esd_sample_timestep_dtype = torch.bfloat16

    def load(self, args):
        if getattr(args, "esd_offload_device", None):
            raise NotImplementedError("esd_offload_device is only supported for model_family='sd3'")
        super().load(args)

    def _encode_prompt(self, prompt):
        emb, pooled_emb, text_ids = self.pipe.encode_prompt(
            prompt, prompt_2=None, device=self.device, max_sequence_length=self._max_seq_len
        )
        return {"emb": emb, "pooled_emb": pooled_emb, "text_ids": text_ids}

    def _latent_image_ids(self, h: int, w: int) -> torch.Tensor:
        return FluxPipeline._prepare_latent_image_ids(1, h // 2, w // 2, self.device, self.weight_dtype)

    def prepare_latents(self, model_input):
        b, c, h, w = model_input.shape
        return {
            "packed_latents": FluxPipeline._pack_latents(model_input, b, c, h, w),
            "latent_image_ids": self._latent_image_ids(h, w),
        }

    def unpack_prediction(self, model_pred, ref_shape):
        _, _, h, w = ref_shape
        vsf = self.pipe.vae_scale_factor
        return FluxPipeline._unpack_latents(model_pred, height=h * vsf, width=w * vsf, vae_scale_factor=vsf)

    def get_attn_text_offset(self, model_input_shape):
        return 0  # joint sequence is [text, image]

    def _forward(self, transformer, latents, timestep, cond, guidance_val, return_attn_maps=False, device=None):
        device = device or self.device
        guidance = None
        if self.transformer_config.guidance_embeds:
            guidance = torch.tensor([guidance_val], device=device, dtype=self.weight_dtype).expand(latents.shape[0])
        out = transformer(
            hidden_states=latents.to(device=device, dtype=self.weight_dtype),
            timestep=(timestep / 1000).to(device),
            guidance=guidance,
            pooled_projections=cond["pooled_emb"].to(device=device, dtype=self.weight_dtype),
            encoder_hidden_states=cond["emb"].to(device=device, dtype=self.weight_dtype),
            txt_ids=cond["text_ids"].to(device=device, dtype=self.weight_dtype),
            img_ids=cond["latent_image_ids"].to(device=device, dtype=self.weight_dtype),
            return_dict=False,
            return_attn_maps=return_attn_maps,
        )
        return out if return_attn_maps else out[0]

    def _latent_hw(self, height: int, width: int):
        # Same rounding as FluxPipeline.prepare_latents; it sets the packed sequence length,
        # which determines the timestep shift `mu`.
        vsf = self.pipe.vae_scale_factor
        return 2 * (int(height) // (vsf * 2)), 2 * (int(width) // (vsf * 2))

    def inference_timesteps(self, num_steps, height, width):
        h, w = self._latent_hw(height, width)
        sched = self.pipe.scheduler
        mu = calculate_shift(
            (h // 2) * (w // 2),
            sched.config.get("base_image_seq_len", 256),
            sched.config.get("max_image_seq_len", 4096),
            sched.config.get("base_shift", 0.5),
            sched.config.get("max_shift", 1.15),
        )
        sigmas = np.linspace(1.0, 1.0 / num_steps, num_steps)
        timesteps, _ = retrieve_timesteps(sched, num_steps, self.device, sigmas=sigmas, mu=mu)
        return timesteps

    def _init_esd_latents(self, height, width):
        h, w = self._latent_hw(height, width)
        c = self.vae.config.latent_channels
        latents = randn_tensor((1, c, h, w), generator=None, dtype=self.weight_dtype)
        latents = FluxPipeline._pack_latents(latents, 1, c, h, w).to(self.device)
        return latents, {"latent_image_ids": self._latent_image_ids(h, w), "latent_shape": (1, c, h, w)}


class SD3Wrapper(ModelWrapper):
    model_family = "sd3"
    attn_map_transformer_cls = SD3AttnMapTransformer
    t5_tokenizer_attr = "tokenizer_3"
    default_max_seq_len = 512
    esd_sample_timestep_dtype = torch.float32

    def __init__(self):
        super().__init__()
        self._esd_offload_transformer = None
        self._esd_offload_device = None

    def load(self, args):
        super().load(args)
        offload_device = getattr(args, "esd_offload_device", None)
        if offload_device:
            # Frozen copy of the base transformer on a second GPU for the no-grad v_0 / v_p
            # forwards of L_esd, to free memory on the training GPU.
            offload_device = torch.device(offload_device)
            upstream = SD3Transformer2DModel.from_pretrained(
                args.pretrained_model_name_or_path, subfolder="transformer", torch_dtype=self.weight_dtype
            )
            offload = self._to_attn_map_transformer(upstream, offload_device)
            offload.requires_grad_(False)
            offload.eval()
            del upstream
            torch.cuda.empty_cache()
            self._esd_offload_transformer = offload
            self._esd_offload_device = offload_device
            print(f"ESD offload: frozen {args.pretrained_model_name_or_path} on {offload_device}")

    def _encode_prompt(self, prompt):
        emb, _, pooled_emb, _ = self.pipe.encode_prompt(
            prompt=prompt, prompt_2=None, prompt_3=None, device=self.device,
            num_images_per_prompt=1, do_classifier_free_guidance=False,
        )
        return {"emb": emb, "pooled_emb": pooled_emb}

    def prepare_latents(self, model_input):
        return {"packed_latents": model_input}

    def get_attn_text_offset(self, model_input_shape):
        # Joint sequence is [image patches, CLIP tokens (padded to 77), T5 tokens].
        _, _, h, w = model_input_shape
        p = self.transformer_config.patch_size
        return (h // p) * (w // p) + self.pipe.tokenizer_max_length

    def _forward(self, transformer, latents, timestep, cond, guidance_val, return_attn_maps=False, device=None):
        device = device or self.device
        out = transformer(
            hidden_states=latents.to(device=device, dtype=self.weight_dtype),
            timestep=timestep.to(device, dtype=torch.float32),
            encoder_hidden_states=cond["emb"].to(device=device, dtype=self.weight_dtype),
            pooled_projections=cond["pooled_emb"].to(device=device, dtype=self.weight_dtype),
            return_dict=False,
            return_attn_maps=return_attn_maps,
        )
        return out if return_attn_maps else out[0]

    def inference_timesteps(self, num_steps, height, width):
        sched = self.pipe.scheduler
        kwargs = {}
        if sched.config.get("use_dynamic_shifting", None):
            vsf, p = int(self.pipe.vae_scale_factor), self.transformer_config.patch_size
            image_seq_len = (int(height) // vsf // p) * (int(width) // vsf // p)
            kwargs["mu"] = calculate_shift(
                image_seq_len,
                sched.config.get("base_image_seq_len", 256),
                sched.config.get("max_image_seq_len", 4096),
                sched.config.get("base_shift", 0.5),
                sched.config.get("max_shift", 1.16),
            )
        timesteps, _ = retrieve_timesteps(sched, num_steps, self.device, **kwargs)
        return timesteps

    def _init_esd_latents(self, height, width):
        vsf = int(self.pipe.vae_scale_factor)
        shape = (1, self.vae.config.latent_channels, int(height) // vsf, int(width) // vsf)
        return randn_tensor(shape, generator=None, dtype=self.weight_dtype).to(self.device), {}

    @property
    def has_esd_offload(self):
        return self._esd_offload_transformer is not None

    def predict_noise_esd_offload(self, z, cond, timestep, guidance_val, extra):
        if self._esd_offload_transformer is None:
            raise RuntimeError("No offload transformer loaded; set `esd_offload_device` in the config.")
        pred = self._forward(
            self._esd_offload_transformer, z, timestep, cond, guidance_val, device=self._esd_offload_device
        )
        return pred.to(self.device)


def create_wrapper(model_family: str) -> ModelWrapper:
    wrappers = {"flux": FluxWrapper, "sd3": SD3Wrapper}
    if model_family not in wrappers:
        raise ValueError(f"Unknown model_family: {model_family!r}. Use 'flux' or 'sd3'.")
    return wrappers[model_family]()
