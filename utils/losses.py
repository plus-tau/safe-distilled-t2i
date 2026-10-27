"""
DICE losses: L_esd (concept erasure), L_attn (keyword attention suppression) and
L_gen (generation preservation). All model-specific work goes through the
`ModelWrapper` interface in `models/wrapper.py`.

Timesteps `t` are on the 0-1000 scale throughout; flow matching interpolates
x_t = (1 - t/1000) * x_0 + (t/1000) * noise and the model predicts the velocity
v = noise - x_0.
"""
from typing import Optional

import torch
import torch.nn.functional as F

# Timestep samplers whose ESD latent is built by fully denoising and then re-noising to t.
FM_RENOISE_SAMPLERS = ("fm_linear_bins", "fm_linear_inference_discrete")


def fm_add_noise(x0: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
    t = (timesteps / 1000).to(x0.device)
    return (1 - t) * x0 + t * noise


def compute_esd_loss(
    wrapper,
    embeddings_p: dict,
    embeddings_0: dict,
    ddim_steps: int,
    num_denoise_steps: int,
    timestep_sampling: str,
    guidance_val: float,
    negative_guidance: float,
    t_enc_ddpm: torch.Tensor,
    criteria,
    height: int = 512,
    width: int = 512,
    embeddings_contrast: Optional[dict] = None,
) -> tuple[torch.Tensor, dict]:
    """ESD loss (Gandikota et al., 2023) for flow-matching models.

    The trained model's velocity for the erase prompt p is pushed towards
        v_0 - negative_guidance * (v_p - v_0),
    where v_p / v_0 are the frozen base model's velocities for p and for the null
    prompt (or `embeddings_contrast`, if given), all evaluated at the same noisy latent z.

    z is a concept latent sampled from the frozen base:
      * FM_RENOISE_SAMPLERS: run the full `ddim_steps` sampler, then re-noise to `t_enc_ddpm`.
      * inference_discrete: stop the sampler after `num_denoise_steps` steps.
    """
    fm_renoise = timestep_sampling in FM_RENOISE_SAMPLERS
    embeddings_neg = embeddings_contrast if embeddings_contrast is not None else embeddings_0

    # z is always built on the trained model with LoRA disabled, even when an offloaded
    # frozen copy exists (only v_0 / v_p use the offloaded copy).
    with torch.no_grad(), wrapper.use_frozen_base():
        if fm_renoise:
            z_clean, extra = wrapper.latent_sample_esd(
                embeddings_p, ddim_steps, ddim_steps, guidance_val, height, width
            )
            z = fm_add_noise(z_clean, torch.randn_like(z_clean), t_enc_ddpm)
        else:
            z, extra = wrapper.latent_sample_esd(
                embeddings_p, ddim_steps, num_denoise_steps, guidance_val, height, width
            )

    with torch.no_grad():
        if wrapper.has_esd_offload:
            e_0 = wrapper.predict_noise_esd_offload(z, embeddings_neg, t_enc_ddpm, guidance_val, extra)
            e_p = wrapper.predict_noise_esd_offload(z, embeddings_p, t_enc_ddpm, guidance_val, extra)
        else:
            with wrapper.use_frozen_base():
                e_0 = wrapper.predict_noise_esd(z, embeddings_neg, t_enc_ddpm, guidance_val, extra)
                e_p = wrapper.predict_noise_esd(z, embeddings_p, t_enc_ddpm, guidance_val, extra)

    e_n = wrapper.predict_noise_esd(z, embeddings_p, t_enc_ddpm, guidance_val, extra)

    target = e_0 - negative_guidance * (e_p - e_0)
    loss_esd = criteria(e_n, target)

    concept_signal = (e_p - e_0).float()
    concept_signal_sq = concept_signal.pow(2)
    diag_stats = {
        "z_norm": z.float().norm().item(),
        "e_0_norm": e_0.float().norm().item(),
        "e_p_norm": e_p.float().norm().item(),
        "e_n_norm": e_n.float().norm().item(),
        "concept_signal_norm": concept_signal.norm().item(),
        "concept_signal_norm_sq": concept_signal_sq.sum().item(),
        "concept_signal_mse": concept_signal_sq.mean().item(),
        "esd_fm_renoise": float(fm_renoise),
        "esd_num_denoise_steps": float(ddim_steps if fm_renoise else num_denoise_steps),
        "esd_contrast_active": float(embeddings_contrast is not None),
    }
    return loss_esd, diag_stats


def compute_attn_loss(attn_maps: torch.Tensor, remove_indices: list) -> torch.Tensor:
    """Attention weights pointing at the keyword tokens `remove_indices` (indices into the
    joint sequence): L2 norm over (batch, heads), summed over all query/keyword positions."""
    if len(remove_indices) == 0:
        return torch.zeros((), device=attn_maps.device)
    mask = torch.zeros_like(attn_maps)
    mask[..., remove_indices] = 1
    return torch.sum(torch.norm(mask * attn_maps, dim=(0, 1)))


def compute_gen_loss(
    model_pred: torch.Tensor,
    noise: torch.Tensor,
    model_input: torch.Tensor,
    teacher_pred: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """L_gen on the target images: match the frozen base model's velocity
    (`teacher_pred`, i.e. `gen_loss_target: teacher`) or the flow-matching
    target noise - x_0 (`gen_loss_target: fm`)."""
    target = teacher_pred if teacher_pred is not None else noise - model_input
    return F.mse_loss(model_pred.float(), target.float())
