"""
Attention processors for Flux and SD3 joint attention that can also return the
softmax attention weights, which `L_attn` needs. Adapted from EraseAnything.

`return_attn_maps` must be an explicit parameter of `__call__` (not **kwargs): diffusers'
`Attention` filters the kwargs it forwards by inspecting the processor's signature.
"""
import math
from typing import Optional

import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention
from diffusers.models.embeddings import apply_rotary_emb


def _project(attn: Attention, x: torch.Tensor, context: bool):
    """Project `x` to per-head (q, k, v) of shape [B, heads, L, head_dim], with QK-norm.
    `context=True` uses the text-stream projections (`add_*_proj`)."""
    if context:
        q, k, v = attn.add_q_proj(x), attn.add_k_proj(x), attn.add_v_proj(x)
        norm_q, norm_k = attn.norm_added_q, attn.norm_added_k
    else:
        q, k, v = attn.to_q(x), attn.to_k(x), attn.to_v(x)
        norm_q, norm_k = attn.norm_q, attn.norm_k
    head_dim = k.shape[-1] // attn.heads
    q, k, v = (t.view(x.shape[0], -1, attn.heads, head_dim).transpose(1, 2) for t in (q, k, v))
    if norm_q is not None:
        q = norm_q(q)
    if norm_k is not None:
        k = norm_k(k)
    return q, k, v


def _attention(q, k, v, return_attn_maps: bool):
    """Returns (output [B, L, heads * head_dim], attention weights or None).

    With `return_attn_maps` the softmax is computed explicitly in fp32 (to stay close to
    fused SDPA in bf16); otherwise the fused `F.scaled_dot_product_attention` is used."""
    attn_maps = None
    if return_attn_maps:
        scale = 1 / math.sqrt(q.shape[-1])
        attn_maps = torch.softmax(q.float() @ k.float().transpose(-2, -1) * scale, dim=-1)
        out = attn_maps.to(v.dtype) @ v
    else:
        out = F.scaled_dot_product_attention(q, k, v)
    b, heads, _, head_dim = out.shape
    return out.transpose(1, 2).reshape(b, -1, heads * head_dim).to(q.dtype), attn_maps


class FluxAttnProcessor2_0:
    """Flux attention. Joint sequence order is [text, image]."""

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        return_attn_maps: bool = False,
    ):
        q, k, v = _project(attn, hidden_states, context=False)
        if encoder_hidden_states is not None:
            cq, ck, cv = _project(attn, encoder_hidden_states, context=True)
            q, k, v = torch.cat([cq, q], dim=2), torch.cat([ck, k], dim=2), torch.cat([cv, v], dim=2)

        if image_rotary_emb is not None:
            q = apply_rotary_emb(q, image_rotary_emb)
            k = apply_rotary_emb(k, image_rotary_emb)

        hidden_states, attn_maps = _attention(q, k, v, return_attn_maps)

        # Single-stream blocks: text and image tokens are already concatenated.
        if encoder_hidden_states is None:
            return hidden_states

        n_txt = encoder_hidden_states.shape[1]
        encoder_hidden_states, hidden_states = hidden_states[:, :n_txt], hidden_states[:, n_txt:]
        hidden_states = attn.to_out[1](attn.to_out[0](hidden_states))
        encoder_hidden_states = attn.to_add_out(encoder_hidden_states)
        if return_attn_maps:
            return hidden_states, encoder_hidden_states, attn_maps
        return hidden_states, encoder_hidden_states


class JointAttnProcessor2_0:
    """SD3 joint attention. Joint sequence order is [image, text]."""

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_attn_maps: bool = False,
    ):
        n_img = hidden_states.shape[1]
        q, k, v = _project(attn, hidden_states, context=False)
        if encoder_hidden_states is not None:
            cq, ck, cv = _project(attn, encoder_hidden_states, context=True)
            q, k, v = torch.cat([q, cq], dim=2), torch.cat([k, ck], dim=2), torch.cat([v, cv], dim=2)

        hidden_states, attn_maps = _attention(q, k, v, return_attn_maps)

        if encoder_hidden_states is not None:
            hidden_states, encoder_hidden_states = hidden_states[:, :n_img], hidden_states[:, n_img:]
            if not attn.context_pre_only:
                encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        hidden_states = attn.to_out[1](attn.to_out[0](hidden_states))

        if encoder_hidden_states is None:
            return hidden_states
        if return_attn_maps:
            return hidden_states, encoder_hidden_states, attn_maps
        return hidden_states, encoder_hidden_states
