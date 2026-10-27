"""
Evaluate the frozen base model with the same transformer that is being trained,
instead of keeping a second copy in VRAM: temporarily overwrite the LoRA
parameters with zeros (so the LoRA delta is zero) and restore them afterwards.
"""
import torch
import torch.nn as nn


def create_zero_lora_parameters(transformer: nn.Module) -> dict:
    return {
        name: torch.zeros_like(param.data)
        for name, param in transformer.named_parameters()
        if "lora" in name.lower() and param.requires_grad
    }


def disable_lora(transformer: nn.Module, zero_params: dict) -> dict:
    """Zero the LoRA parameters in place; returns their original values."""
    original_params = {}
    for name, param in transformer.named_parameters():
        if name in zero_params:
            original_params[name] = param.data.clone()
            param.data.copy_(zero_params[name])
    return original_params


def enable_lora(transformer: nn.Module, original_params: dict) -> None:
    for name, param in transformer.named_parameters():
        if name in original_params:
            param.data.copy_(original_params[name])
