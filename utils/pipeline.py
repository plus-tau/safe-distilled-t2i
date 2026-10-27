import torch
from diffusers import FluxPipeline, StableDiffusion3Pipeline

# Released DICE / SCFM LoRAs: <model>/dice_<concept>.safetensors and flux_scfm/scfm_lora.safetensors.
HF_REPO = "CookedMorsel/dice"

PIPELINES = {"flux": FluxPipeline, "sd3": StableDiffusion3Pipeline}


def load_pipeline(model_family, pretrained_model_name_or_path, scfm_lora_path=None, scfm_lora_weight_name=None,
                  device="cuda"):
    """Load a bf16 base pipeline. For Flux, an optional SCFM LoRA (local dir or HF repo id +
    weight name) is fused into the base weights, so the result behaves like a plain model."""
    if model_family not in PIPELINES:
        raise ValueError(f"Unknown model_family: {model_family!r}; expected one of {sorted(PIPELINES)}")
    pipe = PIPELINES[model_family].from_pretrained(pretrained_model_name_or_path, torch_dtype=torch.bfloat16)
    if scfm_lora_path:
        if model_family != "flux":
            raise ValueError("SCFM LoRA fusion is only supported for model_family='flux'")
        pipe.load_lora_weights(scfm_lora_path, weight_name=scfm_lora_weight_name)
        pipe.fuse_lora()
        pipe.unload_lora_weights()
    return pipe.to(device)
