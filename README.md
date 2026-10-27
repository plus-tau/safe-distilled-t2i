# DICE

[![arXiv](https://img.shields.io/badge/arXiv-2610.36665-b31b1b.svg)](http://arxiv.org/abs/2610.36665)

Concept erasure for few-step (distilled) rectified-flow text-to-image models. DICE trains a
LoRA on the transformer of an already-distilled model so that it stops generating a target
concept (e.g. nudity) while keeping its few-step behaviour on everything else.

Supported base models:

| Name           | Base                                    |
|----------------|-----------------------------------------|
| `flux_schnell` | FLUX.1-schnell                          |
| `flux_scfm`    | FLUX.1-dev + SCFM LoRA (fused)          |
| `sd35_turbo`   | Stable Diffusion 3.5 Large Turbo        |

<p align="center"><img src="example_grids/flux_schnell_nudity_checkpoint.png" width="420"></p>
<p align="center"><em>FLUX.1-schnell + DICE (nudity). Rows: the unsafe prompt "a nude girl, full front body shot", then four benign prompts.</em></p>

## Method

Each step optimizes `L = lamb1 * L_esd + lamb2 * L_attn + lamb3 * L_gen` over the LoRA parameters:

- **`L_esd` (erasure).** A concept latent `z` is sampled with the frozen base model and the
  pipeline's own few-step schedule, then re-noised to a random timestep `t`. The trained
  model's velocity for the concept prompt is regressed onto
  `v_0 - negative_guidance * (v_p - v_0)`, where `v_p` / `v_0` are the frozen base model's
  velocities for the concept prompt and the empty prompt (ESD, adapted to flow matching).
- **`L_attn` (keyword attention).** On target images of the concept, suppresses the joint
  attention weights pointing at the T5 tokens of `key_word` (EraseAnything-style).
- **`L_gen` (preservation).** On the same noisy target images, keeps the trained model's
  velocity close to the frozen base model's (`gen_loss_target: teacher`) or to the
  flow-matching target (`gen_loss_target: fm`).

Timesteps are drawn from one of `ddim_steps` equal-width bins of `[0, 1000)` (`timestep_sampling: fm_linear_bins`),
matching the coarse grid the distilled model is used on. The frozen base model is never
copied: its predictions are obtained by temporarily zeroing the LoRA weights (`utils/lora_utils.py`).

## Setup

```bash
pip install -r requirements.txt   # Python 3.10, CUDA GPU
huggingface-cli login             # FLUX.1-dev / SD3.5 are gated on the HF Hub
```

The released LoRAs are on the HF Hub at [`CookedMorsel/dice`](https://huggingface.co/CookedMorsel/dice):
`<model>/dice_nudity.safetensors` (DICE, nudity) for each model, and `flux_scfm/scfm_lora.safetensors`
(the SCFM LoRA that `flux_scfm` is built on). The scripts download them automatically.

## Quickstart

**Generate a grid with a released DICE LoRA** (rows = prompts, cols = seeds; saved to `output/grids/`):

```bash
python inference.py --model flux_schnell             # DICE LoRA from the HF Hub
python inference.py --model flux_schnell --no_lora   # undefended base model, for comparison
```

Default resolution is 1024 for `sd35_turbo` and 512 for the Flux models; see `python inference.py -h`
for prompts, seeds, steps and guidance.

## Training your own

**1. Generate target images** (10 per model / concept, sampled from the base model itself):

```bash
NUDITY_FLUX="a nude girl with beautiful hair and her breast open to see"
NUDITY_SD35="a nude girl, full front body shot"
VIOLENCE="a brutal scene of violence: people fighting with blood and gore"

python generate_data.py --config config/flux_schnell_nudity.yaml   --prompts "$NUDITY_FLUX" --num_per_prompt 10 --output_dir data/flux_schnell/nudity/
python generate_data.py --config config/flux_scfm_nudity.yaml      --prompts "$NUDITY_FLUX" --num_per_prompt 10 --output_dir data/flux_scfm/nudity/ --height 512 --width 512
python generate_data.py --config config/sd35_turbo_nudity.yaml     --prompts "$NUDITY_SD35" --num_per_prompt 10 --output_dir data/sd35_turbo/nudity/
python generate_data.py --config config/flux_schnell_violence.yaml --prompts "$VIOLENCE"    --num_per_prompt 10 --output_dir data/flux_schnell/violence/
python generate_data.py --config config/flux_scfm_violence.yaml    --prompts "$VIOLENCE"    --num_per_prompt 10 --output_dir data/flux_scfm/violence/ --height 512 --width 512
python generate_data.py --config config/sd35_turbo_violence.yaml   --prompts "$VIOLENCE"    --num_per_prompt 10 --output_dir data/sd35_turbo/violence/
```

**2. Train DICE** (100 steps; the LoRA is written to `output/<model>_<concept>/pytorch_lora_weights.safetensors`):

```bash
python train.py --config config/flux_schnell_nudity.yaml   # add --wandb to log losses and samples
```

Any `config/{flux_schnell,flux_scfm,sd35_turbo}_{nudity,violence}.yaml` works the same way.
The Flux configs fit on one 80 GB GPU. The `sd35_turbo` configs train at 1024px and need two:
`esd_offload_device: cuda:1` keeps a frozen copy of the base transformer on the second GPU for the
`v_0` / `v_p` forwards of `L_esd`.

**3. Render a grid with your LoRA:**

```bash
python inference.py --model flux_schnell --lora output/flux_schnell_nudity/pytorch_lora_weights.safetensors
```

## Configuration

The main keys of `config/*.yaml` (all are read by `train.py`; `generate_data.py` uses the model,
steps, guidance and resolution):

| Key | Meaning |
|-----|---------|
| `model_family` | `flux` or `sd3` |
| `pretrained_model_name_or_path` | Base model (HF repo id or local path) |
| `scfm_lora_path`, `scfm_lora_weight_name` | Flux only: LoRA fused into the base before training (used by `flux_scfm`) |
| `instance_data_dir`, `instance_prompt` | Target images and the prompt paired with them |
| `key_word` | Word in `instance_prompt` whose tokens `L_attn` suppresses |
| `esd_prompt`, `esd_contrast_prompt` | Optional: separate concept prompt for `L_esd`; prompt replacing the empty prompt in `v_0` |
| `lamb1`, `lamb2`, `lamb3` | Weights of `L_esd`, `L_attn`, `L_gen` (a zero weight skips that term's forwards) |
| `ddim_steps`, `guidance_scale` | Sampling steps / distilled guidance of the base model |
| `negative_guidance` | ESD guidance strength |
| `timestep_sampling` | `fm_linear_bins` (default), `inference_discrete` or `fm_linear_inference_discrete` |
| `gen_loss_target` | `teacher` (frozen base velocity) or `fm` (flow-matching target) |
| `rank`, `lora_target_modules` | LoRA rank (alpha = rank) and PEFT target modules |
| `learning_rate`, `max_train_steps`, `max_grad_norm`, `optimizer` | Optimization (`adamw` or `adamw8bit`) |
| `resolution`, `max_sequence_length` | Training resolution and T5 length |
| `gradient_checkpointing`, `esd_offload_device` | Memory savers (`esd_offload_device` is SD3 only) |

## Acknowledgements

Builds on [ESD](https://github.com/rohitgandikota/erasing),
[EraseAnything](https://github.com/tomguluson92/EraseAnything), and [diffusers](https://github.com/huggingface/diffusers).

## License

MIT (see [`LICENSE`](LICENSE)). Code adapted from EraseAnything (BSD-3-Clause) and diffusers
(Apache-2.0) keeps its original license, included at the end of [`LICENSE`](LICENSE).
