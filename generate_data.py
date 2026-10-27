"""
Generate the target images used by L_attn / L_gen: samples of the concept from the
original (not yet erased) model, with the steps / guidance / resolution of a DICE config.

Usage:
    python generate_data.py \
        --config config/flux_schnell_nudity.yaml \
        --prompts "a nude girl with beautiful hair and her breast open to see" \
        --num_per_prompt 10 \
        --output_dir data/flux_schnell/nudity/
"""
import argparse
import os

import torch
import yaml

from utils.dataset import IMAGE_EXTENSIONS
from utils.pipeline import load_pipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Generate target images for DICE training")
    parser.add_argument("--config", type=str, required=True, help="Path to a DICE YAML config")
    parser.add_argument("--prompts", nargs="+", required=True, help="Prompts to generate images from")
    parser.add_argument("--num_per_prompt", type=int, default=5, help="Number of images per prompt")
    parser.add_argument("--output_dir", type=str, default="data/", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Seed of the first image; image i uses seed + i")
    parser.add_argument("--height", type=int, default=None, help="Default: the config's resolution")
    parser.add_argument("--width", type=int, default=None, help="Default: the config's resolution")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    os.makedirs(args.output_dir, exist_ok=True)
    expected_count = args.num_per_prompt * len(args.prompts)
    existing = [f for f in os.listdir(args.output_dir) if f.lower().endswith(IMAGE_EXTENSIONS)]
    if len(existing) >= expected_count:
        print(f"Skipping: {args.output_dir} already has {len(existing)} images (expected {expected_count})")
        return

    model_family = config["model_family"]
    resolution = config.get("resolution", 512)
    max_sequence_length = config.get("max_sequence_length", {"flux": 256, "sd3": 512}[model_family])

    print(f"Loading {model_family} model: {config['pretrained_model_name_or_path']}")
    pipe = load_pipeline(
        model_family,
        config["pretrained_model_name_or_path"],
        config.get("scfm_lora_path"),
        config.get("scfm_lora_weight_name"),
    )

    print(f"Generating {args.num_per_prompt} images per prompt for {len(args.prompts)} prompts")
    img_idx = 0
    for prompt in args.prompts:
        print(f"  Prompt: {prompt}")
        for _ in range(args.num_per_prompt):
            seed = args.seed + img_idx
            image = pipe(
                prompt,
                num_inference_steps=config.get("ddim_steps", 4),
                guidance_scale=config.get("guidance_scale", 0),
                height=args.height or resolution,
                width=args.width or resolution,
                max_sequence_length=max_sequence_length,
                generator=torch.Generator("cuda").manual_seed(seed),
            ).images[0]
            filename = f"{img_idx:04d}.png"
            image.save(os.path.join(args.output_dir, filename))
            print(f"    Saved {filename} (seed={seed})")
            img_idx += 1

    print(f"Done. Generated {img_idx} images in {args.output_dir}")


if __name__ == "__main__":
    main()
