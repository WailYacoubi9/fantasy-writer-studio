"""Generate a base vs. LoRA comparison for one genre with FLUX.1-dev.

Usage:
    python fine-tuning/inference/generate.py <romantic|scifi|fantasy> "<scene description>"

Both images use the same prompt and seed; only the LoRA (and, for Fantasy, the
pivotal-tuning embeddings) differs. Results are saved in fine-tuning/results/.

Memory strategy:
- GPU with 40 GB or more: full bf16 model.
- Smaller GPU (tested on an 8 GB RTX 5070 Laptop): the prompts are encoded first
  on the CPU and the text encoders are freed, then a 4-bit GGUF transformer runs
  on the GPU. No bitsandbytes is needed, which avoids its crashes on Windows.
"""
import gc
import os
import sys

import torch
from PIL import Image, ImageDraw
from diffusers import FluxPipeline, FluxTransformer2DModel
from safetensors.torch import load_file

BASE = "black-forest-labs/FLUX.1-dev"
GGUF_URL = "https://huggingface.co/city96/FLUX.1-dev-gguf/blob/main/flux1-dev-Q4_K_S.gguf"
GENRES = {
    "romantic": ("fine-tuning/loras/romantic", "romantic_style_lora.safetensors", "romantic_style"),
    "scifi": ("fine-tuning/loras/scifi", "pytorch_lora_weights.safetensors", "science_fiction_style"),
    "fantasy": ("fine-tuning/loras/fantasy", "pytorch_lora_weights.safetensors", "<s0><s1>"),
}
SEED = 42
STEPS = 28
WIDTH, HEIGHT = 1024, 768
OUT_DIR = "fine-tuning/results"
DTYPE = torch.bfloat16


def log(message):
    print(message, flush=True)


def free_memory():
    gc.collect()
    torch.cuda.empty_cache()


def encode_prompts(genre, folder, base_prompt, lora_prompt, device):
    """Encode both prompts with CLIP + T5, then free the text encoders."""
    log(f"Encoding prompts on {device}...")
    text_pipe = FluxPipeline.from_pretrained(
        BASE, transformer=None, vae=None, torch_dtype=DTYPE).to(device)

    with torch.no_grad():
        base_embeds = text_pipe.encode_prompt(
            prompt=base_prompt, prompt_2=None, device=device, max_sequence_length=512)

        if genre == "fantasy":
            # Learned <s0><s1> embeddings for the CLIP encoder (pivotal tuning)
            emb = load_file(f"{folder}/fantasy-flux-lora-output_emb.safetensors")
            text_pipe.load_textual_inversion(
                emb["clip_l"], token=["<s0>", "<s1>"],
                text_encoder=text_pipe.text_encoder, tokenizer=text_pipe.tokenizer)

        if lora_prompt == base_prompt:
            lora_embeds = base_embeds
        else:
            lora_embeds = text_pipe.encode_prompt(
                prompt=lora_prompt, prompt_2=None, device=device, max_sequence_length=512)

    del text_pipe
    free_memory()
    # encode_prompt returns (prompt_embeds, pooled_prompt_embeds, text_ids)
    return base_embeds[:2], lora_embeds[:2]


def load_image_pipeline(large_gpu):
    if large_gpu:
        log("Loading full bf16 transformer...")
        transformer = FluxTransformer2DModel.from_pretrained(
            BASE, subfolder="transformer", torch_dtype=DTYPE)
    else:
        from diffusers import GGUFQuantizationConfig

        log("Loading 4-bit GGUF transformer...")
        transformer = FluxTransformer2DModel.from_single_file(
            GGUF_URL,
            quantization_config=GGUFQuantizationConfig(compute_dtype=DTYPE),
            torch_dtype=DTYPE,
            config=BASE,
            subfolder="transformer",
        )

    pipe = FluxPipeline.from_pretrained(
        BASE, transformer=transformer,
        text_encoder=None, text_encoder_2=None, tokenizer=None, tokenizer_2=None,
        torch_dtype=DTYPE)
    pipe.enable_model_cpu_offload()
    return pipe


def generate(pipe, embeds):
    prompt_embeds, pooled_embeds = embeds
    return pipe(
        prompt_embeds=prompt_embeds.to("cuda", DTYPE),
        pooled_prompt_embeds=pooled_embeds.to("cuda", DTYPE),
        num_inference_steps=STEPS, guidance_scale=3.5,
        width=WIDTH, height=HEIGHT,
        generator=torch.Generator().manual_seed(SEED),
    ).images[0]


def main():
    genre, scene = sys.argv[1], sys.argv[2]
    folder, weights, trigger = GENRES[genre]
    os.makedirs(OUT_DIR, exist_ok=True)

    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    large_gpu = vram_gb >= 40
    log(f"GPU: {torch.cuda.get_device_name(0)} ({vram_gb:.0f} GB)")

    # Same scene text for both images; only the LoRA (and its trigger) differs
    if genre == "fantasy":
        base_prompt = f"{scene}, epic fantasy art"
        lora_prompt = f"{trigger} {scene}, epic fantasy art"
    else:
        base_prompt = f"{scene}, {trigger}"
        lora_prompt = base_prompt

    # 1. Text encoders (on the GPU only if it is large enough)
    base_embeds, lora_embeds = encode_prompts(
        genre, folder, base_prompt, lora_prompt, "cuda" if large_gpu else "cpu")

    # 2. Image model
    pipe = load_image_pipeline(large_gpu)

    log(f"[1/2] Base model: {base_prompt}")
    base_image = generate(pipe, base_embeds)
    base_image.save(f"{OUT_DIR}/{genre}_base.png")

    log("Loading LoRA...")
    pipe.load_lora_weights(folder, weight_name=weights)

    log(f"[2/2] With LoRA: {lora_prompt}")
    lora_image = generate(pipe, lora_embeds)
    lora_image.save(f"{OUT_DIR}/{genre}_lora.png")

    # Side-by-side comparison with labels
    comparison = Image.new("RGB", (WIDTH * 2, HEIGHT + 40), "white")
    comparison.paste(base_image, (0, 40))
    comparison.paste(lora_image, (WIDTH, 40))
    draw = ImageDraw.Draw(comparison)
    draw.text((10, 12), "FLUX.1-dev (base)", fill="black")
    draw.text((WIDTH + 10, 12), f"FLUX.1-dev + {genre} LoRA", fill="black")
    comparison.save(f"{OUT_DIR}/{genre}_comparison.png")
    log(f"Saved {OUT_DIR}/{genre}_comparison.png")


if __name__ == "__main__":
    main()
