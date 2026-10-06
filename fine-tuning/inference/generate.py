import sys
import torch
from diffusers import FluxPipeline, FluxTransformer2DModel
from diffusers import BitsAndBytesConfig as DiffusersBnbConfig
from transformers import T5EncoderModel
from transformers import BitsAndBytesConfig as TransformersBnbConfig
from safetensors.torch import load_file

BASE = "black-forest-labs/FLUX.1-dev"
GENRES = {
    "romantic": ("fine-tuning/loras/romantic", "romantic_style_lora.safetensors", "romantic_style"),
    "scifi": ("fine-tuning/loras/scifi", "pytorch_lora_weights.safetensors", "science_fiction_style"),
    "fantasy": ("fine-tuning/loras/fantasy", "pytorch_lora_weights.safetensors", "<s0><s1>"),
}

genre, scene = sys.argv[1], sys.argv[2]
folder, weights, trigger = GENRES[genre]

# 4-bit transformer (~7 GB) and 8-bit T5 so the model fits in 8 GB VRAM / 32 GB RAM
transformer = FluxTransformer2DModel.from_pretrained(
    BASE, subfolder="transformer", torch_dtype=torch.bfloat16,
    quantization_config=DiffusersBnbConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16),
)
text_encoder_2 = T5EncoderModel.from_pretrained(
    BASE, subfolder="text_encoder_2", torch_dtype=torch.bfloat16,
    quantization_config=TransformersBnbConfig(load_in_8bit=True),
)
pipe = FluxPipeline.from_pretrained(
    BASE, transformer=transformer, text_encoder_2=text_encoder_2, torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload()

pipe.load_lora_weights(folder, weight_name=weights)

if genre == "fantasy":
    emb = load_file(f"{folder}/fantasy-flux-lora-output_emb.safetensors")
    pipe.load_textual_inversion(emb["clip_l"], token=["<s0>", "<s1>"],
                                text_encoder=pipe.text_encoder, tokenizer=pipe.tokenizer)
    prompt = f"{trigger} {scene}, epic fantasy art"
else:
    prompt = f"{scene}, {trigger}"

image = pipe(prompt, num_inference_steps=28, guidance_scale=3.5, height=768, width=1024,
             generator=torch.Generator().manual_seed(42)).images[0]
image.save(f"{genre}_test.png")
print(f"Saved {genre}_test.png")