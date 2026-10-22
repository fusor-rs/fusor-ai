"""Golden tensors from the reference pipeline for `realistic-vision parity`: diffusers'
StableDiffusionPipeline at a pinned version, loaded from Realistic Vision's single-file checkpoint
with the SD VAE ft-MSE, in f32 on the CPU, with DPMSolverMultistepScheduler at Karras sigmas,
on one prompt. Its components run here one by one, as the pipeline's `__call__` runs them, so
that every stage's output is kept.

    python dump.py <checkpoint dir> <out.safetensors>

Recorded tensors (f32 unless noted):
    prompt, negative                  [n] u8      the prompts, UTF-8
    prompt_tokens, negative_tokens    [77] i64    their CLIP tokens
    text        [2, 77, 768]   the text encoder's states: the negative prompt's, then the prompt's
    noise       [1, 4, h, w]   the starting latents
    timesteps   [steps]        the timesteps the UNet is asked at
    sigmas      [steps + 1]    the noise levels, the last 0
    guidance    [1]
    predicted   [2, 4, h, w]   the UNet's noise for both prompts at the first step
    latents     [1, 4, h, w]   the latents after the last step
    image       [3, 8h, 8w]    the decoded image, in [-1, 1]
"""

import sys

import torch
from diffusers import AutoencoderKL, DPMSolverMultistepScheduler, StableDiffusionPipeline
from safetensors.torch import save_file

PROMPT = "RAW photo, a portrait of an old fisherman in a yellow raincoat, harbour at dawn, soft light, film grain"
# The model card's recommendation, the page's default (app/src/app.rs).
NEGATIVE = (
    "(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime), "
    "text, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, "
    "mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, "
    "blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross "
    "proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, "
    "too many fingers, long neck"
)
STEPS, GUIDANCE, SEED, SIZE = 12, 5.0, 42, 512
LATENT = 8

checkpoint, out = sys.argv[1:3]
vae = AutoencoderKL.from_single_file(
    f"{checkpoint}/vae-ft-mse-840000-ema-pruned.safetensors",
    config=checkpoint,
    subfolder="vae",
    torch_dtype=torch.float32,
    local_files_only=True,
)
pipe = StableDiffusionPipeline.from_single_file(
    f"{checkpoint}/Realistic_Vision_V6.0_NV_B1_fp16.safetensors",
    config=checkpoint,
    vae=vae,
    safety_checker=None,
    feature_extractor=None,
    requires_safety_checker=False,
    torch_dtype=torch.float32,
    local_files_only=True,
)
scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)

with torch.inference_mode():
    tokens = [
        pipe.tokenizer(text, padding="max_length", max_length=77, truncation=True, return_tensors="pt").input_ids
        for text in (NEGATIVE, PROMPT)
    ]
    text = torch.cat([pipe.text_encoder(ids)[0] for ids in tokens])
    generator = torch.Generator("cpu").manual_seed(SEED)
    noise = torch.randn((1, 4, SIZE // LATENT, SIZE // LATENT), generator=generator, dtype=torch.float32)
    scheduler.set_timesteps(STEPS)
    latents = noise * scheduler.init_noise_sigma
    for step, timestep in enumerate(scheduler.timesteps):
        sample = scheduler.scale_model_input(torch.cat([latents] * 2), timestep)
        predicted = pipe.unet(sample, timestep, encoder_hidden_states=text).sample
        if step == 0:
            first = predicted.clone()
        unconditional, conditional = predicted.chunk(2)
        guided = unconditional + GUIDANCE * (conditional - unconditional)
        latents = scheduler.step(guided, timestep, latents).prev_sample
    image = pipe.vae.decode(latents / pipe.vae.config.scaling_factor).sample[0]

utf8 = lambda value: torch.tensor(list(value.encode()), dtype=torch.uint8)
save_file(
    {
        "prompt": utf8(PROMPT),
        "negative": utf8(NEGATIVE),
        "negative_tokens": tokens[0][0].to(torch.int64).contiguous(),
        "prompt_tokens": tokens[1][0].to(torch.int64).contiguous(),
        "text": text.contiguous(),
        "noise": noise.contiguous(),
        "timesteps": scheduler.timesteps.to(torch.float32).contiguous(),
        "sigmas": scheduler.sigmas.to(torch.float32).contiguous(),
        "guidance": torch.tensor([GUIDANCE], dtype=torch.float32),
        "predicted": first.contiguous(),
        "latents": latents.contiguous(),
        "image": image.contiguous(),
    },
    out,
)
print(f"{SIZE}x{SIZE}, {STEPS} steps, timesteps {scheduler.timesteps.tolist()} -> {out}")
