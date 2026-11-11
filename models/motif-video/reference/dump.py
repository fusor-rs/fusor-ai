"""Golden tensors from Motif-Video-2B's reference for `motif-video parity`: diffusers 0.41.0's
`MotifVideoPipeline` and `MotifVideoImage2VideoPipeline` run stage by stage as their `__call__`
runs them, in f32 on the CPU, with the model card's scheduler workaround (the repository's
`DPMSolverMultistepScheduler`, its `set_timesteps` given the step count alone) and the guider of
`guider/guider_config.json`. Two cases: text to video, and image to video from the repository's
`assets/i2v_sample.jpg`. T5Gemma2's bf16 text layers are upcast to f32 one at a time while they run;
the text encoder is freed before the f32 transformer loads, so the reference fits in 24 GB.

    python -I dump.py <checkpoint dir> <out dir>

Writes <out dir>/golden_f32_t2v.safetensors and golden_f32_i2v.safetensors. Recorded tensors (f32
unless noted; T = (frames - 1) / 4 + 1, h = height / 8, w = width / 8):
    prompt, negative                [n] u8       the prompts, UTF-8
    prompt_tokens, negative_tokens  [L] i64      their tokens, `<bos>` first, without the padding
    prompt_text, negative_text      [L, 2560]    T5Gemma2's last hidden states of those tokens
    noise          [16, T, h, w]     the starting latents
    condition      [16, T, h, w]     the first frame's normalized latents at frame 0, zeros after
                                     (zeros for text to video)
    timesteps      [steps]           the timesteps the transformer is asked at (integers)
    sigmas         [steps + 1]       the noise levels, the last 0
    guidance       [1]
    velocity       [2, 16, T, h, w]  the transformer's prediction at the first step: the prompt's,
                                     then the negative prompt's
    guided         [16, T, h, w]     adaptive projected guidance of those, at the first step
    latents        [16, T, h, w]     the latents after the last step
    video          [3, frames, height, width]  the decoded video, clamped to [-1, 1]
image to video only:
    image          [H0, W0, 3] u8    the first image as Pillow decodes it
    frame          [3, height, width]  the image resized and scaled to [-1, 1] (`video[:, 0]`)
    pixels         [3, 896, 896]     SigLIP's input
    image_states   [4096, 1152]      SigLIP's last hidden states
    moments        [32, h, w]        the autoencoder's encoding of the frame: 16 means, then 16
                                     log-variances (unclamped)
"""

import gc
import json
import os
import sys

import numpy as np
import torch
from diffusers import AutoencoderKLWan, DPMSolverMultistepScheduler, MotifVideoTransformer3DModel
from diffusers.guiders.adaptive_projected_guidance import MomentumBuffer, normalized_guidance
from diffusers.utils import load_image
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from safetensors.torch import save_file
from transformers import AutoTokenizer, SiglipImageProcessor, T5Gemma2Encoder
from transformers.models.t5gemma2.modeling_t5gemma2 import T5Gemma2RotaryEmbedding

NEGATIVE = (
    "text overlay, graphic overlay, watermark, logo, subtitles, timestamp, broadcast graphics, UI "
    "elements, random letters, frozen pose, rigid, static expression, jerky motion, mechanical "
    "motion, discontinuous motion, flat framing, depthless, dull lighting, monotone, crushed "
    "shadows, blown-out highlights, shifting background, fading background, poor continuity, "
    "identity drift, deformation, flickering, ghosting, smearing, duplication, mutated "
    "proportions, inconsistent clothing, flat colors, desaturated, tonally compressed, poor "
    "background separation, exposure shift, uneven brightness, color balance shift"
)
CASES = {
    "t2v": ("A red fox trots through fresh snow in a birch forest at golden hour, cinematic", None, 4),
    "i2v": (
        "Three friends stride through a sun-bleached meadow as a warm breeze ripples the tall dry "
        "grass around their legs.",
        "assets/i2v_sample.jpg",
        2,
    ),
}
WIDTH, HEIGHT, FRAMES, SEED, MAX_TOKENS = 160, 96, 9, 42, 512

checkpoint, out = sys.argv[1:3]


def path(*parts):
    return os.path.join(checkpoint, *parts)


def upcast(module, args):
    """Before a text module runs: its weights in f32, the embedding's scale as an f32 model holds it."""
    module.float()
    if hasattr(module, "scalar_embed_scale"):
        module.embed_scale.fill_(module.scalar_embed_scale)


def downcast(module, args, output):
    module.to(torch.bfloat16)


with torch.inference_mode():
    tokenizer = AutoTokenizer.from_pretrained(path("tokenizer"))
    encoder = T5Gemma2Encoder.from_pretrained(path("text_encoder"), dtype=torch.bfloat16).eval()
    text_model = encoder.text_model
    for module in [text_model.embed_tokens, *text_model.layers, text_model.norm]:
        module.register_forward_pre_hook(upcast)
        module.register_forward_hook(downcast)
    # Its frequencies are buffers the bf16 load rounded; an f32 model computes them in f32.
    text_model.rotary_emb = T5Gemma2RotaryEmbedding(text_model.config)
    vision_tower = encoder.vision_tower.float()

    def encode(text):
        """`_get_prompt_embeds`: 512 padded tokens and their mask; the valid rows apart."""
        inputs = tokenizer(
            [text], padding="max_length", max_length=MAX_TOKENS, truncation=True,
            add_special_tokens=True, return_attention_mask=True, return_tensors="pt",
        )
        states = encoder(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)[0].float()
        length = int(inputs.attention_mask.sum())
        mask = inputs.attention_mask.bool()
        return inputs.input_ids[0, :length].to(torch.int64), states, mask, states[0, :length]

    texts = {text: encode(text) for text in {case[0] for case in CASES.values()} | {NEGATIVE}}

    processor = VideoProcessor(vae_scale_factor=8)
    features = SiglipImageProcessor.from_pretrained(path("feature_extractor"))
    images = {}
    for name, (_, image_file, _) in CASES.items():
        if image_file is None:
            continue
        image = load_image(path(image_file))
        video = processor.preprocess_video([image], height=HEIGHT, width=WIDTH).to(torch.float32)
        frame = video.permute(0, 2, 1, 3, 4)[:, 0]
        vision = ((frame + 1) / 2).clamp(0, 1)
        pixels = features.preprocess(
            images=vision, do_resize=True, do_rescale=False, do_normalize=True,
            do_convert_rgb=True, return_tensors="pt",
        )["pixel_values"].to(torch.float32)
        states = vision_tower(pixel_values=pixels).last_hidden_state
        images[name] = {
            "image": torch.from_numpy(np.array(image)),
            "video": video[:, :, 0:1],
            "frame": frame[0],
            "pixels": pixels[0],
            "image_states": states,
        }
    del encoder, text_model, vision_tower
    gc.collect()

    vae = AutoencoderKLWan.from_pretrained(path("vae"), torch_dtype=torch.float32).eval()
    mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1)
    shape = (1, 16, (FRAMES - 1) // 4 + 1, HEIGHT // 8, WIDTH // 8)
    starts = {}
    for name in CASES:
        generator = torch.Generator("cpu").manual_seed(SEED)
        noise = randn_tensor(shape, generator=generator, device=torch.device("cpu"), dtype=torch.float32)
        condition = torch.zeros(shape)
        mask = torch.zeros((1, 1, *shape[2:]))
        if name in images:
            posterior = vae.encode(images[name]["video"]).latent_dist
            images[name]["moments"] = posterior.parameters[0, :, 0]
            first = (posterior.sample(generator=generator) - mean) / std
            condition = first.repeat(1, 1, shape[2], 1, 1)
            condition[:, :, 1:] = 0
            mask[:, :, 0] = 1.0
        starts[name] = (noise, condition, mask)

    with open(path("guider", "guider_config.json")) as file:
        guider = json.load(file)
    transformer = MotifVideoTransformer3DModel.from_pretrained(
        path("transformer"), torch_dtype=torch.float32
    ).eval()
    results = {}
    for name, (prompt, _, steps) in CASES.items():
        noise, condition, mask = starts[name]
        scheduler = DPMSolverMultistepScheduler.from_pretrained(path("scheduler"))
        # The model card's FlowDPMSolver: the pipeline's `sigmas` and `mu` dropped.
        scheduler.set_timesteps(num_inference_steps=steps)
        momentum = MomentumBuffer(guider["adaptive_projected_guidance_momentum"])
        image_states = images[name]["image_states"] if name in images else None
        branches = [texts[prompt][1:3], texts[NEGATIVE][1:3]]
        latents = noise
        for step, timestep in enumerate(scheduler.timesteps):
            hidden = torch.cat([latents, condition, mask], dim=1)
            predictions = [
                transformer(
                    hidden_states=hidden, timestep=timestep.expand(1), encoder_hidden_states=states,
                    encoder_attention_mask=attention, image_embeds=image_states, return_dict=False,
                )[0]
                for states, attention in branches
            ]
            guided = normalized_guidance(
                predictions[0], predictions[1], guider["guidance_scale"], momentum, guider["eta"],
                guider["adaptive_projected_guidance_rescale"], guider["use_original_formulation"],
                guider["adaptive_projected_guidance_norm_dim"],
            )
            if step == 0:
                velocity, first_guided = torch.stack([p[0] for p in predictions]), guided[0]
            latents = scheduler.step(guided, timestep, latents, return_dict=False)[0]
        results[name] = (scheduler, velocity, first_guided, latents)
    del transformer
    gc.collect()

    for name, (prompt, _, steps) in CASES.items():
        scheduler, velocity, first_guided, latents = results[name]
        video = vae.decode(latents * std + mean, return_dict=False)[0]
        noise, condition, _ = starts[name]
        utf8 = lambda value: torch.tensor(list(value.encode()), dtype=torch.uint8)
        tensors = {
            "prompt": utf8(prompt),
            "negative": utf8(NEGATIVE),
            "prompt_tokens": texts[prompt][0],
            "negative_tokens": texts[NEGATIVE][0],
            "prompt_text": texts[prompt][3],
            "negative_text": texts[NEGATIVE][3],
            "noise": noise[0],
            "condition": condition[0],
            "timesteps": scheduler.timesteps.to(torch.float32),
            "sigmas": scheduler.sigmas.to(torch.float32),
            "guidance": torch.tensor([guider["guidance_scale"]], dtype=torch.float32),
            "velocity": velocity,
            "guided": first_guided,
            "latents": latents[0],
            "video": video[0],
        }
        if name in images:
            tensors.update({key: value for key, value in images[name].items() if key != "video"})
            tensors["image_states"] = tensors["image_states"][0]
        file = os.path.join(out, f"golden_f32_{name}.safetensors")
        save_file({key: value.contiguous() for key, value in tensors.items()}, file)
        print(f"{name}: {WIDTH}x{HEIGHT}, {FRAMES} frames, {steps} steps, timesteps {scheduler.timesteps.tolist()} -> {file}")
