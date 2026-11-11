"""Golden tensors from Wan2.1's reference implementation for `wan2-1 parity`: the modules of the
official repository (github.com/Wan-Video/Wan2.1, at the commit wan2-1-wasm's justfile pins) in
f32 on the CPU, run stage by stage as `WanT2V.generate` runs them with its `dpm++` solver, on one
prompt and Wan's default negative prompt. umT5-XXL's bf16 weights stay memory-mapped, each block
upcast to f32 only while it runs, so the reference fits beside the f32 DiT in 24 GB.

    python -I dump.py <Wan2.1 repository> <checkpoint dir> <out.safetensors>

Recorded tensors (f32 unless noted):
    prompt, negative                [n] u8         the prompts, UTF-8
    prompt_tokens, negative_tokens  [L] i64        their umT5 tokens, the closing </s> included
    prompt_text, negative_text      [L, 4096]      umT5-XXL's last hidden states, after its final norm
    noise       [16, T, h, w]       the starting latents: T = (frames - 1) / 4 + 1, h = height / 8,
                                    w = width / 8
    timesteps   [steps]             the timesteps the DiT is asked at (integers)
    sigmas      [steps + 1]         the noise levels, the last 0
    guidance, shift  [1]
    velocity    [2, 16, T, h, w]    the DiT's prediction at the first step: the prompt's, then the
                                    negative prompt's
    latents     [16, T, h, w]       the latents after the last step
    video       [3, frames, height, width]  the decoded video, clamped to [-1, 1]
"""

import json
import os
import sys
import types

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

PROMPT = "A red fox trots through fresh snow in a birch forest at golden hour, cinematic"
WIDTH, HEIGHT, FRAMES, STEPS, GUIDANCE, SHIFT, SEED = 160, 96, 9, 4, 5.0, 5.0, 42

repo, checkpoint, out = sys.argv[1:4]
# t5.py calls it in a default argument when it is imported; there is no CUDA here.
torch.cuda.current_device = lambda: 0
# The packages' __init__ files import every pipeline and its dependencies; only these modules run.
for name in ("wan", "wan.modules", "wan.utils", "wan.configs"):
    package = types.ModuleType(name)
    package.__path__ = [os.path.join(repo, *name.split("."))]
    sys.modules[name] = package

import wan.modules.model as dit_module
from wan.configs.wan_t2v_1_3B import t2v_1_3B as config
from wan.modules.t5 import umt5_xxl
from wan.modules.tokenizers import HuggingfaceTokenizer
from wan.modules.vae import WanVAE
from wan.utils.fm_solvers import (
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)


def attention(q, k, v, k_lens=None, **_):
    """flash_attention without flash-attn, in f32: every sequence here is whole (one video, one
    context), so `k_lens` masks nothing."""
    out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
    return out.transpose(1, 2).contiguous()


dit_module.flash_attention = attention

tokenizer = HuggingfaceTokenizer(
    name=os.path.join(checkpoint, config.t5_tokenizer), seq_len=config.text_len, clean="whitespace"
)
state = torch.load(
    os.path.join(checkpoint, config.t5_checkpoint), map_location="cpu", mmap=True, weights_only=True
)
encoder = umt5_xxl(encoder_only=True, dtype=torch.float32, device="meta").eval()


def materialize(module, prefix):
    """`module` holding its weights from the checkpoint, upcast to f32."""
    weights = {name: state[prefix + name].float() for name, _ in module.named_parameters()}
    module.load_state_dict(weights, assign=True)
    return module


def encode(text):
    """`T5EncoderModel.__call__`: 512 padded tokens under their mask, a block at a time."""
    ids, mask = tokenizer([text], return_mask=True, add_special_tokens=True)
    length = int(mask.gt(0).sum())
    x = state["token_embedding.weight"][ids[0]].float().unsqueeze(0)
    for index, block in enumerate(encoder.blocks):
        x = materialize(block, f"blocks.{index}.")(x, mask)
        block.to("meta")
    x = materialize(encoder.norm, "norm.")(x)
    return ids[0, :length].to(torch.int64), x[0, :length]


with torch.inference_mode():
    prompt_tokens, prompt_text = encode(PROMPT)
    negative_tokens, negative_text = encode(config.sample_neg_prompt)
    del state, encoder

    shape = (16, (FRAMES - 1) // config.vae_stride[0] + 1, HEIGHT // config.vae_stride[1], WIDTH // config.vae_stride[2])
    tokens = shape[1] * shape[2] * shape[3] // (config.patch_size[1] * config.patch_size[2])
    noise = torch.randn(*shape, dtype=torch.float32, generator=torch.Generator("cpu").manual_seed(SEED))

    with open(os.path.join(checkpoint, "config.json")) as file:
        dimensions = {key: value for key, value in json.load(file).items() if not key.startswith("_")}
    with torch.device("meta"):
        dit = dit_module.WanModel(**dimensions)
    dit.load_state_dict(load_file(os.path.join(checkpoint, "diffusion_pytorch_model.safetensors")), assign=True)
    head_dim = dit.dim // dit.num_heads
    # Built in __init__ (on the meta device here), not stored in the checkpoint.
    dit.freqs = torch.cat(
        [
            dit_module.rope_params(1024, head_dim - 4 * (head_dim // 6)),
            dit_module.rope_params(1024, 2 * (head_dim // 6)),
            dit_module.rope_params(1024, 2 * (head_dim // 6)),
        ],
        dim=1,
    )
    dit.eval()

    scheduler = FlowDPMSolverMultistepScheduler(
        num_train_timesteps=config.num_train_timesteps, shift=1, use_dynamic_shifting=False
    )
    timesteps, _ = retrieve_timesteps(scheduler, device="cpu", sigmas=get_sampling_sigmas(STEPS, SHIFT))
    generator = torch.Generator("cpu").manual_seed(SEED)
    latents = noise
    for step, timestep in enumerate(timesteps):
        at = torch.stack([timestep])
        conditional = dit([latents], t=at, context=[prompt_text], seq_len=tokens)[0]
        unconditional = dit([latents], t=at, context=[negative_text], seq_len=tokens)[0]
        if step == 0:
            velocity = torch.stack([conditional, unconditional])
        guided = unconditional + GUIDANCE * (conditional - unconditional)
        latents = scheduler.step(
            guided.unsqueeze(0), timestep, latents.unsqueeze(0), return_dict=False, generator=generator
        )[0].squeeze(0)
    del dit

    vae = WanVAE(vae_pth=os.path.join(checkpoint, config.vae_checkpoint), device="cpu")
    video = vae.decode([latents])[0]

utf8 = lambda value: torch.tensor(list(value.encode()), dtype=torch.uint8)
save_file(
    {
        "prompt": utf8(PROMPT),
        "negative": utf8(config.sample_neg_prompt),
        "prompt_tokens": prompt_tokens.contiguous(),
        "negative_tokens": negative_tokens.contiguous(),
        "prompt_text": prompt_text.contiguous(),
        "negative_text": negative_text.contiguous(),
        "noise": noise.contiguous(),
        "timesteps": timesteps.to(torch.float32).contiguous(),
        "sigmas": scheduler.sigmas.to(torch.float32).contiguous(),
        "guidance": torch.tensor([GUIDANCE], dtype=torch.float32),
        "shift": torch.tensor([SHIFT], dtype=torch.float32),
        "velocity": velocity.contiguous(),
        "latents": latents.contiguous(),
        "video": video.contiguous(),
    },
    out,
)
print(f"{WIDTH}x{HEIGHT}, {FRAMES} frames, {STEPS} steps, timesteps {timesteps.tolist()} -> {out}")
