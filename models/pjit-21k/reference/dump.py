"""Golden tensors from P-JiT 21K's reference implementation for `pjit-21k parity`: the modules of
the official repository (github.com/Linum-AI/pjit-21k, at the commit pjit-21k-wasm's justfile pins)
in f32 on the CPU, run as `pjit_21k.sampler.generate` runs them (its CUDA noise and bf16 autocast
aside), on one caption and the empty negative prompt. Qwen3.5-4B's bf16 weights stay bf16 in memory,
each decoder layer upcast to f32 only while it runs, and the run stops after the last extracted
layer; the text encoder is freed before the f32 DiT loads, so the two are never resident together.

    python -I dump.py <pjit-21k repository> <checkpoint dir> <out.safetensors>

<checkpoint dir> holds qwen3.5-4b/ and pjit-21k/ as the product's scripts/fetch-checkpoint.sh lays
them out.

Recorded tensors (f32 unless noted):
    prompt, negative                 [n] u8      the caption and the negative prompt, UTF-8
    prompt_tokens, negative_tokens   [L] i64     their Qwen3.5 tokens, chat template applied
    prompt_text, negative_text       [L, 7680]   the hidden states after layers 7, 15 and 27,
                                                 side by side
    noise       [3, H, W]            the standard normal draw; the sampler starts from 4 times it
    seen        [3, H, W]            the DiT's input at the first step: the start rounded to bf16
    times       [steps + 1]          the knots from 1 to 0
    timesteps   [steps]              the integer timesteps the DiT is asked at
    guidance    [1]
    prediction  [2, 3, H, W]         the DiT's clean image at the first step: the caption's, then
                                     the negative prompt's
    sample      [3, H, W]            the image after the last step, before its clamp
"""

import gc
import os
import sys
import types

import torch
from safetensors.torch import save_file

CAPTION = (
    "Close-up portrait of an elderly Middle Eastern woman looking calmly into the lens, her face "
    "filling the tall frame. A cream headscarf and the collar of a dark green cardigan frame her "
    "chin. Soft overcast daylight smooths the fine lines of her skin, and the olive grove behind "
    "melts into silvery blur."
)
NEGATIVE = ""
WIDTH, HEIGHT, STEPS, GUIDANCE, SEED = 200, 288, 4, 15.0, 42

repo, checkpoint, out = sys.argv[1:4]
# The package's __init__ imports the upsampler and every module; only these run.
package = types.ModuleType("pjit_21k")
package.__path__ = [os.path.join(repo, "pjit_21k")]
sys.modules["pjit_21k"] = package

from pjit_21k.config import SamplerConfig
from pjit_21k.model import PyramidJiT
from pjit_21k.sampler import MomentumBuffer, _text_kwargs, adaptive_projected_guidance
from pjit_21k.text_encoder import QwenTextEncoder

SAMPLER = SamplerConfig(height=HEIGHT, width=WIDTH, sampling_steps=STEPS, guidance_scale=GUIDANCE)


class Extracted(Exception):
    """Raised after the last extracted layer: the layers after it do not change the states."""


def floats(value):
    return value.float() if torch.is_tensor(value) and value.is_floating_point() else value


def encode(encoder, caption):
    """`QwenTextEncoder.__call__` on one caption, in f32: its tokens and its states."""
    formatted = encoder.tokenizer.apply_chat_template(
        [{"role": "user", "content": caption}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = encoder.tokenizer(
        [formatted], return_tensors="pt", padding=True, truncation=True, max_length=encoder.max_length
    )
    text_model = encoder.model.model
    last = max(encoder.extraction_layers)
    captured, dtypes = {}, {}

    def upcast(index):
        def hook(module, args, kwargs):
            dtypes[index] = {name: p.dtype for name, p in module.named_parameters()}
            module.float()
            return tuple(floats(a) for a in args), {k: floats(v) for k, v in kwargs.items()}

        return hook

    def restore(index):
        def hook(module, args, kwargs, output):
            for name, parameter in module.named_parameters():
                parameter.data = parameter.data.to(dtypes[index][name])
            if index in encoder.extraction_layers:
                captured[index] = (output[0] if isinstance(output, tuple) else output).clone()
            if index == last:
                raise Extracted

        return hook

    handles = [text_model.embed_tokens.register_forward_hook(lambda module, args, output: output.float())]
    for index, layer in enumerate(text_model.layers[: last + 1]):
        handles.append(layer.register_forward_pre_hook(upcast(index), with_kwargs=True))
        handles.append(layer.register_forward_hook(restore(index), with_kwargs=True))
    try:
        encoder.model(**inputs)
    except Extracted:
        pass
    for handle in handles:
        handle.remove()
    states = torch.cat([captured[index][0] for index in encoder.extraction_layers], dim=-1)
    return inputs["input_ids"][0].to(torch.int64), states


with torch.inference_mode():
    encoder = QwenTextEncoder(
        model_path=os.path.join(checkpoint, "qwen3.5-4b"),
        extraction_layers=(7, 15, 27),
        max_length=256,
        device=torch.device("cpu"),
    )
    prompt_tokens, prompt_text = encode(encoder, CAPTION)
    negative_tokens, negative_text = encode(encoder, NEGATIVE)
    del encoder
    gc.collect()

    model = PyramidJiT.from_pretrained(weights_dir=os.path.join(checkpoint, "pjit-21k"), device="cpu")
    cond = _text_kwargs(text=prompt_text, batch_size=1, text_len=model.text_len)
    uncond = _text_kwargs(text=negative_text, batch_size=1, text_len=model.text_len)
    text = torch.cat([cond["text"], uncond["text"]])
    text_lens = torch.cat([cond["text_lens"], uncond["text_lens"]])

    noise = torch.randn(
        (1, 3, 1, HEIGHT, WIDTH), dtype=torch.float32, generator=torch.Generator("cpu").manual_seed(SEED)
    )
    x = SAMPLER.noise_scale * noise
    timesteps = torch.linspace(1.0, 0.0, STEPS + 1)
    timesteps_int = (timesteps[:-1] * SAMPLER.num_timesteps).long()
    momentum = torch.zeros_like(x)
    for i in range(STEPS):
        t, t_next = timesteps[i], timesteps[i + 1]
        x_in = x.to(torch.bfloat16).float()
        x_pred_cond, x_pred_uncond = model(
            x_t=torch.cat([x_in, x_in]), t=timesteps_int[i].expand(2), text=text, text_lens=text_lens
        ).chunk(2)
        if i == 0:
            seen, prediction = x_in, torch.cat([x_pred_cond, x_pred_uncond])
        v_cond = (x_in - x_pred_cond.float()) / t
        v_uncond = (x_in - x_pred_uncond.float()) / t
        momentum_buffer = MomentumBuffer(momentum=SAMPLER.apg_momentum)
        momentum_buffer.running_average = momentum.to(torch.bfloat16).float()
        v_guided = adaptive_projected_guidance(
            pred_cond=v_cond,
            pred_other=v_uncond,
            guidance_scale=SAMPLER.guidance_scale,
            momentum_buffer=momentum_buffer,
            eta=SAMPLER.apg_eta,
            rescale=SAMPLER.apg_rescale,
        )
        momentum = momentum_buffer.running_average
        x_pred = x_in - t * v_guided
        v = (x - x_pred) / t
        x = x + (t_next - t) * v

utf8 = lambda value: torch.tensor(list(value.encode()), dtype=torch.uint8)
save_file(
    {
        "prompt": utf8(CAPTION),
        "negative": utf8(NEGATIVE),
        "prompt_tokens": prompt_tokens.contiguous(),
        "negative_tokens": negative_tokens.contiguous(),
        "prompt_text": prompt_text.contiguous(),
        "negative_text": negative_text.contiguous(),
        "noise": noise[0, :, 0].contiguous(),
        "seen": seen[0, :, 0].contiguous(),
        "times": timesteps.contiguous(),
        "timesteps": timesteps_int.to(torch.float32).contiguous(),
        "guidance": torch.tensor([GUIDANCE], dtype=torch.float32),
        "prediction": prediction[:, :, 0].contiguous(),
        "sample": x[0, :, 0].contiguous(),
    },
    out,
)
print(f"{WIDTH}x{HEIGHT}, {STEPS} steps, timesteps {timesteps_int.tolist()} -> {out}")
