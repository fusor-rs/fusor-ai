"""Golden tensors from Iris-3B's reference implementation for `iris parity`: the modules of
github.com/speridlabs/iris-3b (at the commit iris-wasm's justfile pins) in f32 on the CPU, one
case per run, so that the text encoder and the diffusion transformer are never in memory
together. Qwen3-VL's bf16 layers are upcast to f32 one at a time; the transformer's blocks are
read from their safetensors file (upcast from bf16 for Turbo) just before each runs and dropped
after it, and every other weight is read once.

    python -I dump.py <iris-3b repository> <checkpoint root> <goldens dir> <case>

<checkpoint root> holds `text/` (Qwen/Qwen3-VL-4B-Instruct) and `iris/` (speridlabs/iris-3b with
its `turbo/`, `depth/` and `upscaler/` folders). `base` and `turbo` read the states `text` wrote.
Each case writes <goldens dir>/golden_f32_<case>.safetensors (f32 unless noted):

text:
    prompt            [n] u8          the prompt, UTF-8
    prompt_tokens     [L] i64         Qwen3-VL's input: the template's prefix, the caption, its suffix
    negative_tokens   [L'] i64        the same for the empty negative prompt
    prompt_states     [R, 12, 2560]   the hidden states after layers 2, 5, ..., 35 of the rows from
                                      the caption on (R = caption + suffix tokens)
    negative_states   [R', 12, 2560]
base:
    noise     [3, H, W]          the starting image
    sigmas    [steps + 1]        the solver's levels, the last 0
    timesteps [steps]            the model time of each step (1000 * level)
    guidance, shift [1]
    adapted   [2, 300, 2560]     the text adapter's output plus the learned positions: the
                                 prompt's, then the negative prompt's
    velocity  [2, 3, H, W]       the first step's predictions: the prompt's, then the negative's
    image     [3, H, W]          the sample, clamped to [-1, 1]
turbo:
    noise [3, H, W], sigmas [4], velocity [3, H, W] (at sigma 1), image [3, H, W]
depth:
    input      [h, w, 3] u8      the photo
    resized    [3, h', w']       the model's input in [-1, 1] at its patch-aligned size
    prediction [h', w']          the depth model's output (after its 1x1 reducer)
    depth      [h, w]            the prediction resized to the photo, relative log depth
    colorized  [h, w, 3] u8      colorize(depth)
upscaler:
    input     [h, w, 3] u8       the photo
    budgeted  [h', w', 3] u8     after fit_budget(short_side=SHORT, long_side=LONG)
    reference [3, 4h', 4w']      the bicubic x4 upsample in [0, 1]
    work      [3, H, W]          the tiles' input: resized, in [-1, 1], zero-padded to the patch
    fused     [3, H, W]          the restored tiles fused
    restored  [3, 4h', 4w']      cropped, in [0, 1], resized back: the colour fix's content
    image     [4h', 4w', 3] u8   the result
"""

import os
import sys

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from safetensors import safe_open
from safetensors.torch import load_file, save_file

PROMPT = "A red fox sleeping in fresh snow, golden hour"
SIZE, SEED = 256, 42
BASE_STEPS, GUIDANCE, SHIFT = 3, 3.0, 4.0
DEPTH_PHOTO, MAX_SIDE = "demo/examples/lowres/57.png", 1024
UPSCALE_PHOTO, SHORT, LONG = "demo/examples/lowres/1.png", 192, 512
LAYERS = [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35]

repo, checkpoints, goldens, case = sys.argv[1:5]
sys.path.insert(0, os.path.join(repo, "src"))

from iris3b.config import TextEncoderConfig, inference_config
from iris3b.downstream import depth as depth_module
from iris3b.downstream import restoration
from iris3b.flow.solver import FlowDPMSolver
from iris3b.models.dit import IrisDiT
from iris3b.sampling import generate, generate_few_step
from iris3b.text.base import TextEncoding

IRIS = os.path.join(checkpoints, "iris")
utf8 = lambda value: torch.tensor(list(value.encode()), dtype=torch.uint8)


def config(folder):
    """The exported config of a folder of the checkpoint, its task section apart."""
    raw = OmegaConf.to_container(OmegaConf.load(os.path.join(folder, "config.yaml")))
    task = raw.pop("task", None)
    return inference_config(raw, []), task


def streamed(model, path, prefix=""):
    """`model` (built on the meta device) reading its weights from `path` in f32: every weight
    outside the blocks now, each block of the transformer under `prefix` just before it runs."""
    handle = safe_open(path, framework="pt")
    blocks = (prefix + "blocks.", prefix + "pixel_blocks.")
    rest = {name: handle.get_tensor(name).float() for name in model.state_dict() if not name.startswith(blocks)}
    model.load_state_dict(rest, strict=False, assign=True)
    dit = model.get_submodule(prefix[:-1]) if prefix else model

    def load(module, at):
        weights = {name: handle.get_tensor(at + name).float() for name in module.state_dict()}
        module.load_state_dict(weights, assign=True)

    def drop(module, args, output):
        module.to("meta")

    for stack in ("blocks", "pixel_blocks"):
        for index, block in enumerate(getattr(dit, stack)):
            at = f"{prefix}{stack}.{index}."
            block.register_forward_pre_hook(lambda module, args, at=at: load(module, at))
            block.register_forward_hook(drop)
    return model.eval()


def transformer(folder):
    """The transformer of a text-to-image folder of the checkpoint."""
    cfg, _ = config(folder)
    with torch.device("meta"):
        model = IrisDiT(cfg.model)
    return streamed(model, os.path.join(folder, "model.safetensors")), cfg


def first_output(module):
    """A list that receives `module`'s output of its first call."""
    seen = []
    module.register_forward_hook(lambda module, args, output: seen.append(output) if not seen else None)
    return seen


class GoldenText:
    """Stands in for the text encoder: the states the `text` case wrote, padded to 300 rows."""

    def __init__(self):
        self.states = load_file(os.path.join(goldens, "golden_f32_text.safetensors"))

    def encoding(self, name):
        states = self.states[f"{name}_states"]
        embeddings = torch.zeros(1, 300, len(LAYERS), states.shape[-1])
        embeddings[0, : states.shape[0]] = states
        mask = torch.zeros(1, 300, dtype=torch.int64)
        mask[0, : states.shape[0]] = 1
        return TextEncoding(embeddings=embeddings, mask=mask)

    def encode(self, prompts):
        return self.encoding("prompt")

    def null(self, negative_prompt=""):
        return self.encoding("negative")


def noise():
    generator = torch.Generator("cpu").manual_seed(SEED)
    return torch.randn(1, 3, SIZE, SIZE, generator=generator, dtype=torch.float32)


def text_case():
    from iris3b.text.qwen3_vl import Qwen3VLTextEncoder

    cfg = TextEncoderConfig(pretrained=os.path.join(checkpoints, "text"), hidden_layers=LAYERS)
    encoder = Qwen3VLTextEncoder(cfg, device="cpu")
    decoder = encoder.decoder
    decoder.embed_tokens.register_forward_hook(lambda module, args, output: output.float())
    for layer in decoder.layers:
        layer.register_forward_pre_hook(lambda module, args: module.float() and None)
        layer.register_forward_hook(lambda module, args, output: module.to(torch.bfloat16) and None)
    inputs = []
    decoder.register_forward_pre_hook(lambda module, args, kwargs: inputs.append(kwargs) and None, with_kwargs=True)

    def encode(text):
        encoding = encoder.encode([text])
        rows = int(encoding.mask.sum())
        call = inputs.pop()
        tokens = call["input_ids"][0][call["attention_mask"][0].bool()]
        return tokens.to(torch.int64), encoding.embeddings[0, :rows].float()

    with torch.inference_mode():
        prompt_tokens, prompt_states = encode(PROMPT)
        negative_tokens, negative_states = encode("")
    return {
        "prompt": utf8(PROMPT),
        "prompt_tokens": prompt_tokens,
        "negative_tokens": negative_tokens,
        "prompt_states": prompt_states,
        "negative_states": negative_states,
    }


def base_case():
    model, cfg = transformer(IRIS)
    adapted = first_output(model.y_embedder)
    velocity = first_output(model)
    start = noise()
    with torch.inference_mode():
        image = generate(
            model, GoldenText(), [PROMPT], SIZE, SIZE, steps=BASE_STEPS, cfg_scale=GUIDANCE, shift=SHIFT,
            negative_prompt="", device="cpu", noise=start, num_train_timesteps=cfg.flow.num_train_timesteps,
            prediction=cfg.flow.prediction,
        )
        positions = model.y_pos_embedding[:, :300].float()
    grid = torch.tensor(FlowDPMSolver.time_grid(BASE_STEPS, SHIFT), dtype=torch.float64)
    # The solver's batch is the negative prompt's, then the prompt's.
    negative, prompt = (adapted[0] + positions).unbind(0)
    unconditional, conditional = velocity[0].x.unbind(0)
    return {
        "noise": start[0],
        "sigmas": grid.float(),
        "timesteps": (grid[:-1] * cfg.flow.num_train_timesteps).float(),
        "guidance": torch.tensor([GUIDANCE]),
        "shift": torch.tensor([SHIFT]),
        "adapted": torch.stack([prompt, negative]),
        "velocity": torch.stack([conditional, unconditional]),
        "image": image[0],
    }


def turbo_case():
    folder = os.path.join(IRIS, "turbo")
    model, cfg = transformer(folder)
    sigmas = list(OmegaConf.load(os.path.join(folder, "distill.yaml")).dmd.sigmas)
    velocity = first_output(model)
    start = noise()
    with torch.inference_mode():
        image = generate_few_step(
            model, GoldenText(), [PROMPT], SIZE, SIZE, sigmas=sigmas, device="cpu", noise=start,
            num_train_timesteps=cfg.flow.num_train_timesteps, prediction=cfg.flow.prediction,
        )
    return {
        "noise": start[0],
        "sigmas": torch.tensor(sigmas, dtype=torch.float32),
        "velocity": velocity[0].x[0],
        "image": image[0],
    }


def depth_case():
    folder = os.path.join(IRIS, "depth")
    cfg, _ = config(folder)
    with torch.device("meta"):
        model = depth_module.IrisDepth(cfg.model)
    model.num_train_timesteps = cfg.flow.num_train_timesteps
    prompt = load_file(os.path.join(folder, "empty_prompt.safetensors"))
    # DepthPredictor.__init__ without load_export, which would read every weight at once.
    predictor = depth_module.DepthPredictor.__new__(depth_module.DepthPredictor)
    predictor.device = torch.device("cpu")
    predictor.model = streamed(model, os.path.join(folder, "model.safetensors"), prefix="pixel.")
    predictor.embeddings = prompt["embeddings"].float()
    predictor.mask = prompt["mask"].bool()
    predictor.patch = cfg.model.patch_size
    calls = []
    model.register_forward_hook(lambda module, args, output: calls.append((args[0], output)) and None)
    photo = Image.open(os.path.join(repo, DEPTH_PHOTO))
    with torch.inference_mode():
        depth = predictor(photo, max_side=MAX_SIDE)
    resized, prediction = calls[0]
    return {
        "input": torch.from_numpy(np.array(photo.convert("RGB"))),
        "resized": resized[0],
        "prediction": prediction[0, 0],
        "depth": torch.from_numpy(depth),
        "colorized": torch.from_numpy(np.array(depth_module.colorize(depth))),
    }


def upscaler_case():
    folder = os.path.join(IRIS, "upscaler")
    cfg, task = config(folder)
    with torch.device("meta"):
        model = IrisDiT(cfg.model)
    prompt = load_file(os.path.join(folder, "empty_prompt.safetensors"))
    # Restorer.__init__ without load_export, which would read every weight at once.
    restorer = restoration.Restorer.__new__(restoration.Restorer)
    restorer.device = torch.device("cpu")
    restorer.model = streamed(model, os.path.join(folder, "model.safetensors"))
    restorer.embeddings = prompt["embeddings"].float()
    restorer.mask = prompt["mask"].bool()
    restorer.patch = cfg.model.patch_size
    restorer.tile = task["tile"]
    restorer.sigma = task["sigma"]
    restorer.time = task["sigma"] * cfg.flow.num_train_timesteps
    seen = {}
    tiled, color_fix = restoration.tiled, restoration.wavelet_color_fix

    def record_tiles(fn, x, tile, stride):
        seen["work"] = x[0].clone()
        seen["fused"] = tiled(fn, x, tile, stride)[0].clone()
        return seen["fused"][None]

    def record_color_fix(content, reference):
        seen["restored"], seen["reference"] = content[0].clone(), reference[0].clone()
        return color_fix(content, reference)

    restoration.tiled, restoration.wavelet_color_fix = record_tiles, record_color_fix
    photo = Image.open(os.path.join(repo, UPSCALE_PHOTO))
    budgeted = restoration.fit_budget(photo, short_side=SHORT, long_side=LONG)
    with torch.inference_mode():
        image = restorer(budgeted)
    return {
        "input": torch.from_numpy(np.array(photo.convert("RGB"))),
        "budgeted": torch.from_numpy(np.array(budgeted.convert("RGB"))),
        **seen,
        "image": torch.from_numpy(np.array(image)),
    }


CASES = {"text": text_case, "base": base_case, "turbo": turbo_case, "depth": depth_case, "upscaler": upscaler_case}
tensors = CASES[case]()
out = os.path.join(goldens, f"golden_f32_{case}.safetensors")
save_file({name: tensor.clone().contiguous() for name, tensor in tensors.items()}, out)
print(f"{case}: {', '.join(f'{name} {list(tensor.shape)}' for name, tensor in tensors.items())} -> {out}")
