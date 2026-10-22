"""The reference for `paddle-ocr parity` and the agreement checks: PaddleOCR-VL-1.6 in transformers
at a pinned version, in f32 on the CPU, with its Pillow image processor. The learned position grid
is resampled to each image with `align_corners=False`, as PaddlePaddle's own code does it (the
checkpoint's `modeling_paddleocr_vl.py`, PaddleX and vLLM); transformers' port uses `True`.

    python dump.py <checkpoint dir> <images dir> <tasks.json> <goldens dir> <references.json>

For every image of `tasks.json`, the text greedy decoding reads off it with its task's prompt, into
`references.json` as `{"<image>": {"task", "text", "tokens"}}`. For the first image, the tensors
of every stage, into `<goldens dir>/golden_f32_<image>.safetensors` (f32 unless noted):

    image       [H,W,3] u8   the image as RGB (an RGBA image's alpha dropped)
    patches     [N,588]      the image's 14×14 patches, channel-major, normalized to [-1, 1]
    grid        [3] i64      the patch grid: frames, rows, columns
    embedded    [N,1152]     patch embeddings plus the interpolated position embeddings
    vision      [N,1152]     the vision encoder's output after its final LayerNorm
    projected   [N/4,1024]   the projector's output: one row per 2×2 patches
    ids         [L] i64      the prompt's token ids
    positions   [3,L] i64    the prompt's temporal, height and width rotary positions
    hidden      [1024]       the language model's final-normed hidden state of the last prompt row
    logits      [V]          and its logits
    tokens      [S] i64      the greedy decoding's tokens, the end token excluded
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors.torch import save_file
from transformers import AutoImageProcessor, AutoModelForImageTextToText, AutoProcessor
from transformers.image_transforms import convert_to_rgb

PROMPTS = {
    "ocr": "OCR:",
    "table": "Table Recognition:",
    "formula": "Formula Recognition:",
    "chart": "Chart Recognition:",
    "seal": "Seal Recognition:",
}
MAX_NEW_TOKENS = 2048
# The model card's limit for every task but spotting: 1280 merged tokens of 28×28 pixels.
MAX_PIXELS = 1280 * 28 * 28

checkpoint, images, tasks, goldens, references = sys.argv[1:6]
tasks = json.loads(Path(tasks).read_text())
torch.manual_seed(0)
model = AutoModelForImageTextToText.from_pretrained(checkpoint, dtype=torch.float32, attn_implementation="eager").eval()
model.model.visual.vision_model.embeddings.interpolation_align_corners = False
processor = AutoProcessor.from_pretrained(checkpoint)
processor.image_processor = AutoImageProcessor.from_pretrained(checkpoint, backend="pil")
image_processor = processor.image_processor


def inputs_of(image, task):
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": PROMPTS[task]}]}]
    return processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        images_kwargs={"size": {"shortest_edge": image_processor.size.shortest_edge, "longest_edge": MAX_PIXELS}},
    )


def captured(module, store, name):
    def keep(_module, _args, output):
        store.setdefault(name, output)

    return module.register_forward_hook(keep)


def captured_positions(module, store):
    def keep(_module, _args, kwargs):
        store.setdefault("positions", kwargs["position_ids"])

    return module.register_forward_pre_hook(keep, with_kwargs=True)


def generate(inputs):
    with torch.inference_mode():
        # The checkpoint's generation config turns the KV cache off, which only slows greedy
        # decoding down: every token would run the vision encoder and the prompt again.
        output = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, use_cache=True)
    tokens = output[0, inputs["input_ids"].shape[-1] :]
    if tokens[-1].item() == model.generation_config.eos_token_id:
        tokens = tokens[:-1]
    return tokens


def stages(inputs):
    store = {}
    parts = model.model
    hooks = [
        captured(parts.visual.vision_model.embeddings, store, "embedded"),
        captured(parts.visual.vision_model.post_layernorm, store, "vision"),
        captured(parts.projector, store, "projected"),
        captured(parts.language_model.norm, store, "hidden"),
        captured_positions(parts.language_model, store),
    ]
    with torch.inference_mode():
        logits = model(**inputs).logits[0, -1]
    for hook in hooks:
        hook.remove()
    positions = store["positions"][-3:, 0]
    return {
        "patches": inputs["pixel_values"].reshape(inputs["pixel_values"].shape[0], -1),
        "grid": inputs["image_grid_thw"][0].to(torch.int64),
        "embedded": store["embedded"].reshape(-1, store["embedded"].shape[-1]),
        "vision": store["vision"].reshape(-1, store["vision"].shape[-1]),
        "projected": store["projected"],
        "ids": inputs["input_ids"][0].to(torch.int64),
        "positions": positions.to(torch.int64),
        "hidden": store["hidden"][0, -1],
        "logits": logits,
    }


read = {}
for index, (name, task) in enumerate(tasks.items()):
    path = next(Path(images).glob(f"{name}.*"))
    inputs = inputs_of(Image.open(path), task)
    tokens = generate(inputs)
    text = processor.decode(tokens)
    read[name] = {"task": task, "text": text, "tokens": tokens.tolist()}
    print(f"{name} ({task}, grid {inputs['image_grid_thw'][0].tolist()}): {text!r}", flush=True)
    if index == 0:
        tensors = {key: value.contiguous().float() if value.is_floating_point() else value.contiguous() for key, value in stages(inputs).items()}
        tensors["tokens"] = tokens.to(torch.int64).contiguous()
        tensors["image"] = torch.from_numpy(np.array(convert_to_rgb(Image.open(path))))
        Path(goldens).mkdir(parents=True, exist_ok=True)
        save_file(tensors, str(Path(goldens) / f"golden_f32_{name}.safetensors"))
Path(references).write_text(json.dumps(read, ensure_ascii=False, indent=2) + "\n")
