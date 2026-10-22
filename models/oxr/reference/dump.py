"""The reference for `oxr parity` and the agreement checks: OXR-1.0's own modeling code (loaded from
the checkpoint directory) in transformers at a pinned version, in f32 on the CPU, with its Pillow
image processor and the pixel budget SoMark serves the model with. The image's indicators are the
visual vocabulary's last words, as SoMark's vLLM plugin embeds them: the modeling code keeps their
indices in a non-persistent buffer that transformers leaves at zeros when it loads the model, which
would embed both as the vocabulary's first word.

    python dump.py <checkpoint dir> <images dir> <tasks.json> <goldens dir> <references.json>

For every image of `tasks.json`, the text greedy decoding reads off it with its task's prompt, into
`references.json` as `{"<image>": {"task", "text", "tokens"}}`. For the first image, the tensors
of every stage, into `<goldens dir>/golden_f32_<image>.safetensors` (f32 unless noted). The
reference orders an image's patches by 2×2 block; the tensors here are row-major over the patch
grid, as the runtime orders them:

    image       [H,W,3] u8   the image as RGB (an RGBA image's alpha dropped)
    patches     [N,768]      the image's 16×16 patches, channel-major, normalized to [-1, 1]
    grid        [3] i64      the patch grid: frames, rows, columns
    embedded    [N,1152]     patch embeddings plus the interpolated position embeddings
    encoded     [N,1152]     the vision encoder's last layer's output (its final norm unused)
    visual      [N/4,1024]   the image's tokens: each 2×2 block's mixture of visual embeddings
    ids         [L] i64      the prompt's token ids, the image's rows its negative placeholders
    hidden      [1024]       the language model's final-normed hidden state of the last prompt row
    logits      [V]          and its logits
    answer      [S] i64      the greedy decoding's tokens, the end token excluded
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors.torch import save_file

PROMPTS = {
    "page": "Recognize the whole page content of the image.",
    "text": "Recognize the content of the image.",
    "table": "Recognize the table of the image into html format.",
    "formula": "Recognize the formula of the image into KaTex format.",
    "code": "Recognize the code of the image into markdown code format.",
    "chemistry": "Recognize the chemical structure of the image into SMILES.",
}
# SoMark's vLLM processor's budget (plugins/vllm/oxr_vllm/oxr_processor.py).
MIN_PIXELS = 200704
MAX_PIXELS = 921600
MAX_NEW_TOKENS = 4096
ENDS = [151643, 151645]

checkpoint, images, tasks, goldens, references = sys.argv[1:6]
tasks = json.loads(Path(tasks).read_text())
torch.manual_seed(0)
# The checkpoint's modeling code as a package of its own: loading it through `trust_remote_code`
# imports its configuration twice, and the vision tower's AutoModel lookup then fails.
spec = importlib.util.spec_from_file_location("oxr_reference", f"{checkpoint}/modeling_oxr.py", submodule_search_locations=[checkpoint])
reference = importlib.util.module_from_spec(spec)
sys.modules["oxr_reference"] = reference
spec.loader.exec_module(reference)
model = reference.OXR.from_pretrained(checkpoint, dtype=torch.float32).eval()
vocabulary = model.config.visual_vocab_size
model.indicator_token_indices = torch.arange(vocabulary - len(reference.INDICATOR_IDS), vocabulary)
vit = model.visual_tokenizer.vit
stride = vit.config.hidden_stride


def inputs_of(image, task):
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": PROMPTS[task]}]}]
    return model.preprocess_inputs(messages, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)


def generate(ids, pixels, grid):
    with torch.inference_mode():
        output = model.generate(ids, pixel_values=pixels, grid_thws=grid, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, eos_token_id=ENDS, pad_token_id=ENDS[0], use_cache=True)
    tokens = output[0]
    if tokens[-1].item() in ENDS:
        tokens = tokens[:-1]
    return tokens


def row_major(grouped, grid):
    """Rows ordered by 2×2 block (the reference's order) reordered row-major over the grid."""
    _, rows, columns = grid[0].tolist()
    blocks = grouped.reshape(rows // stride, columns // stride, stride, stride, -1)
    return blocks.permute(0, 2, 1, 3, 4).reshape(rows * columns, -1)


def stages(ids, pixels, grid):
    store = {}
    hook = model.llm.model.norm.register_forward_hook(lambda _module, _args, output: store.setdefault("hidden", output))
    with torch.inference_mode():
        embedded = vit.vision_model.embeddings(pixels, grid)
        encoded = vit(pixels, grid, output_hidden_states=True, return_dict=True).hidden_states[-1]
        visual = model.vte(model.visual_tokenizer(pixels, grid))
        logits = model(ids, torch.ones_like(ids), pixels, grid).logits[0, -1]
    hook.remove()
    return {
        "patches": row_major(pixels, grid),
        "grid": grid[0].to(torch.int64),
        "embedded": row_major(embedded, grid),
        "encoded": row_major(encoded, grid),
        "visual": visual,
        "ids": ids[0].to(torch.int64),
        "hidden": store["hidden"][0, -1],
        "logits": logits,
    }


read = {}
for index, (name, task) in enumerate(tasks.items()):
    path = next(Path(images).glob(f"{name}.*"))
    ids, pixels, grid = inputs_of(Image.open(path), task)
    tokens = generate(ids, pixels, grid)
    text = model.text_tokenizer.decode(tokens)
    read[name] = {"task": task, "text": text, "tokens": tokens.tolist()}
    print(f"{name} ({task}, grid {grid[0].tolist()}): {text!r}", flush=True)
    if index == 0:
        tensors = {key: value.contiguous().float() if value.is_floating_point() else value.contiguous() for key, value in stages(ids, pixels, grid).items()}
        tensors["answer"] = tokens.to(torch.int64).contiguous()
        tensors["image"] = torch.from_numpy(np.array(Image.open(path).convert("RGB")))
        Path(goldens).mkdir(parents=True, exist_ok=True)
        save_file(tensors, str(Path(goldens) / f"golden_f32_{name}.safetensors"))
Path(references).write_text(json.dumps(read, ensure_ascii=False, indent=2) + "\n")
