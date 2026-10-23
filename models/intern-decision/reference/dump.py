"""The reference for `intern-decision parity` and the agreement checks: the checkpoint's own
`inference.py` (`DecisionEngine`, loaded from the checkpoint directory) on transformers at a
pinned version, in f32 on the CPU, with its torchvision image processor.

    python dump.py <checkpoint dir> <requests.json> <images dir> <goldens dir> <references.json>

`requests.json` maps each case to a request (`{"state", "questions", "images"?}`, the images'
paths relative to `<images dir>`). For every case, its response into `references.json` as
`{"<case>": response}`, and the tensors of every stage into
`<goldens dir>/golden_f32_<case>.safetensors` (f32 unless noted). The reference orders an
image's patches by 2×2 block; the tensors here are row-major over the patch grid, as the
runtime orders them. A case reads at most one image; one without has no image tensors.

    request     [B] u8       the request's JSON (UTF-8), its images' paths included
    image       [H,W,3] u8   the image as RGB
    patches     [N,768]      the image's 16×16 patches, channel-major, normalized to [-1, 1]
                             (one of the two identical frames the processor repeats it in)
    grid        [3] i64      the patch grid: frames, rows, columns
    embedded    [N,1024]     patch embeddings plus the interpolated position embeddings
    encoded     [N,1024]     the vision encoder's last layer's output
    merged      [N/4,2048]   the image's tokens, row-major over its 2×2 blocks
    ids         [L] i64      the prompt's token ids, the image's tokens its <|image_pad|>s
    hidden      [F,2048]     the language model's final-normed state of each question's
                             deciding row (just before its <decision>), in question order
    logits      [S]          each question's logits over its options' symbols, concatenated
    probabilities [S]        each question's calibrated probabilities, concatenated
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors.torch import save_file

CHANNELS = 3
PATCH = 16
MERGE = 2


def load_engine_module(checkpoint):
    spec = importlib.util.spec_from_file_location("inference", checkpoint / "inference.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def row_major(blocks, rows, columns):
    """Rows in the processor's 2×2-block order, reordered row-major over the patch grid."""
    order = []
    for row in range(rows):
        for column in range(columns):
            block = (row // MERGE) * (columns // MERGE) + column // MERGE
            order.append(block * MERGE * MERGE + (row % MERGE) * MERGE + column % MERGE)
    return blocks[torch.tensor(order)]


class Recorder:
    """Forward hooks on the stages' modules, keeping their last inputs and outputs."""

    def __init__(self, model):
        self.seen = {}
        visual = model.model.visual
        self.hook(visual, "pixels", pixels_and_grid)
        self.hook(visual.blocks[0], "embedded", lambda args, kwargs, out: args[0])
        self.hook(visual.blocks[-1], "encoded", lambda args, kwargs, out: out)
        self.hook(visual.merger, "merged", lambda args, kwargs, out: out)
        self.hook(model.model.language_model.norm, "normed", lambda args, kwargs, out: out)
        self.hook(model, "ids", lambda args, kwargs, out: kwargs["input_ids"])

    def hook(self, module, name, keep):
        def record(module, args, kwargs, out):
            self.seen[name] = keep(args, kwargs, out)

        module.register_forward_hook(record, with_kwargs=True)


def pixels_and_grid(args, kwargs, out):
    """The vision tower's input: its patches and their grid, by position or by name."""
    grid = kwargs["grid_thw"] if "grid_thw" in kwargs else args[1]
    return args[0], grid


def image_tensors(seen, image_path):
    pixels, grid = seen["pixels"]
    _, rows, columns = (int(value) for value in grid[0])
    first_frame = pixels.reshape(-1, CHANNELS, 2, PATCH * PATCH)[:, :, 0]
    frames = first_frame.reshape(-1, CHANNELS * PATCH * PATCH)
    with Image.open(image_path) as image:
        rgb = torch.from_numpy(np.array(image.convert("RGB")))
    return {
        "image": rgb,
        "patches": row_major(frames, rows, columns),
        "grid": grid[0].to(torch.int64),
        "embedded": row_major(seen["embedded"], rows, columns),
        "encoded": row_major(seen["encoded"], rows, columns),
        "merged": seen["merged"],
    }


def main():
    checkpoint, requests, images, goldens, references = (Path(arg) for arg in sys.argv[1:6])
    torch.set_grad_enabled(False)
    module = load_engine_module(checkpoint)
    engine = module.DecisionEngine(checkpoint, device="cpu", dtype="float32", media_root=images)
    model, tokenizer = engine.backend.model, engine.tokenizer
    recorder = Recorder(model)
    goldens.mkdir(parents=True, exist_ok=True)
    responses = {}
    for case, request in json.loads(requests.read_text()).items():
        response = engine.predict(request)
        response.pop("timing")
        responses[case] = response
        seen = recorder.seen
        ids = seen["ids"][0]
        decisions = (ids == tokenizer.convert_tokens_to_ids(module.DECISION_TOKEN)).nonzero().flatten() - 1
        symbols = [
            tokenizer.convert_tokens_to_ids(list(module.ANSWER_SYMBOLS[: len(answer["probabilities"])]))
            for answer in response["answers"].values()
        ]
        normed = seen["normed"][0][decisions]
        head = model.lm_head.weight
        logits = torch.cat([normed[field] @ head[options].T for field, options in enumerate(symbols)])
        probabilities = torch.tensor(
            [p for answer in response["answers"].values() for p in answer["probabilities"].values()],
            dtype=torch.float32,
        )
        tensors = {
            "request": torch.frombuffer(bytearray(json.dumps(request).encode()), dtype=torch.uint8),
            "ids": ids.to(torch.int64),
            "hidden": normed,
            "logits": logits,
            "probabilities": probabilities,
        }
        if request.get("images"):
            tensors.update(image_tensors(seen, images / request["images"][0]))
        tensors = {name: tensor.detach().contiguous() for name, tensor in tensors.items()}
        save_file(tensors, goldens / f"golden_f32_{case}.safetensors")
        print(case, json.dumps({field: answer["decision"] for field, answer in response["answers"].items()}))
    references.write_text(json.dumps(responses, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
