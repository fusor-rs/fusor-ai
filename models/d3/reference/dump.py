"""The reference for `d3 parity` and the agreement checks: the checkpoint's own runtime
(`d3_runtime.D3`, loaded from the checkpoint directory) on transformers at a pinned version, in
f32 on the CPU, one question per forward pass, on its plain path (no fused layers).

    python dump.py <checkpoint dir> <requests.json> <media dir> <goldens dir> <references.json>

`requests.json` maps each case to a request (`{"state", "questions", "images"?, "videos"?}`, the
media's paths relative to `<media dir>`). For every case, its response into `references.json`
as `{"<case>": response}`, and the tensors of every stage into
`<goldens dir>/golden_f32_<case>.safetensors` (f32 unless noted). Item `i` is the request's
`i`th image, then each video's frame groups in order; its patches are row-major over its patch
grid, as the runtime orders them (the reference orders them by 2×2 block). Question `q` is the
`q`th question the request answers.

    request          [B] u8       the request's JSON (UTF-8), its media's paths included
    image.{i}        [H,W,3] u8   image `i` as RGB
    video.{v}.frames [F,H,W,3] u8 video `v`'s sampled frames as RGB (an odd last one repeated)
    video.{v}.times  [F] f64      each frame's time in seconds: its index over the video's rate
    item.{i}.grid    [2] i64      the item's patch grid: rows, columns
    item.{i}.patches [N,1536]     its two-frame 16×16 patches, channel-major then frame-major,
                                  normalized to [-1, 1]
    item.{i}.embedded [N,W]       patch embeddings plus the interpolated position embeddings
    item.{i}.encoded [N,W]        the vision encoder's last layer's output
    item.{i}.merged  [N/4,D]      the item's tokens, row-major over its 2×2 blocks
    question.{q}.ids [L] i64      the question's prompt's token ids, the media's tokens their pads
    question.{q}.hidden [D]       the language model's final-normed state of the last row
    question.{q}.probabilities [O] the question's options' probabilities
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors.torch import save_file

MERGE = 2


def row_major(blocks, rows, columns):
    """Rows in the processor's 2×2-block order, reordered row-major over the patch grid."""
    order = []
    for row in range(rows):
        for column in range(columns):
            block = (row // MERGE) * (columns // MERGE) + column // MERGE
            order.append(block * MERGE * MERGE + (row % MERGE) * MERGE + column % MERGE)
    return blocks[torch.tensor(order)]


class Recorder:
    """Forward hooks on the stages' modules, keeping every vision tower call of a question and
    the prompts' ids and final-normed states."""

    def __init__(self, backbone):
        self.towers = []
        self.prompts = []
        visual = backbone.visual
        visual.register_forward_pre_hook(self.tower_input, with_kwargs=True)
        self.keep(visual.blocks[0], lambda args, out: {"embedded": args[0]})
        self.keep(visual.blocks[-1], lambda args, out: {"encoded": out})
        self.keep(visual.merger, lambda args, out: {"merged": out})
        language = backbone.language_model
        language.embed_tokens.register_forward_hook(
            lambda module, args, out: self.prompts.append({"ids": args[0][0]})
        )
        language.norm.register_forward_hook(
            lambda module, args, out: self.prompts[-1].update(hidden=out[0, -1])
        )

    def keep(self, module, kept):
        module.register_forward_hook(
            lambda module, args, out: self.towers[-1].update(kept(args, out))
        )

    def tower_input(self, module, args, kwargs):
        grid = kwargs["grid_thw"] if "grid_thw" in kwargs else args[1]
        self.towers.append({"pixels": args[0], "grid": grid})


def items(towers):
    """Each image and video group's tensors, in prompt order, from the vision tower's calls."""
    out = []
    for call in towers:
        start = merged = 0
        for t, rows, columns in (tuple(int(v) for v in grid) for grid in call["grid"]):
            for _ in range(t):
                patches, tokens = rows * columns, rows * columns // (MERGE * MERGE)
                span = slice(start, start + patches)
                out.append({
                    "grid": torch.tensor([rows, columns], dtype=torch.int64),
                    "patches": row_major(call["pixels"][span], rows, columns),
                    "embedded": row_major(call["embedded"][span], rows, columns),
                    "encoded": row_major(call["encoded"][span], rows, columns),
                    "merged": call["merged"][merged:merged + tokens],
                })
                start, merged = start + patches, merged + tokens
    return out


def main():
    checkpoint, requests, media, goldens, references = (Path(arg) for arg in sys.argv[1:6])
    torch.set_grad_enabled(False)
    sys.path.insert(0, str(checkpoint))
    from d3_runtime import D3, load_video

    runtime = D3(checkpoint, device="cpu", batch_size=1)
    runtime.fast = None
    runtime.backbone.float()
    runtime.readout = runtime.readout.float()
    recorder = Recorder(runtime.backbone)
    goldens.mkdir(parents=True, exist_ok=True)
    responses = {}
    for case, request in json.loads(requests.read_text()).items():
        images = [str(media / path) for path in request.get("images", [])]
        videos = [load_video(str(media / path)) for path in request.get("videos", [])]
        recorder.towers.clear()
        recorder.prompts.clear()
        response = runtime.system_one(
            state=request.get("state"),
            questions=request["questions"],
            images=images or None,
            videos=videos or None,
        )
        responses[case] = response
        tensors = {"request": torch.frombuffer(bytearray(json.dumps(request).encode()), dtype=torch.uint8)}
        for index, path in enumerate(images):
            with Image.open(path) as image:
                tensors[f"image.{index}"] = torch.from_numpy(np.array(image.convert("RGB")))
        for index, video in enumerate(videos):
            times = [i / video.fps for i in video.indices]
            tensors[f"video.{index}.frames"] = torch.from_numpy(np.ascontiguousarray(video.frames))
            tensors[f"video.{index}.times"] = torch.tensor(times, dtype=torch.float64)
        # A video request runs the tower once (its images, then its videos); an image request
        # runs it with every question, the same each time.
        towers = recorder.towers if videos else recorder.towers[:1]
        for index, item in enumerate(items(towers)):
            for name, tensor in item.items():
                tensors[f"item.{index}.{name}"] = tensor
        answered = [answer for answer in response["answers"].values() if "error" not in answer]
        for index, (prompt, answer) in enumerate(zip(recorder.prompts, answered)):
            options = len(answer["probabilities"]) if "probabilities" in answer else 2
            logits = prompt["hidden"] @ runtime.readout[:options].T / runtime.temperature
            tensors[f"question.{index}.ids"] = prompt["ids"].to(torch.int64)
            tensors[f"question.{index}.hidden"] = prompt["hidden"]
            tensors[f"question.{index}.probabilities"] = torch.softmax(logits, -1)
        tensors = {name: tensor.detach().contiguous() for name, tensor in tensors.items()}
        save_file(tensors, goldens / f"golden_f32_{case}.safetensors")
        decided = {name: answer.get("choice", answer.get("noul", answer.get("score"))) for name, answer in response["answers"].items()}
        print(case, json.dumps(decided))
    references.write_text(json.dumps(responses, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
