"""The reference for `d1 parity` and the product's agreement checks: the checkpoint's own code
(`D1Model`, `trust_remote_code`, as its `config.json` names it) under transformers 5.19.0, in
f32 on the CPU, its `system_one` answering each case.

    python dump.py <checkpoint dir> <requests.json> <media dir> <goldens dir> <references.json>

`requests.json` maps each case to the runtime's request (`state`, `questions`) with its images'
paths (`images`) and its clips' (`clips`, 16-bit mono WAV files at 16 kHz, d1-Omni-2B only),
relative to `<media dir>`; a request's images head its state, then its clips. Each case's
answers go into `references.json` as `{"<case>": answers}`, and its tensors into
`<goldens dir>/golden_f32_<case>.safetensors` (f32 unless noted). An image's patches are
row-major over its patch grid, as the runtime orders them (the reference orders them by 2×2
block).

    request            [B] u8       the case's request JSON (UTF-8): its state and questions
    answers            [B] u8       `system_one`'s answers, JSON
    image.{i}          [H,W,3] u8   image `i` as RGB, before the checkpoint's pixel cap
    item.{i}.grid      [2] i64      its patch grid: rows, columns
    item.{i}.patches   [N,1536]     its two-frame 16×16 patches, channel-major then frame-major,
                                    normalized to [-1, 1]
    item.{i}.embedded  [N,1024]     the patch embeddings, RMS-normed
    item.{i}.encoded   [N,1024]     the vision tower's last layer's output
    item.{i}.tokens    [N/4,D]      its tokens, the projector's output, row-major over its 2×2
                                    blocks
    clip.{i}.samples   [S]          clip `i`'s samples
    clip.{i}.features  [F,128]      its log-mel features, frame-major
    clip.{i}.tokens    [⌈F/8⌉,D]    its tokens, the audio projector's output
    pack.{k}.ids       [L] i64      pack `k`'s ids: the state, then up to `d1_set_size` questions
    pack.{k}.hidden    [L,D]        every row's final-normed state
"""

import importlib
import json
import sys
import wave
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors.torch import save_file
from transformers import AutoModel

MERGE = 2
# Mel frames per audio token.
STRIDE = 8
# 2^15: 16-bit samples decode to [-1, 1).
PCM_SCALE = 32768.0


def row_major(blocks, rows, columns):
    """Rows in the processor's 2×2-block order, reordered row-major over the patch grid."""
    order = []
    for row in range(rows):
        for column in range(columns):
            block = (row // MERGE) * (columns // MERGE) + column // MERGE
            order.append(block * MERGE * MERGE + (row % MERGE) * MERGE + column % MERGE)
    return blocks[torch.tensor(order)]


class Recorder:
    """The vision tower's and the audio encoder's inputs and outputs, call by call."""

    def __init__(self, model):
        self.towers, self.encoders = [], []
        body = model.model
        tower = body.vision_tower
        # Ahead of the checkpoint's own hook (`lfm3.py`), which turns `grid_thw` into frame sizes.
        tower.register_forward_pre_hook(self.tower_input, with_kwargs=True, prepend=True)
        self.keep(self.towers, tower.embedding_norm, "embedded", lambda out: out)
        self.keep(self.towers, tower, "encoded", lambda out: out.last_hidden_state)
        self.keep(self.towers, body.multi_modal_projector, "tokens", lambda out: out)
        if hasattr(body, "audio_encoder"):
            body.audio_encoder.register_forward_pre_hook(self.encoder_input)
            self.keep(self.encoders, body.audio_projector, "tokens", lambda out: out)

    def keep(self, calls, module, name, value):
        module.register_forward_hook(lambda module, args, out: calls[-1].__setitem__(name, value(out)))

    def tower_input(self, module, args, kwargs):
        self.towers.append({"pixels": args[0], "grids": kwargs["grid_thw"]})

    def encoder_input(self, module, args):
        self.encoders.append({"features": args[0], "lengths": args[1]})


def items(call):
    """Each image of a vision tower call: its tensors."""
    found, start, token = [], 0, 0
    for _, rows, columns in call["grids"].tolist():
        patches, merged = rows * columns, rows * columns // (MERGE * MERGE)
        span, tokens = slice(start, start + patches), slice(token, token + merged)
        found.append({
            "grid": torch.tensor([rows, columns], dtype=torch.int64),
            "patches": row_major(call["pixels"][span], rows, columns),
            "embedded": row_major(call["embedded"][span], rows, columns),
            "encoded": row_major(call["encoded"][span], rows, columns),
            "tokens": call["tokens"][tokens],
        })
        start, token = start + patches, token + merged
    return found


def clips(call):
    """Each clip of an audio encoder call: its tensors."""
    found, token = [], 0
    for index, frames in enumerate(call["lengths"].tolist()):
        tokens = -(frames // -STRIDE)
        found.append({
            "features": call["features"][index, :, :frames].T,
            "tokens": call["tokens"][token:token + tokens],
        })
        token += tokens
    return found


def read_clip(path):
    """A 16-bit mono WAV file's samples."""
    with wave.open(str(path)) as clip:
        assert clip.getsampwidth() == 2 and clip.getnchannels() == 1, path
        frames = clip.readframes(clip.getnframes())
    return np.frombuffer(frames, dtype="<i2").astype(np.float32) / PCM_SCALE


def packs(model, engine, request, media):
    """Each pack's ids and every row's final-normed state, read as `system_one` reads it."""
    code = type(model).__module__.rsplit(".", 1)[0]
    attention = importlib.import_module(f"{code}.attention")
    media_list = importlib.import_module(f"{code}.media").media_list
    questions = [engine.ask(question) for question in request["questions"].values()]
    sequences = engine._sequences([(request.get("state"), questions, media_list(media))])
    found = []
    for _, _, ids, state, question, _, inputs in sequences:
        rows = torch.tensor([ids])
        with attention.state_attention(model, torch.tensor([state]), torch.tensor([question])):
            out = attention.text_backbone(model)(
                input_ids=rows,
                attention_mask=torch.ones_like(rows),
                use_cache=False,
                **engine._on_device(inputs),
            )
        found.append({"ids": rows[0], "hidden": out.last_hidden_state[0]})
    return found


def main(checkpoint, requests_path, media_dir, goldens_dir, references_path):
    torch.set_grad_enabled(False)
    model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True, dtype=torch.float32).eval()
    engine = model.engine
    recorder = Recorder(model)
    requests = json.loads(Path(requests_path).read_text())
    Path(goldens_dir).mkdir(parents=True, exist_ok=True)
    references = {}
    for case, request in requests.items():
        images = [Image.open(Path(media_dir) / path).convert("RGB") for path in request.get("images", [])]
        clip_paths = [Path(media_dir) / path for path in request.get("clips", [])]
        media = images + [{"type": "audio", "path": str(path)} for path in clip_paths]
        asked = {"state": request.get("state"), "questions": request["questions"]}
        answers = model.system_one(asked["state"], asked["questions"], media=media)["answers"]
        references[case] = answers
        recorder.towers.clear()
        recorder.encoders.clear()
        read = packs(model, engine, asked, media)
        tensors = {
            "request": torch.frombuffer(bytearray(json.dumps(asked).encode()), dtype=torch.uint8),
            "answers": torch.frombuffer(bytearray(json.dumps(answers).encode()), dtype=torch.uint8),
        }
        for index, image in enumerate(images):
            tensors[f"image.{index}"] = torch.from_numpy(np.asarray(image).copy())
        first_tower = items(recorder.towers[0]) if recorder.towers else []
        for index, item in enumerate(first_tower):
            tensors |= {f"item.{index}.{name}": value for name, value in item.items()}
        first_encoder = clips(recorder.encoders[0]) if recorder.encoders else []
        for index, (path, clip) in enumerate(zip(clip_paths, first_encoder)):
            tensors[f"clip.{index}.samples"] = torch.from_numpy(read_clip(path))
            tensors |= {f"clip.{index}.{name}": value for name, value in clip.items()}
        for index, pack in enumerate(read):
            tensors |= {f"pack.{index}.{name}": value for name, value in pack.items()}
        tensors = {name: value.contiguous() for name, value in tensors.items()}
        save_file(tensors, str(Path(goldens_dir) / f"golden_f32_{case}.safetensors"))
    Path(references_path).parent.mkdir(parents=True, exist_ok=True)
    Path(references_path).write_text(json.dumps(references, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main(*sys.argv[1:])
