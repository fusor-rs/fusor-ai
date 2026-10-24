"""The reference for `qwen3-5 parity` and the products' agreement checks: transformers'
`Qwen3_5ForConditionalGeneration` and the checkpoint's processor at a pinned version, in f32 on the
CPU, each conversation turn by turn as a chat client sends it (the whole conversation again each
turn, its earlier replies as generated), each reply greedy.

    python dump.py <checkpoint dir> <conversations.json> <media dir> <goldens dir> <references.json>

`conversations.json` maps each case to `{"turns": [...]}`, each turn the runtime's message
(`text`, `thinking`, `max_tokens`) with its media's paths relative to `<media dir>` (`images`,
`videos`); a video is read as the frames `frames.py` samples beside it (`<video>.frames/`). The
processor reads images at 65,536 to 1,638,400 pixels and videos within 6,422,528 pixels, each frame
capped, as the runtime does. For every case, each turn's reply text into `references.json` as
`{"<case>": ["<reply>", ...]}`, and the tensors of every stage into
`<goldens dir>/golden_f32_<case>.safetensors` (f32 unless noted). Item `i` is the conversation's
`i`th image, then each video's frame groups in order, turn after turn; its patches are row-major
over its patch grid, as the runtime orders them (the reference orders them by 2×2 block).

    conversation     [B] u8       the case's JSON (UTF-8), its media's paths included
    image.{i}        [H,W,3] u8   image `i` as RGB
    video.{v}.frames [F,H,W,3] u8 video `v`'s sampled frames as RGB
    video.{v}.times  [F] f64      each frame's time in seconds: its index over the video's rate
    item.{i}.grid    [2] i64      the item's patch grid: rows, columns
    item.{i}.patches [N,1536]     its two-frame 16×16 patches, channel-major then frame-major,
                                  normalized to [-1, 1]
    item.{i}.embedded [N,W]       patch embeddings plus the interpolated position embeddings
    item.{i}.encoded [N,W]        the vision encoder's last layer's output
    item.{i}.merged  [N/4,D]      the item's tokens, row-major over its 2×2 blocks
    turn.{t}.ids     [L] i64      turn `t`'s prompt ids, the whole conversation's, the media's
                                  tokens their pads
    turn.{t}.reply   [A] i64      its greedy reply, its end token included when it ended
    turn.{t}.hidden  [A,D]        the language model's final-normed state of the prompt's last
                                  row, then of each reply token's row but the last's
    turn.{t}.logits  [V]          the logits of the prompt's last row
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors.torch import save_file
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

MERGE = 2
IMAGE_SIZE = {"shortest_edge": 65_536, "longest_edge": 1_638_400}
VIDEO_SIZE = {"shortest_edge": 4_096, "longest_edge": 6_422_528}
# <|im_end|>, <|endoftext|>
ENDS = [248_046, 248_044]


def row_major(blocks, rows, columns):
    """Rows in the processor's 2×2-block order, reordered row-major over the patch grid."""
    order = []
    for row in range(rows):
        for column in range(columns):
            block = (row // MERGE) * (columns // MERGE) + column // MERGE
            order.append(block * MERGE * MERGE + (row % MERGE) * MERGE + column % MERGE)
    return blocks[torch.tensor(order)]


class Recorder:
    """Forward hooks on the stages' modules, keeping the vision tower's calls and the final norm's
    output of every forward pass of a generation."""

    def __init__(self, model):
        self.towers = []
        self.states = []
        visual = model.model.visual
        visual.register_forward_pre_hook(self.tower_input, with_kwargs=True)
        self.keep(visual.blocks[0], lambda args, out: {"embedded": args[0]})
        self.keep(visual.blocks[-1], lambda args, out: {"encoded": out})
        self.keep(visual.merger, lambda args, out: {"merged": out})
        model.model.language_model.norm.register_forward_hook(
            lambda module, args, out: self.states.append(out[0, -1])
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


def frames(media, path):
    """A video's sampled frames (`frames.py`) as `[F, H, W, 3]` RGB, its metadata and times."""
    directory = media / f"{path}.frames"
    sampled = json.loads((directory / "sampled.json").read_text())
    video = np.stack([
        np.array(Image.open(directory / f"{index:03}.png").convert("RGB"))
        for index in range(len(sampled["indices"]))
    ])
    height, width = video.shape[1:3]
    metadata = {
        "total_num_frames": sampled["total"],
        "fps": sampled["fps"],
        "width": width,
        "height": height,
        "duration": sampled["total"] / sampled["fps"],
        "frames_indices": sampled["indices"],
    }
    times = [index / sampled["fps"] for index in sampled["indices"]]
    return video, metadata, times


def user_message(said):
    content = [{"type": "image"} for _ in said.get("images", [])]
    content += [{"type": "video"} for _ in said.get("videos", [])]
    content.append({"type": "text", "text": said["text"]})
    return {"role": "user", "content": content}


def reply(model, processor, recorder, messages, media, said):
    """One turn: its prompt through the processor, and the greedy reply."""
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=said.get("thinking", False)
    )
    images, videos, metadata = media
    inputs = processor(
        text=[text],
        images=images or None,
        videos=videos or None,
        images_kwargs={"size": IMAGE_SIZE},
        videos_kwargs={"size": VIDEO_SIZE, "cap_pixels_per_frame": True, "do_sample_frames": False,
                       "video_metadata": metadata} if videos else {},
        return_tensors="pt",
    )
    recorder.states.clear()
    generated = model.generate(
        **inputs,
        max_new_tokens=said["max_tokens"],
        do_sample=False,
        eos_token_id=ENDS,
        output_logits=True,
        return_dict_in_generate=True,
    )
    ids = inputs["input_ids"][0]
    answer = generated.sequences[0, len(ids):]
    return ids, answer, torch.stack(recorder.states[:len(answer)]), generated.logits[0][0]


def main():
    checkpoint, conversations, media, goldens, references = (Path(arg) for arg in sys.argv[1:6])
    torch.set_grad_enabled(False)
    processor = AutoProcessor.from_pretrained(checkpoint)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(checkpoint, dtype=torch.float32).eval()
    recorder = Recorder(model)
    goldens.mkdir(parents=True, exist_ok=True)
    replies = {}
    for case, conversation in json.loads(conversations.read_text()).items():
        tensors = {"conversation": torch.frombuffer(bytearray(json.dumps(conversation).encode()), dtype=torch.uint8)}
        messages, images, videos, metadata = [], [], [], []
        recorder.towers.clear()
        seen = 0
        replies[case] = []
        for index, said in enumerate(conversation["turns"]):
            for path in said.get("images", []):
                with Image.open(media / path) as image:
                    images.append(image.convert("RGB"))
                tensors[f"image.{len(images) - 1}"] = torch.from_numpy(np.array(images[-1]))
            for path in said.get("videos", []):
                video, meta, times = frames(media, path)
                videos.append(video)
                metadata.append(meta)
                tensors[f"video.{len(videos) - 1}.frames"] = torch.from_numpy(video)
                tensors[f"video.{len(videos) - 1}.times"] = torch.tensor(times, dtype=torch.float64)
            messages.append(user_message(said))
            calls = len(recorder.towers)
            ids, answer, states, logits = reply(model, processor, recorder, messages, (images, videos, metadata), said)
            # The tower reads the whole conversation's media each turn: its new items are the last.
            new = items(recorder.towers[calls:])[seen:]
            for item in new:
                for name, tensor in item.items():
                    tensors[f"item.{seen}.{name}"] = tensor
                seen += 1
            tensors[f"turn.{index}.ids"] = ids.to(torch.int64)
            tensors[f"turn.{index}.reply"] = answer.to(torch.int64)
            tensors[f"turn.{index}.hidden"] = states
            tensors[f"turn.{index}.logits"] = logits
            written = answer[:-1] if int(answer[-1]) in ENDS else answer
            text = processor.decode(written, skip_special_tokens=False)
            messages.append({"role": "assistant", "content": text})
            replies[case].append(text)
            print(case, index, json.dumps(text))
        tensors = {name: tensor.detach().contiguous() for name, tensor in tensors.items()}
        save_file(tensors, goldens / f"golden_f32_{case}.safetensors")
    references.write_text(json.dumps(replies, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
