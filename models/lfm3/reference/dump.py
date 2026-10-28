"""The reference for `lfm3 parity` and the product's agreement checks: the checkpoint's own
modeling code (`Lfm3ForCausalLM`, `Lfm3VlForConditionalGeneration`,
`Lfm3AudioForConditionalGeneration` or `Lfm3OmniForConditionalGeneration`, `trust_remote_code`,
as its `config.json` names it) under transformers 5.21.0 and its tokenizer or processor, in f32
on the CPU, each conversation turn by turn as a chat client sends it (the whole conversation again
each turn, its earlier replies' text as generated), each reply greedy, its speech's codes too.

    python dump.py <checkpoint dir> <conversations.json> <media dir> <goldens dir> <references.json>

`conversations.json` maps each case to `{"turns": [...]}`, each turn the runtime's message
(`text`, `system`, `thinking`, `mode`, `max_tokens`, `seed`) with its images' paths
(`images`, LFM3-VL-2B and LFM3-Omni-2B) and its 16-bit 16 kHz WAV clips' paths (`audio`,
LFM3-Audio-2B and LFM3-Omni-2B) relative to `<media dir>`; the processor reads images at 65,536 to
1,638,400 pixels, as the runtime does. For every case, each turn's reply text into
`references.json` as `{"<case>": ["<reply>", ...]}`, and its tensors into `<goldens
dir>/golden_f32_<case>.safetensors` (f32 unless noted). Item `i` is the conversation's `i`th
image; its patches are row-major over its patch grid, as the runtime orders them (the reference
orders them by 2×2 block). Clip `c` is its `c`th clip.

    conversation      [B] u8       the case's JSON (UTF-8), its images' and clips' paths included
    image.{i}         [H,W,3] u8   image `i` as RGB
    item.{i}.grid     [2] i64      its patch grid: rows, columns
    item.{i}.patches  [N,1536]     its two-frame 16×16 patches, channel-major then frame-major,
                                   normalized to [-1, 1]
    item.{i}.embedded [N,1024]     the patch embeddings, RMS-normed
    item.{i}.encoded  [N,1024]     the vision tower's last layer's output
    item.{i}.tokens   [N/4,D]      its tokens, the projector's output, row-major over its 2×2
                                   blocks
    clip.{c}          [S]          clip `c`'s samples
    clip.{c}.features [F,128]      its log-mel features, frame-major
    clip.{c}.encoded  [T,768]      the audio encoder's output, final-normed
    clip.{c}.tokens   [T,D]        its tokens, the audio projector's output
    turn.{t}.ids      [L] i64      turn `t`'s prompt ids, the whole conversation's
    turn.{t}.reply    [A] i64      its greedy reply, `<|audio_out|>` for each frame it spoke, its
                                   end token included when it ended
    turn.{t}.hidden   [A,D]        the final-normed state of the prompt's last row, then of each
                                   reply row's but the last's
    turn.{t}.logits   [V]          the logits of the prompt's last row
    turn.{t}.depth    [12,2017]    a reply that speaks: the depth decoder's logits over its first
                                   frame, every codebook masked
    turn.{t}.codes    [F,12] i64   its frames, greedy, the one that ends its speech left out
    turn.{t}.speech   [S]          the codec's 24 kHz samples of them
"""

import json
import sys
import wave
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors.torch import save_file
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoModelForMultimodalLM,
    AutoProcessor,
    AutoTokenizer,
)

MERGE = 2
IMAGE_SIZE = {"shortest_edge": 65_536, "longest_edge": 1_638_400}
# <|im_end|>
END = 124900
# <|text_end|> and <|audio_out|>: like <|im_end|>, a reply's text leaves them out.
TEXT_END, AUDIO_OUT = 125014, 125032
# Mel frames per audio token.
STRIDE = 8
# 2^15: 16-bit samples decode to [-1, 1).
PCM_SCALE = 32768.0
# What an assistant's turn opens with in a mode that speaks.
MARKERS = {"speech": "<|audio_start|>", "interleaved": "<|mixed_start|>"}


def row_major(blocks, rows, columns):
    """Rows in the processor's 2×2-block order, reordered row-major over the patch grid."""
    order = []
    for row in range(rows):
        for column in range(columns):
            block = (row // MERGE) * (columns // MERGE) + column // MERGE
            order.append(block * MERGE * MERGE + (row % MERGE) * MERGE + column % MERGE)
    return blocks[torch.tensor(order)]


class Recorder:
    """The final-normed states of the language model's calls, the vision tower's and the audio
    encoder's inputs and outputs, of one reply."""

    def __init__(self, model, variant):
        self.states, self.towers, self.encoders = [], [], []
        language = model.model if variant == "lfm3" else model.model.language_model
        language.embedding_norm.register_forward_hook(
            lambda module, args, out: self.states.append(out[0, -1])
        )
        if variant in ("lfm3_vl", "lfm3_omni"):
            tower = model.model.vision_tower
            tower.register_forward_pre_hook(self.tower_input, with_kwargs=True)
            self.keep(self.towers, tower.embedding_norm, "embedded", lambda out: out)
            self.keep(self.towers, tower, "encoded", lambda out: out.last_hidden_state)
            self.keep(self.towers, model.model.multi_modal_projector, "tokens", lambda out: out)
        if variant in ("lfm3_audio", "lfm3_omni"):
            encoder = model.model.audio_encoder
            encoder.register_forward_pre_hook(self.encoder_input)
            self.keep(self.encoders, encoder, "encoded", lambda out: out[0])
            self.keep(self.encoders, model.model.audio_projector, "tokens", lambda out: out)

    def clear(self):
        self.states.clear()
        self.towers.clear()
        self.encoders.clear()

    def keep(self, calls, module, name, value):
        module.register_forward_hook(
            lambda module, args, out: calls[-1].__setitem__(name, value(out))
        )

    def tower_input(self, module, args, kwargs):
        self.towers.append({"pixels": args[0], "grids": kwargs["grid_thw"]})

    def encoder_input(self, module, args):
        self.encoders.append({"features": args[0], "lengths": args[1]})


def items(call, count):
    """The last `count` images of the vision tower's first call: each one's tensors."""
    found, start, token = [], 0, 0
    for _, rows, columns in call["grids"].tolist():
        patches = rows * columns
        merged = patches // (MERGE * MERGE)
        span, tokens = slice(start, start + patches), slice(token, token + merged)
        found.append({
            "grid": torch.tensor([rows, columns], dtype=torch.int64),
            "patches": row_major(call["pixels"][span], rows, columns),
            "embedded": row_major(call["embedded"][span], rows, columns),
            "encoded": row_major(call["encoded"][span], rows, columns),
            "tokens": call["tokens"][tokens],
        })
        start, token = start + patches, token + merged
    return found[len(found) - count:]


def clips(call, count):
    """The last `count` clips of the audio encoder's first call: each one's tensors."""
    found, token = [], 0
    for index, frames in enumerate(call["lengths"].tolist()):
        tokens = -(frames // -STRIDE)
        found.append({
            "features": call["features"][index, :, :frames].T,
            "encoded": call["encoded"][index, :tokens],
            "tokens": call["tokens"][token:token + tokens],
        })
        token += tokens
    return found[len(found) - count:]


def read_clip(path):
    """A 16-bit mono WAV file's samples."""
    with wave.open(str(path)) as clip:
        assert clip.getsampwidth() == 2 and clip.getnchannels() == 1, path
        frames = clip.readframes(clip.getnframes())
    return np.frombuffer(frames, dtype="<i2").astype(np.float32) / PCM_SCALE


def user_message(turn):
    content = [{"type": "image"} for _ in turn.get("images", [])]
    content += [{"type": "audio"} for _ in turn.get("audio", [])]
    content.append({"type": "text", "text": turn["text"]})
    return {"role": "user", "content": content}


def render(reader, messages, turn, variant):
    """The prompt's text: a reply that speaks, and LFM3-Audio-2B's text reply, prefilled with
    their opener; any other from the template's generation prompt."""
    mode = turn.get("mode", "text")
    opener = MARKERS.get(mode)
    if opener is None and mode == "text" and variant == "lfm3_audio":
        opener = "<think>" if turn.get("thinking") else "<think></think>"
    if opener is not None:
        prefilled = messages + [{"role": "assistant", "content": opener}]
        return reader.apply_chat_template(prefilled, tokenize=False, continue_final_message=True)
    return reader.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=turn.get("thinking", False),
    )


def reply(model, reader, recorder, messages, media, turn, variant):
    """One turn: its prompt through the chat template (and the processor), and the greedy reply."""
    text = render(reader, messages, turn, variant)
    images, audio = media
    kwargs = {"add_special_tokens": False, "return_tensors": "pt"}
    if images:
        kwargs |= {"images": images, "images_kwargs": {"size": IMAGE_SIZE}}
    if audio:
        kwargs |= {"audio": audio}
    inputs = reader(text=text, **kwargs)
    recorder.clear()
    torch.manual_seed(turn["seed"])
    if variant in ("lfm3_audio", "lfm3_omni"):
        speaks = turn.get("mode", "text") in MARKERS
        generated = model.generate(
            **inputs,
            max_new_tokens=turn["max_tokens"],
            do_sample=False,
            audio_temperature=0,
            generation_mode=None if speaks else "text",
            return_dict_in_generate=True,
        )
        logits = model.lm_head(recorder.states[0])
    else:
        generated = model.generate(
            **inputs,
            max_new_tokens=turn["max_tokens"],
            do_sample=False,
            # Greedy as the fixtures ask: the checkpoints' generation_config.json penalizes
            # repetition by 1.1, which the runtime's greedy replies don't.
            repetition_penalty=1.0,
            eos_token_id=END,
            output_logits=True,
            return_dict_in_generate=True,
        )
        logits = generated.logits[0][0]
    ids = inputs["input_ids"]
    answer = generated.sequences[0, ids.shape[1]:]
    states = torch.stack(recorder.states[:len(answer)])
    return ids[0], answer, states, logits, generated


def spoken(model, codec, hidden, generated):
    """A reply's speech: the depth decoder's first pass from the prompt's last state, its
    frames, and the codec's samples of them."""
    codebooks = model.config.num_codebooks
    masked = torch.ones((1, codebooks), dtype=torch.bool)
    codes = torch.zeros((1, codebooks), dtype=torch.long)
    depth = model.model.depth_decoder(hidden[None], codes, masked)[0]
    frames = generated.audio_codes[0, :, :int(generated.audio_lengths[0])]
    speech = codec.decode(frames[None]).audio_values[0]
    return {"depth": depth, "codes": frames.T.to(torch.int64), "speech": speech.float()}


def load(checkpoint):
    """The model, its tokenizer or processor, the codec of a model that speaks, and the variant."""
    variant = json.loads((checkpoint / "config.json").read_text())["model_type"]
    loaders = {
        "lfm3": AutoModelForCausalLM,
        "lfm3_vl": AutoModelForImageTextToText,
        "lfm3_audio": AutoModelForMultimodalLM,
        "lfm3_omni": AutoModelForMultimodalLM,
    }
    if variant == "lfm3":
        reader = AutoTokenizer.from_pretrained(checkpoint)
    else:
        reader = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True)
    codec = getattr(reader, "audio_tokenizer", None)
    if codec is not None:
        codec = codec.float().eval()
    model = loaders[variant].from_pretrained(checkpoint, dtype=torch.float32, trust_remote_code=True)
    return model.eval(), reader, codec, variant


def media(turn, root, tensors, taken):
    """The turn's images and clips, read and recorded as the conversation's next ones."""
    images, audio = taken
    for path in turn.get("images", []):
        with Image.open(root / path) as image:
            images.append(image.convert("RGB"))
        tensors[f"image.{len(images) - 1}"] = torch.from_numpy(np.array(images[-1]))
    for path in turn.get("audio", []):
        audio.append(read_clip(root / path))
        tensors[f"clip.{len(audio) - 1}"] = torch.from_numpy(audio[-1])


def stages(recorder, tensors, turn, taken):
    """The tensors of the turn's new images and clips, from the recorded calls."""
    images, audio = taken
    new = len(turn.get("images", []))
    for offset, item in enumerate(items(recorder.towers[0], new) if new else []):
        for name, tensor in item.items():
            tensors[f"item.{len(images) - new + offset}.{name}"] = tensor
    new = len(turn.get("audio", []))
    for offset, clip in enumerate(clips(recorder.encoders[0], new) if new else []):
        for name, tensor in clip.items():
            tensors[f"clip.{len(audio) - new + offset}.{name}"] = tensor


def main():
    checkpoint, conversations, root, goldens, references = (Path(arg) for arg in sys.argv[1:6])
    torch.set_grad_enabled(False)
    model, reader, codec, variant = load(checkpoint)
    recorder = Recorder(model, variant)
    goldens.mkdir(parents=True, exist_ok=True)
    replies = {}
    for case, conversation in json.loads(conversations.read_text()).items():
        encoded = bytearray(json.dumps(conversation).encode())
        tensors = {"conversation": torch.frombuffer(encoded, dtype=torch.uint8)}
        messages, taken = [], ([], [])
        if conversation["turns"][0].get("system"):
            messages.append({"role": "system", "content": conversation["turns"][0]["system"]})
        replies[case] = []
        for index, turn in enumerate(conversation["turns"]):
            media(turn, root, tensors, taken)
            messages.append(user_message(turn))
            ids, answer, hidden, logits, generated = reply(
                model, reader, recorder, messages, taken, turn, variant
            )
            stages(recorder, tensors, turn, taken)
            tensors[f"turn.{index}.ids"] = ids.to(torch.int64)
            tensors[f"turn.{index}.reply"] = answer.to(torch.int64)
            tensors[f"turn.{index}.hidden"] = hidden
            tensors[f"turn.{index}.logits"] = logits
            if turn.get("mode", "text") in MARKERS:
                for name, tensor in spoken(model, codec, hidden[0], generated).items():
                    tensors[f"turn.{index}.{name}"] = tensor
            written = [token for token in answer.tolist() if token not in (END, TEXT_END, AUDIO_OUT)]
            text = reader.decode(written, skip_special_tokens=False)
            messages.append({"role": "assistant", "content": text})
            replies[case].append(text)
            print(case, index, json.dumps(text))
        tensors = {name: tensor.detach().contiguous() for name, tensor in tensors.items()}
        save_file(tensors, goldens / f"golden_f32_{case}.safetensors")
    references.parent.mkdir(parents=True, exist_ok=True)
    references.write_text(json.dumps(replies, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
