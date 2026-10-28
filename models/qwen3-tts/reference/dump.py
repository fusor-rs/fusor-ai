"""The reference for `qwen3-tts parity` and the product's checks: `qwen-tts` 0.1.1's own modeling
code (`Qwen3TTSModel`, transformers 4.57.3) in f32 on the CPU with eager attention, a voice cloned
as its `generate_voice_clone` does, greedy everywhere (`do_sample=False`, `subtalker_dosample=
False`, the repetition penalty of its generation config).

    python -I dump.py <checkpoint dir> <cases.json> <media dir> <goldens dir>

`cases.json` maps each case to `{"text", "language", "reference_text"}`: a voice cloned from the
clip's frames and transcript (in-context learning) when `reference_text` is given, from its speaker
embedding alone when it is empty. The voice is `<media dir>/voice-24k.wav`, 24 kHz float32 of whole
1,920-sample frames. Each case's tensors go into `<goldens dir>/golden_f32_<case>.safetensors`
(f32 unless noted):

    case                [B] u8       the case's JSON (UTF-8)
    reference           [S]          the voice's samples
    speaker.features    [F,128]      its log-mel features, frame-major
    speaker.embedding   [1024]       the speaker encoder's embedding of them
    codec.latent        [T,512]      with a transcript: the speech tokenizer's encoder before its
                                     quantizer, 12.5 Hz
    codec.codes         [T,16] i64   with a transcript: the voice's frames
    text.ids            [L] i64      the text's token ids in the speaking turn
    reference.ids       [R] i64      with a transcript: its token ids in the reference turn
    prompt              [P,1024]     the rows the talker reads first
    trailing            [N,1024]     the text rows read along with the frames after (none when the
                                     prompt holds all the text and the padding follows)
    pad                 [1024]       the row that follows the frames once the text is out
    talker.logits       [3072]       the prompt's last row's logits, raw
    talker.hidden       [F,1024]     the final-normed state of the prompt's last row, then of each
                                     frame's row but the last's: the code predictor's input
    codes               [F,16] i64   the speech's frames, the one that ends it left out
    ended               [1] u8       whether the speech ended, rather than ran out of frames
                                     (a greedy reply can loop on)
    predictor.logits    [15,2048]    the code predictor's logits over the first frame, teacher-forced
    speech              [F*1920]     the speech tokenizer's decoder's 24 kHz samples of the frames,
                                     after the voice's own frames when it has any
"""

import json
import sys
from pathlib import Path

import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel
from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram
from safetensors.torch import save_file
from transformers import AutoTokenizer

SAMPLE_RATE = 24_000
FRAME_SAMPLES = 1920
# Frames of a reply, at the most (the reply's last first code is drawn but never expanded into a
# frame, so a reply that runs out has one frame fewer).
MAX_FRAMES = 200
# The first code that ends the speech.
END_CODE = 2150
# Frames the speech tokenizer's decoder decodes in one pass; past them it decodes in chunks, which
# the runtime's single stream does not follow.
CHUNK_FRAMES = 300
# Tokens of `<|im_start|>assistant\n` and of what closes the text: `<|im_end|>\n<|im_start|>
# assistant\n`, and a transcript: `<|im_end|>\n`.
ROLE, REPLY, CLOSING = 3, 5, 2


def bytes_of(text):
    return torch.frombuffer(bytearray(text.encode()), dtype=torch.uint8)


def features_of(wav):
    mels = mel_spectrogram(
        torch.from_numpy(wav).unsqueeze(0),
        n_fft=1024,
        num_mels=128,
        sampling_rate=SAMPLE_RATE,
        hop_size=256,
        win_size=1024,
        fmin=0,
        fmax=12000,
    )
    return mels.transpose(1, 2)[0]


def latent_of(tokenizer_model, wav):
    """The encoder's rows before the quantizer: SEANet, the transformer, the downsampling conv."""
    encoder = tokenizer_model.encoder
    rows = encoder.encoder(torch.from_numpy(wav)[None, None]).transpose(1, 2)
    rows = encoder.encoder_transformer(rows)[0].transpose(1, 2)
    return encoder.downsample(rows)[0].transpose(0, 1)


def recording(model):
    """Wraps the talker's `generate` and first `forward` to keep its prompt, its trailing rows and
    padding, whether it ended, and the prompt's last row's logits."""
    kept = {}
    generate = model.talker.generate

    def record(*args, **kwargs):
        kept["prompt"] = kwargs["inputs_embeds"][0]
        kept["trailing"] = kwargs["trailing_text_hidden"][0]
        kept["pad"] = kwargs["tts_pad_embed"].reshape(-1)
        result = generate(*args, **kwargs)
        kept["ended"] = bool(result.sequences[0, -1] == END_CODE)
        return result

    def first_logits(module, args, kwargs, output):
        kept.setdefault("logits", output.logits[0, -1].clone())

    model.talker.generate = record
    model.talker.register_forward_hook(first_logits, with_kwargs=True)
    return kept


def predictor_logits(model, hidden, frame):
    """The code predictor's logits over `frame`, read teacher-forced: the talker's state, the first
    code's embedding, then each code but the last through its codebook's table."""
    predictor = model.talker.code_predictor
    rows = [hidden, model.talker.get_input_embeddings()(frame[:1])[0]]
    tables = predictor.get_input_embeddings()
    for index in range(14):
        rows.append(tables[index](frame[index + 1 : index + 2])[0])
    return predictor.forward_finetune(inputs_embeds=torch.stack(rows)[None]).logits[0]


def dump(tts, kept, case, wav):
    model = tts.model
    icl = bool(case["reference_text"])
    items = tts.create_voice_clone_prompt(
        ref_audio=(wav, SAMPLE_RATE),
        ref_text=case["reference_text"],
        x_vector_only_mode=not icl,
    )
    prompt = tts._prompt_items_to_voice_clone_prompt(items)
    input_ids = tts._tokenize_texts([tts._build_assistant_text(case["text"])])
    ref_ids = [tts._tokenize_texts([tts._build_ref_text(case["reference_text"])])[0]] if icl else [None]
    settings = tts._merge_generate_kwargs(
        do_sample=False, subtalker_dosample=False, max_new_tokens=MAX_FRAMES
    )
    kept.clear()
    codes, hiddens = model.generate(
        input_ids=input_ids,
        ref_ids=ref_ids,
        voice_clone_prompt=prompt,
        languages=[case["language"]],
        non_streaming_mode=False,
        **settings,
    )
    codes, hidden = codes[0], hiddens[0]
    reference = prompt["ref_code"][0] if icl else None
    tensors = {
        "case": bytes_of(json.dumps(case)),
        "reference": torch.from_numpy(wav),
        "speaker.features": features_of(wav),
        "speaker.embedding": prompt["ref_spk_embedding"][0].reshape(-1),
        "text.ids": input_ids[0][0, ROLE:-REPLY],
        "prompt": kept["prompt"],
        "pad": kept["pad"],
        "talker.logits": kept["logits"],
        "talker.hidden": hidden,
        "codes": codes,
        "ended": torch.tensor([int(kept["ended"])], dtype=torch.uint8),
        "predictor.logits": predictor_logits(model, hidden[0], codes[0]),
    }
    padded = kept["trailing"].shape[0] == 1 and torch.equal(kept["trailing"][0], kept["pad"])
    tensors["trailing"] = kept["trailing"][:0] if padded else kept["trailing"]
    played = codes
    if icl:
        tensors["codec.latent"] = latent_of(model.speech_tokenizer.model, wav)
        tensors["codec.codes"] = reference
        tensors["reference.ids"] = ref_ids[0][0, ROLE:-CLOSING]
        played = torch.cat([reference, codes])
    assert len(played) <= CHUNK_FRAMES, f"{len(played)} frames decode in chunks"
    speech = model.speech_tokenizer.decode([{"audio_codes": played}])[0][0]
    skipped = len(reference) * FRAME_SAMPLES if icl else 0
    tensors["speech"] = torch.from_numpy(speech[skipped:]).float()
    return tensors


def main():
    checkpoint, cases, media, goldens = (Path(arg) for arg in sys.argv[1:5])
    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    tts = Qwen3TTSModel.from_pretrained(
        str(checkpoint), dtype=torch.float32, device_map="cpu", attn_implementation="eager"
    )
    # `from_pretrained` asks for transformers' Mistral pre-tokenizer regex for a local checkpoint
    # of this transformers version; loaded by hub id, as the model card does, Qwen's stays.
    tts.processor.tokenizer = AutoTokenizer.from_pretrained(str(checkpoint))
    kept = recording(tts.model)
    wav, rate = sf.read(media / "voice-24k.wav", dtype="float32")
    assert rate == SAMPLE_RATE and len(wav) % FRAME_SAMPLES == 0
    goldens.mkdir(parents=True, exist_ok=True)
    for name, case in json.loads(cases.read_text()).items():
        tensors = dump(tts, kept, case, wav)
        ended = bool(tensors["ended"][0])
        print(name, len(tensors["codes"]), "frames", "ended" if ended else "ran out")
        tensors = {key: tensor.detach().contiguous() for key, tensor in tensors.items()}
        save_file(tensors, goldens / f"golden_f32_{name}.safetensors")


if __name__ == "__main__":
    main()
