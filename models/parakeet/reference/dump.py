"""Golden tensors from the reference runtime for `parakeet parity`: Moondream's kestrel (the engine
behind Photon) at a pinned version, its ternary weights expanded to f32 (the oracle its own kernels
are tested against) and its torch conformer ops, on one recording.

    python dump.py <checkpoint dir> <recording.wav> <out.safetensors>

Recorded tensors (f32 unless noted):
    pcm         [N]          the recording, 16 kHz mono
    features    [N/160+1,128] normalized log-mel features (the last frame is the zero past the end)
    subsampled  [T,1024]     the subsampler's output, every row
    speech      [L]          the speech head's probability of each encoder frame
    encoded     [L,640]      the encoder's frames through the joint's encoder projection
    tokens      [S] i64      the greedy decoder's decisions: token (8192 is the blank)
    durations   [S] i64      and frames advanced
"""

import sys

import numpy as np
import soundfile
import torch
from safetensors.torch import save_file

import kestrel_kernels.ternary as ternary
from kestrel_kernels.conformer_ops import torch_conformer_runtime
from kestrel_kernels.runtime import get_runtime
from kestrel.models.parakeet_tdt.features import parakeet_features
from kestrel.models.parakeet_tdt.weights import load_parakeet_tdt

checkpoint, recording, out = sys.argv[1:4]
ternary.resident_form = lambda device: "dense"
get_runtime("cpu").conformer = torch_conformer_runtime("conv1d")

model = load_parakeet_tdt(checkpoint, device="cpu", dtype=torch.float32, local_files_only=True).model
pcm, rate = soundfile.read(recording, dtype="float32")
assert rate == 16_000 and pcm.ndim == 1, f"{recording}: 16 kHz mono expected"

with torch.inference_mode():
    features, mask = parakeet_features(pcm)
    subsampled, valid = model.encoder.subsampling(features, mask)
    speech, _ = model.speech_probabilities(features, mask)
    encoded, valid = model.encode(features, mask)
    frames = int(valid.sum())
    output = model.generate_encoded(encoded, valid)

# The first decision is the blank the decoder starts from, advancing no frame.
tokens = output.sequences[0, 1:].to(torch.int64)
durations = output.durations[0, 1:].to(torch.int64)
save_file(
    {
        "pcm": torch.from_numpy(np.ascontiguousarray(pcm)),
        "features": features[0].contiguous(),
        "subsampled": subsampled[0].contiguous(),
        "speech": speech[0, :frames].contiguous(),
        "encoded": encoded[0, :frames].contiguous(),
        "tokens": tokens.contiguous(),
        "durations": durations.contiguous(),
    },
    out,
)
print(f"{len(pcm) / 16_000:.2f}s, {frames} frames, {int((tokens != 8192).sum())} tokens -> {out}")
