"""The reference for `kronos parity` and the agreement checks: the Kronos repository's own
`model` package (`KronosTokenizer`, `Kronos`, `KronosPredictor`) at a pinned commit, in f32 on
the CPU, forecasting with each case's sampler, one sample, seeded as its regression test.

    python dump.py <kronos source> <checkpoint dir> <cases.json> <series dir> <goldens dir> <references.json>

`<checkpoint dir>` holds Kronos-base in `model/` and Kronos-Tokenizer-base in `tokenizer/`.
`cases.json` maps each case to `{"series", "lookback", "horizon", "end", "sampling"}`: the CSV
(relative to `<series dir>`), the rows of context, the rows to forecast, the rows before the
forecast, and the sampler's `temperature`, `top_k` and `top_p` (greedy, `top_k` 1, for checks
that compare forecasts).
For every case, its forecast into `references.json` as `{"<case>": {"times", "values"}}`, and
the tensors of every stage into `<goldens dir>/golden_f32_<case>.safetensors` (f32 unless
noted):

    rows        [L,6]       the context's rows as f32: open, high, low, close, volume, amount
    normalized  [L,6]       the rows normalized by their columns' mean and deviation, clipped
    mean        [6]         each column's mean
    deviation   [6]         each column's standard deviation
    stamps      [L+H,5] i64 each row's minute, hour, weekday, day and month: the context's,
                            then the forecast's
    codes       [L,20]      the tokenizer's code of each row, before its signs are kept
    s1, s2      [L] i64     the context's tokens
    s1_logits   [1024]      the first step's s1 logits (the last context row's)
    s2_logits   [1024]      the first step's s2 logits for its greedy s1
    drawn_s1, drawn_s2 [H] i64  the tokens drawn
    decoded     [W,6]       the tokenizer's decoding of the last window of tokens, normalized
    forecast    [H,6]       the forecast in the series' units
"""

import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from safetensors.torch import save_file

COLUMNS = ["open", "high", "low", "close", "volume", "amount"]
MAX_CONTEXT = 512
SEED = 123


def load(source, checkpoint):
    sys.path.insert(0, str(source))
    from model import Kronos, KronosPredictor, KronosTokenizer

    tokenizer = KronosTokenizer.from_pretrained(checkpoint / "tokenizer").eval()
    model = Kronos.from_pretrained(checkpoint / "model").eval()
    return KronosPredictor(model, tokenizer, device="cpu", max_context=MAX_CONTEXT)


def read_series(path):
    frame = pd.read_csv(path, encoding="utf-8-sig")
    frame["timestamps"] = pd.to_datetime(frame["timestamps"])
    return frame


class Recorder:
    """Records the tensors the predictor's calls pass through, one forecast at a time."""

    def __init__(self, predictor):
        self.seen = {"codes": [], "s1_logits": [], "s2_logits": []}
        tokenizer, model = predictor.tokenizer, predictor.model
        tokenizer.quant_embed.register_forward_hook(
            lambda module, args, out: self.seen["codes"].append(out.detach().clone())
        )
        decode_s1, decode_s2 = model.decode_s1, model.decode_s2

        def recorded_s1(*args, **kwargs):
            logits, context = decode_s1(*args, **kwargs)
            self.seen["s1_logits"].append(logits[:, -1].detach().clone())
            return logits, context

        def recorded_s2(*args, **kwargs):
            logits = decode_s2(*args, **kwargs)
            self.seen["s2_logits"].append(logits[:, -1].detach().clone())
            return logits

        model.decode_s1, model.decode_s2 = recorded_s1, recorded_s2
        self.decode = tokenizer.decode
        self.decoded = None

        def recorded_decode(tokens, half=False):
            self.drawn = tokens
            out = self.decode(tokens, half)
            self.decoded = out.detach().clone()
            return out

        tokenizer.decode = recorded_decode


def stamps(times):
    return np.stack(
        [times.dt.minute, times.dt.hour, times.dt.day_of_week, times.dt.day, times.dt.month], axis=1
    ).astype(np.int64)


def dump_case(predictor, case, series_dir):
    frame = read_series(series_dir / case["series"])
    end, lookback, horizon = case["end"], case["lookback"], case["horizon"]
    sampling = case["sampling"]
    context = frame.iloc[end - lookback : end].reset_index(drop=True)
    future = frame["timestamps"].iloc[end : end + horizon].reset_index(drop=True)
    recorder = Recorder(predictor)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    with torch.no_grad():
        forecast = predictor.predict(
            df=context[COLUMNS],
            x_timestamp=context["timestamps"],
            y_timestamp=future,
            pred_len=horizon,
            T=sampling["temperature"],
            top_k=sampling["top_k"],
            top_p=sampling["top_p"],
            sample_count=1,
            verbose=False,
        )
    rows = context[COLUMNS].values.astype(np.float32)
    mean, deviation = np.mean(rows, axis=0), np.std(rows, axis=0)
    normalized = np.clip((rows - mean) / (deviation + 1e-5), -5, 5)
    codes = recorder.seen["codes"][0][0]
    bits = (codes > 0).long()
    weights = 2 ** torch.arange(10)
    drawn = recorder.drawn
    tensors = {
        "rows": torch.from_numpy(rows),
        "normalized": torch.from_numpy(normalized.astype(np.float32)),
        "mean": torch.from_numpy(mean),
        "deviation": torch.from_numpy(deviation),
        "stamps": torch.from_numpy(np.concatenate([stamps(context["timestamps"]), stamps(future)])),
        "codes": codes,
        "s1": (bits[:, :10] * weights).sum(-1),
        "s2": (bits[:, 10:] * weights).sum(-1),
        "s1_logits": recorder.seen["s1_logits"][0][0],
        "s2_logits": recorder.seen["s2_logits"][0][0],
        "drawn_s1": drawn[0][0, -horizon:],
        "drawn_s2": drawn[1][0, -horizon:],
        "decoded": recorder.decoded[0],
        "forecast": torch.from_numpy(forecast[COLUMNS].values.astype(np.float32)),
    }
    reference = {
        "times": [time.strftime("%Y-%m-%d %H:%M:%S") for time in future],
        "values": [[float(value) for value in row] for row in forecast[COLUMNS].values],
    }
    return {name: tensor.contiguous() for name, tensor in tensors.items()}, reference


def main():
    source, checkpoint, cases, series_dir, goldens, references = map(Path, sys.argv[1:7])
    torch.set_num_threads(8)
    goldens.mkdir(parents=True, exist_ok=True)
    out = {}
    for name, case in json.loads(cases.read_text()).items():
        predictor = load(source, checkpoint)
        tensors, out[name] = dump_case(predictor, case, series_dir)
        save_file(tensors, str(goldens / f"golden_f32_{name}.safetensors"))
        print(f"{name}: {len(out[name]['times'])} rows forecast")
    references.write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
