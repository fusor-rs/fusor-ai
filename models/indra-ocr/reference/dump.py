"""The reference for `indra-ocr parity` and the agreement checks: Datalab's indra-ocr package at a
pinned version, its networks on ONNX Runtime with one thread, reading images as its CLI does with
`--mode ocr` (never a PDF's text layer).

    python dump.py <model dir> <pages.json> <images dir> <goldens dir> <references.json>

`pages.json` maps each case to `{"image": <file>, "math": <bool>}`. For every case, the page as
the package's JSON output writes it (lines, blocks, markdown, rotation, skew) into
`references.json`, and into `<goldens dir>/golden_f32_<case>.safetensors` every network's calls in
order and the pipeline's steps between them (f32 unless noted):

    image                 [H,W] u8      the page as read, grayscale (OpenCV's conversion)
    skew                  [1]           degrees the page is straightened by (0 under 0.3)
    turn                  [1] i64       quarter turns undone (4: vertical columns)
    upright               [H',W'] u8    the page turned and straightened, as detected and read
    boxes                 [N,5]         the detected lines: x0, y0, x1, y1 in page pixels, score
    page                  [B] u8        the page's JSON (UTF-8), as `references.json` holds it
    <net>.<i>.<name>      the i-th call of network <net> (`det`, `orient`, `script`, `rec`,
                          `rec_cjk`, `rec_rtl`, `rec_world`, `order`, `mathcls`, `math_enc`,
                          `math_dec_init`, `math_dec_step`): each of its inputs and outputs by
                          the ONNX graph's names, integer tensors as i64 and boolean as u8; a
                          decoder step's caches left out (its token, position and logits kept)
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import onnxruntime
from safetensors.numpy import save_file

import indra_ocr
from indra_ocr import pipeline

# A decoder step's key and value caches: the runtime rebuilds them from the steps' tokens.
CACHES = {"self_k", "self_v", "cross_k", "cross_v", "new_k", "new_v"}

class Recorder:
    """Every call of every ONNX Runtime session, by the session's model name, in order."""

    def __init__(self):
        self.calls = []
        run = onnxruntime.InferenceSession.run
        recorder = self

        def recorded(session, names, feed, *args, **kwargs):
            outputs = run(session, names, feed, *args, **kwargs)
            wanted = names or [output.name for output in session.get_outputs()]
            net = Path(session._model_path).stem
            recorder.calls.append((net, dict(feed), dict(zip(wanted, outputs))))
            return outputs

        onnxruntime.InferenceSession.run = recorded

    def take(self):
        """The calls recorded since the last take, by `<net>.<i>.<name>`."""
        calls, self.calls = self.calls, []
        counts = {}
        out = {}
        for net, feed, outputs in calls:
            index = counts.get(net, 0)
            counts[net] = index + 1
            for name, value in {**feed, **outputs}.items():
                if net == "math_dec_step" and name in CACHES:
                    continue
                out[f"{net}.{index}.{name}"] = stored(np.asarray(value))
        return out


def stored(array):
    if array.dtype == np.bool_:
        return array.astype(np.uint8)
    if np.issubdtype(array.dtype, np.integer):
        return array.astype(np.int64)
    return np.ascontiguousarray(array.astype(np.float32))


class Steps:
    """The pipeline's results between the networks, kept from the page's read."""

    def __init__(self, ocr):
        self.seen = {}
        steps = self
        turn, detect = ocr._turn, ocr.detect

        def turned(gray, k, a, small=None):
            page, k, skew = turn(gray, k, a, small)
            steps.seen.update(upright=page, turn=k, skew=skew)
            return page, k, skew

        def detected(gray):
            boxes = detect(gray)
            steps.seen["boxes"] = boxes
            return boxes

        ocr._turn, ocr.detect = turned, detected


def page_json(page):
    return {
        "source": page.source,
        "rotation": page.rotation,
        "skew": page.skew,
        "vertical": page.vertical,
        "lines": [{"box": line.box, "text": line.text, "conf": line.conf} for line in page.lines],
        "blocks": page.blocks,
        "markdown": page.markdown,
    }


def main():
    models, pages, images, goldens, references = (Path(arg) for arg in sys.argv[1:6])
    goldens.mkdir(parents=True, exist_ok=True)
    recorder = Recorder()
    out = {}
    for case, spec in json.loads(pages.read_text()).items():
        converter = indra_ocr.load(path=models, threads=1, mode="ocr", math=spec["math"])
        steps = Steps(converter.ocr)
        gray = pipeline.read_images(images / spec["image"])[0]
        [(_, page)] = list(converter.convert(images / spec["image"]))
        written = page_json(page)
        out[case] = written
        seen = steps.seen
        calls = recorder.take()
        tensors = {
            "image": gray,
            "skew": np.array([seen["skew"]], np.float32),
            "turn": np.array([seen["turn"]], np.int64),
            "upright": np.ascontiguousarray(seen["upright"]),
            "boxes": np.asarray(seen["boxes"], np.float32).reshape(-1, 5),
            "page": np.frombuffer(json.dumps(written, ensure_ascii=False).encode(), np.uint8),
            **calls,
        }
        save_file(tensors, str(goldens / f"golden_f32_{case}.safetensors"))
        print(case, f"{len(page.lines)} lines, {len(calls)} tensors of network calls")
    references.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
