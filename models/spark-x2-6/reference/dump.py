"""The reference for `spark-x2-6 parity` and the products' agreement checks: the checkpoint's own
modeling code (`Spark2_5ForCausalLM`, `trust_remote_code`) under transformers 4.57.1 and its
tokenizer, in f32 on the CPU, each conversation turn by turn as a chat client sends it (the whole
conversation again each turn, its earlier replies' answers as generated), each reply greedy.

    python dump.py <checkpoint dir> <conversations.json> <goldens dir> <references.json>

`conversations.json` maps each case to `{"turns": [...]}`, each turn the runtime's message
(`text`, `system`, `thinking`, `max_tokens`, `seed`). For every case, each turn's reply text into
`references.json` as `{"<case>": ["<reply>", ...]}`, and its tensors into
`<goldens dir>/golden_f32_<case>.safetensors` (f32 unless noted):

    conversation     [B] u8       the case's JSON (UTF-8)
    turn.{t}.ids     [L] i64      turn `t`'s prompt ids, the whole conversation's
    turn.{t}.reply   [A] i64      its greedy reply, its end token included when it ended
    turn.{t}.hidden  [A,D]        the final-normed state of the prompt's last row, then of each
                                  reply token's row but the last's
    turn.{t}.logits  [V]          the logits of the prompt's last row
"""

import json
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

# <｜end▁of▁sentence｜>
END = 1
THINK_END = "</think>"


def messages_of(turn, history):
    """The chat template's messages: the conversation's system text, then each earlier turn and
    its reply's answer (what follows its reasoning), then the turn's message."""
    messages = [{"role": "system", "content": turn["system"]}] if turn.get("system") else []
    for said, reply in history:
        messages.append({"role": "user", "content": said})
        messages.append({"role": "assistant", "content": reply.rsplit(THINK_END, 1)[-1]})
    messages.append({"role": "user", "content": turn["text"]})
    return messages


def reply(model, tokenizer, states, turn, history):
    """One turn: its prompt through the chat template, and the greedy reply."""
    text = tokenizer.apply_chat_template(
        messages_of(turn, history),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=turn.get("thinking", False),
    )
    ids = tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"]
    states.clear()
    generated = model.generate(
        input_ids=ids,
        max_new_tokens=turn["max_tokens"],
        do_sample=False,
        eos_token_id=END,
        output_logits=True,
        return_dict_in_generate=True,
    )
    answer = generated.sequences[0, ids.shape[1]:]
    return ids[0], answer, torch.stack(states[:len(answer)]), generated.logits[0][0]


def main():
    checkpoint, conversations, goldens, references = (Path(arg) for arg in sys.argv[1:5])
    torch.set_grad_enabled(False)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint, dtype=torch.float32, trust_remote_code=True
    ).eval()
    states = []
    model.model.norm.register_forward_hook(lambda module, args, out: states.append(out[0, -1]))
    goldens.mkdir(parents=True, exist_ok=True)
    replies = {}
    for case, conversation in json.loads(conversations.read_text()).items():
        encoded = bytearray(json.dumps(conversation).encode())
        tensors = {"conversation": torch.frombuffer(encoded, dtype=torch.uint8)}
        history = []
        replies[case] = []
        for index, turn in enumerate(conversation["turns"]):
            ids, answer, hidden, logits = reply(model, tokenizer, states, turn, history)
            tensors[f"turn.{index}.ids"] = ids.to(torch.int64)
            tensors[f"turn.{index}.reply"] = answer.to(torch.int64)
            tensors[f"turn.{index}.hidden"] = hidden
            tensors[f"turn.{index}.logits"] = logits
            written = answer[:-1] if int(answer[-1]) == END else answer
            text = tokenizer.decode(written, skip_special_tokens=False)
            history.append((turn["text"], text))
            replies[case].append(text)
            print(case, index, json.dumps(text))
        tensors = {name: tensor.detach().contiguous() for name, tensor in tensors.items()}
        save_file(tensors, goldens / f"golden_f32_{case}.safetensors")
    references.write_text(json.dumps(replies, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
