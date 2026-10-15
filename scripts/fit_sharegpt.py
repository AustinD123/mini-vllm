"""Refit the ShareGPT length parameters in minivllm/bench/workload.py from the raw dataset.

Source
    Hugging Face dataset anon8231489123/ShareGPT_Vicuna_unfiltered, file
    ShareGPT_V3_unfiltered_cleaned_split.json. The script records the file's SHA-256.

Filtering
    A conversation counts when its first turn is from the human and its second is from the
    assistant ("gpt"). The prompt is the first human message and the output is the first
    assistant reply. Conversations with an empty prompt or reply are dropped, because the fit
    needs lengths of at least one token.

Tokenizer
    Qwen/Qwen2.5-0.5B-Instruct, add_special_tokens=False.

Cap
    None in the fit: lengths are untruncated. make_workload clips at max_len when it draws
    requests (256 in the notebook), and the fit does not change that.

Usage (from the repo root)
    python scripts/fit_sharegpt.py path/to/ShareGPT_V3_unfiltered_cleaned_split.json

Prints the counts and the fitted mu and sigma, and writes results/sharegpt/fit.json.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

sys.path.insert(0, ".")
from minivllm.bench.fit_lengths import fit_lognormal  # noqa: E402

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
OUT = Path("results") / "sharegpt" / "fit.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def first_pair(conversation: dict) -> tuple[str, str] | None:
    turns = conversation.get("conversations") or []
    if len(turns) < 2 or turns[0].get("from") != "human" or turns[1].get("from") != "gpt":
        return None
    return turns[0].get("value") or "", turns[1].get("value") or ""


def main(path: Path) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    pairs = [p for p in (first_pair(c) for c in data) if p is not None]
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    prompts = [p for p, _ in pairs]
    replies = [r for _, r in pairs]
    prompt_ids = tokenizer(prompts, add_special_tokens=False)["input_ids"]
    reply_ids = tokenizer(replies, add_special_tokens=False)["input_ids"]
    prompt_lens = [len(ids) for ids in prompt_ids]
    output_lens = [len(ids) for ids in reply_ids]
    kept = [(p, o) for p, o in zip(prompt_lens, output_lens) if p >= 1 and o >= 1]

    prompt_mu, prompt_sigma = fit_lognormal([p for p, _ in kept])
    output_mu, output_sigma = fit_lognormal([o for _, o in kept])
    record = {
        "source": "anon8231489123/ShareGPT_Vicuna_unfiltered / ShareGPT_V3_unfiltered_cleaned_split.json",
        "file": path.name,
        "file_bytes": path.stat().st_size,
        "file_sha256": sha256(path),
        "tokenizer": MODEL_ID,
        "add_special_tokens": False,
        "filter": "first turn human, second turn gpt; empty prompt or reply dropped",
        "cap": None,
        "conversations_in_file": len(data),
        "conversations_with_first_pair": len(pairs),
        "conversations_fitted": len(kept),
        "prompt_mu": prompt_mu, "prompt_sigma": prompt_sigma,
        "output_mu": output_mu, "output_sigma": output_sigma,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(record, indent=2), encoding="utf-8")
    for key, value in record.items():
        print(f"{key}: {value}")
    print("wrote", OUT)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
