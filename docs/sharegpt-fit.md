# ShareGPT length fit: provenance

The lognormal length parameters in `minivllm/bench/workload.py` (`LENGTH_DEFAULTS`) and the notebook's
`CONFIG` come from this fit. Rerun it with `scripts/fit_sharegpt.py` on the raw file. Output:
`results/sharegpt/fit.json`.

## Source

- Hugging Face dataset `anon8231489123/ShareGPT_Vicuna_unfiltered`, file `ShareGPT_V3_unfiltered_cleaned_split.json`
- 672,837,942 bytes, SHA-256 `35f0e213ce091ed9b9af2a1f0755e9d39f9ccec34ab281cd4ca60d70f6479ba4`
- 94,145 conversations in the file

## Method

- Filter: a conversation counts when its first turn is `human` and its second is `gpt`. Prompt = first
  human message, output = first assistant reply. Empty prompts or replies are dropped (the fit needs at least one token).
- Tokenizer: `Qwen/Qwen2.5-0.5B-Instruct`, `add_special_tokens=False`.
- Cap: none. The fit is untruncated. `make_workload` clips lengths at `max_len` (256 in the notebook).
- Fit: `minivllm/bench/fit_lengths.py::fit_lognormal`. Mean and population standard deviation of the log of the lengths.

## Result

| | Reproduced here | In `workload.py` |
|---|---|---|
| conversations fitted | 58,766 (of 58,787 with a human/gpt first pair) | 58,778 (note in the file) |
| prompt mu | 3.835436 | 3.835046 |
| prompt sigma | 1.749920 | 1.750131 |
| output mu | 5.183726 | 5.183535 |
| output sigma | 1.160612 | 1.160675 |

## Status

**Not reproduced exactly.** The fit matches to about 0.01% on each parameter, and the conversation count is 12 lower
than the original note. The original filter is not recorded, so the difference is most likely a small difference in
how the first pair or empty turns were handled. The constants in `workload.py` are kept unchanged.
