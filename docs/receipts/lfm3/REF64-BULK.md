# Float64 reference rows in bulk: LFM3-1.7B MLX 4-bit

- Checkpoint: [LiquidAI/LFM3-1.7B-MLX-4bit](https://huggingface.co/LiquidAI/LFM3-1.7B-MLX-4bit) at revision `d0ca73dcfa70ce18733754a492cdc1fb0123262d`, dequantized exactly (MLX affine).
- Reference: `tools/lfm3/ref64.py`'s numpy float64 forward (no torch), driven by `tools/lfm3/ref64_bulk.py`; numpy 2.2.6, tokenizers 0.23.3, on a GB10's CPU, 2026-11-01.
- 20 prompts, 7253 rows: 12 short ones (prose, maths, German, lists, chat turns, JSON, C, a table) and 8 of 668 to 1,459 tokens that cross the 256-key sliding window (an inventory, a sensor log, generated code, a story, keyed codes asked early and late, a 25-turn chat).
- Per prompt: the text, its ids (BOS, then the tokenizer's encoding), every row's top-8 ids and float64 logits, the top-1 to top-2 margin, and a sha256 of the full float64 logits.
- 52 rows have a top-1 margin under 0.05: an engine may pick either token there without being wrong.

| prompt | tokens | margin < 0.05 |
|---|---|---|
| short_fact | 7 | 1 |
| short_math | 13 | 0 |
| short_de | 6 | 0 |
| short_list | 7 | 0 |
| chat_recipe | 21 | 0 |
| chat_code | 18 | 0 |
| json | 26 | 2 |
| code_c | 42 | 0 |
| poem | 18 | 0 |
| units | 24 | 1 |
| table | 20 | 0 |
| dialog_de | 24 | 1 |
| long_items_q7 | 994 | 8 |
| long_items_q61 | 998 | 7 |
| long_log | 1459 | 7 |
| long_code | 683 | 1 |
| long_story | 802 | 4 |
| long_keys_early | 668 | 5 |
| long_keys_late | 668 | 5 |
| long_chat | 755 | 10 |

Reproduce: `python3 -I tools/lfm3/ref64_bulk.py MODEL_DIR out.json` (about 10 minutes on 12 threads; 24 GB of memory is enough).
