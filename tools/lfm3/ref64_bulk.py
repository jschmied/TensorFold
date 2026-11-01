"""Float64 reference rows in bulk: for each prompt, the top-8 logits of every row (teacher-forced on the prompt), numpy path.

Usage: python3 -I tools/lfm3/ref64_bulk.py MODEL_DIR OUT.json
"""
import hashlib, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import ref64  # tools/lfm3/ref64.py, beside this file
from tokenizers import Tokenizer

COLORS = "red green blue amber violet".split()

def prompts():
    p = []
    p.append(("short_fact", "The chemical symbol for gold is"))
    p.append(("short_math", "Question: What is 17 times 23? Answer:"))
    p.append(("short_de", "Die Hauptstadt von Österreich ist"))
    p.append(("short_list", "Monday, Tuesday, Wednesday,"))
    p.append(("chat_recipe", "<|im_start|>user\nGive me a three-step recipe for scrambled eggs.<|im_end|>\n<|im_start|>assistant\n"))
    p.append(("chat_code", "<|im_start|>user\nWrite a Python function that reverses a string.<|im_end|>\n<|im_start|>assistant\n"))
    p.append(("json", '{"name": "Ada", "age": 36, "languages": ["English", "French"], "city": "'))
    p.append(("code_c", "#include <stdio.h>\n\nint main(void) {\n    for (int i = 0; i < 10; i++) {\n        printf(\"%d\\n\", i * i);\n    }\n    return"))
    p.append(("poem", "Roses are red,\nViolets are blue,\nSugar is sweet,\nAnd"))
    p.append(("units", "One kilometre is 1000 metres. One metre is 100 centimetres. So one kilometre is"))
    p.append(("table", "| fruit | colour |\n|---|---|\n| banana | yellow |\n| cherry | red |\n| lime |"))
    p.append(("dialog_de", "<|im_start|>user\nErkläre in einem Satz, warum der Himmel blau ist.<|im_end|>\n<|im_start|>assistant\n"))
    # long prompts: past the 256-key sliding window
    inv = " ".join(f"Item {i}: the {COLORS[i % 5]} box holds {7 * i % 23} stones." for i in range(80))
    p.append(("long_items_q7", inv + " Question: how many stones does item 7 hold? Answer: item 7 holds"))
    p.append(("long_items_q61", inv + " Question: what colour is the box of item 61? Answer: the box of item 61 is"))
    log = "\n".join(f"2026-10-{1 + i % 28:02d} 12:{i % 60:02d}:00 sensor-{i % 9} temperature {18 + (i * 7) % 11}.{i % 10} C status {'OK' if i % 13 else 'ALARM'}" for i in range(60))
    p.append(("long_log", log + "\nThe first ALARM in the log above was raised by sensor-"))
    code = "\n".join(f"def f{i}(x):\n    return x * {i} + {i % 7}\n" for i in range(45))
    p.append(("long_code", code + "\n# f12(3) returns"))
    story = " ".join(["Mara walked to the old lighthouse at dawn, carrying a lantern and a map her grandfather had drawn."] * 2 +
                     [f"On day {d} she counted {d * 3} gulls and wrote the number in her notebook beside a small drawing of the sea." for d in range(1, 31)])
    p.append(("long_story", story + " On day 12 she had counted"))
    keys = " ".join(f"The code for room {r} is {(r * 7919) % 10000:04d}." for r in range(100, 160))
    p.append(("long_keys_early", keys + " The code for room 103 is"))
    p.append(("long_keys_late", keys + " The code for room 157 is"))
    chat = "<|im_start|>system\nYou are a careful assistant.<|im_end|>\n" + "".join(
        f"<|im_start|>user\nNote {i}: my favourite number this week is {(i * 37) % 100}.<|im_end|>\n<|im_start|>assistant\nNoted: {(i * 37) % 100}.<|im_end|>\n" for i in range(25))
    p.append(("long_chat", chat + "<|im_start|>user\nWhat was my favourite number in note 4?<|im_end|>\n<|im_start|>assistant\n"))
    return p

def main(model_dir, out):
    tok = Tokenizer.from_file(f"{model_dir}/tokenizer.json")
    bos = json.load(open(f"{model_dir}/config.json"))["bos_token_id"]
    m = ref64.NpModel(model_dir)
    res = json.load(open(out)) if os.path.exists(out) else {}
    for name, text in prompts():
        if name in res:
            continue
        ids = [bos] + tok.encode(text).ids
        t0 = time.time()
        lg = m.logits(ids, len(ids))
        top = np.argsort(-lg, axis=-1)[:, :8]
        vals = np.take_along_axis(lg, top, -1)
        res[name] = {"text": text, "ids": ids, "tokens": len(ids),
                     "top8_ids": top.tolist(), "top8_logits": [[float(v) for v in r] for r in vals],
                     "margin12": [float(r[0] - r[1]) for r in vals],
                     "logits_sha256": hashlib.sha256(np.ascontiguousarray(lg).tobytes()).hexdigest()}
        json.dump(res, open(out + ".tmp", "w")); os.replace(out + ".tmp", out)
        print(f"{name}: {len(ids)} tokens, {time.time() - t0:.0f} s, near-ties(<0.05) {sum(x < 0.05 for x in res[name]['margin12'])}", flush=True)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
