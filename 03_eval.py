import json
from collections import Counter
from mlx_lm import load, generate

LABELS = ["Benign Status", "DDoS Attack", "Dictionary Attack", "Other Attack"]
model, tok = load("mlx-community/Qwen3-8B-4bit", adapter_path="./adapters")

def head(text):
    return next((l for l in LABELS if text.startswith(l)), "Unknown")

rows = [json.loads(l) for l in open("data/test.jsonl")]
rows = rows[::max(1, len(rows) // 300)][:300]       # spread sample across the test set
conf = Counter()
for r in rows:
    m = r["messages"]
    prompt = tok.apply_chat_template(m[:2], add_generation_prompt=True, tokenize=False, enable_thinking=False)
    pred = head(generate(model, tok, prompt=prompt, max_tokens=30, verbose=False).strip())
    conf[(head(m[2]["content"]), pred)] += 1

print("accuracy:", round(sum(v for (g, p), v in conf.items() if g == p) / len(rows), 3))
for (g, p), v in sorted(conf.items()):
    print(f"true={g:18s} pred={p:18s} {v}")