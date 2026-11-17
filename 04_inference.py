import json
from mlx_lm import load, generate

LABELS = ["Benign Status", "DDoS Attack", "Dictionary Attack", "Other Attack"]
SYSTEM = open("data/system_prompt.txt").read()
COLS = json.load(open("data/feature_cols.json"))

model, tok = load("mlx-community/Qwen3-8B-4bit", adapter_path="./adapters")

def classify_flow(features: dict) -> dict:
    text = ", ".join(f"{c}={float(features.get(c, 0)):.4g}" for c in COLS)   # same format as training
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": "Classify this network flow:\n" + text}]
    prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
    out = generate(model, tok, prompt=prompt, max_tokens=30, verbose=False).strip()
    status = next((l for l in LABELS if out.startswith(l)), "Unknown")
    return {"status": status, "raw": out}

if __name__ == "__main__":
    demo = {c: 0 for c in COLS}
    demo.update({"Rate": 45.2, "IAT": 0.02, "SSH": 1, "TCP": 1})
    print(classify_flow(demo))
    # In your chatbot: if result["status"] != "Benign Status": send_alert(result)