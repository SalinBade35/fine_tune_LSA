import json, random, sys
from pathlib import Path
import numpy as np
import pandas as pd

# ---------- CELL 1: settings ----------
ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "Downloads" / "CSV"
OUT = Path("data")
TARGET_PER_CLASS = 4000                            # rows per class before train/valid/test split
CAP = {"train": 3000, "valid": 400, "test": 800}   # max rows per class in each split
SEED = 42

LABELS = ["Benign Status", "DDoS Attack", "Dictionary Attack", "Other Attack"]

# ---------- CELL 2: folder name -> class + type ----------
def map_folder(name):
    n = name.lower()
    pretty = name.replace("-", " ").replace("_", " ")
    if n.startswith("benign"):
        return "Benign Status", "normal traffic"
    if n.startswith("ddos") or n.startswith("dos-"):
        return "DDoS Attack", pretty
    if n.startswith("dictionary"):
        return "Dictionary Attack", "password guessing"
    return "Other Attack", pretty

# ---------- CELL 3: find folders and count per class ----------
folders = [p for p in sorted(ROOT.iterdir()) if p.is_dir() and list(p.rglob("*.csv"))]
mapping = {f: map_folder(f.name) for f in folders}
folders_per_class = {l: sum(1 for v in mapping.values() if v[0] == l) for l in LABELS}
print("Folders per class:", folders_per_class)
if min(folders_per_class.values()) == 0:
    sys.exit("A class has no folders. Check the ROOT path: " + str(ROOT))

# ---------- CELL 4: read files, sample rows, assign splits ----------
rng = random.Random(SEED)
rows = {"train": [], "valid": [], "test": []}      # items: (label, type, text)
feature_cols = None

for folder in folders:
    label, sub = mapping[folder.name and folder]
    files = sorted(folder.rglob("*.csv"))
    per_folder = max(1, TARGET_PER_CLASS // folders_per_class[label])
    per_file = max(1, per_folder // len(files))
    by_file = len(files) >= 3                       # split by file to avoid near-duplicate leakage
    print(f"{folder.name:32s} -> {label:18s} files={len(files):3d} rows/file={per_file}")

    for i, f in enumerate(files):
        df = pd.read_csv(f)
        if all(str(c).replace(".", "", 1).replace("-", "", 1).isdigit() for c in df.columns):
            sys.exit(f"{f} looks like it has NO header row. Add the column names first.")
        df = df.drop(columns=[c for c in df.columns if str(c).strip().lower() in ("label", "class")])
        df = df.select_dtypes("number").replace([np.inf, -np.inf], np.nan).fillna(0)
        if feature_cols is None:
            feature_cols = list(df.columns)
            print(f"  -> {len(feature_cols)} feature columns: {feature_cols[:5]} ... {feature_cols[-3:]}")
        elif list(df.columns) != feature_cols:
            print(f"  WARNING: columns differ in {f.name}; aligning to the first file")
        df = df.reindex(columns=feature_cols, fill_value=0)
        df = df.sample(n=min(per_file, len(df)), random_state=SEED)

        for vals in df.to_numpy():
            text = ", ".join(f"{c}={v:.4g}" for c, v in zip(feature_cols, vals))
            if by_file:
                split = "test" if i == len(files) - 1 else "valid" if i == len(files) - 2 else "train"
            else:
                r = rng.random()
                split = "train" if r < .8 else "valid" if r < .9 else "test"
            rows[split].append((label, sub, text))

# ---------- CELL 5: cap per class, write JSONL ----------
SYSTEM = ("You are a network security analyst. Given numeric network-flow features, reply with "
          f"exactly one status ({', '.join(repr(l) for l in LABELS)}) followed by the traffic type.")

OUT.mkdir(exist_ok=True)
for split, items in rows.items():
    rng.shuffle(items)
    kept, seen = [], {}
    for l, sub, text in items:
        if seen.get(l, 0) < CAP[split]:
            seen[l] = seen.get(l, 0) + 1
            kept.append((l, sub, text))
    with open(OUT / f"{split}.jsonl", "w") as fh:
        for l, sub, text in kept:
            fh.write(json.dumps({"messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": "Classify this network flow:\n" + text},
                {"role": "assistant", "content": f"{l}. Type: {sub}"},
            ]}) + "\n")
    print(f"{split:6s} {len(kept):6d}  {seen}")

(OUT / "system_prompt.txt").write_text(SYSTEM)
(OUT / "feature_cols.json").write_text(json.dumps(feature_cols))
print("Done. Files are in", OUT.resolve())