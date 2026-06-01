"""
FractalNova-Pro · preparazione dataset per il fine-tuning (multilingua + multi-task).

Trasforma esempi grezzi in conversazioni chat pronte per l'SFT. Lo stesso modello
impara a: scrivere, continuare, umanizzare, creare titoli/sinossi, generare SEO e
tradurre -- nella STESSA lingua della richiesta (IT/EN/ES/FR/DE/PT/...).

Schemi di input accettati (uno o piu file .jsonl):
  A) {"instruction": "...", "input": "...", "output": "...", "lang": "it", "task": "write"}
  B) {"messages": [{"role": "...", "content": "..."}, ...]}

Output: train.jsonl / val.jsonl con campo "messages".

Esempio:
    python training/prepare_dataset.py \
        --inputs training/data/sample_books_it.jsonl training/data/sample_books_multi.jsonl \
        --out-dir training/data --val-ratio 0.1
"""
import argparse
import glob
import json
import os
import random

SYSTEM_PROMPT = (
    "Sei FractalNova, autore ed editor professionista. Scrivi in modo naturale e umano, "
    "con voce e ritmo curati. Rispondi SEMPRE nella stessa lingua della richiesta. "
    "Sai scrivere e continuare narrativa, umanizzare e correggere testi, proporre titoli e "
    "sinossi, generare metadati SEO e tradurre, mantenendo qualita' editoriale."
)


def load_rows(paths):
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            for ln, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"[warn] {path}:{ln} JSON non valido: {e}")


def to_messages(row):
    # Schema B: gia' in formato chat
    if "messages" in row and isinstance(row["messages"], list):
        msgs = row["messages"]
        if msgs and msgs[0].get("role") != "system":
            msgs = [{"role": "system", "content": SYSTEM_PROMPT}] + msgs
        return msgs
    # Schema A: instruction/input/output
    instruction = (row.get("instruction") or "").strip()
    user_input = (row.get("input") or "").strip()
    output = (row.get("output") or "").strip()
    if not instruction or not output:
        return None
    user = instruction if not user_input else f"{instruction}\n\n{user_input}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
        {"role": "assistant", "content": output},
    ]


def expand_inputs(inputs):
    paths = []
    for item in inputs:
        if os.path.isdir(item):
            paths.extend(sorted(glob.glob(os.path.join(item, "**", "*.jsonl"), recursive=True)))
        else:
            paths.append(item)
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="file .jsonl o cartelle")
    ap.add_argument("--out-dir", default="training/data")
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    paths = expand_inputs(args.inputs)
    if not paths:
        raise SystemExit("Nessun file .jsonl trovato negli input forniti.")

    examples, skipped, seen = [], 0, set()
    for row in load_rows(paths):
        msgs = to_messages(row)
        if not msgs:
            skipped += 1
            continue
        key = json.dumps(msgs, ensure_ascii=False)
        if key in seen:  # dedup esatto
            continue
        seen.add(key)
        examples.append({"messages": msgs})

    if not examples:
        raise SystemExit("Nessun esempio valido prodotto. Controlla lo schema dei dati.")

    random.Random(args.seed).shuffle(examples)
    n_val = max(1, int(len(examples) * args.val_ratio)) if len(examples) > 10 else 0
    val, train = examples[:n_val], examples[n_val:]

    os.makedirs(args.out_dir, exist_ok=True)
    train_path = os.path.join(args.out_dir, "train.jsonl")
    val_path = os.path.join(args.out_dir, "val.jsonl")
    _dump(train, train_path)
    if val:
        _dump(val, val_path)

    print(f"[dataset] esempi: {len(examples)} (train {len(train)}, val {len(val)}), scartati {skipped}")
    print(f"[dataset] scritto: {train_path}" + (f" e {val_path}" if val else " (val non creato: dataset piccolo)"))


def _dump(rows, path):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
