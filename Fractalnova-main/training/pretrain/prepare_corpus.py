"""
FractalNova-Core · preparazione del corpus per il pretraining.

Tokenizza il corpus col TUO tokenizer (tokenizer.json) e scrive i token in file
binari memmappabili (train.bin / val.bin) in stile nanoGPT, piu' meta.json.

Prerequisito: aver gia' creato il tokenizer con tokenizer_train.py.

Esempio:
    python training/pretrain/prepare_corpus.py \
        --input training/data/corpus \
        --artifacts training/pretrain/artifacts \
        --val-ratio 0.1
"""
import argparse
import glob
import json
import os

import numpy as np
from tokenizers import Tokenizer

EOS_TOKEN = "<|endoftext|>"


def _text_from_obj(obj) -> str:
    if isinstance(obj, str):
        return obj
    if "text" in obj:
        return str(obj["text"])
    if "messages" in obj:
        return "\n".join(str(m.get("content", "")) for m in obj["messages"])
    return "\n".join(str(obj.get(k, "")) for k in ("instruction", "input", "output") if obj.get(k))


def iter_texts(paths):
    for path in paths:
        if path.endswith(".jsonl"):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield _text_from_obj(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        else:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                yield f.read()


def collect_paths(input_path):
    if os.path.isfile(input_path):
        return [input_path]
    files = []
    for pat in ("*.txt", "*.md", "*.jsonl"):
        files.extend(glob.glob(os.path.join(input_path, "**", pat), recursive=True))
    return sorted(files)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--artifacts", default="training/pretrain/artifacts")
    ap.add_argument("--val-ratio", type=float, default=0.1)
    args = ap.parse_args()

    tok_path = os.path.join(args.artifacts, "tokenizer.json")
    if not os.path.exists(tok_path):
        raise SystemExit(f"tokenizer.json non trovato in {args.artifacts}. Esegui prima tokenizer_train.py")
    tokenizer = Tokenizer.from_file(tok_path)
    eos_id = tokenizer.token_to_id(EOS_TOKEN)
    vocab_size = tokenizer.get_vocab_size()
    if vocab_size >= 65536:
        raise SystemExit("vocab_size >= 65536: cambia dtype in uint32 (qui usiamo uint16).")

    paths = collect_paths(args.input)
    if not paths:
        raise SystemExit(f"Nessun file di corpus in {args.input}")

    ids = []
    n_docs = 0
    for text in iter_texts(paths):
        if not text:
            continue
        ids.extend(tokenizer.encode(text).ids)
        if eos_id is not None:
            ids.append(eos_id)
        n_docs += 1

    arr = np.array(ids, dtype=np.uint16)
    n_val = int(len(arr) * args.val_ratio)
    val, train = arr[:n_val], arr[n_val:]

    os.makedirs(args.artifacts, exist_ok=True)
    train.tofile(os.path.join(args.artifacts, "train.bin"))
    val.tofile(os.path.join(args.artifacts, "val.bin"))
    with open(os.path.join(args.artifacts, "meta.json"), "w", encoding="utf-8") as f:
        json.dump({"vocab_size": vocab_size, "eos_id": eos_id}, f, ensure_ascii=False, indent=2)

    print(f"[corpus] doc={n_docs} | token totali={len(arr):,} (train {len(train):,}, val {len(val):,})")
    print(f"[corpus] scritti train.bin / val.bin / meta.json in {args.artifacts}")


if __name__ == "__main__":
    main()
