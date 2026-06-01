"""
FractalNova-Core · addestramento del tokenizer BPE (da zero).

Crea un tokenizer Byte-Level BPE TUO a partire dal tuo corpus di testo.
Output: <out_dir>/tokenizer.json

Esempio:
    python training/pretrain/tokenizer_train.py \
        --input training/data/corpus \
        --out training/pretrain/artifacts \
        --vocab-size 32000
"""
import argparse
import json
import os
import glob

from tokenizers import ByteLevelBPETokenizer

EOS_TOKEN = "<|endoftext|>"


def iter_texts(paths):
    for path in paths:
        if path.endswith(".jsonl"):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    yield _text_from_obj(obj)
        else:  # .txt / .md / altro testo
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                yield f.read()


def _text_from_obj(obj) -> str:
    if isinstance(obj, str):
        return obj
    if "text" in obj:
        return str(obj["text"])
    if "messages" in obj:
        return "\n".join(str(m.get("content", "")) for m in obj["messages"])
    parts = [str(obj.get(k, "")) for k in ("instruction", "input", "output")]
    return "\n".join(p for p in parts if p)


def collect_paths(input_path):
    if os.path.isfile(input_path):
        return [input_path]
    patterns = ("*.txt", "*.md", "*.jsonl")
    files = []
    for pat in patterns:
        files.extend(glob.glob(os.path.join(input_path, "**", pat), recursive=True))
    return sorted(files)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="file o cartella con il corpus (.txt/.md/.jsonl)")
    ap.add_argument("--out", default="training/pretrain/artifacts")
    ap.add_argument("--vocab-size", type=int, default=32000)
    ap.add_argument("--min-frequency", type=int, default=2)
    args = ap.parse_args()

    paths = collect_paths(args.input)
    if not paths:
        raise SystemExit(f"Nessun file di testo trovato in: {args.input}")
    print(f"[tokenizer] {len(paths)} file di corpus")

    os.makedirs(args.out, exist_ok=True)
    tok = ByteLevelBPETokenizer()
    tok.train_from_iterator(
        iter_texts(paths),
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=[EOS_TOKEN],
    )
    out_path = os.path.join(args.out, "tokenizer.json")
    tok.save(out_path)
    print(f"[tokenizer] salvato in {out_path} | vocab={tok.get_vocab_size()} | eos_id={tok.token_to_id(EOS_TOKEN)}")


if __name__ == "__main__":
    main()
