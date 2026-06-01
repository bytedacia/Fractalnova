"""
FractalNova-Pro · inferenza interattiva del modello fine-tunato (multilingua, multi-task).

Esempi:
    # modello gia' unito (merged)
    python training/infer.py --model training/outputs/fractalnova-pro-merged \
        --prompt "Scrivi l'incipit di un giallo ambientato a Trieste."

    # base + adapter in 4-bit (senza merge, sta in 16GB)
    python training/infer.py --base Qwen/Qwen3-4B \
        --adapter training/outputs/fractalnova-qlora --load-4bit \
        --prompt "Write the synopsis of a sci-fi novel about a lighthouse on Mars."
"""
import argparse
import sys
import os

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import SYSTEM_PROMPT, load_model  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="directory di un modello pronto (es. merged)")
    ap.add_argument("--base", help="modello base (se usi un adapter)")
    ap.add_argument("--adapter", help="adapter LoRA da applicare al base")
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--system", default=SYSTEM_PROMPT)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--trust-remote-code", action="store_true")
    args = ap.parse_args()

    model, tokenizer = load_model(
        model_path=args.model, base=args.base, adapter=args.adapter,
        load_4bit=args.load_4bit, trust_remote_code=args.trust_remote_code,
    )

    messages = [
        {"role": "system", "content": args.system},
        {"role": "user", "content": args.prompt},
    ]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        out = model.generate(
            inputs, max_new_tokens=args.max_new_tokens, do_sample=True,
            temperature=args.temperature, top_p=args.top_p,
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(out[0, inputs.shape[-1]:], skip_special_tokens=True)
    print(text.strip())


if __name__ == "__main__":
    main()
