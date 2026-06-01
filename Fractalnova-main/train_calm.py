import argparse
import json
from calm import CALMConfig, train_calm


def main():
    parser = argparse.ArgumentParser(description="Train FractalNova CALM bridge")
    parser.add_argument("--anchor", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--augmenting", type=str, default="HuggingFaceTB/SmolLM2-1.7B-Instruct")
    parser.add_argument("--bridge-layers", type=int, nargs="+", default=[8, 16, 24])
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--output-dir", type=str, default="calm_output")
    parser.add_argument("--train-file", type=str, required=True, help="JSONL file with 'text' field per line")
    parser.add_argument("--eval-file", type=str, default=None, help="Optional JSONL eval file")
    parser.add_argument("--config", type=str, default=None, help="JSON config file (overrides CLI)")

    args = parser.parse_args()

    if args.config:
        with open(args.config) as f:
            config_dict = json.load(f)
        config = CALMConfig(**config_dict)
    else:
        config = CALMConfig(
            anchor_model_name=args.anchor,
            augmenting_model_name=args.augmenting,
            bridge_layers=args.bridge_layers,
            learning_rate=args.lr,
            batch_size=args.batch_size,
            grad_accumulation_steps=args.grad_accum,
            max_steps=args.max_steps,
            output_dir=args.output_dir,
        )

    print("=" * 60)
    print("FractalNova CALM Training")
    print("=" * 60)
    print(f"Anchor:      {config.anchor_model_name}")
    print(f"Augmenting:  {config.augmenting_model_name}")
    print(f"Bridge:      {len(config.bridge_layers)} layers @ {config.bridge_layers}")
    print(f"Max steps:   {config.max_steps}")
    print(f"Batch:       {config.batch_size} x {config.grad_accumulation_steps} accum")
    print("=" * 60)

    print("Loading training data...")
    train_texts = []
    with open(args.train_file) as f:
        for line in f:
            data = json.loads(line)
            text = data.get("text", "")
            if text:
                train_texts.append(text)
    print(f"Loaded {len(train_texts)} training examples")

    eval_texts = None
    if args.eval_file:
        eval_texts = []
        with open(args.eval_file) as f:
            for line in f:
                data = json.loads(line)
                text = data.get("text", "")
                if text:
                    eval_texts.append(text)
        print(f"Loaded {len(eval_texts)} eval examples")

    model = train_calm(train_texts, eval_texts, config)
    print(f"Training complete. Bridge saved to {config.output_dir}")


if __name__ == "__main__":
    main()
