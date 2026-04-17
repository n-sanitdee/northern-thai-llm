"""
finetune.py
Fine-tunes a model using LoRA via MLX-LM on Apple Silicon.
Designed for 16GB RAM — uses conservative settings to avoid OOM.

Usage:
  python scripts/finetune.py --model typhoon2
  python scripts/finetune.py --model typhoon2 --iters 500 --batch_size 2

After training, the LoRA adapter is saved to outputs/<model>/adapters/
The adapter is small (~50MB) and can be pushed to GitHub unlike full weights.
"""

import argparse
import subprocess
import os
import json
from datetime import datetime

# ── model registry ────────────────────────────────────────────────────────────

MODELS = {
    "typhoon2": "models/typhoon2",
    "llama":    "models/llama",
    "seallm":   "models/seallm",
    "qwen":     "models/qwen",
}

# ── LoRA config ───────────────────────────────────────────────────────────────
# Conservative settings for 16GB RAM M4
# If you get OOM errors: reduce batch_size to 1, reduce max_seq_length to 512

LORA_CONFIG = {
    "num_layers": 8,        # number of layers to apply LoRA to (fewer = less RAM)
    "lora_parameters": {
        "rank": 8,          # LoRA rank (lower = less RAM, less expressive)
        "alpha": 16,        # scaling factor (usually 2x rank)
        "dropout": 0.05,
        "scale": 10.0,
    }
}

def save_config(output_dir: str, args: argparse.Namespace):
    """Save run config for reproducibility"""
    config = {
        "model": args.model,
        "model_path": MODELS[args.model],
        "iters": args.iters,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "max_seq_length": args.max_seq_length,
        "lora_config": LORA_CONFIG,
        "train_data": args.data_dir + "/train.jsonl",
        "valid_data": args.data_dir + "/valid.jsonl",
        "timestamp": datetime.now().isoformat(),
    }
    config_path = os.path.join(output_dir, "run_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Config saved → {config_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",          choices=MODELS.keys(), default="typhoon2")
    parser.add_argument("--iters",          type=int,   default=300,
                        help="Training iterations (300–500 is a good start for 500 examples)")
    parser.add_argument("--batch_size",     type=int,   default=2,
                        help="Batch size — reduce to 1 if you get memory errors")
    parser.add_argument("--lr",             type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--max_seq_length", type=int,   default=512,
                        help="Max token length — reduce to 256 if memory is tight")
    parser.add_argument("--data_dir",       default="data",
                        help="Directory containing train.jsonl and valid.jsonl")
    parser.add_argument("--steps_per_eval", type=int,   default=50,
                        help="How often to evaluate on validation set")
    parser.add_argument("--save_every",     type=int,   default=100,
                        help="Save adapter checkpoint every N steps")
    args = parser.parse_args()

    model_path = MODELS[args.model]
    output_dir = os.path.join("outputs", args.model)
    adapter_dir = os.path.join(output_dir, "adapters")
    os.makedirs(adapter_dir, exist_ok=True)

    # Save lora config file that mlx_lm expects
    lora_config_path = os.path.join(output_dir, "lora_config.yaml")
    with open(lora_config_path, "w") as f:
        f.write(f"num_layers: {LORA_CONFIG['num_layers']}\n")
        f.write(f"lora_parameters:\n")
        for k, v in LORA_CONFIG["lora_parameters"].items():
            f.write(f"  {k}: {v}\n")

    save_config(output_dir, args)

    print(f"\nStarting LoRA fine-tuning")
    print(f"  Model:      {model_path}")
    print(f"  Adapter:    {adapter_dir}")
    print(f"  Iterations: {args.iters}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Max seq:    {args.max_seq_length}")
    print(f"\n  Tip: Watch RAM usage in Activity Monitor.")
    print(f"  If training crashes, restart with --batch_size 1 --max_seq_length 256\n")

    # Build the mlx_lm.lora command
    cmd = [
        "python", "-m", "mlx_lm", "lora",
        "--model",          model_path,
        "--train",
        "--data",           args.data_dir,
        "--iters",          str(args.iters),
        "--batch-size",     str(args.batch_size),
        "--learning-rate",  str(args.lr),
        "--max-seq-length", str(args.max_seq_length),
        "--steps-per-eval", str(args.steps_per_eval),
        "--save-every",     str(args.save_every),
        "--adapter-path",   adapter_dir,
        "--num-layers",     str(LORA_CONFIG["num_layers"]),
    ]

    print("Running:", " ".join(cmd), "\n")

    # Run training
    result = subprocess.run(cmd)

    if result.returncode == 0:
        print(f"\nTraining complete.")
        print(f"Adapter saved to: {adapter_dir}")
        print(f"\nNext steps:")
        print(f"  Test:  python scripts/evaluate.py --model {args.model}")
        print(f"  Fuse:  python scripts/finetune.py --fuse --model {args.model}  (optional, for deployment)")
    else:
        print("\nTraining failed. Common fixes:")
        print("  - OOM: try --batch_size 1 --max_seq_length 256")
        print("  - Model not found: run python scripts/download_model.py --model", args.model)

if __name__ == "__main__":
    main()
