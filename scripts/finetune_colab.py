"""
finetune_colab.py

Fine-tuning script using Unsloth + LoRA.
Works on both Google Colab and Kaggle (NVIDIA GPU required).
Uses 4-bit quantization to fit 7-8B models in 15GB VRAM.

Usage:
    python scripts/finetune_colab.py --model typhoon2
    python scripts/finetune_colab.py --model typhoon2 --iters 300 --batch_size 4

Models available:
    typhoon2  — scb10x/llama3.1-typhoon2-8b-instruct  (Thai-specific, primary target)
    seallm    — SeaLLMs/SeaLLMs-v3-7B-Chat             (SEA-specific)
    qwen      — Qwen/Qwen2.5-7B-Instruct               (Asian multilingual)
    llama     — meta-llama/Llama-3.1-8B-Instruct       (General, requires HF login)

Outputs:
    outputs/<model>/adapters/      — LoRA adapter weights (~50-100MB)
    outputs/<model>/run_config.json — hyperparameters for reproducibility
"""

import argparse
import json
import os
import torch
from datetime import datetime

# ── Model registry ────────────────────────────────────────────────────────────

MODELS = {
    "typhoon2": "scb10x/llama3.1-typhoon2-8b-instruct",
    "seallm":   "SeaLLMs/SeaLLMs-v3-7B-Chat",
    "qwen":     "Qwen/Qwen2.5-7B-Instruct",
    "llama":    "meta-llama/Llama-3.1-8B-Instruct",
}

# ── Data formatter ────────────────────────────────────────────────────────────

def format_example(example):
    """
    Converts the messages format from your JSONL into a single
    training string the model can learn from.

    Input format (from prepare_data.py):
        {"messages": [
            {"role": "system",    "content": "..."},
            {"role": "user",      "content": "..."},
            {"role": "assistant", "content": "..."}
        ]}

    Output format:
        "<|system|>\n...\n<|user|>\n...\n<|assistant|>\n...\n"
    """
    messages = example["messages"]
    text = ""
    for m in messages:
        if m["role"] == "system":
            text += f"<|system|>\n{m['content']}\n"
        elif m["role"] == "user":
            text += f"<|user|>\n{m['content']}\n"
        elif m["role"] == "assistant":
            text += f"<|assistant|>\n{m['content']}\n"
    return {"text": text}

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune an LLM on Northern Thai dialect data using LoRA"
    )
    parser.add_argument(
        "--model",
        choices=MODELS.keys(),
        default="typhoon2",
        help="Which model to fine-tune"
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=300,
        help="Number of training steps (300 is good for ~500 examples)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size per GPU (reduce to 2 if you get OOM errors)"
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=512,
        help="Maximum token length (reduce to 256 if OOM)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=16,
        help="LoRA rank (higher = more expressive but more memory)"
    )
    parser.add_argument(
        "--data_dir",
        default="data",
        help="Directory containing train.jsonl and valid.jsonl"
    )
    parser.add_argument(
        "--output_dir",
        default="outputs",
        help="Where to save adapter and checkpoints"
    )
    args = parser.parse_args()

    # ── Paths ─────────────────────────────────────────────────────────────────

    model_id    = MODELS[args.model]
    run_dir     = os.path.join(args.output_dir, args.model)
    adapter_dir = os.path.join(run_dir, "adapters")
    os.makedirs(adapter_dir, exist_ok=True)

    # ── System check ──────────────────────────────────────────────────────────

    print("="*60)
    print(f"Fine-tuning: {model_id}")
    print("="*60)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "No GPU detected. "
            "On Kaggle: Session options → Accelerator → GPU T4 x2. "
            "On Colab: Runtime → Change runtime type → T4 GPU."
        )

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"GPU count: {torch.cuda.device_count()}")

    # ── Check data files ──────────────────────────────────────────────────────

    train_path = os.path.join(args.data_dir, "train.jsonl")
    valid_path = os.path.join(args.data_dir, "valid.jsonl")

    for path in [train_path, valid_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Data file not found: {path}\n"
                "Run prepare_data.py first:\n"
                "  python scripts/prepare_data.py --task multitask "
                "--input data/Master_Dataset.xlsx"
            )

    # ── Save run config ───────────────────────────────────────────────────────

    config = {
        "model_id":       model_id,
        "model_key":      args.model,
        "iters":          args.iters,
        "batch_size":     args.batch_size,
        "max_seq_length": args.max_seq_length,
        "learning_rate":  args.lr,
        "lora_rank":      args.lora_rank,
        "lora_alpha":     args.lora_rank * 2,
        "quantization":   "4bit",
        "gpu":            torch.cuda.get_device_name(0),
        "timestamp":      datetime.now().isoformat(),
        "train_data":     train_path,
        "valid_data":     valid_path,
    }

    config_path = os.path.join(run_dir, "run_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"\nConfig saved → {config_path}")

    # ── Load model ────────────────────────────────────────────────────────────

    print(f"\nLoading {model_id}...")
    print("(This may take 5-10 minutes for the first download)")

    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_id,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,      # 4-bit quantization: fits 8B model in ~5GB VRAM
        dtype=None,             # auto-detect best dtype for your GPU
    )

    # ── Apply LoRA ────────────────────────────────────────────────────────────

    print("\nApplying LoRA adapters...")

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=args.lora_rank * 2,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing=True,   # saves VRAM during backprop
    )

    # Count trainable parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable:,} / {total:,} "
          f"({100 * trainable / total:.3f}%)")

    # ── Load dataset ──────────────────────────────────────────────────────────

    print("\nLoading dataset...")

    from datasets import load_dataset

    dataset = load_dataset(
        "json",
        data_files={
            "train":      train_path,
            "validation": valid_path,
        }
    )
    dataset = dataset.map(format_example)

    print(f"  Train examples:      {len(dataset['train'])}")
    print(f"  Validation examples: {len(dataset['validation'])}")

    # Preview one example
    print("\nSample training text:")
    print(dataset["train"][0]["text"][:300])

    # ── Training arguments ────────────────────────────────────────────────────

    from transformers import TrainingArguments
    from trl import SFTTrainer

    training_args = TrainingArguments(
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,      # effective batch = batch_size * 4
        max_steps=args.iters,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",         # gradually reduce LR during training

        # Mixed precision — use bf16 if supported, else fp16
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),

        # Logging and evaluation
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,

        # Saving checkpoints
        save_strategy="steps",
        save_steps=50,                      # save every 50 steps
        save_total_limit=3,                 # keep last 3 checkpoints

        # Output
        output_dir=run_dir,
        report_to="none",                   # disable wandb/tensorboard
        seed=42,
    )

    # ── Trainer ───────────────────────────────────────────────────────────────

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        args=training_args,
    )

    # ── Train ─────────────────────────────────────────────────────────────────

    print(f"\nStarting training for {args.iters} steps...")
    print("You will see loss printed every 10 steps.")
    print("Loss should decrease over time — if it stays flat something is wrong.\n")

    trainer_stats = trainer.train()

    # Print training summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Total steps:    {trainer_stats.global_step}")
    print(f"Training loss:  {trainer_stats.training_loss:.4f}")
    runtime_mins = trainer_stats.metrics.get('train_runtime', 0) / 60
    print(f"Time taken:     {runtime_mins:.1f} minutes")

    # ── Save adapter ──────────────────────────────────────────────────────────

    print(f"\nSaving adapter to {adapter_dir}...")
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    # Verify saved files
    print("\nAdapter files saved:")
    for f in os.listdir(adapter_dir):
        size_mb = os.path.getsize(os.path.join(adapter_dir, f)) / (1024 * 1024)
        print(f"  {f}: {size_mb:.1f} MB")

    print("\n" + "="*60)
    print("DONE")
    print("="*60)
    print(f"Adapter saved to: {adapter_dir}")
    print("\nNext steps:")
    print("  1. Save adapter to Drive/output before session ends")
    print("  2. Run evaluate.py to compare base vs fine-tuned model")
    print("  3. Repeat for other models (seallm, qwen, llama)")

if __name__ == "__main__":
    main()