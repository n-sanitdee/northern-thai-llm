"""
evaluate.py
Tests the fine-tuned model on your test set and compares against the base model.
Saves results to outputs/<model>/results.jsonl for analysis.

Usage:
  python scripts/evaluate.py --model typhoon2
  python scripts/evaluate.py --model typhoon2 --compare_base
"""

import argparse
import json
import os
from mlx_lm import load, generate

MODELS = {
    "typhoon2": "models/typhoon2",
    "llama":    "models/llama",
    "seallm":   "models/seallm",
    "qwen":     "models/qwen",
}

def load_test_data(path: str) -> list:
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            examples.append(json.loads(line.strip()))
    return examples

def extract_prompt_and_answer(example: dict):
    """Pull user prompt and expected answer from a JSONL example"""
    messages = example["messages"]
    user_msg = next(m["content"] for m in messages if m["role"] == "user")
    expected = next(m["content"] for m in messages if m["role"] == "assistant")
    system   = next((m["content"] for m in messages if m["role"] == "system"), "")
    return system, user_msg, expected

def run_inference(model, tokenizer, system: str, user_msg: str, max_tokens=256) -> str:
    """Run inference using MLX-LM generate"""
    prompt = f"<|system|>\n{system}\n<|user|>\n{user_msg}\n<|assistant|>\n"
    response = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        verbose=False,
    )
    return response.strip()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",       choices=MODELS.keys(), default="typhoon2")
    parser.add_argument("--test_data",   default="data/test.jsonl")
    parser.add_argument("--max_samples", type=int, default=50,
                        help="How many test examples to evaluate (keep low for speed)")
    parser.add_argument("--compare_base", action="store_true",
                        help="Also run inference on the base model for comparison")
    args = parser.parse_args()

    model_path   = MODELS[args.model]
    adapter_dir  = os.path.join("outputs", args.model, "adapters")
    results_path = os.path.join("outputs", args.model, "results.jsonl")

    os.makedirs(os.path.dirname(results_path), exist_ok=True)

    print(f"Loading fine-tuned model from {model_path}")
    print(f"Using adapter: {adapter_dir}\n")

    # Load fine-tuned model (base + LoRA adapter)
    ft_model, ft_tokenizer = load(model_path, adapter_path=adapter_dir)

    # Optionally load base model for comparison
    base_model, base_tokenizer = None, None
    if args.compare_base:
        print("Loading base model for comparison...")
        base_model, base_tokenizer = load(model_path)

    # Load test data
    test_data = load_test_data(args.test_data)
    test_data = test_data[:args.max_samples]
    print(f"Evaluating on {len(test_data)} examples...\n")

    results = []
    for i, example in enumerate(test_data):
        system, user_msg, expected = extract_prompt_and_answer(example)

        ft_output = run_inference(ft_model, ft_tokenizer, system, user_msg)

        result = {
            "id": i,
            "input":     user_msg,
            "expected":  expected,
            "ft_output": ft_output,
        }

        if base_model:
            base_output = run_inference(base_model, base_tokenizer, system, user_msg)
            result["base_output"] = base_output

        results.append(result)

        # Print progress
        print(f"[{i+1}/{len(test_data)}]")
        print(f"  Input:    {user_msg[:80]}...")
        print(f"  Expected: {expected[:80]}")
        print(f"  FT model: {ft_output[:80]}")
        if base_model:
            print(f"  Base:     {result['base_output'][:80]}")
        print()

    # Save results
    with open(results_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Results saved to {results_path}")
    print(f"\nNext: open this file and review outputs manually,")
    print(f"or run human evaluation with your native speaker annotators.")

if __name__ == "__main__":
    main()
