"""
download_model.py
Downloads a model from Hugging Face into the local models/ directory.
Usage: python scripts/download_model.py --model typhoon2
"""

import argparse
from huggingface_hub import snapshot_download
import os

MODELS = {
    "typhoon2": "scb10x/llama3.1-typhoon2-8b-instruct",
    "llama":    "meta-llama/Llama-3.1-8B-Instruct",
    "seallm":   "SeaLLMs/SeaLLMs-v3-7B-Chat",
    "qwen":     "Qwen/Qwen2.5-7B-Instruct",
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS.keys(), required=True,
                        help="Which model to download")
    parser.add_argument("--output_dir", default="models",
                        help="Where to save the model")
    args = parser.parse_args()

    model_id = MODELS[args.model]
    local_dir = os.path.join(args.output_dir, args.model)

    print(f"Downloading {model_id} → {local_dir}")
    print("This may take a while depending on your connection (~15GB)...")

    snapshot_download(
        repo_id=model_id,
        local_dir=local_dir,
        ignore_patterns=["*.pt", "original/*"],  # skip redundant formats
    )

    print(f"Done. Model saved to: {local_dir}")

if __name__ == "__main__":
    main()
