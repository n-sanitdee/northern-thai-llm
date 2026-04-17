"""
prepare_data.py
Converts your Excel dataset into JSONL format for MLX-LM fine-tuning.
Supports multiple task formats: translation, intent classification, reverse translation.

Usage: python scripts/prepare_data.py --task translation --split 0.8
"""

import pandas as pd
import json
import argparse
import random
import os

# ── prompt templates ──────────────────────────────────────────────────────────

def make_translation_example(ntd: str, std: str) -> dict:
    """NTD → Standard Thai translation"""
    return {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant that translates Northern Thai dialect "
                    "(ภาษาเหนือ / คำเมือง) into Standard Thai (ภาษาไทยกลาง). "
                    "Preserve the meaning and tone of the original sentence."
                )
            },
            {
                "role": "user",
                "content": f"แปลประโยคภาษาเหนือนี้เป็นภาษาไทยกลาง:\n{ntd}"
            },
            {
                "role": "assistant",
                "content": std
            }
        ]
    }

def make_reverse_translation_example(ntd: str, std: str) -> dict:
    """Standard Thai → NTD (reverse direction, same data)"""
    return {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant that translates Standard Thai (ภาษาไทยกลาง) "
                    "into Northern Thai dialect (ภาษาเหนือ / คำเมือง). "
                    "Use authentic Northern Thai vocabulary and particles."
                )
            },
            {
                "role": "user",
                "content": f"แปลประโยคภาษาไทยกลางนี้เป็นภาษาเหนือ:\n{std}"
            },
            {
                "role": "assistant",
                "content": ntd
            }
        ]
    }

def make_intent_example(ntd: str, intent: str) -> dict:
    """Intent classification"""
    return {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a linguistic analyst specializing in Northern Thai dialect. "
                    "Classify the communicative intent of Northern Thai sentences. "
                    "Choose from: complaint, joke, invitation, sarcasm, question, "
                    "agreement, advice, information."
                )
            },
            {
                "role": "user",
                "content": f"ประโยคนี้มีเจตนาสื่อสารอะไร:\n{ntd}"
            },
            {
                "role": "assistant",
                "content": intent
            }
        ]
    }

def make_multitask_example(row: pd.Series) -> list:
    """
    From one row, generate multiple training examples.
    Always generates translation pair. Adds intent if annotated.
    """
    examples = []
    ntd = str(row.get("Text_Northern", "")).strip()
    std = str(row.get("Text_Standard_Thai", "")).strip()

    if not ntd or not std or ntd == "nan" or std == "nan":
        return []

    # Forward translation
    examples.append(make_translation_example(ntd, std))

    # Reverse translation (free augmentation from same data)
    examples.append(make_reverse_translation_example(ntd, std))

    # Intent classification if available
    intent = str(row.get("Intent", "")).strip()
    if intent and intent != "nan":
        examples.append(make_intent_example(ntd, intent))

    return examples

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",    default="data/Master_Dataset.xlsx",
                        help="Path to your Excel dataset")
    parser.add_argument("--sheet",    default="natural",
                        help="Sheet name to use")
    parser.add_argument("--task",
                        choices=["translation", "reverse", "intent", "multitask"],
                        default="multitask",
                        help="Which task format to generate")
    parser.add_argument("--split",    type=float, default=0.8,
                        help="Train/valid split ratio (default 0.8)")
    parser.add_argument("--seed",     type=int,   default=42)
    parser.add_argument("--output_dir", default="data")
    args = parser.parse_args()

    random.seed(args.seed)

    print(f"Loading {args.input} (sheet: {args.sheet})...")
    df = pd.read_excel(args.input, sheet_name=args.sheet)
    print(f"  Loaded {len(df)} rows")

    # Build examples
    examples = []
    skipped = 0

    for _, row in df.iterrows():
        ntd = str(row.get("Text_Northern", "")).strip()
        std = str(row.get("Text_Standard_Thai", "")).strip()

        if not ntd or not std or ntd == "nan" or std == "nan":
            skipped += 1
            continue

        if args.task == "translation":
            examples.append(make_translation_example(ntd, std))
        elif args.task == "reverse":
            examples.append(make_reverse_translation_example(ntd, std))
        elif args.task == "intent":
            intent = str(row.get("Intent", "")).strip()
            if intent and intent != "nan":
                examples.append(make_intent_example(ntd, intent))
        elif args.task == "multitask":
            examples.extend(make_multitask_example(row))

    print(f"  Generated {len(examples)} examples ({skipped} rows skipped)")

    # Shuffle and split
    random.shuffle(examples)
    n_train = int(len(examples) * args.split)
    n_valid = int(len(examples) * ((1 - args.split) / 2))

    train_data = examples[:n_train]
    valid_data = examples[n_train:n_train + n_valid]
    test_data  = examples[n_train + n_valid:]

    print(f"  Train: {len(train_data)} | Valid: {len(valid_data)} | Test: {len(test_data)}")

    # Write JSONL files
    def write_jsonl(data, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"  Wrote {len(data)} examples → {path}")

    write_jsonl(train_data, os.path.join(args.output_dir, "train.jsonl"))
    write_jsonl(valid_data, os.path.join(args.output_dir, "valid.jsonl"))
    write_jsonl(test_data,  os.path.join(args.output_dir, "test.jsonl"))

    print("\nDone. Data ready for fine-tuning.")

if __name__ == "__main__":
    main()
