%%writefile scripts/finetune_hf.py

"""
finetune_hf.py

Pure HuggingFace + PEFT fine-tuning with integrated checkpoint evaluation.
Fixes applied:
  - Direction flag: evaluates NTD->STD and STD->NTD separately
  - max_new_tokens increased to 256 for complete translations
  - Whitespace normalization before ChrF scoring
  - Prints only 10 sample translations in cell output, saves all 100 to file
  - Dataset quality check: warns about rows where Gold == NTD (untranslated)
  - Label masking, deterministic generation, ChrF metric, early stopping

Usage:
    python scripts/finetune_hf.py --model typhoon2 --task multitask
    python scripts/finetune_hf.py --model typhoon2 --task translation
"""

import argparse
import json
import math
import os
import random
import re
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    TrainerCallback,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from datasets import load_dataset
from sacrebleu.metrics import CHRF as ChrF

# ── Model registry ────────────────────────────────────────────────────────────

MODELS = {
    "typhoon2": "scb10x/llama3.1-typhoon2-8b-instruct",
    "seallm":   "SeaLLMs/SeaLLMs-v3-7B-Chat",
    "qwen":     "Qwen/Qwen2.5-7B-Instruct",
    "llama":    "meta-llama/Llama-3.1-8B-Instruct",
}

# ── Reproducibility ───────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    os.environ["PYTHONHASHSEED"]       = str(seed)

# ── Text normalization ────────────────────────────────────────────────────────

def normalize_text(text: str) -> str:
    """
    Normalize whitespace before ChrF scoring.
    Prevents extra spaces from artificially lowering scores.
    e.g. 'กิน  ข้าว' -> 'กิน ข้าว'
    ChrF is character-level so extra spaces do affect scores.
    """
    if not text:
        return ""
    # Collapse multiple spaces to single space
    text = re.sub(r' +', ' ', text)
    return text.strip()

def compute_chrf(hypothesis: str, reference: str) -> float:
    """ChrF score with whitespace normalization."""
    chrf = ChrF()
    h = normalize_text(hypothesis)
    r = normalize_text(reference)
    if not h or not r:
        return 0.0
    return round(chrf.corpus_score([h], [[r]]).score, 2)

# ── Label masking ─────────────────────────────────────────────────────────────

def mask_prompt_labels(input_ids: list, tokenizer) -> list:
    """
    Mask all tokens except assistant response with -100.
    Model only learns to generate translations, not reproduce prompts.
    """
    labels     = [-100] * len(input_ids)
    marker     = "<|assistant|>"
    marker_ids = tokenizer.encode(marker, add_special_tokens=False)
    marker_len = len(marker_ids)
    asst_start = None

    for i in range(len(input_ids) - marker_len):
        if input_ids[i:i + marker_len] == marker_ids:
            asst_start = i + marker_len
            break

    if asst_start is not None:
        labels[asst_start:] = input_ids[asst_start:]
    else:
        labels = input_ids.copy()
    return labels

def format_and_tokenize(example, tokenizer, max_length: int):
    messages = example["messages"]
    text = ""
    for m in messages:
        if m["role"] == "system":
            text += f"<|system|>\n{m['content']}\n"
        elif m["role"] == "user":
            text += f"<|user|>\n{m['content']}\n"
        elif m["role"] == "assistant":
            text += f"<|assistant|>\n{m['content']}\n"
    tokenized = tokenizer(
        text, truncation=True, max_length=max_length, padding=False
    )
    tokenized["labels"] = mask_prompt_labels(tokenized["input_ids"], tokenizer)
    return tokenized

# ── Inference ─────────────────────────────────────────────────────────────────

SYSTEM_NTD_TO_STD = (
    "You are a translation assistant specializing in Northern Thai dialect "
    "(ภาษาเหนือ / คำเมือง). "
    "Translate the given Northern Thai sentence into natural Standard Thai "
    "(ภาษาไทยกลาง). "
    "Output only the translated sentence, nothing else."
)

SYSTEM_STD_TO_NTD = (
    "You are a translation assistant specializing in Northern Thai dialect "
    "(ภาษาเหนือ / คำเมือง). "
    "Translate the given Standard Thai sentence into natural Northern Thai dialect "
    "(ภาษาเหนือ). "
    "Output only the translated sentence, nothing else."
)

def run_inference(model, tokenizer, system: str, user_msg: str,
                  max_new_tokens: int = 256) -> str:
    """
    Deterministic inference (do_sample=False) for reproducible outputs.
    max_new_tokens=256 to ensure complete translations of longer sentences.
    """
    prompt = (
        f"<|system|>\n{system}\n"
        f"<|user|>\n{user_msg}\n"
        f"<|assistant|>\n"
    )
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=512
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,            # deterministic
            pad_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "<|assistant|>" in decoded:
        return decoded.split("<|assistant|>")[-1].strip()
    return decoded.strip()

# ── Output flag ───────────────────────────────────────────────────────────────

def flag_output(source: str, output: str) -> str:
    if not output or output.strip() == "":
        return "EMPTY"
    if normalize_text(output) == normalize_text(source):
        return "ECHO"
    if not any('\u0e00' <= c <= '\u0e7f' for c in output):
        return "NO_THAI"
    if len(output.strip()) < 3:
        return "TOO_SHORT"
    # Truncated: output is much shorter than source and ends abruptly
    if len(output) < len(source) * 0.3 and not output.endswith((".", "?", "!", "ค่ะ", "ครับ", "จ้าว", "เน้อ")):
        return "TRUNCATED"
    if len(output) > len(source) * 4:
        return "TOO_LONG"
    return "OK"

# ── Generation sampler callback ───────────────────────────────────────────────

SAMPLE_SENTENCES = [
    # (NTD, STD gold)
    ("กิ๋นข้าวล่ะ",              "กินข้าวหรือยัง"),
    ("วันนี้อากาศหนาวแต้ๆ เน้อ", "วันนี้อากาศหนาวมากนะ"),
    ("ตี้ไหนมีน้ำขายพ้องจ้าว",   "ที่ไหนมีน้ำขายบ้าง"),
    ("บ่ฮู้จะเยียะจะใดดี",        "ไม่รู้จะทำอะไรดี"),
    ("เจ้าไปไหนมาก้า",           "คุณไปไหนมา"),
]

class GenerationSamplerCallback(TrainerCallback):
    """Saves 5 fixed sentence translations every N steps for trajectory analysis."""

    def __init__(self, tokenizer, sample_sentences, output_dir, every_n_steps=50):
        self.tokenizer        = tokenizer
        self.sample_sentences = sample_sentences
        self.output_dir       = output_dir
        self.every_n_steps    = every_n_steps
        os.makedirs(output_dir, exist_ok=True)

    def _generate(self, model, step: int) -> list:
        model.eval()
        samples = []
        with torch.no_grad():
            for ntd, gold in self.sample_sentences:
                prompt = (
                    f"<|system|>\n{SYSTEM_NTD_TO_STD}\n"
                    f"<|user|>\nแปลประโยคภาษาเหนือต่อไปนี้เป็นภาษาไทยกลาง:\n{ntd}\n"
                    f"<|assistant|>\n"
                )
                inputs = self.tokenizer(
                    prompt, return_tensors="pt",
                    truncation=True, max_length=256,
                ).to(model.device)
                try:
                    out = model.generate(
                        **inputs, max_new_tokens=128,
                        do_sample=False,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )
                    decoded = self.tokenizer.decode(out[0], skip_special_tokens=True)
                    output  = decoded.split("<|assistant|>")[-1].strip() \
                              if "<|assistant|>" in decoded else decoded.strip()
                except Exception as e:
                    output = f"[ERROR: {str(e)[:40]}]"

                samples.append({
                    "ntd_input":   ntd,
                    "gold":        gold,
                    "output":      output,
                    "chrf":        compute_chrf(output, gold),
                    "exact_match": normalize_text(output) == normalize_text(gold),
                })
        model.train()
        return samples

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step % self.every_n_steps == 0:
            samples  = self._generate(model, state.global_step)
            avg_chrf = sum(s["chrf"] for s in samples) / len(samples)

            print(f"\n── Step {state.global_step} | Avg ChrF: {avg_chrf:.1f} ──")
            for s in samples:
                print(f"  [{s['chrf']:5.1f}] {s['ntd_input']}")
                print(f"          Gold:  {s['gold']}")
                print(f"          Model: {s['output']}")

            path = os.path.join(
                self.output_dir,
                f"samples_step_{state.global_step:04d}.json"
            )
            with open(path, "w", encoding="utf-8") as f:
                json.dump({
                    "step": state.global_step,
                    "avg_chrf": avg_chrf,
                    "samples": samples,
                }, f, ensure_ascii=False, indent=2)

# ── Data quality check ────────────────────────────────────────────────────────

def check_data_quality(test_data: list):
    """
    Warn about rows where gold == source (untranslated rows in dataset).
    These are the rows where you copied NTD into the Gold column.
    They inflate ECHO flags and lower ChrF scores misleadingly.
    """
    total      = len(test_data)
    echo_count = sum(
        1 for item in test_data
        if normalize_text(item["gold"]) == normalize_text(item["source"])
    )
    if echo_count > 0:
        pct = echo_count / total * 100
        print(f"\n⚠ DATA QUALITY WARNING:")
        print(f"  {echo_count}/{total} ({pct:.0f}%) test items have Gold == Source")
        print(f"  These are untranslated rows (Gold was copied from NTD column)")
        print(f"  They will show as ECHO in results and lower ChrF scores")
        print(f"  Fix: complete the translations in your Excel before retraining\n")
    else:
        print(f"  Data quality: OK — no untranslated rows detected")

# ── Test data loading ─────────────────────────────────────────────────────────

def load_test_data(path: str) -> dict:
    """
    Load test JSONL and separate by translation direction.
    Returns dict with keys 'ntd_to_std' and 'std_to_ntd'.
    """
    ntd_to_std = []
    std_to_ntd = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex       = json.loads(line)
            messages = ex["messages"]
            system   = next((m["content"] for m in messages if m["role"] == "system"), "")
            user_msg = next(m["content"] for m in messages if m["role"] == "user")
            expected = next(m["content"] for m in messages if m["role"] == "assistant")

            # Detect translation direction from user prompt
            if "แปลประโยคภาษาเหนือ" in user_msg:
                # NTD -> STD direction
                source = user_msg.split("\n")[-1].strip() \
                         if "\n" in user_msg else user_msg
                ntd_to_std.append({
                    "direction": "NTD->STD",
                    "system":    SYSTEM_NTD_TO_STD,
                    "user_msg":  user_msg,
                    "source":    source,
                    "gold":      expected,
                })
            elif "แปลประโยคภาษาไทยกลาง" in user_msg or "แปลประโยคนี้เป็นภาษาเหนือ" in user_msg:
                # STD -> NTD direction
                source = user_msg.split("\n")[-1].strip() \
                         if "\n" in user_msg else user_msg
                std_to_ntd.append({
                    "direction": "STD->NTD",
                    "system":    SYSTEM_STD_TO_NTD,
                    "user_msg":  user_msg,
                    "source":    source,
                    "gold":      expected,
                })

    print(f"  NTD->STD items: {len(ntd_to_std)}")
    print(f"  STD->NTD items: {len(std_to_ntd)}")
    return {"ntd_to_std": ntd_to_std, "std_to_ntd": std_to_ntd}

# ── Checkpoint evaluation ─────────────────────────────────────────────────────

def evaluate_checkpoint(
    base_model_id: str,
    adapter_path:  str,
    test_items:    list,
    label:         str,
    max_samples:   int = 100,
    print_samples: int = 10,      # how many to print in cell output
) -> tuple:
    """
    Evaluate one checkpoint on test_items.
    Prints only print_samples to terminal, saves all max_samples to file.
    """
    chrf_metric = ChrF()
    results     = []
    items       = test_items[:max_samples]

    print(f"\n{'='*55}")
    print(f"Checkpoint: {label}")
    print(f"Items:      {len(items)} (printing {print_samples} to terminal)")
    print(f"{'='*55}")

    # Load model + adapter
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        base_model_id, quantization_config=bnb_config,
        device_map="auto", torch_dtype=torch.float16,
    )
    if adapter_path and os.path.exists(adapter_path):
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    hypotheses = []
    references = []
    printed    = 0

    for i, item in enumerate(items):
        try:
            output = run_inference(
                model, tokenizer,
                system=item["system"],
                user_msg=item["user_msg"],
                max_new_tokens=256,       # increased for complete translations
            )
            f = flag_output(item["source"], output)
        except Exception as e:
            output = ""
            f      = "ERROR"

        # Normalize before scoring
        output_norm = normalize_text(output)
        gold_norm   = normalize_text(item["gold"])
        item_chrf   = compute_chrf(output_norm, gold_norm) if output else 0.0

        hypotheses.append(output_norm)
        references.append(gold_norm)

        # Print only first print_samples to terminal
        if printed < print_samples:
            flag_marker = f" [{f}]" if f != "OK" else ""
            print(f"\n  [{i+1:3d}/{len(items)}]{flag_marker}")
            print(f"    Source: {item['source'][:65]}")
            print(f"    Gold:   {item['gold'][:65]}")
            print(f"    Model:  {output[:65]}")
            print(f"    ChrF:   {item_chrf:.1f}")
            printed += 1
        elif printed == print_samples:
            print(f"\n  ... ({len(items) - print_samples} more saved to file)")
            printed += 1  # prevent repeated message

        results.append({
            "checkpoint":  label,
            "direction":   item.get("direction", "NTD->STD"),
            "source":      item["source"],
            "gold":        item["gold"],
            "translation": output,
            "chrf":        item_chrf,
            "flag":        f,
            "human_score_accuracy":     None,
            "human_score_naturalness":  None,
            "human_score_dialect_loss": None,
            "human_notes":              "",
        })

    # Corpus ChrF
    valid = [(h, r) for h, r in zip(hypotheses, references) if h]
    if valid:
        h_v, r_v = zip(*valid)
        corpus_chrf = chrf_metric.corpus_score(list(h_v), [list(r_v)]).score
    else:
        corpus_chrf = 0.0

    print(f"\n  Corpus ChrF ({len(valid)}/{len(items)} valid): {corpus_chrf:.2f}")

    del model
    torch.cuda.empty_cache()

    return results, corpus_chrf

def evaluate_all_checkpoints(
    base_model_id: str,
    run_dir:       str,
    test_data:     dict,
    adapter_dir:   str,
    max_samples:   int = 100,
    print_samples: int = 10,
    direction:     str = "ntd_to_std",
):
    """
    Evaluate base model + all saved checkpoints + best adapter.
    direction: 'ntd_to_std' or 'std_to_ntd' or 'both'
    """
    test_items = []
    if direction in ("ntd_to_std", "both"):
        test_items.extend(test_data["ntd_to_std"])
    if direction in ("std_to_ntd", "both"):
        test_items.extend(test_data["std_to_ntd"])

    if not test_items:
        print(f"No test items found for direction: {direction}")
        print("Check that prepare_data.py generated the correct task types")
        return [], []

    print(f"\nEvaluating direction: {direction} | {len(test_items)} items")
    check_data_quality(test_items)

    all_results  = []
    summary_rows = []

    # Base model
    base_results, base_chrf = evaluate_checkpoint(
        base_model_id, None, test_items,
        "Base (step 0)", max_samples, print_samples
    )
    all_results.extend(base_results)
    summary_rows.append({
        "checkpoint": "Base (step 0)", "step": 0,
        "chrf": round(base_chrf, 2), "is_best": False,
        "direction": direction,
    })

    # Intermediate checkpoints
    checkpoints = []
    for name in sorted(os.listdir(run_dir)):
        if name.startswith("checkpoint-"):
            try:
                step = int(name.split("-")[-1])
                checkpoints.append((step, os.path.join(run_dir, name)))
            except ValueError:
                pass

    for step, ckpt_path in sorted(checkpoints):
        results, ckpt_chrf = evaluate_checkpoint(
            base_model_id, ckpt_path, test_items,
            f"Checkpoint step {step}", max_samples, print_samples
        )
        all_results.extend(results)
        summary_rows.append({
            "checkpoint": f"Checkpoint step {step}", "step": step,
            "chrf": round(ckpt_chrf, 2), "is_best": False,
            "direction": direction,
        })

    # Best adapter
    if os.path.exists(adapter_dir):
        results, best_chrf = evaluate_checkpoint(
            base_model_id, adapter_dir, test_items,
            "Best adapter (final)", max_samples, print_samples
        )
        all_results.extend(results)
        summary_rows.append({
            "checkpoint": "Best adapter (final)", "step": 999,
            "chrf": round(best_chrf, 2), "is_best": True,
            "direction": direction,
        })

    return all_results, summary_rows

def save_evaluation_results(
    all_results:  list,
    summary_rows: list,
    output_dir:   str,
    base_chrf:    float,
):
    os.makedirs(output_dir, exist_ok=True)

    # Full JSONL
    jsonl_path = os.path.join(output_dir, "checkpoint_eval_results.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in all_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nFull results ({len(all_results)} rows) -> {jsonl_path}")

    # Summary table
    summary_df = pd.DataFrame(summary_rows).sort_values("step")
    summary_df["delta"] = (summary_df["chrf"] - base_chrf).round(2)

    summary_path = os.path.join(output_dir, "checkpoint_eval_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    print(f"\n── ChrF across checkpoints ─────────────────────────")
    print(f"  {'Checkpoint':<30} {'ChrF':>8} {'Delta':>8}")
    print("  " + "-"*48)
    for _, row in summary_df.iterrows():
        delta_str = f"{row['delta']:+.2f}" if row["step"] > 0 else "    —"
        best_mark = " ← best" if row["is_best"] else ""
        print(f"  {row['checkpoint']:<30} {row['chrf']:>8.2f} {delta_str:>8}{best_mark}")

    print(f"\n  Summary -> {summary_path}")

    # Gold vs translation CSV (all 100 per checkpoint)
    df = pd.DataFrame(all_results)
    sidebyside_path = os.path.join(output_dir, "gold_vs_translation.csv")
    df[[
        "checkpoint", "direction", "source", "gold", "translation",
        "chrf", "flag",
        "human_score_accuracy", "human_score_naturalness",
        "human_score_dialect_loss", "human_notes",
    ]].to_csv(sidebyside_path, index=False, encoding="utf-8-sig")
    print(f"  Gold vs translation -> {sidebyside_path}")

    # Human eval sheet (best adapter only, flagged items prioritized)
    best_df  = df[df["checkpoint"] == "Best adapter (final)"]
    flagged  = best_df[~best_df["flag"].isin(["OK", "ERROR"])]
    ok_items = best_df[best_df["flag"] == "OK"]
    n_flag   = min(15, len(flagged))
    n_ok     = min(30 - n_flag, len(ok_items))

    frames = []
    if n_flag > 0:
        frames.append(flagged.sample(n_flag, random_state=42))
    if n_ok > 0:
        frames.append(ok_items.sample(n_ok, random_state=42))

    if frames:
        human_eval = pd.concat(frames)[[
            "checkpoint", "direction", "source", "gold", "translation",
            "flag", "human_score_accuracy", "human_score_naturalness",
            "human_score_dialect_loss", "human_notes",
        ]]
        human_path = os.path.join(output_dir, "human_eval_sheet.csv")
        human_eval.to_csv(human_path, index=False, encoding="utf-8-sig")
        print(f"  Human eval sheet -> {human_path}")
        print(f"    ({n_flag} flagged + {n_ok} OK items)")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",            choices=MODELS.keys(), default="typhoon2")
    parser.add_argument("--task",
                        choices=["multitask", "translation"],
                        default="multitask")
    parser.add_argument("--iters",            type=int,   default=300)
    parser.add_argument("--batch_size",       type=int,   default=2)
    parser.add_argument("--max_seq_length",   type=int,   default=512)
    parser.add_argument("--lr",               type=float, default=2e-4)
    parser.add_argument("--lora_rank",        type=int,   default=8)
    parser.add_argument("--sample_every",     type=int,   default=50)
    parser.add_argument("--early_stopping",   type=int,   default=3)
    parser.add_argument("--eval_checkpoints", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max_eval_samples", type=int,   default=100,
                        help="Total items to evaluate per checkpoint (saved to file)")
    parser.add_argument("--print_samples",    type=int,   default=10,
                        help="How many translations to print in terminal output")
    parser.add_argument("--eval_direction",
                        choices=["ntd_to_std", "std_to_ntd", "both"],
                        default="ntd_to_std",
                        help="Which translation direction to evaluate")
    parser.add_argument("--test_data",        default="data/test.jsonl")
    parser.add_argument("--data_dir",         default="data")
    parser.add_argument("--output_dir",       default="outputs")
    parser.add_argument("--seed",             type=int,   default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    model_id    = MODELS[args.model]
    run_name    = f"{args.model}_{args.task}"
    run_dir     = os.path.join(args.output_dir, run_name)
    adapter_dir = os.path.join(run_dir, "adapters")
    samples_dir = os.path.join(run_dir, "generation_samples")
    eval_dir    = os.path.join(run_dir, "checkpoint_evaluation")
    os.makedirs(adapter_dir, exist_ok=True)
    os.makedirs(samples_dir, exist_ok=True)
    os.makedirs(eval_dir,    exist_ok=True)

    print("="*60)
    print(f"Model:     {model_id}")
    print(f"Task:      {args.task}")
    print(f"Direction: {args.eval_direction}")
    print(f"Dir:       {run_dir}")
    print("="*60)

    if not torch.cuda.is_available():
        raise RuntimeError("No GPU. Kaggle: Session options -> Accelerator -> GPU T4 x2")

    print(f"GPU:  {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    # Save config
    config = {
        "model_id": model_id, "task": args.task,
        "eval_direction": args.eval_direction,
        "iters": args.iters, "batch_size": args.batch_size,
        "max_seq_length": args.max_seq_length, "lr": args.lr,
        "lora_rank": args.lora_rank, "lora_alpha": args.lora_rank * 2,
        "label_masking": True, "deterministic_generation": True,
        "whitespace_normalization": True, "max_new_tokens": 256,
        "early_stopping_patience": args.early_stopping,
        "seed": args.seed, "gpu": torch.cuda.get_device_name(0),
        "timestamp": datetime.now().isoformat(),
    }
    with open(os.path.join(run_dir, "run_config.json"), "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    # ── Load model ────────────────────────────────────────────────────────────
    print("\nLoading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "right"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb_config,
        device_map="auto", torch_dtype=torch.float16,
    )
    model.config.use_cache = False

    # ── LoRA ──────────────────────────────────────────────────────────────────
    print("Applying LoRA...")
    lora_config = LoraConfig(
        r=args.lora_rank, lora_alpha=args.lora_rank * 2,
        target_modules=["q_proj","k_proj","v_proj","o_proj",
                        "gate_proj","up_proj","down_proj"],
        lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── Dataset ───────────────────────────────────────────────────────────────
    train_path = os.path.join(args.data_dir, f"train_{args.task}.jsonl")
    valid_path = os.path.join(args.data_dir, f"valid_{args.task}.jsonl")
    if not os.path.exists(train_path):
        train_path = os.path.join(args.data_dir, "train.jsonl")
    if not os.path.exists(valid_path):
        valid_path = os.path.join(args.data_dir, "valid.jsonl")

    print(f"\nLoading dataset: {train_path}")
    dataset = load_dataset(
        "json",
        data_files={"train": train_path, "validation": valid_path}
    )
    tokenize_fn = lambda x: format_and_tokenize(x, tokenizer, args.max_seq_length)
    dataset = dataset.map(
        tokenize_fn, remove_columns=dataset["train"].column_names,
        desc="Tokenizing",
    )
    print(f"  Train: {len(dataset['train'])} | Valid: {len(dataset['validation'])}")

    # ── Training ──────────────────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=run_dir,
        max_steps=args.iters,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        fp16=True,
        max_grad_norm=1.0,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=20,
        logging_steps=10,
        logging_dir=os.path.join(run_dir, "logs"),
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=4,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=args.seed,
        data_seed=args.seed,
        report_to="none",
        remove_unused_columns=False,
    )

    callbacks = [
        GenerationSamplerCallback(
            tokenizer=tokenizer,
            sample_sentences=SAMPLE_SENTENCES,
            output_dir=samples_dir,
            every_n_steps=args.sample_every,
        )
    ]
    if args.early_stopping > 0:
        callbacks.append(EarlyStoppingCallback(
            early_stopping_patience=args.early_stopping
        ))

    data_collator = DataCollatorForSeq2Seq(
        tokenizer, model=model, padding=True,
        pad_to_multiple_of=8, label_pad_token_id=-100,
    )

    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
        callbacks=callbacks,
    )

    print(f"\nStarting training for {args.iters} steps...")
    print(f"  Label masking:      ON (trains only on assistant tokens)")
    print(f"  Generation:         deterministic (do_sample=False)")
    print(f"  Whitespace norm:    ON (ChrF normalized before scoring)")
    print(f"  Max new tokens:     256 (for complete translations)")
    print(f"  Samples every:      {args.sample_every} steps\n")

    trainer_stats = trainer.train()

    best_ckpt    = trainer.state.best_model_checkpoint or "N/A"
    best_loss    = trainer.state.best_metric or 0.0
    perplexity   = math.exp(best_loss) if best_loss > 0 else 0.0
    runtime_mins = trainer_stats.metrics.get("train_runtime", 0) / 60

    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Best checkpoint: {best_ckpt}")
    print(f"Best eval loss:  {best_loss:.4f}")
    print(f"Perplexity:      {perplexity:.2f}")
    print(f"Training loss:   {trainer_stats.training_loss:.4f}")
    print(f"Time:            {runtime_mins:.1f} minutes")

    print(f"\nSaving best adapter -> {adapter_dir}")
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    # Compile trajectory
    all_snapshots = []
    for fname in sorted(os.listdir(samples_dir)):
        if fname.endswith(".json"):
            with open(os.path.join(samples_dir, fname)) as f:
                all_snapshots.append(json.load(f))

    with open(os.path.join(run_dir, "generation_trajectory.json"), "w",
              encoding="utf-8") as f:
        json.dump(all_snapshots, f, ensure_ascii=False, indent=2)

    print(f"\n── Learning trajectory (ChrF over steps) ────────────")
    for snap in all_snapshots:
        bar = "█" * int(snap.get("avg_chrf", 0) / 5)
        print(f"  Step {snap['step']:4d}: {snap.get('avg_chrf', 0):5.1f} {bar}")

    metrics = {
        "model": args.model, "task": args.task,
        "training_loss": trainer_stats.training_loss,
        "best_eval_loss": best_loss, "perplexity": round(perplexity, 2),
        "best_checkpoint": best_ckpt, "total_steps": trainer_stats.global_step,
        "runtime_minutes": runtime_mins,
    }
    with open(os.path.join(run_dir, "training_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # ── Post-training checkpoint evaluation ───────────────────────────────────
    if args.eval_checkpoints and os.path.exists(args.test_data):
        print("\n" + "="*60)
        print("POST-TRAINING CHECKPOINT EVALUATION")
        print("="*60)
        print(f"Direction:     {args.eval_direction}")
        print(f"Items saved:   {args.max_eval_samples} per checkpoint")
        print(f"Items printed: {args.print_samples} per checkpoint\n")

        del model
        torch.cuda.empty_cache()

        print("Loading test data...")
        test_data = load_test_data(args.test_data)

        all_eval, summary_rows = evaluate_all_checkpoints(
            base_model_id=model_id,
            run_dir=run_dir,
            test_data=test_data,
            adapter_dir=adapter_dir,
            max_samples=args.max_eval_samples,
            print_samples=args.print_samples,
            direction=args.eval_direction,
        )

        if summary_rows:
            base_chrf = next(r["chrf"] for r in summary_rows if r["step"] == 0)
            save_evaluation_results(all_eval, summary_rows, eval_dir, base_chrf)

    elif args.eval_checkpoints:
        print(f"\nTest data not found at {args.test_data} — skipping evaluation")

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("ALL OUTPUTS")
    print("="*60)
    print(f"\n{run_dir}/")
    print(f"  adapters/                    <- best LoRA weights")
    print(f"  generation_samples/          <- 5-sentence snapshots every {args.sample_every} steps")
    print(f"  generation_trajectory.json   <- learning trajectory")
    print(f"  training_metrics.json        <- loss, perplexity, runtime")
    print(f"  run_config.json              <- all hyperparameters")
    print(f"\n{eval_dir}/")
    print(f"  checkpoint_eval_summary.csv  <- ChrF per checkpoint (goes in paper)")
    print(f"  gold_vs_translation.csv      <- all 100 items: source | gold | translation")
    print(f"  human_eval_sheet.csv         <- 30 items for native speaker annotation")
    print(f"  checkpoint_eval_results.jsonl <- full raw results")
    print(f"\nDownload {run_dir}/ from Kaggle Output panel before session ends.")

if __name__ == "__main__":
    main()