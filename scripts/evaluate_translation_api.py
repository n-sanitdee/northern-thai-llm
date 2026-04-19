"""
evaluate_translation_api.py

Evaluates NTD -> Standard Thai translation across GPT-4o, Gemini, Claude, and DeepSeek.
Tests both zero-shot and few-shot conditions.
Samples 50 CI and 50 CD items from your dataset.

Usage:
    python scripts/evaluate_translation_api.py --anthropic_key YOUR_KEY
    python scripts/evaluate_translation_api.py --gemini_key YOUR_KEY
    python scripts/evaluate_translation_api.py --openai_key YOUR_KEY
    python scripts/evaluate_translation_api.py --deepseek_key YOUR_KEY

Or set environment variables:
    export ANTHROPIC_API_KEY=...
    export GEMINI_API_KEY=...
    export OPENAI_API_KEY=...
    export DEEPSEEK_API_KEY=...
    python scripts/evaluate_translation_api.py

Results saved to:
    outputs/translation/results.jsonl       <- all raw outputs
    outputs/translation/summary.csv         <- ChrF scores per model/condition
    outputs/translation/for_human_eval.csv  <- subset for your annotators
"""

import os
import json
import time
import random
import argparse
import pandas as pd
from sacrebleu.metrics import CHRF as ChrF

# ── Output quality flags ──────────────────────────────────────────────────────

def flag_output(ntd_input: str, model_output: str) -> str:
    """
    Flags potentially problematic outputs for human reviewer attention.
    Categories:
        OK             - looks like a normal translation
        EMPTY          - model returned nothing
        ECHO           - model repeated the input unchanged
        ECHO_REORDERED - same words as input, different order
        NO_THAI        - output has no Thai characters (e.g. responded in English)
        TOO_SHORT      - suspiciously short (less than 3 chars)
        TOO_LONG       - suspiciously long (model explained instead of translating)
        ERROR          - API call failed
    """
    if not model_output or model_output.strip() == "":
        return "EMPTY"
    if model_output.strip() == ntd_input.strip():
        return "ECHO"
    if not any('\u0e00' <= c <= '\u0e7f' for c in model_output):
        return "NO_THAI"
    if len(model_output.strip()) < 3:
        return "TOO_SHORT"
    if len(model_output) > len(ntd_input) * 4:
        return "TOO_LONG"
    input_words  = set(ntd_input.strip().split())
    output_words = set(model_output.strip().split())
    if len(input_words) > 2 and input_words == output_words:
        return "ECHO_REORDERED"
    return "OK"

# ── API clients ───────────────────────────────────────────────────────────────

def call_openai(prompt: str, system: str, api_key: str) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt},
        ],
        temperature=0.1,
        max_tokens=256,
    )
    return response.choices[0].message.content.strip()

def call_gemini(prompt: str, system: str, api_key: str) -> str:
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        config=types.GenerateContentConfig(
            system_instruction=system,
            temperature=0.1,
            max_output_tokens=256,
        ),
        contents=prompt,
    )
    return response.text.strip()

def call_claude(prompt: str, system: str, api_key: str) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=256,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text.strip()

def call_deepseek(prompt: str, system: str, api_key: str) -> str:
    from openai import OpenAI
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
    )
    response = client.chat.completions.create(
        model="deepseek-chat",  # DeepSeek-V3
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt},
        ],
        temperature=0.1,
        max_tokens=256,
    )
    return response.choices[0].message.content.strip()

def call_thaillm(prompt: str, system: str, api_key: str) -> str:
    from openai import OpenAI
    client = OpenAI(
        api_key=api_key,
        base_url="http://thaillm.or.th/api/typhoon/v1"
    )
    response = client.chat.completions.create(
        model="/model",
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt},
        ],
        temperature=0.1,
        max_tokens=256,
    )
    return response.choices[0].message.content.strip()

# ── Model registry ────────────────────────────────────────────────────────────
# Add or remove models here — no other changes needed

MODELS = {
    "gpt4o":    "GPT-4o",
    "gemini":   "Gemini 2.0 Flash",
    "claude":   "Claude Sonnet",
    "deepseek": "DeepSeek-V3",
}

API_CALLERS = {
    "gpt4o":    call_openai,
    "gemini":   call_gemini,
    "claude":   call_claude,
    "deepseek": call_deepseek,
}

# ── Prompt builders ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a translation assistant specializing in Northern Thai dialect "
    "(ภาษาเหนือ / คำเมือง). "
    "Translate the given Northern Thai sentence into natural Standard Thai "
    "(ภาษาไทยกลาง). "
    "Output only the translated sentence, nothing else."
)

def build_zero_shot_prompt(ntd_text: str) -> str:
    """
    Zero-shot: just the sentence, no examples.
    Tests whether the model can translate NTD without any guidance.
    """
    return f"แปลประโยคภาษาเหนือต่อไปนี้เป็นภาษาไทยกลาง:\n{ntd_text}"

def build_few_shot_prompt(ntd_text: str, examples: list) -> str:
    """
    Few-shot: 5 example pairs shown before the target sentence.
    Tests whether examples help the model recognize and translate NTD better.
    Examples are drawn from the training set, not the test set.
    """
    header = "ต่อไปนี้คือตัวอย่างการแปลภาษาเหนือเป็นภาษาไทยกลาง:\n\n"
    shots  = ""
    for ex in examples[:5]:
        shots += f"ภาษาเหนือ: {ex['ntd']}\nภาษาไทยกลาง: {ex['std_gold']}\n\n"
    task = (
        f"แปลประโยคต่อไปนี้เป็นภาษาไทยกลาง:\n"
        f"ภาษาเหนือ: {ntd_text}\n"
        f"ภาษาไทยกลาง:"
    )
    return header + shots + task

# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(excel_path: str, n_per_type: int = 50, seed: int = 42):
    """
    Loads dataset and samples n_per_type CI and CD items for testing.
    Remaining CI items form the few-shot example pool.
    """
    random.seed(seed)
    df = pd.read_excel(excel_path, sheet_name="natural")

    df = df[
        df["Text_Northern"].notna() &
        df["Text_Standard_Thai"].notna()
    ].copy()

    ci = df[df["Contextual_Dependency"] == "CI"]
    cd = df[df["Contextual_Dependency"] == "CD"]

    print(f"Available CI: {len(ci)} | Available CD: {len(cd)}")

    ci_test = ci.sample(min(n_per_type, len(ci)), random_state=seed)
    cd_test = cd.sample(min(n_per_type, len(cd)), random_state=seed)
    test_df = pd.concat([ci_test, cd_test])

    # Few-shot pool = CI items NOT in test set
    few_shot_df = ci[~ci.index.isin(ci_test.index)]

    def df_to_items(df):
        items = []
        for _, row in df.iterrows():
            items.append({
                "id":           str(row.get("ID", "")),
                "context_type": str(row.get("Contextual_Dependency", "")),
                "ntd":          str(row["Text_Northern"]).strip(),
                "std_gold":     str(row["Text_Standard_Thai"]).strip(),
                "head_post":    str(row.get("Head_Post_Text", "")).strip(),
                "intent":       str(row.get("Intent", "")),
                "tone":         str(row.get("Tone", "")),
            })
        return items

    test_items        = df_to_items(test_df)
    few_shot_pool     = df_to_items(few_shot_df)
    few_shot_examples = random.sample(few_shot_pool, min(5, len(few_shot_pool)))

    print(f"Test set: {len(test_items)} items ({len(ci_test)} CI + {len(cd_test)} CD)")
    print(f"Few-shot examples: {len(few_shot_examples)}")

    return test_items, few_shot_examples

# ── Evaluation loop ───────────────────────────────────────────────────────────

def run_evaluation(
    test_items:        list,
    few_shot_examples: list,
    keys:              dict,
    output_dir:        str,
    delay:             float = 1.0,
):
    os.makedirs(output_dir, exist_ok=True)
    results    = []
    chrf       = ChrF()
    conditions = ["zero_shot", "few_shot"]

    for model_key, model_name in MODELS.items():

        if not keys.get(model_key):
            print(f"\nSkipping {model_name} — no API key provided")
            continue

        caller = API_CALLERS[model_key]

        for condition in conditions:
            print(f"\n{'='*50}")
            print(f"Model: {model_name} | Condition: {condition}")
            print(f"{'='*50}")

            hypotheses  = []
            references  = []
            quota_error = False

            for i, item in enumerate(test_items):

                # Build user prompt based on condition
                if condition == "zero_shot":
                    prompt = build_zero_shot_prompt(item["ntd"])
                else:
                    prompt = build_few_shot_prompt(item["ntd"], few_shot_examples)

                # For CD items, prepend the original post as context
                if item["context_type"] == "CD" and item["head_post"] not in ("", "nan"):
                    prompt = f"[โพสต์ต้นฉบับ]: {item['head_post']}\n" + prompt

                # Call the API
                try:
                    output = caller(prompt, SYSTEM_PROMPT, keys[model_key])
                    flag   = flag_output(item["ntd"], output)

                    flag_marker = f" [{flag}]" if flag != "OK" else ""
                    print(f"  [{i+1}/{len(test_items)}] {item['id']}{flag_marker}")
                    print(f"    NTD:   {item['ntd'][:70]}")
                    print(f"    Gold:  {item['std_gold'][:70]}")
                    print(f"    Model: {output[:70]}")

                except Exception as e:
                    error_msg = str(e)
                    print(f"  [{i+1}] ERROR for {item['id']}: {error_msg[:120]}")

                    # Stop immediately on quota/billing errors
                    if any(kw in error_msg for kw in
                           ["insufficient_quota", "429", "credit balance is too low",
                            "Insufficient Balance"]):
                        print(f"  x Quota/billing error — skipping rest of {model_name}")
                        quota_error = True
                        break

                    output = ""
                    flag   = "ERROR"

                hypotheses.append(output)
                references.append(item["std_gold"])

                results.append({
                    "id":           item["id"],
                    "context_type": item["context_type"],
                    "model":        model_name,
                    "condition":    condition,
                    "ntd_input":    item["ntd"],
                    "std_gold":     item["std_gold"],
                    "model_output": output,
                    "output_flag":  flag,
                    "head_post":    item["head_post"],
                    "intent":       item["intent"],
                    "tone":         item["tone"],
                    # Human annotators fill these in:
                    "human_score_accuracy":     None,
                    "human_score_naturalness":  None,
                    "human_score_dialect_loss": None,
                    "human_notes":              "",
                })

                time.sleep(delay)

            if quota_error:
                break

            # ChrF score for this model/condition
            valid = [(h, r) for h, r in zip(hypotheses, references) if h]
            if valid:
                h_valid, r_valid = zip(*valid)
                score = chrf.corpus_score(list(h_valid), [list(r_valid)]).score
                print(f"\n  ChrF score ({len(valid)} valid items): {score:.2f}")

    return results

# ── Save results ──────────────────────────────────────────────────────────────

def save_results(results: list, output_dir: str):
    if not results:
        print("No results to save.")
        return

    # 1. Full results JSONL
    jsonl_path = os.path.join(output_dir, "results.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nFull results -> {jsonl_path}")

    df   = pd.DataFrame(results)
    chrf = ChrF()

    # 2. Flag summary
    print("\n── Output flag summary ──────────────────────────")
    flag_summary = (
        df.groupby(["model", "condition", "output_flag"])
        .size()
        .unstack(fill_value=0)
    )
    print(flag_summary.to_string())

    # 3. ChrF summary table
    summary = []
    for model in df["model"].unique():
        for condition in df["condition"].unique():
            for ctx in ["CI", "CD", "ALL"]:
                subset = df[
                    (df["model"] == model) &
                    (df["condition"] == condition) &
                    (df["model_output"] != "")
                ]
                if ctx != "ALL":
                    subset = subset[subset["context_type"] == ctx]
                if len(subset) == 0:
                    continue

                score = chrf.corpus_score(
                    subset["model_output"].tolist(),
                    [subset["std_gold"].tolist()]
                ).score

                summary.append({
                    "model":      model,
                    "condition":  condition,
                    "context":    ctx,
                    "n_items":    len(subset),
                    "chrf_score": round(score, 2),
                })

    summary_df   = pd.DataFrame(summary)
    summary_path = os.path.join(output_dir, "summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\n── ChrF Summary ─────────────────────────────────")
    print(summary_df.to_string(index=False))
    print(f"\nSummary -> {summary_path}")

    # 4. Human evaluation CSV
    # Prioritize flagged items so annotators review failure cases first
    flagged  = df[df["output_flag"] != "OK"]
    ok_items = df[df["output_flag"] == "OK"]
    n_flagged = min(15, len(flagged))
    n_ok      = min(30 - n_flagged, len(ok_items))

    human_eval = pd.concat([
        flagged.sample(n_flagged,  random_state=42) if n_flagged > 0 else pd.DataFrame(),
        ok_items.sample(n_ok, random_state=42)      if n_ok      > 0 else pd.DataFrame(),
    ])[[
        "id", "context_type", "model", "condition", "output_flag",
        "ntd_input", "std_gold", "model_output",
        "human_score_accuracy", "human_score_naturalness",
        "human_score_dialect_loss", "human_notes",
    ]]

    human_path = os.path.join(output_dir, "for_human_eval.csv")
    human_eval.to_csv(human_path, index=False, encoding="utf-8-sig")
    print(f"Human eval sheet -> {human_path}")
    print(f"  ({n_flagged} flagged + {n_ok} OK items — flagged items prioritized)")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate NTD->STD translation across multiple LLMs"
    )
    parser.add_argument("--input",         default="data/Master_Dataset.xlsx",
                        help="Path to your Excel dataset")
    parser.add_argument("--n_per_type",    type=int,   default=50,
                        help="Number of CI and CD items each to sample")
    parser.add_argument("--output_dir",    default="outputs/translation",
                        help="Where to save results")
    parser.add_argument("--openai_key",    default=os.getenv("OPENAI_API_KEY"),
                        help="OpenAI API key for GPT-4o")
    parser.add_argument("--gemini_key",    default=os.getenv("GEMINI_API_KEY"),
                        help="Google API key for Gemini")
    parser.add_argument("--anthropic_key", default=os.getenv("ANTHROPIC_API_KEY"),
                        help="Anthropic API key for Claude")
    parser.add_argument("--deepseek_key",  default=os.getenv("DEEPSEEK_API_KEY"),
                        help="DeepSeek API key for DeepSeek-V3")
    parser.add_argument("--delay",         type=float, default=1.0,
                        help="Seconds between API calls (avoid rate limits)")
    parser.add_argument("--seed",          type=int,   default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    keys = {
        "gpt4o":    args.openai_key,
        "gemini":   args.gemini_key,
        "claude":   args.anthropic_key,
        "deepseek": args.deepseek_key,
    }

    # Report which models will run
    for name, key in keys.items():
        status = "ready" if key else "no key — will be skipped"
        print(f"  {MODELS[name]}: {status}")

    # Load data
    test_items, few_shot_examples = load_data(args.input, args.n_per_type, args.seed)

    print("\nFew-shot examples selected:")
    for ex in few_shot_examples:
        print(f"  {ex['ntd'][:60]} -> {ex['std_gold'][:60]}")

    # Run evaluation
    results = run_evaluation(
        test_items, few_shot_examples, keys, args.output_dir, args.delay
    )

    # Save everything
    save_results(results, args.output_dir)

    print("\nDone.")
    print("  1. Check outputs/translation/summary.csv for automatic scores")
    print("  2. Send outputs/translation/for_human_eval.csv to your annotators")
    print("  3. Run again after fine-tuning to compare before/after")

if __name__ == "__main__":
    main()
