# Northern Thai Dialect in LLMs — Fine-tuning Pipeline

Reproducible fine-tuning pipeline for evaluating and improving LLM performance on Northern Thai dialect (คำเมือง).

## Requirements

- macOS with Apple Silicon (M1–M5) **or** Linux with CUDA GPU
- Python 3.11+
- ~15GB disk space per model

## Setup

```bash
git clone https://github.com/yourusername/northern-thai-llm
cd northern-thai-llm
pip install -r requirements.txt
```

## Workflow

### 1. Download a model

```bash
python scripts/download_model.py --model typhoon2
```

Available models: `typhoon2`, `llama`, `seallm`, `qwen`

> Note: Llama requires accepting Meta's license at huggingface.co first.

### 2. Prepare your data

Place your dataset at `data/Master_Dataset.xlsx`, then:

```bash
python scripts/prepare_data.py --task multitask
```

This generates `data/train.jsonl`, `data/valid.jsonl`, `data/test.jsonl`.

Task options:
- `multitask` — translation + reverse translation + intent (recommended)
- `translation` — NTD → Standard Thai only
- `reverse` — Standard Thai → NTD only
- `intent` — intent classification only (requires Intent column to be annotated)

### 3. Fine-tune

```bash
python scripts/finetune.py --model typhoon2
```

**If you get memory errors (16GB RAM):**
```bash
python scripts/finetune.py --model typhoon2 --batch_size 1 --max_seq_length 256
```

LoRA adapters are saved to `outputs/typhoon2/adapters/`. These are small (~50MB) and committed to Git. Full model weights are gitignored.

### 4. Evaluate

```bash
# Fine-tuned model only
python scripts/evaluate.py --model typhoon2

# Fine-tuned vs base model comparison
python scripts/evaluate.py --model typhoon2 --compare_base
```

Results saved to `outputs/typhoon2/results.jsonl`.

## Project Structure

```
northern-thai-llm/
├── data/
│   ├── train.jsonl          # generated, gitignored
│   ├── valid.jsonl
│   └── test.jsonl
├── scripts/
│   ├── download_model.py    # download from Hugging Face
│   ├── prepare_data.py      # Excel → JSONL
│   ├── finetune.py          # LoRA fine-tuning via MLX-LM
│   └── evaluate.py          # inference + comparison
├── outputs/
│   └── typhoon2/
│       ├── adapters/        # LoRA weights (committed)
│       ├── run_config.json  # hyperparameters for reproducibility
│       └── results.jsonl    # evaluation outputs
├── models/                  # full model weights (gitignored)
├── requirements.txt
└── README.md
```

## Citation

If you use this pipeline, please cite:

```
[Your paper citation here]
```

## Notes on 16GB RAM (Apple Silicon)

- Use `--batch_size 1` and `--max_seq_length 256` if training crashes
- Close other applications before training
- Monitor RAM in Activity Monitor — keep free RAM above ~2GB
- For larger runs, consider Google Colab Pro (A100, ~$10/session)
