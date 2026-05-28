# Text-to-SQL Fine-Tuning with Qwen 2.5-3B + LoRA on Apple MLX

Fine-tuning a 3B-parameter LLM for natural-language-to-SQL generation on consumer hardware (MacBook Pro M4, 16GB unified memory) using **LoRA** and **Apple's MLX framework** — trained in **~33 minutes** on **0.216%** of parameters.

> **Why this matters:** Most Text-to-SQL fine-tuning research uses 7B+ models on data-center GPUs. This project demonstrates that **meaningful accuracy gains are achievable on a laptop** when you choose the right architecture (LoRA + 4-bit quantization + MLX's unified-memory advantage).

---

## 📊 Results

Evaluated on a held-out 50-sample test set from `gretelai/synthetic_text_to_sql`.

| Metric | Base Model | Fine-Tuned | Δ |
|---|---|---|---|
| **Exact Match** | 26.0% | **30.0%** | **+4.0%** |
| **BLEU Score** | 0.504 | **0.566** | **+12.3%** |
| **Keyword Overlap** | 73.0% | **77.4%** | **+4.4%** |

### Breakdown by SQL complexity

| Category | Exact Match | BLEU |
|---|---|---|
| Basic SQL (n=30) | 40.0% → 43.3% | 0.630 → 0.701 |
| **Aggregation (n=9)** ✨ | **11.1% → 22.2% (2×)** | **0.471 → 0.573 (+21.6%)** |
| Single Join (n=5) | 0% → 0% | 0.095 → 0.177 |

**Key finding:** Strongest gains on **aggregation queries** — the fine-tuned model learned to avoid unnecessary self-joins and produce cleaner, more idiomatic SQL.

### Case study

**Question:** *"How many distinct genres per location?"*

| | SQL |
|---|---|
| **Base model** ❌ | `SELECT T1.location, COUNT(DISTINCT T1.genre) FROM media AS T1 JOIN media AS T2 ON T1.location = T2.location` |
| **Fine-tuned** ✅ | `SELECT location, COUNT(DISTINCT genre) FROM media GROUP BY location` |

Fine-tuned model dropped the unnecessary self-join — cleaner, faster, semantically equivalent to ground truth.

---

## 🏗️ Architecture

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Data Prep  │ →  │  LoRA Train │ →  │ Fuse Model  │ →  │  Evaluate   │ →  │ Gradio Demo │
│ 5K JSONL    │    │ 1000 iters  │    │ Merge LoRA  │    │ 50 samples  │    │Interactive  │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

### Tech stack

| Layer | Choice | Why |
|---|---|---|
| Base model | **Qwen 2.5-3B Instruct** (4-bit) | Best 3B-class instruction follower; MLX-quantized available |
| Method | **LoRA** (rank adaptation) | Train 6.65M / 3.09B params (0.216%) — fits in 16GB |
| Framework | **MLX** (Apple Silicon) | Native Metal acceleration; unified memory avoids CPU↔GPU copies |
| Dataset | `gretelai/synthetic_text_to_sql` (Apache 2.0) | 105K records, 100 domains, 7 SQL complexity categories |
| Eval | Exact Match + BLEU + Keyword Overlap | Triangulate — exact match too strict alone |
| Demo UI | Gradio + Ollama (base) / MLX (fine-tuned) | Side-by-side comparison |

---

## ⚙️ Training Configuration

```yaml
model: mlx-community/Qwen2.5-3B-Instruct-4bit
adapter_path: adapters
data: data

# LoRA
lora_layers: 16
learning_rate: 1.0e-05

# Training
iters: 1000
batch_size: 1
max_seq_length: 1024
save_every: 200
steps_per_eval: 100
steps_per_report: 50
val_batches: 25
seed: 42
```

**Training cost:** ~33 minutes on MacBook Pro M4, 16GB unified memory.

---

## 🚀 Reproduce

### Prerequisites
- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.10+
- ~5GB disk space

### Install
```bash
pip install mlx-lm datasets gradio nltk
```

### Run end-to-end
```bash
# 1. Prepare data (downloads gretelai/synthetic_text_to_sql, writes train/valid/test.jsonl)
jupyter execute notebooks/01_data_prep.ipynb

# 2. Train LoRA adapters (~33 min on M4)
mlx_lm.lora --config lora_config.yaml

# 3. Fuse adapters into model
mlx_lm.fuse --model mlx-community/Qwen2.5-3B-Instruct-4bit \
            --adapter-path adapters \
            --save-path fused_model

# 4. Evaluate
jupyter execute notebooks/03_evaluation.ipynb

# 5. Launch Gradio demo
python demo/app.py
```

### Inference (after fusing)
```python
from mlx_lm import load, generate

model, tokenizer = load("fused_model")

prompt = """You are a SQL expert. Given the schema and question, generate SQL.

Schema:
CREATE TABLE media (id INT, title VARCHAR(50), location VARCHAR(50), genre VARCHAR(50));

Question: How many distinct genres per location?

SQL:"""

response = generate(model, tokenizer, prompt=prompt, max_tokens=200)
print(response)
```

---

## 📁 Repo Structure

```
text-to-sql-lora-mlx/
├── README.md                          # You are here
├── lora_config.yaml                   # MLX-LM LoRA training config
├── Modelfile                          # Ollama deployment config
├── notebooks/
│   ├── 01_data_prep.ipynb            # Load Gretel dataset → JSONL
│   ├── 02_fine_tuning_full.ipynb     # End-to-end training pipeline
│   └── 03_evaluation.ipynb           # EM / BLEU / Keyword Overlap
├── results/
│   ├── evaluation_results.json       # Per-sample base vs fine-tuned
│   └── presentation.pdf              # Project summary slides
├── demo/
│   └── app.py                         # Gradio side-by-side UI
└── data/
    └── .gitkeep                       # (data files gitignored)
```

---

## 🔍 Design Decisions & Trade-offs

**Why LoRA over full fine-tuning?**
Full fine-tuning a 3B model requires ~24GB VRAM and hours on a discrete GPU. LoRA trains only the adapter matrices (rank decomposition), reducing trainable params by **~99.8%** with minimal accuracy loss for downstream tasks.

**Why 3B and not 7B?**
3B fits comfortably in 16GB unified memory with headroom for inference. 7B is feasible on M4 Pro/Max but training time scales ~2.3×, and the marginal accuracy gain for syntactic tasks like SQL generation isn't proportional.

**Why MLX over PyTorch on MPS?**
MLX is purpose-built for Apple Silicon — unified memory model means no CPU↔GPU tensor copies. In practice, ~1.5–2× faster training than PyTorch+MPS for this workload.

**Why both BLEU and Exact Match?**
Exact Match is **overly strict** — semantically equivalent SQL (different column aliases, JOIN ordering, whitespace) scores 0. BLEU captures partial credit. Keyword Overlap is a sanity check that the model is at least producing the right tables/columns.

---

## ⚠️ Limitations & Future Work

- **3B model has limited capacity for complex SQL** (subqueries, window functions). Scaling to 7B would likely improve `Multiple Joins` and `Window Functions` categories.
- **Only 5,000 training samples used** (out of 105K available). Full dataset would yield stronger results.
- **Exact Match is too strict** — future work: add **execution-based evaluation** (run the SQL against a database, compare result sets instead of syntax).
- **No subquery / window function specialization** — current dataset sampling under-represents these.

---

## 📜 License & Attribution

- **Code:** MIT License
- **Base model:** [Qwen 2.5-3B Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) (Apache 2.0)
- **Dataset:** [gretelai/synthetic_text_to_sql](https://huggingface.co/datasets/gretelai/synthetic_text_to_sql) (Apache 2.0)
- **Framework:** [Apple MLX](https://github.com/ml-explore/mlx)

---

## 👤 Author

**Renna Wu** — MS in Information Systems, Northeastern University (Toronto)
[LinkedIn](https://linkedin.com/in/renna-wu) · [GitHub](https://github.com/RennaWu)

Open to Fall 2026 Co-op opportunities in **Data Engineering**, **Applied AI**, and **LLM Engineering**.
