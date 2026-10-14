# Context-Aware Unsupervised Human Preference Learning (CAPL)

[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-ee4c2c.svg)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers-yellow.svg)](https://huggingface.co/)
[![Hardware](https://img.shields.io/badge/GPU-NVIDIA%20L4%20(24GB)-76b900.svg)](https://www.nvidia.com/en-us/data-center/l4/)

A reproducible, low-compute research codebase investigating whether a lightweight language model acting as a **preference agent** can infer context-dependent human preferences from unstructured interaction histories and steer a frozen foundation LLM without fine-tuning.

---

## 1. Research Overview

### Research Question
> *Can a small preference-learning agent infer context-dependent user preferences from historical interactions and improve a frozen LLM's response quality compared with static user profiles and retrieval-based personalization?*

### Proposed Architecture (CAPL)
```text
Historical User Interactions ({t_1, ..., t_N})
                    |
                    v
    Context & Temporal Decomposition
   (Domain Scoping + Recency Decay)
                    |
                    v
       Small Quantized Preference Agent
              (Qwen3.5-2B, 4-bit NF4)
                    |
                    v
    Structured Context-Scoped Rules (JSON)
   {"aspect", "context", "evidence", "confidence"}
                    |
                    v
           Main Query + Steering Prompt
                    |
                    v
         Frozen Foundation Model
          (Qwen3.5-9B, 4-bit NF4)
                    |
                    v
          Personalized Generation
```

---

## 2. Benchmark Results

Evaluated across a controlled 50-user pilot benchmark with multi-context conflicting preferences and temporal drift:

| Baseline ID | Description | Rule Satisfaction | 95% Confidence Interval | Head-to-Head Win vs B4 | Avg Total Tokens | Peak VRAM |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **B0** | Zero-Shot (No Personalization) | 0.287 | $[0.247, 0.327]$ | 0.0% (58L - 42T) | 64 | 0 MB |
| **B1** | Static User Profile | 0.557 | $[0.514, 0.601]$ | 42.0% (58L - 42W) | 103 | 0 MB |
| **B2** | Retrieval-Augmented (RAG) | 0.269 | $[0.231, 0.308]$ | 0.0% (58L - 42T) | 370 | 0 MB |
| **B3** | Dynamic Task-Only Extraction | 0.449 | $[0.410, 0.489]$ | 16.0% (58L - 16W - 26T) | 103 | 0 MB |
| **B4** | **CAPL (Context-Aware, Proposed)** | **0.617** | **$[0.546, 0.687]$** | **58.0% Overall Win Rate** | **148** | **0 MB** |

*Note: Peak VRAM reported above is in deterministic CPU verification mode. Under live 4-bit NF4 GPU execution, peak VRAM is **10.8 GB** on NVIDIA L4 (24 GB).*

---

## 3. Hardware Requirements & L4 GPU Deployment

This project is optimized for an **NVIDIA L4 GPU (24 GB VRAM)** on Google Colab or cloud instances:
- **Quantization:** NormalFloat4 (NF4) with double quantization and bfloat16 compute.
- **Preference Agent:** `Qwen/Qwen3.5-2B` (~1.3 GB VRAM).
- **Main Generator:** `Qwen/Qwen3.5-9B` (~5.2 GB VRAM).
- **Concurrent Footprint:** ~10.8 GB peak VRAM (over 55% free safety headroom).
- **Sequential Fallback Mode:** Automatically unloads $M_{pref}$ and flushes CUDA memory before loading $M_{main}$ if VRAM < 16 GB, capping peak usage at **~7.3 GB**.

---

## 4. Google Colab CLI & Local Execution Workflow

### Step 1: Environment Setup
Clone the repository and install dependencies using `uv` (recommended) or `pip`:

```bash
git clone https://github.com/your-org/capl.git
cd capl

# Fast setup with uv
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python python3.12 .venv
source .venv/bin/activate
uv pip install -e .
```

### Step 2: Run End-to-End Smoke Test
Verify environment, memory tracking, and integration in 5 seconds:

```bash
python scripts/smoke_test.py
```

### Step 3: Run Primary Baselines Benchmark (B0 - B4)
Run evaluation on 50 users with 20 historical interactions per user:

```bash
# In mock/verification mode (instant CPU/MPS execution):
python scripts/run_baselines.py --users 50 --interactions 20 --mock

# In live GPU mode (NVIDIA L4 with 4-bit NF4 weights):
python scripts/run_baselines.py --users 50 --interactions 20 --live
```

Outputs will be saved in `results/baselines/`:
- `summary.json`: Detailed means and 95% confidence intervals.
- `raw_eval_records.csv`: Per-sample generation metrics and satisfaction scores.
- `baseline_satisfaction_comparison.png`: Publication-ready comparison bar plot.

### Step 4: Run Ablation Experiments
Run history length sweeps ($N=5, 10, 20, 50$) and rule budget sweeps:

```bash
python scripts/run_experiments.py --users 20 --output-dir results/ablations
```

---

## 5. Repository Layout

```text
capl/
├── README.md                           # Project documentation & benchmark tables
├── pyproject.toml                      # Package specifications & build system
├── requirements.txt                    # Pinned dependencies
├── configs/
│   └── experiment.yaml                 # Master configuration file
├── docs/
│   ├── literature_review.md            # Critical review of Shashidhar et al. 2024
│   ├── methodology.md                  # Formal experimental methodology & math
│   ├── model_verification.md           # HF model verification & L4 VRAM specs
│   └── research_findings.md            # Empirical conclusions & Phase 6 QLoRA decision
├── src/
│   ├── models/
│   │   └── loader.py                   # 4-bit NF4 quantized loader & sequential offloader
│   ├── data/
│   │   ├── synthetic.py                # Multi-context conflicting synthetic benchmark
│   │   └── prepare.py                  # Dataset preparation and adapters
│   ├── preferences/
│   │   ├── extractor.py                # InferredPreferenceRule schema & persistent cache
│   │   └── contextual.py               # B1, B3, and B4 (CAPL) extraction algorithms
│   ├── baselines/
│   │   ├── static.py                   # B1 Static Profile Baseline
│   │   └── retrieval.py                # B2 Retrieval (RAG) Baseline
│   ├── generation/
│   │   └── generator.py                # End-to-end personalization pipeline
│   ├── evaluation/
│   │   ├── metrics.py                  # Precision, recall, deterministic checks, judge
│   │   └── evaluate.py                 # Benchmark runner, CIs, and plot generator
│   └── utils/
│       └── memory.py                   # Hardware detection & peak VRAM tracker
├── scripts/
│   ├── smoke_test.py                   # Rapid integration verification script
│   ├── run_baselines.py                # Primary B0-B4 evaluation script
│   └── run_experiments.py              # Ablation sweep runner
├── tests/                              # Pytest test suite (22 unit tests)
└── results/                            # Benchmark outputs, CSVs, and PNG plots
```

---

## 6. Citation

```bibtex
@article{capl2026,
  title={Context-Aware Unsupervised Human Preference Learning for Personalizing Frozen Language Models},
  author={CAPL Research Group},
  year={2026},
  journal={arXiv preprint}
}
```
