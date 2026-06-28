# Unsupervised Anomaly Detection on Ethereum Transactions Using Graph-Based Features

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey)]()

This repository contains the full implementation for the thesis research on unsupervised anomaly detection in Ethereum blockchain transaction networks. The pipeline constructs a transaction graph from on-chain data, extracts graph topological features alongside transactional features, and evaluates six unsupervised anomaly detection models under a rigorous multi-seed cross-validation framework with statistical significance testing.

---

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Data Source](#data-source)
- [Usage](#usage)
  - [Mode 1: Demo with Synthetic Data](#mode-1-demo-with-synthetic-data)
  - [Mode 2: Real Ethereum Data via BigQuery](#mode-2-real-ethereum-data-via-bigquery)
  - [Mode 3: Cross-Feature-Set Comparison](#mode-3-cross-feature-set-comparison)
- [Feature Sets](#feature-sets)
- [Models](#models)
- [Evaluation Framework](#evaluation-framework)
- [Output Files](#output-files)
- [Reproducing Paper Results](#reproducing-paper-results)
- [Citation](#citation)

---

## Overview

Anomaly detection on blockchain networks is challenging due to the absence of ground-truth labels, high transaction volume, and the complex relational structure of address interactions. This work proposes a graph-based unsupervised framework that:

1. Constructs a directed weighted transaction graph from raw Ethereum transaction data.
2. Extracts **11 graph topological features** per address node (PageRank, betweenness centrality, closeness centrality, eigenvector centrality, HITS authority/hub scores, k-core number, in/out degree, weighted degree).
3. Evaluates three feature sets — **transactional**, **graph**, and **hybrid** — across **six unsupervised models**.
4. Applies a **30-seed multi-fold cross-validation** protocol with Friedman χ² and Wilcoxon signed-rank post-hoc tests (Bonferroni-corrected) to substantiate model comparison claims.
5. Validates findings against externally labeled addresses (Etherscan phishing tags) via Precision@k, Recall@k, and AUC-ROC.

---

## Repository Structure

```
.
├── eth_anomaly_detection_graph_based_cv_novelty.py   # Core pipeline: BigQuery integration + graph feature extraction
├── fetch_bq_realtime_and_analyze.py                  # Fetch real-time Ethereum data from BigQuery and run analysis
├── run_enhanced_pipeline.py                           # Enhanced single-file pipeline with demo mode (no BigQuery needed)
├── run_all_feature_sets.py                            # Wrapper: runs all three feature sets and produces combined tables
├── requirements.txt                                   # Python dependencies
└── README.md
```

---

## Requirements

| Package | Version | Role |
|---------|---------|------|
| Python | ≥ 3.9 | Runtime |
| numpy | ≥ 1.24 | Numerical operations |
| pandas | ≥ 2.0 | Data manipulation |
| scipy | ≥ 1.10 | Statistical tests |
| scikit-learn | ≥ 1.3 | ML models, cross-validation |
| networkx | ≥ 3.0 | Graph construction and centrality |
| matplotlib | ≥ 3.7 | Publication-ready figures |
| pyod | ≥ 2.0 | COPOD baseline |
| shap | ≥ 0.43 | SHAP feature importance |
| psutil | ≥ 5.9 | Memory/runtime profiling |
| scikit-posthocs | ≥ 0.7 | Nemenyi post-hoc test |

**BigQuery dependencies** (only required for real data mode):

```
google-cloud-bigquery>=3.13
pyarrow>=14.0
db-dtypes>=1.2
google-auth-oauthlib>=1.1
```

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/14240002Nuri/ethereumUnsupervised.git
cd ethereumUnsupervised

# 2. Create and activate a virtual environment (recommended)
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Data Source

This study uses the **Google BigQuery public dataset** `bigquery-public-data.crypto_ethereum.transactions`, which contains all Ethereum mainnet transactions. The dataset is publicly available and queryable at no cost within BigQuery free-tier limits (1 TB/month).

To access real data, you need:
1. A [Google Cloud](https://cloud.google.com/) project with BigQuery API enabled.
2. An OAuth 2.0 Desktop App credential file (`client_secret.json`) downloaded from the Google Cloud Console.
3. Place `client_secret.json` in the project root directory.

First-time authentication will open a browser window for Google account authorization. Credentials are cached locally in `bigquery_user_token.json` for subsequent runs.

> **Privacy note:** Do not commit `client_secret.json` or `bigquery_user_token.json` to version control. These files are excluded via `.gitignore`.

---

## Usage

### Mode 1: Demo with Synthetic Data

No BigQuery account required. Generates synthetic Ethereum-like transaction data, injects known anomalies, and runs the full pipeline.

```bash
python run_enhanced_pipeline.py
```

**Custom parameters:**

```bash
python run_enhanced_pipeline.py \
    --n-nodes 5000 \
    --n-tx 50000 \
    --seeds 42,0,1,7,100,2023,2024 \
    --n-splits 5 \
    --contamination 0.02 \
    --contamination-grid 0.005,0.01,0.02,0.05,0.1 \
    --feature-set hybrid \
    --output-dir outputs_my_run
```

**Skip heavy stages for a quick test:**

```bash
python run_enhanced_pipeline.py --skip sensitivity,shap
```

Available stages to skip: `sensitivity`, `shap`, `modern`, `external`.

Expected runtime: ~30 seconds (default synthetic size).

---

### Mode 2: Real Ethereum Data via BigQuery

#### Step 2a — Fetch and analyze in one command

```bash
python fetch_bq_realtime_and_analyze.py
```

Fetches the most recent 7 days of Ethereum transactions (up to 500,000 records) and runs the full detection pipeline. Output is saved in `outputs_realtime_YYYYMMDD_HHMMSS/`.

**Custom date range and volume:**

```bash
python fetch_bq_realtime_and_analyze.py \
    --start-date 2025-01-01 \
    --end-date 2025-03-31 \
    --limit 1000000 \
    --feature-set hybrid
```

**Use cached data (skip re-querying BigQuery):**

```bash
python fetch_bq_realtime_and_analyze.py --use-cache
```

#### Step 2b — Core pipeline with an existing CSV

If you already have a transaction CSV with columns `tx_hash, value, gas, gas_price, block_timestamp, from_address, to_address`:

```bash
python run_enhanced_pipeline.py --input-csv path/to/transactions.csv
```

**With externally labeled addresses for validation** (e.g., Etherscan phishing tags):

```bash
python run_enhanced_pipeline.py \
    --input-csv transactions.csv \
    --labels-csv etherscan_labels.csv
```

Expected format for `labels-csv`:

```
address,label
0xabc123...,phishing
0xdef456...,exchange-hack
```

**Estimated runtimes for real data:**

| Dataset size | Estimated time |
|---|---|
| 100,000 addresses | ~5–10 minutes |
| 500,000 addresses | ~30–60 minutes |
| 1,000,000 addresses | ~2–4 hours |

For large datasets, start with `--max-edges 50000` and `--skip sensitivity` to verify the pipeline before a full run.

---

### Mode 3: Cross-Feature-Set Comparison

Runs all three feature sets (transactional, graph, hybrid) sequentially and produces unified comparison tables required for Hypothesis H2 validation.

```bash
python run_all_feature_sets.py
```

**With an existing parquet cache:**

```bash
python run_all_feature_sets.py \
    --from-parquet bq_cache_eth_transactions/transactions_2025-01-01_2025-01-31.parquet
```

**Reuse an existing hybrid run to save time:**

```bash
python run_all_feature_sets.py \
    --from-parquet bq_cache_eth_transactions/transactions.parquet \
    --reuse-hybrid outputs_realtime_20260506_155941
```

Estimated runtime: 4–8 hours (all three feature sets, 30 seeds).

---

## Feature Sets

| Feature Set | Description | Dimensionality |
|---|---|---|
| `transactional` | Raw transaction statistics per address: total value, gas usage, transaction count, active days, avg inter-arrival time | 8 features |
| `graph` | Graph topological features derived from the directed weighted transaction graph | 11 features |
| `hybrid` | Concatenation of transactional and graph features | 19 features |

**Graph topological features (11):**

| Feature | Description |
|---|---|
| PageRank | Global influence score |
| Betweenness centrality | Bridge/relay position in network |
| Closeness centrality | Average shortest path to all nodes |
| Eigenvector centrality | Connection to high-influence nodes |
| HITS authority score | Receive-side importance |
| HITS hub score | Send-side importance |
| K-core number | Participation in densely connected subgraphs |
| In-degree / out-degree | Number of unique senders/receivers |
| Weighted in-degree / out-degree | Total ETH received/sent |

---

## Models

| Model | Algorithm type | Library |
|---|---|---|
| Isolation Forest | Ensemble tree-based | scikit-learn |
| LOF | Density-based | scikit-learn |
| PCA Reconstruction | Linear dimensionality reduction | scikit-learn |
| One-Class SVM | Kernel-based boundary | scikit-learn |
| COPOD | Copula-based | PyOD |
| Autoencoder (MLP) | Neural reconstruction | scikit-learn (MLPRegressor) |

All models are trained in a fully unsupervised setting (no labels used during training).

---

## Evaluation Framework

### Multi-Seed Cross-Validation
- **30 random seeds** (42, 0, 1, 7, 100, 2023, 2024, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103)
- **5-fold cross-validation** per seed
- Primary metric: **Jaccard similarity** between top-k anomaly sets across folds
- Reported as: `mean ± std` over 30 seeds

### Statistical Testing
1. **Friedman χ² test** — non-parametric test for overall model ranking differences
2. **Wilcoxon signed-rank test** — pairwise post-hoc comparison
3. **Bonferroni correction** — family-wise error rate control

### External Validation (if labeled data available)
- Precision@k (k = 10, 50, 100)
- Recall@k
- AUC-ROC

### Sensitivity Analysis
- Contamination ratio sweep: {0.005, 0.01, 0.02, 0.05, 0.10}

---

## Output Files

```
outputs_*/
├── multi_seed_results.csv           # Per-seed × per-model Jaccard scores
├── statistical_tests.json           # Friedman χ², Wilcoxon p-values
├── contamination_sensitivity.csv    # Model stability across contamination levels
├── full_pred_with_baselines.csv     # Anomaly scores for all addresses
├── external_validation.json         # Precision@k, Recall@k, AUC-ROC
├── feature_importance.csv           # SHAP + permutation importance rankings
├── computational_profile.csv        # Runtime and memory per pipeline stage
├── experiment_metadata.json         # Library versions, data hash (reproducibility)
├── plots/
│   ├── fig01_seed_distribution.png  # Boxplot: score stability per model
│   ├── fig02_model_comparison.png   # Barplot: mean ± std across seeds
│   ├── fig03_sensitivity.png        # Line plot: contamination sensitivity
│   ├── fig04_feature_importance.png # Top-15 features
│   └── fig05_external_validation.png
└── paper_tables/
    ├── table_seed_summary.csv       # Table: multi-seed results (paper-ready)
    ├── table_friedman.csv           # Table: statistical test results
    ├── table_sensitivity.csv        # Table: sensitivity analysis (wide format)
    ├── table_external_val.csv       # Table: external validation metrics
    └── table_top10_anomalies.csv    # Top 10 detected anomalous addresses
```

**Cross-feature-set comparison** (from `run_all_feature_sets.py`):

```
outputs_all_feature_sets/
├── combined_multi_seed_results.csv  # 3 feature sets × 6 models × 30 seeds
├── combined_statistical_tests.json  # Friedman per feature set
├── combined_table_seed_summary.csv  # 18-row paper table
└── by_feature_set/
    ├── transactional/
    ├── graph/
    └── hybrid/
```

---

## Reproducing Paper Results

To reproduce the exact results reported in the paper:

```bash
# Step 1: Fetch data for the study period
python fetch_bq_realtime_and_analyze.py \
    --start-date 2025-01-01 \
    --end-date 2025-03-31 \
    --limit 500000

# Step 2: Run all feature sets with 30 seeds
python run_all_feature_sets.py \
    --from-parquet bq_cache_eth_transactions/transactions_2025-01-01_2025-03-31.parquet

# Step 3: Check experiment_metadata.json to verify data hash and library versions
```

The file `experiment_metadata.json` in each output directory records:
- SHA-256 hash of the input dataset
- Python and library versions
- Random seeds used
- Timestamp of the experiment

This enables verification that results were produced from the same data and environment.

---

## Citation

If you use this code or methodology in your research, please cite the following work:

```
[Author]. (2026). Unsupervised Anomaly Detection on Ethereum Transactions
Using Graph-Based Features. [Journal Name]. [DOI pending]
```

This implementation also builds on the following prior work:

```
Liu, F.T., Ting, K.M., & Zhou, Z.H. (2008). Isolation Forest.
  In Proc. ICDM 2008. https://doi.org/10.1109/ICDM.2008.17

Breunig, M.M., Kriegel, H.P., Ng, R.T., & Sander, J. (2000).
  LOF: Identifying Density-Based Local Outliers. In Proc. SIGMOD 2000.

Schölkopf, B., Platt, J.C., Shawe-Taylor, J., Smola, A.J., & Williamson, R.C. (2001).
  Estimating the Support of a High-Dimensional Distribution.
  Neural Computation, 13(7), 1443–1471.

Li, Z., Zhao, Y., Botta, N., Ionescu, C., & Hu, X. (2020).
  COPOD: Copula-Based Outlier Detection. In Proc. ICDM 2020.

Zhao, Y., Nasrullah, Z., & Li, Z. (2019).
  PyOD: A Python Toolbox for Scalable Outlier Detection.
  Journal of Machine Learning Research, 20(96), 1–7.

Demšar, J. (2006). Statistical Comparisons of Classifiers over Multiple Data Sets.
  Journal of Machine Learning Research, 7, 1–30.

Hagberg, A.A., Schult, D.A., & Swart, P.J. (2008).
  Exploring Network Structure, Dynamics, and Function using NetworkX.
  In Proc. SciPy 2008.
```

---

## License

This project is released under the MIT License. You are free to use, modify, and distribute this code for academic and commercial purposes, provided that appropriate credit is given.
