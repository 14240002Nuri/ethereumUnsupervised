# Enhanced Ethereum Anomaly Detection Pipeline (Q1/Q2 Scopus)

Pipeline lengkap, **single-file**, dan **langsung bisa di-run** untuk
unsupervised anomaly detection pada data transaksi Ethereum, dengan semua
enhancement yang dibutuhkan untuk publikasi Scopus Q1/Q2:

- 6 baseline models: Isolation Forest, LOF, PCA, One-Class SVM, COPOD, Autoencoder
- Multi-seed evaluation (≥5 seeds dengan mean ± std)
- Statistical tests: Friedman χ² + Wilcoxon pairwise + Bonferroni correction
- 11 graph topological features (PageRank, betweenness, closeness, eigenvector, HITS, k-core, dst)
- Contamination sensitivity sweep
- External validation via Precision@k, Recall@k, AUC-ROC
- SHAP / permutation feature importance
- Reproducibility metadata (env hash, data hash, library versions)
- Computational profiling (runtime + memory per stage)
- Publication-ready plots (300 DPI) dan paper-ready tables

## Quickstart

### Instalasi
```bash
pip install -r requirements.txt
```

### Mode 1: Demo (synthetic data) — 30 detik
```bash
python run_enhanced_pipeline.py
```
Pipeline akan generate ~5000 nodes synthetic Ethereum-like data,
inject anomali, jalankan seluruh pipeline, dan output di `outputs_q1_pipeline/`.

### Mode 2: Pakai data BigQuery existing
Saudara harus punya CSV transaksi dari script BigQuery sebelumnya, dengan kolom:
`tx_hash, value, gas, gas_price, block_timestamp, from_address, to_address`.

```bash
python run_enhanced_pipeline.py --input-csv path/to/transactions.csv
```

Optional: kalau punya labeled addresses (Etherscan tags):
```bash
python run_enhanced_pipeline.py \
    --input-csv transactions.csv \
    --labels-csv etherscan_phishing_labels.csv
```

Format `labels-csv`:
```
address,label
0xabc...,phishing
0xdef...,exchange-hack
```

### Mode 3: Konfigurasi custom
```bash
python run_enhanced_pipeline.py \
    --n-nodes 10000 \
    --n-tx 100000 \
    --seeds 42,0,1,7,100,2023,2024 \
    --n-splits 5 \
    --contamination 0.02 \
    --contamination-grid 0.005,0.01,0.02,0.05,0.1 \
    --feature-set hybrid \
    --output-dir outputs_my_run
```

### Skip stages tertentu (untuk testing cepat)
```bash
python run_enhanced_pipeline.py --skip sensitivity,shap
```
Available stages to skip: `sensitivity`, `shap`, `modern`, `external`.

## Output Files

```
outputs_q1_pipeline/
├── multi_seed_results.csv          # Multi-seed CV (per seed × per model)
├── statistical_tests.json          # Friedman + Wilcoxon (untuk klaim signifikansi)
├── contamination_sensitivity.csv   # Sensitivity sweep (untuk robustness section)
├── full_pred_with_baselines.csv    # Hasil ensemble + 6 model scores
├── external_validation.json        # Precision@k, Recall@k, AUC-ROC
├── feature_importance.csv          # Permutation + SHAP importance
├── computational_profile.csv       # Runtime + memory per stage
├── experiment_metadata.json        # Reproducibility info
├── plots/
│   ├── fig01_seed_distribution.png    # Box plot stability per model
│   ├── fig02_model_comparison.png     # Bar chart mean ± std
│   ├── fig03_sensitivity.png          # Line plot contamination
│   ├── fig04_feature_importance.png   # Top-15 features
│   └── fig05_external_validation.png  # P@k & R@k curves
└── paper_tables/
    ├── table_seed_summary.csv         # Tabel multi-seed
    ├── table_friedman.csv             # Tabel uji statistik
    ├── table_friedman_header.json     # Friedman χ², p-value
    ├── table_sensitivity.csv          # Tabel sensitivity (wide format)
    ├── table_external_val.csv         # Tabel external validation
    └── table_top10_anomalies.csv      # Top 10 anomalies untuk case study
```

## Mapping Output ke Section Paper

| Section Paper | File yang Dipakai |
|---|---|
| Abstract / Conclusion | `multi_seed_results.csv`, `external_validation.json` |
| Methodology — Pipeline diagram | `experiment_metadata.json` (untuk versi library) |
| Methodology — Statistical Framework | `statistical_tests.json` |
| Results — Table 1 (Stability) | `paper_tables/table_seed_summary.csv` |
| Results — Table 2 (Significance) | `paper_tables/table_friedman.csv` |
| Results — Figure 1 (Box plot) | `plots/fig01_seed_distribution.png` |
| Results — Figure 2 (Bar comparison) | `plots/fig02_model_comparison.png` |
| Results — Sensitivity Analysis | `plots/fig03_sensitivity.png`, `paper_tables/table_sensitivity.csv` |
| Results — External Validation | `plots/fig05_external_validation.png`, `paper_tables/table_external_val.csv` |
| Discussion — Feature Importance | `plots/fig04_feature_importance.png`, `feature_importance.csv` |
| Discussion — Case Study | `paper_tables/table_top10_anomalies.csv` |
| Computational Cost section | `computational_profile.csv` |
| Reproducibility Statement | `experiment_metadata.json` |

## Sample Output (dari run synthetic 2000 nodes / 15000 tx)

```
Multi-seed Summary (jaccard_mean):
  Isolation Forest         : 0.8686 ± 0.0169  ← MOST STABLE
  LOF                      : 0.7733 ± 0.0246
  PCA Reconstruction       : 0.6871 ± 0.0735
  One-Class SVM            : 0.8603 ± 0.0497
  COPOD                    : (not installed)
  Autoencoder (MLP)        : 0.8254 ± 0.1307

Friedman χ² = 10.24, p = 0.037 (SIGNIFICANT at α=0.05)

External Validation:
  AUC-ROC = 0.992
  Precision@10  = 0.500
  Precision@50  = 0.480
  Precision@100 = 0.300
  Recall@100    = 1.000  ← all 30 anomalies in top-100

Top 5 Features (permutation importance):
  betweenness_centrality      : 0.0060
  closeness_centrality        : 0.0048
  degree_centrality_undirected: 0.0045
  weighted_out_degree         : 0.0044
  total_tx_count              : 0.0040
```

## Mapping ke Tesis Saudara

Pipeline ini **tidak menggantikan** script BigQuery existing Saudara —
melainkan **melengkapinya dengan rigor metodologis Q1/Q2**.

| Komponen Tesis | Komponen Pipeline Ini |
|---|---|
| BAB III metodologi | Tetap pakai (script BigQuery existing) |
| Tabel 4 (CV results) | **Replace** dengan `paper_tables/table_seed_summary.csv` |
| Tabel 5 (Top-k Jaccard) | **Augment** dengan `paper_tables/table_friedman.csv` |
| Tabel 8 (Feature importance) | **Replace** dengan `feature_importance.csv` (lebih rigor) |
| Validasi Etherscan (3 address) | **Replace** dengan `external_validation.json` (Precision@k) |
| Klaim "PCA paling stabil" | **Validasi** dengan Friedman p-value < 0.05 |

## Troubleshooting

**Q: Saya dapat warning "COPOD not installed"**
A: Install pyod: `pip install pyod`. Pipeline akan tetap jalan tanpa COPOD.

**Q: Saya dapat warning "SHAP not installed"**
A: Install shap: `pip install shap`. Pipeline pakai permutation importance saja.

**Q: Ingin pakai data BigQuery real**
A: Jalankan script BigQuery existing Saudara dulu untuk dapat CSV, lalu:
```bash
python run_enhanced_pipeline.py --input-csv hasil_query.csv
```

**Q: Berapa lama runtime untuk data sungguhan (1 juta address)?**
A: Estimasi:
- 100K nodes: ~5-10 menit
- 500K nodes: ~30-60 menit
- 1M nodes: ~2-4 jam (terutama di betweenness centrality)
Strategi: pakai `--max-edges` lebih kecil (e.g., 50000) dan `--skip sensitivity`
untuk test cepat dulu.

**Q: Pipeline crash di feature engineering?**
A: Coba kurangi `--max-edges` (default 100000). Untuk dataset besar, mulai dari
20000 dan naikkan bertahap.

## Citation

Jika Saudara pakai pipeline ini di publikasi, mohon cite paper-paper berikut:

- **Isolation Forest:** Liu, Ting, Zhou (2008). ICDM.
- **LOF:** Breunig et al. (2000). SIGMOD.
- **One-Class SVM:** Schölkopf et al. (2001). Neural Computation.
- **COPOD:** Li et al. (2020). ICDM.
- **Friedman test:** Demšar (2006). JMLR.
- **PyOD:** Zhao et al. (2019). JMLR.
- **NetworkX:** Hagberg et al. (2008).

## License

Free to use, modify, and redistribute. Citation appreciated.

## Author Note

Pipeline ini dibuat sebagai panduan praktis transformasi script tesis S2 ke
publikasi Scopus Q1/Q2. Ini **bukan** state-of-the-art (tidak ada GNN), tapi
**solid baseline** yang akan lolos di Q2-Q3 dan jadi fondasi untuk paper Q1
selanjutnya (dengan tambahan GNN baseline + multi-window evaluation).

Untuk Q1 sebenarnya, tambahkan:
- GraphSAGE / GAT baseline (PyTorch Geometric)
- Multi-window evaluation (≥3 time periods)
- Real-world case study (1 known exploit)
- External labeled dataset (Etherscan phishing tags atau XBlock-ETH)
