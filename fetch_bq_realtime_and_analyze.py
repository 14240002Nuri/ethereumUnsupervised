"""
================================================================================
fetch_bq_realtime_and_analyze.py
================================================================================
Ambil data transaksi Ethereum TERBARU langsung dari Google BigQuery public
dataset, lalu jalankan enhanced anomaly detection pipeline (Q1/Q2 Scopus).

Cara pakai:

  Mode default (7 hari terakhir, 500K tx):
      python fetch_bq_realtime_and_analyze.py

  Tentukan rentang tanggal sendiri:
      python fetch_bq_realtime_and_analyze.py --start-date 2025-03-01 --end-date 2025-03-07

  Tambah jumlah transaksi (lebih lama, lebih akurat):
      python fetch_bq_realtime_and_analyze.py --limit 1000000

  Pakai cache yang sudah ada (tidak query ulang ke BQ):
      python fetch_bq_realtime_and_analyze.py --use-cache

  Skip analisis berat (untuk test cepat):
      python fetch_bq_realtime_and_analyze.py --skip sensitivity,shap

Output di folder outputs_realtime_YYYYMMDD_HHMMSS/ :
  - eth_transactions_realtime.csv    : data mentah dari BigQuery
  - multi_seed_results.csv           : stabilitas 7 seed × 6 model
  - statistical_tests.json           : Friedman + Wilcoxon + Bonferroni
  - full_pred_with_baselines.csv     : skor anomali semua alamat
  - feature_importance.csv           : SHAP + permutation importance
  - anomaly_report.txt               : laporan ringkas siap baca
  - plots/                           : grafik publikasi 300 DPI
  - paper_tables/                    : tabel siap LaTeX/Word

Dependensi BigQuery:
    pip install google-cloud-bigquery pyarrow db-dtypes google-auth-oauthlib
================================================================================
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

# ── Tangkap error import awal dengan pesan jelas ──────────────────────────────
try:
    import numpy as np
    import pandas as pd
except ImportError as e:
    print(f"[ERROR] Dependensi inti tidak ada: {e}")
    print("  pip install numpy pandas")
    sys.exit(1)

try:
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.cloud import bigquery
except ImportError as e:
    print(f"[ERROR] Paket BigQuery tidak ada: {e}")
    print("  pip install google-cloud-bigquery pyarrow db-dtypes google-auth-oauthlib")
    sys.exit(1)


# =============================================================================
# CONFIG
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
GCP_PROJECT_ID = "tesis-gaptek"
BQ_SCOPES = ["https://www.googleapis.com/auth/bigquery"]
TOKEN_FILE = SCRIPT_DIR / "bigquery_user_token.json"
CLIENT_SECRET_FILES = [
    SCRIPT_DIR / "client_secret_desktopapp.json",
    SCRIPT_DIR / "client_secret.json",
]
BQ_DATASET = "bigquery-public-data.crypto_ethereum.transactions"
DEFAULT_LIMIT = 500_000
DEFAULT_DAYS = 7
CACHE_DIR = SCRIPT_DIR / "bq_cache_realtime"
DEFAULT_PAPER_SEEDS = (
    "42,0,1,7,100,2023,2024,11,13,17,19,23,29,31,37,"
    "41,43,47,53,59,61,67,71,73,79,83,89,97,101,103"
)


# =============================================================================
# UTILITIES
# =============================================================================

def log(msg: str, level: str = "INFO") -> None:
    print(f"[{level}] {msg}", flush=True)


def banner(text: str) -> None:
    line = "=" * 70
    print(f"\n{line}\n{text.center(70)}\n{line}")


# =============================================================================
# BAGIAN 1: AUTENTIKASI BIGQUERY (OAuth2 Desktop App)
# =============================================================================

def get_credentials() -> Credentials:
    """
    Ambil/refresh OAuth2 credentials.
    - Jika token.json ada dan valid → pakai langsung
    - Jika expired + punya refresh_token → auto-refresh
    - Jika tidak ada / tidak bisa refresh → buka browser login
    """
    creds: Credentials | None = None

    if TOKEN_FILE.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_FILE), scopes=BQ_SCOPES)

    if creds and creds.valid:
        log("Token BigQuery valid, tidak perlu login ulang.")
        return creds

    if creds and creds.expired and creds.refresh_token:
        log("Token expired, me-refresh otomatis...")
        try:
            creds.refresh(Request())
            TOKEN_FILE.write_text(creds.to_json(), encoding="utf-8")
            log(f"Token berhasil di-refresh, disimpan ke {TOKEN_FILE.name}")
            return creds
        except Exception as e:
            log(f"Refresh gagal ({e}). Token lama dihapus, login ulang diperlukan.", level="WARN")
            # Hapus token yang sudah tidak bisa di-refresh
            try:
                TOKEN_FILE.unlink()
            except Exception:
                pass
            creds = None

    # Butuh login baru
    client_secret_path = None
    for p in CLIENT_SECRET_FILES:
        if p.exists():
            client_secret_path = p
            break

    if client_secret_path is None:
        raise RuntimeError(
            "File OAuth client tidak ditemukan. Letakkan "
            f"client_secret_desktopapp.json atau client_secret.json di {SCRIPT_DIR}"
        )

    log(f"Membuka browser untuk login Google ({client_secret_path.name})...")
    flow = InstalledAppFlow.from_client_secrets_file(str(client_secret_path), scopes=BQ_SCOPES)
    creds = flow.run_local_server(port=0)
    TOKEN_FILE.write_text(creds.to_json(), encoding="utf-8")
    log(f"Token baru tersimpan di {TOKEN_FILE.name}")
    return creds


def get_bq_client() -> bigquery.Client:
    creds = get_credentials()
    return bigquery.Client(project=GCP_PROJECT_ID, credentials=creds)


# =============================================================================
# BAGIAN 2: QUERY BUILDER + FETCH DATA
# =============================================================================

def get_date_range(days: int) -> tuple[str, str]:
    """Hitung rentang tanggal: (today - days) s.d. (today - 1) agar data lengkap."""
    today = datetime.now(timezone.utc).date()
    end = today - timedelta(days=1)
    start = end - timedelta(days=days - 1)
    return str(start), str(end)


def build_eth_query(start_date: str, end_date: str, limit: int) -> str:
    limit_clause = f"\nLIMIT {int(limit)}" if limit > 0 else ""
    return f"""
SELECT
  `hash`                        AS tx_hash,
  value,
  gas,
  gas_price,
  receipt_gas_used,
  max_fee_per_gas,
  max_priority_fee_per_gas,
  nonce,
  transaction_index,
  block_number,
  transaction_type,
  block_timestamp,
  from_address,
  to_address
FROM `{BQ_DATASET}`
WHERE DATE(block_timestamp) BETWEEN '{start_date}' AND '{end_date}'
  AND from_address IS NOT NULL
  AND to_address   IS NOT NULL
  AND receipt_gas_used IS NOT NULL
ORDER BY block_timestamp ASC{limit_clause}
""".strip()


def dry_run_bytes(client: bigquery.Client, query: str) -> int:
    cfg = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
    job = client.query(query, job_config=cfg)
    return int(job.total_bytes_processed or 0)


def fetch_from_bigquery(
    start_date: str,
    end_date: str,
    limit: int,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Fetch data dari BigQuery, dengan file cache Parquet per (start, end, limit).
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_key = f"{start_date}_{end_date}_lim{limit}"
    cache_parquet = CACHE_DIR / f"eth_tx_{cache_key}.parquet"
    cache_meta = CACHE_DIR / f"eth_tx_{cache_key}.json"

    if use_cache and cache_parquet.exists() and cache_meta.exists():
        log(f"Memakai cache: {cache_parquet.name}")
        df = pd.read_parquet(cache_parquet)
        meta = json.loads(cache_meta.read_text(encoding="utf-8"))
        log(f"  {len(df):,} baris, di-cache {meta.get('cached_at', 'unknown')}")
        return df

    log(f"Menghubungi BigQuery: {start_date} s.d. {end_date}, limit {limit:,}")
    client = get_bq_client()
    query = build_eth_query(start_date, end_date, limit)

    # Dry-run untuk estimasi biaya
    try:
        bytes_est = dry_run_bytes(client, query)
        gb_est = bytes_est / (1024 ** 3)
        log(f"  Estimasi data di-scan: {gb_est:.2f} GB "
            f"({'dalam free tier 1TB/bulan' if gb_est < 1024 else 'melebihi 1TB!'})")
    except Exception as e:
        log(f"  Dry-run gagal ({e}), lanjut fetch...", level="WARN")

    log("  Menjalankan query (bisa 30–120 detik untuk data besar)...")
    t0 = time.time()
    df = client.query(query).to_dataframe()
    elapsed = time.time() - t0
    log(f"  Fetch selesai: {len(df):,} baris dalam {elapsed:.1f}s")

    # Simpan cache
    df.to_parquet(cache_parquet, index=False)
    cache_meta.write_text(json.dumps({
        "start_date": start_date,
        "end_date": end_date,
        "limit": limit,
        "row_count": len(df),
        "cached_at": datetime.now(timezone.utc).isoformat(),
        "query_seconds": round(elapsed, 1),
    }, indent=2), encoding="utf-8")
    log(f"  Cache disimpan: {cache_parquet.name}")
    return df


# =============================================================================
# BAGIAN 3: TRANSFORMASI DATA BQ → FORMAT PIPELINE
# =============================================================================

def transform_to_pipeline_format(df: pd.DataFrame) -> pd.DataFrame:
    """
    Petakan kolom BigQuery ke format yang diharapkan run_enhanced_pipeline.py.

    BQ columns → pipeline columns:
      value (Wei string/Decimal) → value + value_eth (float)
      receipt_gas_used           → gas_used
      gas_used * gas_price / 1e18 → gas_fee_native
      block_timestamp.hour       → hour
      block_timestamp.dayofweek  → dayofweek
    """
    log("Transformasi kolom BigQuery ke format pipeline...")

    tx = df.copy()

    # --- value (Wei) → float → ETH ---
    tx["value"] = pd.to_numeric(tx["value"], errors="coerce").fillna(0)
    tx["value_eth"] = tx["value"] / 1e18

    # --- gas_used ---
    tx["gas_used"] = pd.to_numeric(tx["receipt_gas_used"], errors="coerce").fillna(tx["gas"])

    # --- gas_price: jika 0 atau NaN, pakai max_fee_per_gas ---
    tx["gas_price"] = pd.to_numeric(tx["gas_price"], errors="coerce").fillna(0)
    mf = pd.to_numeric(tx.get("max_fee_per_gas", pd.Series(dtype=float)), errors="coerce").fillna(0)
    tx["gas_price"] = tx["gas_price"].where(tx["gas_price"] > 0, mf)

    # --- gas_fee_native (ETH) ---
    tx["gas_fee_native"] = tx["gas_used"] * tx["gas_price"] / 1e18

    # --- gas (int) ---
    tx["gas"] = pd.to_numeric(tx["gas"], errors="coerce").fillna(21000).astype(int)

    # --- timestamp ---
    tx["block_timestamp"] = pd.to_datetime(tx["block_timestamp"], utc=True)
    tx["hour"] = tx["block_timestamp"].dt.hour
    tx["dayofweek"] = tx["block_timestamp"].dt.dayofweek

    # --- pastikan kolom wajib ada ---
    required = ["tx_hash", "value", "value_eth", "gas", "gas_price",
                "gas_used", "gas_fee_native", "block_timestamp",
                "from_address", "to_address", "block_number", "hour", "dayofweek"]
    for col in required:
        if col not in tx.columns:
            tx[col] = 0

    tx = tx[required].copy()
    tx = tx.dropna(subset=["from_address", "to_address"])
    tx = tx[tx["from_address"] != ""].copy()
    tx = tx[tx["to_address"] != ""].copy()

    log(f"  Hasil transformasi: {len(tx):,} baris, "
        f"{tx['from_address'].nunique():,} pengirim unik, "
        f"{tx['to_address'].nunique():,} penerima unik")
    log(f"  Rentang waktu: {tx['block_timestamp'].min()} → {tx['block_timestamp'].max()}")
    log(f"  Nilai ETH: min={tx['value_eth'].min():.6f}, "
        f"max={tx['value_eth'].max():.2f}, "
        f"median={tx['value_eth'].median():.6f}")
    return tx


# =============================================================================
# BAGIAN 4: JALANKAN ENHANCED PIPELINE
# =============================================================================

def run_pipeline(
    tx_df: pd.DataFrame,
    output_dir: Path,
    contamination: float = 0.02,
    seeds: list[int] = None,
    contamination_grid: list[float] = None,
    skip: set[str] = None,
    feature_set: str = "hybrid",
    max_edges: int = 100_000,
    n_splits: int = 5,
    top_k_ratio: float = 0.05,
    large_graph_threshold: int = 999_999,
    max_fit_samples: int = 999_999,
) -> dict:
    """
    Import dan jalankan semua fungsi dari run_enhanced_pipeline secara langsung.
    Kembalikan dict berisi semua hasil untuk laporan.
    """
    import importlib.util, sys

    # Import run_enhanced_pipeline sebagai modul
    spec = importlib.util.spec_from_file_location(
        "pipeline", SCRIPT_DIR / "run_enhanced_pipeline.py"
    )
    pipeline = importlib.util.module_from_spec(spec)
    sys.modules["pipeline"] = pipeline
    spec.loader.exec_module(pipeline)

    if seeds is None:
        seeds = [int(s) for s in DEFAULT_PAPER_SEEDS.split(",")]
    if contamination_grid is None:
        contamination_grid = [0.005, 0.01, 0.02, 0.05, 0.1]
    if skip is None:
        skip = set()

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    profiler = pipeline.PipelineProfiler()
    results = {}

    # ── Stage 1: Feature Engineering ─────────────────────────────────────────
    banner("STAGE 1/8: FEATURE ENGINEERING (DATA REAL BIGQUERY)")
    with pipeline.timed_stage("feature_engineering", profiler):
        edge_df = pipeline.build_edge_features(tx_df)
        node_df = pipeline.build_node_features(tx_df)
        node_df = pipeline.add_topological_features(
            node_df, edge_df, max_edges=max_edges,
            large_graph_threshold=large_graph_threshold,
        )
        feature_sets = pipeline.get_feature_sets(node_df)

    feature_cols = feature_sets[feature_set]
    log(f"Feature set '{feature_set}': {len(feature_cols)} fitur")
    results["node_df"] = node_df
    results["feature_cols"] = feature_cols

    # ── Stage 2: Multi-seed Evaluation ───────────────────────────────────────
    banner("STAGE 2/8: MULTI-SEED STABILITY EVALUATION")
    include_modern = "modern" not in skip
    with pipeline.timed_stage("multi_seed", profiler):
        seed_df = pipeline.multi_seed_evaluation_full(
            node_df, feature_cols, seeds,
            contamination=contamination, n_splits=n_splits,
            top_k_ratio=top_k_ratio, include_modern=include_modern,
            feature_set_name=feature_set,
            max_fit_samples=max_fit_samples,
        )
    seed_df.to_csv(output_dir / "multi_seed_results.csv", index=False)
    results["seed_df"] = seed_df

    # ── Stage 3: Statistical Tests ────────────────────────────────────────────
    banner("STAGE 3/8: STATISTICAL TESTS (FRIEDMAN + WILCOXON)")
    models_used = pipeline.ALL_MODELS if include_modern else pipeline.MODELS_CLASSICAL
    with pipeline.timed_stage("statistical_tests", profiler):
        stat_results = pipeline.friedman_with_posthoc(seed_df, models_used)
    (output_dir / "statistical_tests.json").write_text(
        json.dumps(stat_results, indent=2, default=str), encoding="utf-8"
    )
    results["stat_results"] = stat_results
    if "friedman_p_value" in stat_results:
        log(f"Friedman chi2={stat_results['friedman_chi2']:.4f}, "
            f"p={stat_results['friedman_p_value']:.4e} "
            f"({'SIGNIFIKAN' if stat_results['friedman_significant_at_alpha'] else 'tidak signifikan'})")

    # ── Stage 4: Sensitivity Analysis ────────────────────────────────────────
    sens_df = pd.DataFrame()
    if "sensitivity" not in skip:
        banner("STAGE 4/8: SENSITIVITY ANALYSIS")
        with pipeline.timed_stage("sensitivity", profiler):
            sens_df = pipeline.contamination_sensitivity_sweep(
                node_df, feature_cols, contamination_grid,
                n_splits=n_splits, seed=seeds[0], top_k_ratio=top_k_ratio,
                max_fit_samples=max_fit_samples,
            )
        sens_df.to_csv(output_dir / "contamination_sensitivity.csv", index=False)
    results["sens_df"] = sens_df

    # ── Stage 5: Ensemble Scoring ─────────────────────────────────────────────
    banner("STAGE 5/8: ENSEMBLE FULL-DATASET SCORING")
    with pipeline.timed_stage("ensemble_scoring", profiler):
        full_pred = pipeline.ensemble_full_dataset_scoring(
            node_df, feature_cols, contamination=contamination,
            seed=seeds[0], include_modern=include_modern,
        )
    full_pred.to_csv(output_dir / "full_pred_with_baselines.csv", index=False)
    results["full_pred"] = full_pred

    # ── Stage 6: External Validation (tidak ada label real, skip) ────────────
    log("Skipping external validation (tidak ada ground-truth labels untuk data real)")
    ext_val: dict = {}
    results["ext_val"] = ext_val

    # ── Stage 7: Feature Importance ───────────────────────────────────────────
    imp_df = pd.DataFrame()
    if "shap" not in skip:
        banner("STAGE 7/8: FEATURE IMPORTANCE (SHAP + PERMUTATION)")
        with pipeline.timed_stage("feature_importance", profiler):
            imp_df = pipeline.compute_feature_importance(
                node_df, feature_cols, contamination=contamination, seed=seeds[0],
            )
        imp_df.to_csv(output_dir / "feature_importance.csv", index=False)
    results["imp_df"] = imp_df

    # ── Stage 8: Plots ────────────────────────────────────────────────────────
    banner("STAGE 8/8: GENERATING PUBLICATION-READY PLOTS")
    with pipeline.timed_stage("plotting", profiler):
        pipeline.plot_seed_distribution(seed_df, models_used, plots_dir / "fig01_seed_distribution.png")
        pipeline.plot_model_comparison_bars(seed_df, models_used, plots_dir / "fig02_model_comparison.png")
        if not sens_df.empty:
            pipeline.plot_sensitivity(sens_df, plots_dir / "fig03_sensitivity.png")
        if not imp_df.empty:
            pipeline.plot_feature_importance(imp_df, plots_dir / "fig04_feature_importance.png")

    # ── Paper Tables ─────────────────────────────────────────────────────────
    banner("GENERATING PAPER TABLES")
    pipeline.generate_paper_tables(
        seed_df, stat_results, sens_df, ext_val, full_pred,
        output_dir, models_used,
    )

    # ── Profiler ──────────────────────────────────────────────────────────────
    profiler_df = profiler.to_df()
    profiler_df.to_csv(output_dir / "computational_profile.csv", index=False)
    results["profiler_df"] = profiler_df
    results["models_used"] = models_used

    return results


# =============================================================================
# BAGIAN 5: LAPORAN ANALISIS
# =============================================================================

def generate_report(
    tx_df: pd.DataFrame,
    results: dict,
    output_dir: Path,
    start_date: str,
    end_date: str,
) -> str:
    """Buat laporan teks komprehensif dalam Bahasa Indonesia."""
    node_df: pd.DataFrame = results["node_df"]
    full_pred: pd.DataFrame = results["full_pred"]
    seed_df: pd.DataFrame = results["seed_df"]
    stat_results: dict = results["stat_results"]
    imp_df: pd.DataFrame = results["imp_df"]
    models_used: list = results["models_used"]
    profiler_df: pd.DataFrame = results["profiler_df"]

    # Gunakan pipeline yang sudah diimport (hindari re-import konflik)
    import sys
    pl = sys.modules.get("pipeline")
    if pl is None:
        import importlib.util
        spec = importlib.util.spec_from_file_location("pipeline", SCRIPT_DIR / "run_enhanced_pipeline.py")
        pl = importlib.util.module_from_spec(spec)
        sys.modules["pipeline"] = pl
        spec.loader.exec_module(pl)

    sep = "=" * 70
    lines = [
        sep,
        "LAPORAN ANALISIS ANOMALI ETHEREUM — DATA REAL BIGQUERY".center(70),
        sep,
        f"Tanggal analisis : {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"Rentang data     : {start_date} s.d. {end_date}",
        "",
        "─" * 70,
        "A. STATISTIK DATA",
        "─" * 70,
        f"  Total transaksi        : {len(tx_df):>12,}",
        f"  Total alamat unik      : {node_df['address'].nunique():>12,}",
        f"  Pengirim unik          : {tx_df['from_address'].nunique():>12,}",
        f"  Penerima unik          : {tx_df['to_address'].nunique():>12,}",
        f"  Nilai ETH total        : {tx_df['value_eth'].sum():>14,.2f} ETH",
        f"  Nilai ETH median       : {tx_df['value_eth'].median():>14.6f} ETH",
        f"  Nilai ETH maks         : {tx_df['value_eth'].max():>14.4f} ETH",
        f"  Rentang waktu          : {tx_df['block_timestamp'].min().strftime('%Y-%m-%d %H:%M')} UTC",
        f"                           s.d. {tx_df['block_timestamp'].max().strftime('%Y-%m-%d %H:%M')} UTC",
        "",
        "─" * 70,
        "B. HASIL DETEKSI ANOMALI (ENSEMBLE)",
        "─" * 70,
    ]

    n_flagged = int((full_pred["ensemble_consistent_anomaly"] == 1).sum())
    pct_flagged = n_flagged / len(full_pred) * 100
    lines += [
        f"  Alamat dianalisis      : {len(full_pred):>12,}",
        f"  Terdeteksi anomali     : {n_flagged:>12,}  ({pct_flagged:.2f}%)",
        "",
        "  Top-10 Alamat Paling Anomali:",
        f"  {'Rank':<5} {'Alamat':<45} {'Vote':<6} {'Skor Pct':<10}",
        f"  {'-'*5} {'-'*44} {'-'*6} {'-'*10}",
    ]

    top10 = full_pred.nlargest(10, "ensemble_score_percentile_mean")
    for rank, (_, row) in enumerate(top10.iterrows(), 1):
        addr = str(row["address"])
        addr_short = addr[:6] + "..." + addr[-6:] if len(addr) > 15 else addr
        lines.append(
            f"  {rank:<5} {addr_short:<45} {int(row['ensemble_vote_count']):<6} "
            f"{row['ensemble_score_percentile_mean']:.4f}"
        )

    lines += [
        "",
        "─" * 70,
        "C. STABILITAS MODEL (TOP-K JACCARD STABILITY)",
        "─" * 70,
        f"  {'Model':<28} {'Mean Jaccard':>12} {'Std':>8}",
        f"  {'-'*28} {'-'*12} {'-'*8}",
    ]
    for m in models_used:
        col = f"{m}_jaccard_mean"
        if col in seed_df.columns:
            vals = seed_df[col].dropna()
            if len(vals) > 0:
                lines.append(
                    f"  {pl.MODEL_LABELS[m]:<28} {vals.mean():>12.4f} {vals.std():>8.4f}"
                )

    lines += ["", "─" * 70, "D. UJI STATISTIK", "─" * 70]
    if "friedman_p_value" in stat_results:
        sig = "SIGNIFIKAN" if stat_results["friedman_significant_at_alpha"] else "tidak signifikan"
        lines += [
            f"  Friedman chi2  = {stat_results['friedman_chi2']:.4f}",
            f"  p-value        = {stat_results['friedman_p_value']:.4e}",
            f"  Kesimpulan     : Perbedaan antar model {sig} (alpha=0.05)",
            "",
            "  Pairwise Wilcoxon (Bonferroni-corrected):",
        ]
        for pw in stat_results.get("pairwise_wilcoxon", []):
            if "p_value" in pw:
                sig_sym = "SIGNIFIKAN" if pw.get("significant") else "n.s.      "
                lines.append(
                    f"  [{sig_sym}]  {pw['model_a'][:22]:<22} vs {pw['model_b'][:22]:<22}  "
                    f"p={pw['p_value_bonferroni']:.4e}"
                )
    else:
        lines.append(f"  {stat_results.get('error', 'Tidak dapat dihitung')}")

    if not imp_df.empty:
        lines += ["", "─" * 70, "E. TOP-10 FITUR TERPENTING", "─" * 70]
        perm = imp_df[imp_df["method"] == "permutation_iforest"].nlargest(10, "importance")
        for _, row in perm.iterrows():
            bar = "█" * int(row["importance"] / perm["importance"].max() * 20)
            lines.append(f"  {row['feature']:<35} {row['importance']:.4f}  {bar}")

    lines += ["", "─" * 70, "F. PROFIL KOMPUTASI", "─" * 70]
    total_sec = profiler_df["duration_sec"].sum()
    lines.append(f"  Total waktu eksekusi: {total_sec:.1f} detik ({total_sec/60:.1f} menit)")
    for _, row in profiler_df.iterrows():
        mem_str = f"{row['memory_mb_peak']:.0f} MB" if pd.notna(row.get("memory_mb_peak")) else "N/A"
        lines.append(
            f"  {row['stage']:<30} {row['duration_sec']:>7.1f}s   {mem_str}"
        )

    lines += [
        "",
        "─" * 70,
        "G. OUTPUT FILES",
        "─" * 70,
    ]
    for f in sorted(output_dir.rglob("*")):
        if f.is_file():
            size_kb = f.stat().st_size // 1024
            rel = f.relative_to(output_dir)
            lines.append(f"  {str(rel):<50} {size_kb:>5} KB")

    lines += [
        "",
        sep,
        "SELESAI — Lihat folder output untuk semua hasil.".center(70),
        sep,
    ]

    report_text = "\n".join(lines)
    report_path = output_dir / "anomaly_report.txt"
    report_path.write_text(report_text, encoding="utf-8")
    log(f"Laporan tersimpan: {report_path}")
    return report_text


# =============================================================================
# MAIN
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fetch data Ethereum dari BigQuery + jalankan anomaly detection pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--start-date", default=None,
                   help="Tanggal mulai (YYYY-MM-DD). Default: hari ini - (days+1).")
    p.add_argument("--end-date", default=None,
                   help="Tanggal akhir (YYYY-MM-DD). Default: kemarin.")
    p.add_argument("--days", type=int, default=DEFAULT_DAYS,
                   help="Berapa hari ke belakang (diabaikan jika --start-date diisi).")
    p.add_argument("--limit", type=int, default=DEFAULT_LIMIT,
                   help="Maksimum jumlah transaksi yang di-fetch dari BigQuery.")
    p.add_argument("--use-cache", action="store_true",
                   help="Pakai cache lokal jika ada (tidak re-query BigQuery).")
    p.add_argument("--no-cache", action="store_true",
                   help="Selalu fetch ulang dari BigQuery meski cache ada.")
    p.add_argument("--output-dir", default=None,
                   help="Folder output. Default: outputs_realtime_YYYYMMDD_HHMMSS/")
    p.add_argument("--contamination", type=float, default=0.02)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--top-k-ratio", type=float, default=0.05)
    p.add_argument("--feature-set", choices=["transactional", "graph", "hybrid"], default="hybrid")
    p.add_argument("--max-edges", type=int, default=100_000)
    p.add_argument("--skip", type=str, default="",
                   help="Stage yang dilewati (comma-separated): sensitivity,shap,modern")
    p.add_argument("--seeds", type=str, default=DEFAULT_PAPER_SEEDS,
                   help="Comma-separated random seeds. Default paper run uses 30 seeds "
                        "so Bonferroni-corrected Wilcoxon post-hoc tests have enough power.")
    p.add_argument("--contamination-grid", type=str, default="0.005,0.01,0.02,0.05,0.1")
    p.add_argument("--from-parquet", type=str, default=None,
                   help="Load langsung dari file .parquet (skip BigQuery fetch). "
                        "Contoh: bq_cache_eth_transactions/transactions_2025-01-01_2025-01-31.parquet")
    p.add_argument("--large-graph-threshold", type=int, default=50_000,
                   help="Node threshold untuk skip closeness+hits (O(V^2)). Default: 50000.")
    p.add_argument("--max-fit-samples", type=int, default=50_000,
                   help="Maksimum sampel untuk training model (LOF, dll). Default: 50000.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Output dir
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else SCRIPT_DIR / f"outputs_realtime_{ts}"
    output_dir.mkdir(parents=True, exist_ok=True)

    skip = set(s.strip() for s in args.skip.split(",") if s.strip())
    seeds = [int(s) for s in args.seeds.split(",")]
    contamination_grid = [float(c) for c in args.contamination_grid.split(",")]

    banner("FETCH REAL-TIME BIGQUERY + ETHEREUM ANOMALY DETECTION PIPELINE")
    log(f"Output             : {output_dir.absolute()}")
    log(f"large_graph_threshold : {args.large_graph_threshold:,} nodes")
    log(f"max_fit_samples       : {args.max_fit_samples:,}")
    log(f"Skip               : {skip if skip else 'none'}")

    t_fetch_start = time.time()

    # ── STAGE 0: Load data (dari parquet lokal ATAU BigQuery) ─────────────────
    if args.from_parquet:
        parquet_path = Path(args.from_parquet)
        if not parquet_path.is_absolute():
            parquet_path = SCRIPT_DIR / parquet_path
        banner("STAGE 0: LOAD DATA DARI PARQUET LOKAL")
        log(f"Membaca: {parquet_path}")
        bq_df = pd.read_parquet(parquet_path)
        log(f"  {len(bq_df):,} baris dimuat")
        start_date = str(pd.to_datetime(bq_df["block_timestamp"]).min().date())
        end_date   = str(pd.to_datetime(bq_df["block_timestamp"]).max().date())
        log(f"  Rentang: {start_date} s.d. {end_date}")
    else:
        if args.start_date and args.end_date:
            start_date, end_date = args.start_date, args.end_date
        else:
            start_date, end_date = get_date_range(args.days)

        log(f"Project  : {GCP_PROJECT_ID}")
        log(f"Dataset  : {BQ_DATASET}")
        log(f"Rentang  : {start_date} s.d. {end_date}")
        log(f"Limit    : {args.limit:,} transaksi")

        banner("STAGE 0: FETCH DATA DARI GOOGLE BIGQUERY")
        bq_df = fetch_from_bigquery(
            start_date, end_date, args.limit,
            use_cache=not args.no_cache,
        )

        if bq_df.empty:
            log("TIDAK ADA DATA yang di-fetch dari BigQuery!", level="ERROR")
            sys.exit(1)

        raw_csv = output_dir / "eth_transactions_realtime.csv"
        bq_df.to_csv(raw_csv, index=False)
        log(f"Data mentah disimpan: {raw_csv.name} ({raw_csv.stat().st_size // 1024} KB)")

    # ── Transformasi ──────────────────────────────────────────────────────────
    tx_df = transform_to_pipeline_format(bq_df)

    if len(tx_df) < 100:
        log(f"Data terlalu sedikit ({len(tx_df)} baris).", level="ERROR")
        sys.exit(1)

    # ── STAGE 1-8: Jalankan Pipeline ─────────────────────────────────────────
    results = run_pipeline(
        tx_df=tx_df,
        output_dir=output_dir,
        contamination=args.contamination,
        seeds=seeds,
        contamination_grid=contamination_grid,
        skip=skip,
        feature_set=args.feature_set,
        max_edges=args.max_edges,
        n_splits=args.n_splits,
        top_k_ratio=args.top_k_ratio,
        large_graph_threshold=args.large_graph_threshold,
        max_fit_samples=args.max_fit_samples,
    )

    # ── Laporan Akhir ─────────────────────────────────────────────────────────
    banner("GENERATING FINAL ANALYSIS REPORT")
    report = generate_report(tx_df, results, output_dir, start_date, end_date)
    print(report)

    banner("PIPELINE SELESAI")
    total_min = (time.time() - t_fetch_start) / 60
    log(f"Total waktu (fetch + analisis): {total_min:.1f} menit")
    log(f"Output lengkap: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
