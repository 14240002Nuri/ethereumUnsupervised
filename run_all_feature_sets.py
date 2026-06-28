"""
================================================================================
run_all_feature_sets.py
================================================================================
Wrapper script untuk menjalankan fetch_bq_realtime_and_analyze.py pada KETIGA
feature set (transactional, graph, hybrid) secara otomatis, lalu menggabungkan
hasil multi-seed dari ketiganya menjadi satu tabel komprehensif siap-paper.

Mengisi GAP penting: pipeline default hanya jalan di SATU feature set saja,
sehingga klaim H2 (perbedaan antar feature set) tidak terdukung empiris.

Cara pakai:

    python run_all_feature_sets.py

    # Pakai parquet cache yang sudah ada (tidak perlu BigQuery):
    python run_all_feature_sets.py --from-parquet bq_cache_eth_transactions\\transactions_2025-01-01_2025-01-31.parquet

    # Jika hybrid sudah pernah jalan, copy hasilnya agar tidak re-run:
    python run_all_feature_sets.py --from-parquet ... --reuse-hybrid outputs_realtime_20260506_155941

Output (di folder outputs_all_feature_sets/):
  - combined_multi_seed_results.csv      : multi-seed x 3 feature set x 6 model
  - combined_statistical_tests.json      : Friedman + Wilcoxon per feature set
  - combined_table_seed_summary.csv      : 18 baris (3 fset x 6 model) siap paper
  - combined_friedman_per_fset.csv       : 3 Friedman tests (1 per feature set)
  - by_feature_set/                      : sub-output per feature set
      - transactional/
      - graph/
      - hybrid/

Estimasi runtime:
  - Hybrid sudah ada (--reuse-hybrid): ~4-5 jam (transactional + graph saja)
  - Semua dari awal: ~7-8 jam
================================================================================
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
FEATURE_SETS = ["transactional", "graph", "hybrid"]
DEFAULT_PAPER_SEEDS = (
    "42,0,1,7,100,2023,2024,11,13,17,19,23,29,31,37,"
    "41,43,47,53,59,61,67,71,73,79,83,89,97,101,103"
)

MODEL_LABELS = {
    "iforest": "Isolation Forest",
    "lof": "LOF",
    "pca": "PCA Reconstruction",
    "ocsvm": "One-Class SVM",
    "copod": "COPOD",
    "autoencoder": "Autoencoder (MLP)",
}


def run_pipeline_for_feature_set(
    feature_set: str,
    args: argparse.Namespace,
    output_subdir: Path,
) -> dict:
    """Run fetch_bq_realtime_and_analyze.py untuk satu feature set."""
    print(f"\n{'='*70}")
    print(f"  RUNNING FEATURE SET: {feature_set.upper()}")
    print(f"{'='*70}\n")

    output_subdir.mkdir(parents=True, exist_ok=True)
    pipeline_script = SCRIPT_DIR / "fetch_bq_realtime_and_analyze.py"
    if not pipeline_script.exists():
        raise FileNotFoundError(f"fetch_bq_realtime_and_analyze.py tidak ditemukan di {SCRIPT_DIR}")

    cmd = [
        sys.executable, str(pipeline_script),
        "--feature-set", feature_set,
        "--output-dir", str(output_subdir),
        "--seeds", args.seeds,
        "--n-splits", str(args.n_splits),
        "--top-k-ratio", str(args.top_k_ratio),
        "--contamination", str(args.contamination),
        "--max-edges", str(args.max_edges),
        "--large-graph-threshold", str(args.large_graph_threshold),
        "--max-fit-samples", str(args.max_fit_samples),
        "--skip", "sensitivity",  # skip sensitivity untuk hemat waktu
    ]

    if args.from_parquet:
        cmd.extend(["--from-parquet", args.from_parquet])
    else:
        cmd.extend([
            "--start-date", args.start_date,
            "--end-date", args.end_date,
            "--limit", str(args.limit),
        ])

    print(f"[CMD] python fetch_bq_realtime_and_analyze.py --feature-set {feature_set} ...")
    t_start = time.time()
    result = subprocess.run(cmd, capture_output=False, text=True)
    duration = time.time() - t_start

    if result.returncode != 0:
        print(f"[ERROR] Pipeline FAILED untuk {feature_set} setelah {duration:.1f}s")
        return {"feature_set": feature_set, "status": "failed", "duration": duration}

    print(f"[DONE] Pipeline selesai untuk {feature_set} dalam {duration/60:.1f} menit")
    return {"feature_set": feature_set, "status": "ok", "duration": duration}


def _seed_count(seeds: str) -> int:
    return len([s for s in seeds.split(",") if s.strip()])


def copy_hybrid_results(src_dir: Path, dst_dir: Path, expected_n_seeds: int | None = None) -> bool:
    """Copy hasil hybrid dari run sebelumnya jika ada."""
    needed = ["multi_seed_results.csv", "statistical_tests.json", "full_pred_with_baselines.csv"]
    optional = ["feature_importance.csv"]
    if not all((src_dir / f).exists() for f in needed):
        print(f"[WARN] Tidak semua file hybrid ditemukan di {src_dir}, akan re-run hybrid.")
        return False

    if expected_n_seeds is not None:
        try:
            stat_json = json.loads((src_dir / "statistical_tests.json").read_text(encoding="utf-8"))
            actual_n_seeds = int(stat_json.get("n_seeds", 0))
        except Exception:
            actual_n_seeds = 0
        if actual_n_seeds != expected_n_seeds:
            print(
                f"[WARN] Hybrid reuse ditolak: {src_dir} berisi n_seeds={actual_n_seeds}, "
                f"sementara run ini meminta n_seeds={expected_n_seeds}. Akan re-run hybrid."
            )
            return False

    dst_dir.mkdir(parents=True, exist_ok=True)
    for f in needed:
        shutil.copy2(src_dir / f, dst_dir / f)
    for f in optional:
        if (src_dir / f).exists():
            shutil.copy2(src_dir / f, dst_dir / f)
    # Copy plots jika ada
    if (src_dir / "plots").exists():
        shutil.copytree(src_dir / "plots", dst_dir / "plots", dirs_exist_ok=True)
    print(f"[COPY] Hasil hybrid disalin dari {src_dir.name} → {dst_dir}")
    return True


def aggregate_results(output_dir: Path) -> None:
    """Aggregate hasil dari 3 feature set menjadi tabel siap-paper."""
    print(f"\n{'='*70}")
    print("  AGGREGATING RESULTS ACROSS FEATURE SETS")
    print(f"{'='*70}\n")

    all_seed_results = []
    all_stat_tests = {}

    for fset in FEATURE_SETS:
        fset_dir = output_dir / "by_feature_set" / fset
        seed_csv = fset_dir / "multi_seed_results.csv"
        stat_json = fset_dir / "statistical_tests.json"

        if seed_csv.exists():
            df = pd.read_csv(seed_csv)
            df["feature_set"] = fset
            all_seed_results.append(df)
            print(f"  Loaded {fset}: {len(df)} seed rows")
        else:
            print(f"  [WARN] {seed_csv} TIDAK DITEMUKAN, skip {fset}")

        if stat_json.exists():
            all_stat_tests[fset] = json.loads(stat_json.read_text(encoding="utf-8"))

    if not all_seed_results:
        print("[ERROR] Tidak ada hasil untuk di-aggregate")
        return

    combined_seed_df = pd.concat(all_seed_results, ignore_index=True)
    combined_seed_df.to_csv(output_dir / "combined_multi_seed_results.csv", index=False)
    print(f"\n[SAVED] combined_multi_seed_results.csv ({len(combined_seed_df)} rows)")

    (output_dir / "combined_statistical_tests.json").write_text(
        json.dumps(all_stat_tests, indent=2, default=str), encoding="utf-8"
    )
    print("[SAVED] combined_statistical_tests.json")

    # ── Tabel summary: 3 fsets x 6 models = 18 baris ────────────────────────
    summary_rows = []
    models = ["iforest", "lof", "pca", "ocsvm", "copod", "autoencoder"]
    for fset in FEATURE_SETS:
        sub_df = combined_seed_df[combined_seed_df["feature_set"] == fset]
        for m in models:
            col_mean = f"{m}_jaccard_mean"
            if col_mean not in sub_df.columns:
                continue
            vals = sub_df[col_mean].dropna()
            if len(vals) == 0:
                continue
            summary_rows.append({
                "Feature Set": fset.capitalize(),
                "Model": MODEL_LABELS.get(m, m),
                "Mean Jaccard": round(vals.mean(), 4),
                "Std Jaccard": round(vals.std(), 4),
                "Min": round(vals.min(), 4),
                "Max": round(vals.max(), 4),
                "n_seeds": len(vals),
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "combined_table_seed_summary.csv", index=False)
    print(f"[SAVED] combined_table_seed_summary.csv ({len(summary_df)} baris)")
    print("\nSUMMARY TABLE (Tabel utama paper):")
    print(summary_df.to_string(index=False))

    # ── Friedman per feature set ─────────────────────────────────────────────
    friedman_rows = []
    for fset, stats in all_stat_tests.items():
        if "friedman_chi2" in stats:
            friedman_rows.append({
                "Feature Set": fset.capitalize(),
                "Friedman chi2": round(stats["friedman_chi2"], 4),
                "p-value": f"{stats['friedman_p_value']:.4e}",
                "Significant (alpha=0.05)": stats.get("friedman_significant_at_alpha", False),
                "n_seeds": stats.get("n_seeds", "N/A"),
                "n_models": len(stats.get("valid_models", [])),
            })

    if friedman_rows:
        friedman_df = pd.DataFrame(friedman_rows)
        friedman_df.to_csv(output_dir / "combined_friedman_per_fset.csv", index=False)
        print(f"\n[SAVED] combined_friedman_per_fset.csv")
        print("\nFRIEDMAN TESTS PER FEATURE SET:")
        print(friedman_df.to_string(index=False))

    # ── Cross-feature-set Friedman (apakah feature set berpengaruh?) ─────────
    _cross_fset_friedman(combined_seed_df, models, output_dir)

    print(f"\n{'='*70}")
    print("  AGGREGATION COMPLETE")
    print(f"{'='*70}\n")


def _cross_fset_friedman(combined_df: pd.DataFrame, models: list, output_dir: Path):
    """Uji apakah feature set berbeda secara signifikan (H2 paper)."""
    try:
        from scipy.stats import friedmanchisquare
    except ImportError:
        return

    print("\n[H2] Cross-feature-set Friedman test (apakah fset berpengaruh?):")
    results = []
    for m in models:
        col = f"{m}_jaccard_mean"
        rows_per_fset = []
        for fset in FEATURE_SETS:
            sub = combined_df[combined_df["feature_set"] == fset][col].dropna().values
            rows_per_fset.append(sub)
        min_len = min(len(r) for r in rows_per_fset)
        if min_len < 3:
            continue
        rows_trimmed = [r[:min_len] for r in rows_per_fset]
        try:
            stat, p = friedmanchisquare(*rows_trimmed)
            sig = "SIGNIFIKAN" if p < 0.05 else "n.s."
            print(f"  {MODEL_LABELS.get(m, m):<28}: chi2={stat:.3f}, p={p:.4e} [{sig}]")
            results.append({"model": MODEL_LABELS.get(m, m), "chi2": stat, "p_value": p,
                            "significant": p < 0.05})
        except Exception:
            pass

    if results:
        pd.DataFrame(results).to_csv(output_dir / "cross_fset_friedman.csv", index=False)
        print("[SAVED] cross_fset_friedman.csv")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run multi-seed evaluation across all 3 feature sets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--from-parquet", type=str,
                   default="bq_cache_eth_transactions/transactions_2025-01-01_2025-01-31.parquet",
                   help="Path ke parquet cache. Default: cache 3M record Jan 2025.")
    p.add_argument("--start-date", type=str, default="2025-01-01",
                   help="Tanggal mulai (diabaikan jika --from-parquet diisi).")
    p.add_argument("--end-date", type=str, default="2025-01-03",
                   help="Tanggal akhir (diabaikan jika --from-parquet diisi).")
    p.add_argument("--limit", type=int, default=3_000_000)
    p.add_argument("--seeds", type=str, default=DEFAULT_PAPER_SEEDS,
                   help="Comma-separated random seeds. Default paper run uses 30 seeds "
                        "so Bonferroni-corrected Wilcoxon post-hoc tests have enough power.")
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--top-k-ratio", type=float, default=0.05)
    p.add_argument("--contamination", type=float, default=0.02)
    p.add_argument("--max-edges", type=int, default=100_000)
    p.add_argument("--large-graph-threshold", type=int, default=50_000)
    p.add_argument("--max-fit-samples", type=int, default=50_000)
    p.add_argument("--output-dir", type=str, default="outputs_all_feature_sets_30seeds")
    p.add_argument("--reuse-hybrid", type=str, default="",
                   help="Path folder hasil hybrid yang sudah ada (skip re-run hybrid).")
    p.add_argument("--feature-sets-to-run", type=str, default="transactional,graph,hybrid",
                   help="Feature set yang akan dijalankan (comma-separated).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fsets_to_run = [s.strip() for s in args.feature_sets_to_run.split(",") if s.strip()]

    print(f"\n{'='*70}")
    print("  MULTI-FEATURE-SET PIPELINE WRAPPER")
    print(f"{'='*70}")
    print(f"  Output dir       : {output_dir.absolute()}")
    print(f"  Feature sets     : {fsets_to_run}")
    print(f"  Parquet cache    : {args.from_parquet}")
    print(f"  Seeds            : {args.seeds}")
    print(f"  n_seeds          : {_seed_count(args.seeds)}")
    print(f"  max_fit_samples  : {args.max_fit_samples:,}")
    print(f"  large_graph_thr  : {args.large_graph_threshold:,}")

    # Estimasi runtime
    n_to_run = len(fsets_to_run)
    if args.reuse_hybrid and "hybrid" in fsets_to_run:
        n_to_run -= 1
    print(f"  Est. runtime     : ~{n_to_run * 2.5:.0f}–{n_to_run * 3.5:.0f} jam")
    print(f"{'='*70}\n")

    statuses = []
    for fset in fsets_to_run:
        if fset not in FEATURE_SETS:
            print(f"[WARN] Skipping unknown feature set: {fset}")
            continue

        sub_dir = output_dir / "by_feature_set" / fset

        # Cek apakah bisa reuse hasil hybrid yang sudah ada
        if fset == "hybrid" and args.reuse_hybrid:
            reuse_src = Path(args.reuse_hybrid)
            if copy_hybrid_results(reuse_src, sub_dir, expected_n_seeds=_seed_count(args.seeds)):
                statuses.append({"feature_set": "hybrid", "status": "reused", "duration": 0})
                continue

        status = run_pipeline_for_feature_set(fset, args, sub_dir)
        statuses.append(status)

    aggregate_results(output_dir)

    print("\nFINAL STATUS:")
    total_min = sum(s.get("duration", 0) for s in statuses) / 60
    for s in statuses:
        dur = f"{s.get('duration',0)/60:.1f} menit" if s.get("duration",0) > 0 else "reused"
        print(f"  {s['feature_set']:15s}: {s['status']:8s} ({dur})")
    print(f"\n  Total aktual: {total_min:.1f} menit")
    print(f"  Output      : {output_dir.absolute()}")


if __name__ == "__main__":
    main()
