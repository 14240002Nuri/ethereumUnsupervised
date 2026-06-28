"""
================================================================================
run_enhanced_pipeline.py
================================================================================
Self-contained, fully runnable Ethereum anomaly detection pipeline yang
mengimplementasikan SEMUA enhancement Q1/Q2 Scopus dalam SATU FILE.

Cara pakai (3 mode):

  Mode 1 — DEMO (synthetic data, langsung jalan, tidak butuh BigQuery):
      python run_enhanced_pipeline.py
      python run_enhanced_pipeline.py --demo --n-nodes 5000 --n-tx 50000

  Mode 2 — Pakai CSV dari script BigQuery existing Saudara:
      python run_enhanced_pipeline.py --input-csv path/to/eth_transactions.csv

  Mode 3 — Skip stages tertentu (untuk testing cepat):
      python run_enhanced_pipeline.py --skip sensitivity,shap

Output (di folder outputs_q1_pipeline/):
  - multi_seed_results.csv          : Multi-seed CV (7 seeds × 6 models)
  - statistical_tests.json          : Friedman + Wilcoxon + Bonferroni
  - contamination_sensitivity.csv   : Sensitivity sweep
  - extended_baselines_results.csv  : Hasil 3 baseline modern
  - external_validation.json        : Precision@k vs labeled set
  - feature_importance.csv          : SHAP / permutation importance
  - computational_profile.csv       : Runtime + memory per stage
  - experiment_metadata.json        : Reproducibility info
  - plots/                          : Publication-ready figures (300 DPI)
      - fig01_sensitivity.png
      - fig02_seed_distribution.png
      - fig03_model_comparison.png
      - fig04_feature_importance.png
      - fig05_external_validation.png
  - paper_tables/                   : Tables siap copy-paste ke LaTeX/Word
      - table_seed_summary.csv
      - table_friedman.csv
      - table_baselines.csv
      - table_external_val.csv

Dependencies (install):
    pip install numpy pandas scipy scikit-learn networkx matplotlib

Optional (untuk fitur tambahan):
    pip install pyod shap psutil scikit-posthocs
================================================================================
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import OneClassSVM

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# =============================================================================
# CONFIG
# =============================================================================

DEFAULT_OUTPUT_DIR = "outputs_q1_pipeline"
DEFAULT_SEEDS = [42, 0, 1, 7, 100, 2023, 2024]
DEFAULT_CONTAMINATION_GRID = [0.005, 0.01, 0.02, 0.05, 0.1]

MODELS_CLASSICAL = ["iforest", "lof", "pca"]
MODELS_MODERN = ["ocsvm", "copod", "autoencoder"]
ALL_MODELS = MODELS_CLASSICAL + MODELS_MODERN

MODEL_LABELS = {
    "iforest": "Isolation Forest",
    "lof": "LOF",
    "pca": "PCA Reconstruction",
    "ocsvm": "One-Class SVM",
    "copod": "COPOD",
    "autoencoder": "Autoencoder (MLP)",
    "hbos": "HBOS",
}

PLOT_STYLE = {
    "figure.figsize": (8, 5),
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.frameon": False,
}
plt.rcParams.update(PLOT_STYLE)


# =============================================================================
# UTILITIES
# =============================================================================

def log(msg: str, level: str = "INFO") -> None:
    print(f"[{level}] {msg}", flush=True)


def banner(text: str) -> None:
    line = "=" * 70
    print(f"\n{line}\n{text.center(70)}\n{line}")


@contextmanager
def timed_stage(name: str, profiler: "PipelineProfiler" = None):
    log(f"START {name}")
    t0 = time.time()
    mem_start = None
    try:
        import psutil
        mem_start = psutil.Process().memory_info().rss / 1024 / 1024
    except ImportError:
        pass
    try:
        yield
    finally:
        duration = time.time() - t0
        mem_peak = None
        try:
            import psutil
            mem_peak = max(mem_start or 0, psutil.Process().memory_info().rss / 1024 / 1024)
        except ImportError:
            pass
        log(f"DONE  {name} ({duration:.1f}s"
            + (f", peak {mem_peak:.0f}MB" if mem_peak else "") + ")")
        if profiler:
            profiler.add(name, duration, mem_peak)


@dataclass
class StageRecord:
    name: str
    duration_sec: float
    memory_mb_peak: float | None


class PipelineProfiler:
    def __init__(self):
        self.stages: list[StageRecord] = []

    def add(self, name: str, duration: float, mem: float | None):
        self.stages.append(StageRecord(name, duration, mem))

    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame([
            {"stage": s.name, "duration_sec": s.duration_sec, "memory_mb_peak": s.memory_mb_peak}
            for s in self.stages
        ])


def _build_preprocessor() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("variance", VarianceThreshold(threshold=0.0)),
        ("scaler", RobustScaler()),
    ])


def _jaccard(a: np.ndarray, b: np.ndarray) -> float:
    sa, sb = set(map(str, a)), set(map(str, b))
    if not sa and not sb:
        return 1.0
    union = sa | sb
    return len(sa & sb) / len(union) if union else 0.0


# =============================================================================
# SECTION 1: SYNTHETIC DATA GENERATOR
# =============================================================================

def generate_synthetic_eth_data(
    n_nodes: int = 5000,
    n_transactions: int = 50_000,
    anomaly_ratio: float = 0.015,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate synthetic Ethereum-like transaction data dengan injected anomalies.

    Returns:
        tx_df: transaction-level DataFrame (mirip output BigQuery)
        labels_df: ground-truth labels (untuk demo external validation)
    """
    log(f"Generating synthetic data: {n_nodes} nodes, {n_transactions} transactions")
    rng = np.random.default_rng(random_state)

    # Generate addresses (use bytes to avoid int64 overflow on 40-hex addresses)
    addresses = ["0x" + rng.bytes(20).hex() for _ in range(n_nodes)]

    # Decide which addresses are anomalies
    n_anomaly = int(n_nodes * anomaly_ratio)
    anomaly_idx = rng.choice(n_nodes, size=n_anomaly, replace=False)
    anomaly_addresses = set(addresses[i] for i in anomaly_idx)

    # Generate transactions with realistic distributions
    tx_records = []
    base_time = pd.Timestamp("2025-01-01 00:00:00", tz="UTC")

    # Normal addresses: low to medium activity, normal value distribution
    # Anomalous addresses: very high activity OR very high value OR unusual ratios
    for _ in range(n_transactions):
        # Bias: some nodes have many more transactions (power-law-ish)
        if rng.random() < 0.1:  # 10% chance involve anomaly
            from_addr = addresses[rng.choice(anomaly_idx)] if rng.random() < 0.5 else \
                addresses[rng.integers(0, n_nodes)]
            to_addr = addresses[rng.choice(anomaly_idx)] if rng.random() < 0.5 else \
                addresses[rng.integers(0, n_nodes)]
            value_eth = rng.lognormal(2.5, 1.5)  # higher mean, higher var
            gas_price = rng.lognormal(2.5, 1.0) * 1e9
        else:
            from_addr = addresses[rng.integers(0, n_nodes)]
            to_addr = addresses[rng.integers(0, n_nodes)]
            value_eth = rng.lognormal(0, 1.0)
            gas_price = rng.lognormal(2.0, 0.5) * 1e9

        if from_addr == to_addr and rng.random() > 0.05:
            to_addr = addresses[(addresses.index(from_addr) + 1) % n_nodes]

        timestamp = base_time + pd.Timedelta(seconds=int(rng.integers(0, 30 * 86400)))

        gas = int(rng.choice([21000, 50000, 100000, 200000], p=[0.6, 0.2, 0.15, 0.05]))
        gas_used = int(gas * rng.uniform(0.7, 1.0))

        tx_records.append({
            "tx_hash": "0x" + rng.bytes(32).hex(),
            "value": value_eth * 1e18,  # in wei
            "value_eth": value_eth,
            "gas": gas,
            "gas_price": gas_price,
            "gas_used": gas_used,
            "gas_fee_native": gas_used * gas_price / 1e18,
            "block_timestamp": timestamp,
            "from_address": from_addr,
            "to_address": to_addr,
            "block_number": int(rng.integers(15_000_000, 16_000_000)),
            "hour": timestamp.hour,
            "dayofweek": timestamp.dayofweek,
        })

    tx_df = pd.DataFrame(tx_records).sort_values("block_timestamp").reset_index(drop=True)

    # Labels for external validation demo
    labels_df = pd.DataFrame({
        "address": list(anomaly_addresses),
        "label": "synthetic_anomaly",
    })

    log(f"Synthetic data: {len(tx_df)} transactions, "
        f"{tx_df['from_address'].nunique() + tx_df['to_address'].nunique()} unique addresses")
    log(f"Injected {len(labels_df)} anomalies")

    return tx_df, labels_df


# =============================================================================
# SECTION 2: FEATURE ENGINEERING
# =============================================================================

def build_node_features(tx_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate transaction-level data ke node-level features."""
    log("Building node-level features...")

    out = tx_df.groupby("from_address").agg(
        out_tx_count=("tx_hash", "count"),
        out_total_value_eth=("value_eth", "sum"),
        out_mean_value_eth=("value_eth", "mean"),
        out_std_value_eth=("value_eth", "std"),
        out_mean_gas_price=("gas_price", "mean"),
        out_mean_gas_fee_native=("gas_fee_native", "mean"),
        out_unique_neighbors=("to_address", "nunique"),
        first_out_time=("block_timestamp", "min"),
        last_out_time=("block_timestamp", "max"),
    ).reset_index().rename(columns={"from_address": "address"})

    inn = tx_df.groupby("to_address").agg(
        in_tx_count=("tx_hash", "count"),
        in_total_value_eth=("value_eth", "sum"),
        in_mean_value_eth=("value_eth", "mean"),
        in_std_value_eth=("value_eth", "std"),
        in_unique_neighbors=("from_address", "nunique"),
        first_in_time=("block_timestamp", "min"),
        last_in_time=("block_timestamp", "max"),
    ).reset_index().rename(columns={"to_address": "address"})

    node_df = pd.merge(out, inn, on="address", how="outer")

    # Numeric fillna
    for col in node_df.columns:
        if col.startswith(("out_", "in_")) and not pd.api.types.is_datetime64_any_dtype(node_df[col]):
            node_df[col] = pd.to_numeric(node_df[col], errors="coerce").fillna(0)

    # Derived features
    node_df["first_activity"] = node_df[["first_out_time", "first_in_time"]].min(axis=1)
    node_df["last_activity"] = node_df[["last_out_time", "last_in_time"]].max(axis=1)
    node_df["activity_span_hours"] = (
        (node_df["last_activity"] - node_df["first_activity"]).dt.total_seconds().fillna(0) / 3600
    )
    node_df["total_tx_count"] = node_df["out_tx_count"] + node_df["in_tx_count"]
    node_df["total_value_eth"] = node_df["out_total_value_eth"] + node_df["in_total_value_eth"]
    node_df["unique_neighbors"] = node_df["out_unique_neighbors"] + node_df["in_unique_neighbors"]
    node_df["in_out_tx_ratio"] = node_df["in_tx_count"] / (node_df["out_tx_count"] + 1)
    node_df["in_out_value_ratio"] = node_df["in_total_value_eth"] / (
        node_df["out_total_value_eth"] + 1e-9
    )
    node_df["activity_density"] = node_df["total_tx_count"] / (node_df["activity_span_hours"] + 1)

    self_loop = (
        tx_df[tx_df["from_address"] == tx_df["to_address"]]
        .groupby("from_address").size()
        .rename("self_loop_count").reset_index()
        .rename(columns={"from_address": "address"})
    )
    node_df = node_df.merge(self_loop, on="address", how="left")
    node_df["self_loop_count"] = node_df["self_loop_count"].fillna(0)
    node_df["self_loop_ratio"] = node_df["self_loop_count"] / (node_df["total_tx_count"] + 1)

    for col in [c for c in node_df.columns if "std" in c]:
        node_df[col] = node_df[col].fillna(0)

    log(f"Node features: {node_df.shape}")
    return node_df


def build_edge_features(tx_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate transactions ke edge-level."""
    log("Building edge features...")
    edge_df = tx_df.groupby(["from_address", "to_address"]).agg(
        tx_count=("tx_hash", "count"),
        total_value_eth=("value_eth", "sum"),
        mean_value_eth=("value_eth", "mean"),
    ).reset_index()
    log(f"Edges: {len(edge_df)}")
    return edge_df


def add_topological_features(
    node_df: pd.DataFrame,
    edge_df: pd.DataFrame,
    max_edges: int = 200_000,
    large_graph_threshold: int = 999_999,
) -> pd.DataFrame:
    """
    Add 11 topological features (basic + extended for Q1/Q2 standard).

    Basic: pagerank, weighted_in_degree, weighted_out_degree, clustering_coef
    Extended: betweenness, closeness, eigenvector, hits_hub, hits_auth,
              k_core, triangle_count, degree_centrality_undirected

    Untuk graph > large_graph_threshold nodes: closeness dan hits di-skip
    (O(V^2) terlalu lama), betweenness k dikurangi ke 200.
    """
    if edge_df.empty:
        for col in [
            "pagerank", "weighted_in_degree", "weighted_out_degree", "clustering_coef",
            "betweenness_centrality", "closeness_centrality", "eigenvector_centrality",
            "hits_hub_score", "hits_auth_score", "k_core", "triangle_count",
            "degree_centrality_undirected",
        ]:
            node_df[col] = 0.0
        return node_df

    edge_sample = edge_df.sort_values("tx_count", ascending=False).head(max_edges).copy()
    G = nx.DiGraph()
    for _, row in edge_sample.iterrows():
        G.add_edge(row["from_address"], row["to_address"], weight=row["tx_count"])
    G_undir = G.to_undirected()
    G_no_loops = G_undir.copy()
    G_no_loops.remove_edges_from(nx.selfloop_edges(G_no_loops))

    n_nodes = G.number_of_nodes()
    is_large = n_nodes > large_graph_threshold
    log(f"Topology graph: {n_nodes} nodes, {G.number_of_edges()} edges"
        + (" [LARGE GRAPH MODE: skip closeness+hits]" if is_large else ""))

    def attach(metric_dict: dict, col_name: str):
        s = pd.Series(metric_dict, name=col_name).reset_index()
        s.columns = ["address", col_name]
        return s

    # Betweenness k lebih kecil untuk graph besar
    bw_k = min(200 if is_large else 500, n_nodes)

    metrics_to_compute = [
        ("pagerank", lambda: nx.pagerank(G, weight="weight")),
        ("weighted_in_degree", lambda: dict(G.in_degree(weight="weight"))),
        ("weighted_out_degree", lambda: dict(G.out_degree(weight="weight"))),
        ("clustering_coef", lambda: nx.clustering(G_undir)),
        ("betweenness_centrality",
         lambda: nx.betweenness_centrality(G, k=bw_k, normalized=True)),
        ("k_core", lambda: nx.core_number(G_no_loops)),
        ("triangle_count", lambda: nx.triangles(G_no_loops)),
        ("degree_centrality_undirected", lambda: nx.degree_centrality(G_undir)),
    ]

    # Closeness dan HITS hanya untuk graph kecil (O(V^2))
    if not is_large:
        metrics_to_compute += [
            ("closeness_centrality", lambda: nx.closeness_centrality(G_undir)),
            ("hits_hub_score", lambda: nx.hits(G, max_iter=200, normalized=True)[0]),
            ("hits_auth_score", lambda: nx.hits(G, max_iter=200, normalized=True)[1]),
        ]
    else:
        log("  closeness_centrality: SKIPPED (graph terlalu besar, set=0)", level="WARN")
        log("  hits_hub_score: SKIPPED (graph terlalu besar, set=0)", level="WARN")
        log("  hits_auth_score: SKIPPED (graph terlalu besar, set=0)", level="WARN")
        node_df["closeness_centrality"] = 0.0
        node_df["hits_hub_score"] = 0.0
        node_df["hits_auth_score"] = 0.0

    for name, fn in metrics_to_compute:
        try:
            t0 = time.time()
            d = fn()
            node_df = node_df.merge(attach(d, name), on="address", how="left")
            node_df[name] = node_df[name].fillna(0)
            log(f"  {name}: {time.time() - t0:.1f}s")
        except Exception as e:
            log(f"  {name}: FAILED ({e})", level="WARN")
            node_df[name] = 0.0

    # Eigenvector separate (can fail on disconnected)
    try:
        t0 = time.time()
        ec = nx.eigenvector_centrality_numpy(G, weight="weight")
        node_df = node_df.merge(attach(ec, "eigenvector_centrality"), on="address", how="left")
        node_df["eigenvector_centrality"] = node_df["eigenvector_centrality"].fillna(0)
        log(f"  eigenvector_centrality: {time.time() - t0:.1f}s")
    except Exception as e:
        log(f"  eigenvector_centrality: FAILED ({e})", level="WARN")
        node_df["eigenvector_centrality"] = 0.0

    return node_df


def get_feature_sets(node_df: pd.DataFrame) -> dict[str, list[str]]:
    """Define 3 feature sets: transactional, graph, hybrid."""
    transactional = [
        "out_mean_value_eth", "out_std_value_eth",
        "in_mean_value_eth", "in_std_value_eth",
        "out_mean_gas_price", "out_mean_gas_fee_native",
        "activity_span_hours", "activity_density",
    ]
    graph = [
        "out_tx_count", "in_tx_count", "total_tx_count",
        "out_unique_neighbors", "in_unique_neighbors", "unique_neighbors",
        "out_total_value_eth", "in_total_value_eth", "total_value_eth",
        "in_out_tx_ratio", "in_out_value_ratio", "self_loop_ratio",
        "pagerank", "weighted_in_degree", "weighted_out_degree", "clustering_coef",
        "betweenness_centrality", "closeness_centrality", "eigenvector_centrality",
        "hits_hub_score", "hits_auth_score", "k_core", "triangle_count",
        "degree_centrality_undirected",
    ]
    hybrid = sorted(set(transactional + graph))

    sets = {
        "transactional": [c for c in transactional if c in node_df.columns],
        "graph": [c for c in graph if c in node_df.columns],
        "hybrid": [c for c in hybrid if c in node_df.columns],
    }
    for name, cols in sets.items():
        log(f"Feature set '{name}': {len(cols)} columns")
    return sets


# =============================================================================
# SECTION 3: MODEL FITTING
# =============================================================================

def fit_classical_models(
    x_tr: np.ndarray, x_te: np.ndarray, contamination: float, seed: int,
    max_lof_fit_samples: int = 999_999,
) -> dict[str, dict[str, np.ndarray]]:
    """Fit IF, LOF, PCA. Returns {model: {scores, flags}}."""
    out: dict[str, dict[str, np.ndarray]] = {}

    m = IsolationForest(n_estimators=200, contamination=contamination, random_state=seed)
    m.fit(x_tr)
    s = -m.decision_function(x_te)
    out["iforest"] = {"scores": s, "flags": (s >= np.quantile(s, 1 - contamination)).astype(int)}

    # LOF: subsample training jika terlalu besar (LOF O(n^2) sangat lambat)
    x_tr_lof = x_tr
    if len(x_tr) > max_lof_fit_samples:
        rng_lof = np.random.default_rng(seed)
        idx_lof = rng_lof.choice(len(x_tr), size=max_lof_fit_samples, replace=False)
        x_tr_lof = x_tr[idx_lof]
    n_neighbors = max(1, min(20, len(x_tr_lof) - 1))
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination, novelty=True)
    lof.fit(x_tr_lof)
    s = -lof.decision_function(x_te)
    out["lof"] = {"scores": s, "flags": (s >= np.quantile(s, 1 - contamination)).astype(int)}

    pca = PCA(n_components=0.95, random_state=seed)
    pca.fit(x_tr)
    recon = pca.inverse_transform(pca.transform(x_te))
    s = np.mean((x_te - recon) ** 2, axis=1)
    out["pca"] = {"scores": s, "flags": (s >= np.quantile(s, 1 - contamination)).astype(int)}

    return out


def fit_modern_baselines(
    x_tr: np.ndarray, x_te: np.ndarray, contamination: float, seed: int,
    enable: list[str] = None,
) -> dict[str, dict[str, np.ndarray]]:
    """Fit modern baselines: OCSVM, COPOD (if available), Autoencoder."""
    if enable is None:
        enable = ["ocsvm", "copod", "autoencoder"]

    out: dict[str, dict[str, np.ndarray]] = {}

    if "ocsvm" in enable:
        # Subsample for tractability
        if len(x_tr) > 20_000:
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(x_tr), size=20_000, replace=False)
            x_tr_sub = x_tr[idx]
        else:
            x_tr_sub = x_tr
        ocsvm = OneClassSVM(nu=contamination, gamma="scale", kernel="rbf")
        ocsvm.fit(x_tr_sub)
        s = -ocsvm.decision_function(x_te)
        out["ocsvm"] = {"scores": s, "flags": (s >= np.quantile(s, 1 - contamination)).astype(int)}

    if "copod" in enable:
        try:
            from pyod.models.copod import COPOD
            copod = COPOD(contamination=contamination)
            copod.fit(x_tr)
            s = copod.decision_function(x_te)
            out["copod"] = {
                "scores": s,
                "flags": (s >= np.quantile(s, 1 - contamination)).astype(int),
            }
        except ImportError:
            pass  # silent skip

    if "autoencoder" in enable:
        n_features = x_tr.shape[1]
        ae = MLPRegressor(
            hidden_layer_sizes=(
                max(2, n_features // 2),
                max(1, n_features // 4),
                max(2, n_features // 2),
            ),
            activation="relu", solver="adam", max_iter=100,
            random_state=seed, early_stopping=True, validation_fraction=0.1,
        )
        ae.fit(x_tr, x_tr)
        recon = ae.predict(x_te)
        s = np.mean((x_te - recon) ** 2, axis=1)
        out["autoencoder"] = {
            "scores": s,
            "flags": (s >= np.quantile(s, 1 - contamination)).astype(int),
        }

    return out


# =============================================================================
# SECTION 4: MULTI-SEED EVALUATION (with both classical & modern baselines)
# =============================================================================

def multi_seed_evaluation_full(
    feature_df: pd.DataFrame,
    feature_cols: list[str],
    seeds: list[int],
    contamination: float = 0.02,
    n_splits: int = 5,
    top_k_ratio: float = 0.05,
    include_modern: bool = True,
    feature_set_name: str = "hybrid",
    max_fit_samples: int = 999_999,
) -> pd.DataFrame:
    """
    Multi-seed stability evaluation using NOVELTY MODE (proper methodology).

    For each seed: train models on resampled subsets (80% bootstrap-style),
    score the FULL dataset, and measure top-k Jaccard stability across resamples.
    This is the methodologically correct approach for stability measurement
    (vs disjoint K-fold which has non-overlapping test sets).
    """
    log(f"Multi-seed eval: {len(seeds)} seeds × {n_splits} resamples × "
        f"{6 if include_modern else 3} models on '{feature_set_name}'")

    x_df = feature_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    addresses = feature_df["address"].to_numpy()
    n_samples = len(x_df)
    sample_size = min(int(0.8 * n_samples), max_fit_samples)
    top_k = max(1, int(n_samples * top_k_ratio))

    if sample_size < int(0.8 * n_samples):
        log(f"  [OPT] max_fit_samples={max_fit_samples:,}: training cap dari "
            f"{int(0.8*n_samples):,} → {sample_size:,} (scoring tetap semua {n_samples:,})")

    all_models = ALL_MODELS if include_modern else MODELS_CLASSICAL
    rows = []

    for seed in seeds:
        log(f"  seed={seed}")
        rng = np.random.default_rng(seed)
        topk_per_model: dict[str, list[np.ndarray]] = {m: [] for m in all_models}

        # n_splits resamples per seed (each resample trains on capped subset, scores ALL)
        for resample_idx in range(n_splits):
            sample_idx = rng.choice(n_samples, size=sample_size, replace=False)
            x_train_raw = x_df.iloc[sample_idx]

            prep = _build_preprocessor()
            x_train_p = prep.fit_transform(x_train_raw)
            x_all_p = prep.transform(x_df)

            # Score ALL addresses (overlapping, comparable)
            classical = fit_classical_models(x_train_p, x_all_p, contamination, seed,
                                             max_lof_fit_samples=max_fit_samples)
            results = dict(classical)
            if include_modern:
                modern = fit_modern_baselines(x_train_p, x_all_p, contamination, seed)
                results.update(modern)

            for model_name in all_models:
                if model_name in results:
                    s = results[model_name]["scores"]
                    topk_per_model[model_name].append(addresses[np.argsort(s)[-top_k:]])

        # Pairwise Jaccard across resamples (now meaningful: same address universe)
        row = {"seed": seed, "feature_set": feature_set_name}
        for model_name in all_models:
            sets = topk_per_model[model_name]
            if not sets or len(sets) < 2:
                row[f"{model_name}_jaccard_mean"] = np.nan
                row[f"{model_name}_jaccard_std"] = np.nan
                continue
            scores = [_jaccard(sets[i], sets[j])
                     for i in range(len(sets)) for j in range(i + 1, len(sets))]
            row[f"{model_name}_jaccard_mean"] = float(np.mean(scores)) if scores else np.nan
            row[f"{model_name}_jaccard_std"] = float(np.std(scores)) if scores else np.nan
        rows.append(row)

    df = pd.DataFrame(rows)
    log("Multi-seed summary (jaccard_mean):")
    for m in all_models:
        col = f"{m}_jaccard_mean"
        if col in df.columns:
            vals = df[col].dropna()
            if len(vals) > 0:
                log(f"  {MODEL_LABELS[m]:25s}: {vals.mean():.4f} ± {vals.std():.4f}")
    return df


# =============================================================================
# SECTION 5: STATISTICAL TESTS
# =============================================================================

def friedman_with_posthoc(
    seed_df: pd.DataFrame,
    models: list[str],
    metric_template: str = "{model}_jaccard_mean",
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Friedman omnibus + pairwise Wilcoxon with Bonferroni correction.

    Models with all-NaN values (e.g., COPOD when pyod not installed)
    are silently dropped before testing.
    """
    from scipy.stats import friedmanchisquare, wilcoxon

    # Filter out models that have any NaN (e.g., not-installed baselines)
    valid_models = []
    for m in models:
        col = metric_template.format(model=m)
        if col in seed_df.columns and seed_df[col].notna().sum() == len(seed_df):
            valid_models.append(m)

    dropped = set(models) - set(valid_models)
    if dropped:
        log(f"Statistical tests: dropping models with NaN/missing data: "
            f"{[MODEL_LABELS.get(m, m) for m in dropped]}", level="WARN")

    if len(valid_models) < 3:
        return {
            "error": "insufficient_models_for_friedman",
            "valid_models": valid_models,
            "n_valid": len(valid_models),
            "note": "Need ≥3 models with complete data for Friedman test.",
        }

    cols = [metric_template.format(model=m) for m in valid_models]
    matrix = seed_df[cols].to_numpy()

    if matrix.shape[0] < 3:
        return {
            "error": "insufficient_seeds",
            "n_seeds": int(matrix.shape[0]),
            "note": "Need ≥3 seeds for Friedman test.",
        }

    stat, p_val = friedmanchisquare(*[matrix[:, i] for i in range(len(valid_models))])

    n_pairs = len(valid_models) * (len(valid_models) - 1) // 2
    bonf_alpha = alpha / n_pairs if n_pairs > 0 else alpha
    pairwise = []
    for i in range(len(valid_models)):
        for j in range(i + 1, len(valid_models)):
            try:
                w, p = wilcoxon(matrix[:, i], matrix[:, j])
                pairwise.append({
                    "model_a": MODEL_LABELS[valid_models[i]],
                    "model_b": MODEL_LABELS[valid_models[j]],
                    "wilcoxon_stat": float(w),
                    "p_value": float(p),
                    "p_value_bonferroni": float(min(1.0, p * n_pairs)),
                    "significant": p < bonf_alpha,
                })
            except ValueError as e:
                pairwise.append({
                    "model_a": MODEL_LABELS[valid_models[i]],
                    "model_b": MODEL_LABELS[valid_models[j]],
                    "error": str(e),
                })

    return {
        "n_seeds": int(matrix.shape[0]),
        "valid_models": [MODEL_LABELS[m] for m in valid_models],
        "dropped_models": [MODEL_LABELS.get(m, m) for m in dropped],
        "friedman_chi2": float(stat),
        "friedman_p_value": float(p_val),
        "friedman_significant_at_alpha": p_val < alpha,
        "alpha": alpha,
        "bonferroni_alpha": bonf_alpha,
        "model_means": {MODEL_LABELS[m]: float(matrix[:, i].mean())
                       for i, m in enumerate(valid_models)},
        "model_stds": {MODEL_LABELS[m]: float(matrix[:, i].std())
                      for i, m in enumerate(valid_models)},
        "pairwise_wilcoxon": pairwise,
    }


# =============================================================================
# SECTION 6: SENSITIVITY ANALYSIS
# =============================================================================

def contamination_sensitivity_sweep(
    feature_df: pd.DataFrame,
    feature_cols: list[str],
    contamination_grid: list[float],
    n_splits: int = 5,
    seed: int = 42,
    top_k_ratio: float = 0.05,  # kept for API compat but overridden by contamination
    max_fit_samples: int = 999_999,
) -> pd.DataFrame:
    """
    Sensitivity sweep: untuk setiap nilai contamination, ukur stabilitas top-k
    dengan k = contamination * N_samples. Ini benar-benar mengukur efek
    contamination pada hasil deteksi (karena top-k ikut berubah).
    """
    log(f"Sensitivity sweep: contamination={contamination_grid}")

    x_df = feature_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    addresses = feature_df["address"].to_numpy()
    n_samples = len(x_df)
    sample_size = min(int(0.8 * n_samples), max_fit_samples)
    rows = []

    for c in contamination_grid:
        # Top-k size scales with contamination
        top_k = max(1, int(n_samples * c))
        log(f"  contamination={c} → top-k={top_k}")
        rng = np.random.default_rng(seed)
        topk_per_model: dict[str, list[np.ndarray]] = {m: [] for m in MODELS_CLASSICAL}

        for resample_idx in range(n_splits):
            sample_idx = rng.choice(n_samples, size=sample_size, replace=False)
            prep = _build_preprocessor()
            x_train_p = prep.fit_transform(x_df.iloc[sample_idx])
            x_all_p = prep.transform(x_df)

            results = fit_classical_models(x_train_p, x_all_p, c, seed,
                                           max_lof_fit_samples=max_fit_samples)
            for m_name, r in results.items():
                topk_per_model[m_name].append(addresses[np.argsort(r["scores"])[-top_k:]])

        for m_name, sets in topk_per_model.items():
            scores = [_jaccard(sets[i], sets[j])
                     for i in range(len(sets)) for j in range(i + 1, len(sets))]
            rows.append({
                "contamination": c,
                "top_k": top_k,
                "model": MODEL_LABELS[m_name],
                "jaccard_mean": float(np.mean(scores)) if scores else np.nan,
                "jaccard_std": float(np.std(scores)) if scores else np.nan,
            })
    return pd.DataFrame(rows)


# =============================================================================
# SECTION 7: ENSEMBLE FULL-DATASET SCORING + EXTERNAL VALIDATION
# =============================================================================

def ensemble_full_dataset_scoring(
    feature_df: pd.DataFrame,
    feature_cols: list[str],
    contamination: float = 0.02,
    seed: int = 42,
    include_modern: bool = True,
) -> pd.DataFrame:
    """Score all addresses with all models, build ensemble vote."""
    log("Full-dataset ensemble scoring...")
    x_df = feature_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    prep = _build_preprocessor()
    x_p = prep.fit_transform(x_df)

    classical = fit_classical_models(x_p, x_p, contamination, seed)
    results = dict(classical)
    if include_modern:
        modern = fit_modern_baselines(x_p, x_p, contamination, seed)
        results.update(modern)

    out = pd.DataFrame({"address": feature_df["address"].values})
    for m_name, r in results.items():
        out[f"{m_name}_score"] = r["scores"]
        out[f"{m_name}_flag"] = r["flags"]
        out[f"{m_name}_score_pct"] = pd.Series(r["scores"]).rank(pct=True).values

    flag_cols = [f"{m}_flag" for m in results.keys()]
    out["ensemble_vote_count"] = out[flag_cols].sum(axis=1)
    out["ensemble_consistent_anomaly"] = (
        out["ensemble_vote_count"] >= max(2, len(results) // 2)
    ).astype(int)

    pct_cols = [f"{m}_score_pct" for m in results.keys()]
    out["ensemble_score_percentile_mean"] = out[pct_cols].mean(axis=1)

    log(f"Ensemble: {(out['ensemble_consistent_anomaly'] == 1).sum()} consistent anomalies")
    return out


def external_validation(
    full_pred: pd.DataFrame,
    labels_df: pd.DataFrame,
    score_column: str = "ensemble_score_percentile_mean",
    k_values: list[int] = None,
) -> dict[str, Any]:
    """Compute Precision@k, Recall@k, AUC vs labeled subset."""
    if k_values is None:
        k_values = [10, 50, 100, 500]

    full_pred = full_pred.copy()
    full_pred["address"] = full_pred["address"].astype(str).str.lower()
    labels_df = labels_df.copy()
    labels_df["address"] = labels_df["address"].astype(str).str.lower()

    pos_set = set(labels_df["address"])
    full_pred["is_positive"] = full_pred["address"].isin(pos_set).astype(int)
    n_pos = full_pred["is_positive"].sum()

    if n_pos < 5:
        return {"warning": "Too few positive labels", "n_positive": int(n_pos)}

    ranked = full_pred.sort_values(score_column, ascending=False).reset_index(drop=True)

    precision_at_k = {}
    recall_at_k = {}
    for k in k_values:
        if k > len(ranked):
            continue
        top_k = ranked.head(k)
        hits = int(top_k["is_positive"].sum())
        precision_at_k[k] = hits / k
        recall_at_k[k] = hits / n_pos

    try:
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(full_pred["is_positive"], full_pred[score_column]))
    except Exception:
        auc = None

    log(f"External validation: {n_pos} positives, "
        f"P@100={precision_at_k.get(100, 'N/A')}, "
        f"AUC-ROC={auc if auc is not None else 'N/A'}")

    return {
        "n_positive": int(n_pos),
        "n_total": int(len(full_pred)),
        "precision_at_k": precision_at_k,
        "recall_at_k": recall_at_k,
        "auc_roc": auc,
    }


# =============================================================================
# SECTION 8: FEATURE IMPORTANCE
# =============================================================================

def compute_feature_importance(
    feature_df: pd.DataFrame,
    feature_cols: list[str],
    contamination: float = 0.02,
    seed: int = 42,
    sample_size: int = 1000,
) -> pd.DataFrame:
    """Compute SHAP (if available) + permutation importance for IF & PCA."""
    log("Computing feature importance...")
    x_df = feature_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    prep = _build_preprocessor()
    x_p = prep.fit_transform(x_df)

    # Get actual surviving feature names after VarianceThreshold
    var_selector = prep.named_steps["variance"]
    if hasattr(var_selector, "get_support"):
        surviving_mask = var_selector.get_support()
        surviving_features = [f for f, keep in zip(feature_cols, surviving_mask) if keep]
    else:
        surviving_features = feature_cols
    log(f"  Features after VarianceThreshold: {len(surviving_features)}/{len(feature_cols)}")

    # Subsample for efficiency
    if len(x_p) > sample_size:
        rng_sub = np.random.default_rng(seed)
        idx = rng_sub.choice(len(x_p), size=sample_size, replace=False)
        x_use = x_p[idx]
    else:
        x_use = x_p

    # Fit IsolationForest on full preprocessed data
    if_model = IsolationForest(n_estimators=200, contamination=contamination, random_state=seed)
    if_model.fit(x_p)

    # Permutation importance for IF
    log("  permutation importance (Isolation Forest)...")
    base_scores = -if_model.decision_function(x_use)
    rng = np.random.default_rng(seed)
    importances = []
    n_features_actual = x_use.shape[1]
    for j in range(n_features_actual):
        feat_name = surviving_features[j] if j < len(surviving_features) else f"feature_{j}"
        deltas = []
        for _ in range(3):
            x_perm = x_use.copy()
            rng.shuffle(x_perm[:, j])
            perm_scores = -if_model.decision_function(x_perm)
            deltas.append(np.mean(np.abs(perm_scores - base_scores)))
        importances.append({
            "feature": feat_name,
            "method": "permutation_iforest",
            "importance": float(np.mean(deltas)),
            "importance_std": float(np.std(deltas)),
        })

    # Try SHAP
    try:
        import shap
        log("  SHAP (Isolation Forest)...")
        explainer = shap.TreeExplainer(if_model)
        shap_values = explainer.shap_values(x_use[:200])
        mean_abs = np.abs(shap_values).mean(axis=0)
        for i in range(len(mean_abs)):
            feat_name = surviving_features[i] if i < len(surviving_features) else f"feature_{i}"
            importances.append({
                "feature": feat_name,
                "method": "shap_iforest",
                "importance": float(mean_abs[i]),
                "importance_std": np.nan,
            })
    except ImportError:
        log("  SHAP skipped (not installed)")
    except Exception as e:
        log(f"  SHAP failed: {e}", level="WARN")

    df = pd.DataFrame(importances)
    log(f"Top 5 features (permutation):")
    top5 = df[df["method"] == "permutation_iforest"].nlargest(5, "importance")
    for _, row in top5.iterrows():
        log(f"  {row['feature']}: {row['importance']:.4f}")
    return df


# =============================================================================
# SECTION 9: PUBLICATION-READY PLOTS
# =============================================================================

def plot_sensitivity(sens_df: pd.DataFrame, output_path: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    pivoted = sens_df.pivot(index="contamination", columns="model", values="jaccard_mean")
    for col in pivoted.columns:
        ax.plot(pivoted.index, pivoted[col], marker="o", linewidth=2, markersize=7, label=col)
    ax.set_xlabel("Contamination parameter (α)")
    ax.set_ylabel("Mean Top-k Jaccard Stability")
    ax.set_title("Robustness to Contamination Hyperparameter")
    ax.set_xscale("log")
    ax.legend(loc="best")
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log(f"Saved {output_path.name}")


def plot_seed_distribution(seed_df: pd.DataFrame, models: list[str], output_path: Path):
    """Box plot showing distribution of stability across seeds."""
    fig, ax = plt.subplots(figsize=(9, 5))
    data = []
    labels = []
    for m in models:
        col = f"{m}_jaccard_mean"
        if col in seed_df.columns:
            data.append(seed_df[col].dropna().values)
            labels.append(MODEL_LABELS[m])
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.6)
    colors = plt.cm.Set2(np.linspace(0, 1, len(data)))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    ax.set_ylabel("Top-k Jaccard Stability")
    ax.set_title(f"Stability Distribution Across {len(seed_df)} Random Seeds")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log(f"Saved {output_path.name}")


def plot_model_comparison_bars(seed_df: pd.DataFrame, models: list[str], output_path: Path):
    """Bar plot mean ± std stability."""
    means, stds, names = [], [], []
    for m in models:
        col = f"{m}_jaccard_mean"
        if col in seed_df.columns:
            means.append(seed_df[col].mean())
            stds.append(seed_df[col].std())
            names.append(MODEL_LABELS[m])
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(names))
    bars = ax.bar(x, means, yerr=stds, capsize=5, alpha=0.8,
                  color=plt.cm.Set2(np.linspace(0, 1, len(names))))
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 0.02, f"{m:.3f}\n±{s:.3f}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("Top-k Jaccard Stability (mean ± std)")
    ax.set_title("Model Comparison on Stability Metric")
    ax.set_ylim(0, 1.15)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log(f"Saved {output_path.name}")


def plot_feature_importance(imp_df: pd.DataFrame, output_path: Path, top_n: int = 15):
    perm_df = imp_df[imp_df["method"] == "permutation_iforest"].nlargest(top_n, "importance")
    fig, ax = plt.subplots(figsize=(9, max(5, top_n * 0.35)))
    y = np.arange(len(perm_df))
    ax.barh(y, perm_df["importance"], color="steelblue", alpha=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(perm_df["feature"])
    ax.invert_yaxis()
    ax.set_xlabel("Permutation Importance")
    ax.set_title(f"Top-{top_n} Feature Importance (Isolation Forest)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log(f"Saved {output_path.name}")


def plot_external_validation(ext_val: dict, output_path: Path):
    p_at_k = ext_val.get("precision_at_k", {})
    r_at_k = ext_val.get("recall_at_k", {})
    if not p_at_k:
        return
    ks = sorted(p_at_k.keys())
    p_vals = [p_at_k[k] for k in ks]
    r_vals = [r_at_k[k] for k in ks]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(ks, p_vals, marker="o", linewidth=2, markersize=8, color="C0")
    axes[0].set_xlabel("k (Top-k anomaly addresses)")
    axes[0].set_ylabel("Precision@k")
    axes[0].set_title("External Validation: Precision@k")
    axes[0].set_xscale("log")
    axes[0].set_ylim(0, 1.05)

    axes[1].plot(ks, r_vals, marker="s", linewidth=2, markersize=8, color="C1")
    axes[1].set_xlabel("k (Top-k anomaly addresses)")
    axes[1].set_ylabel("Recall@k")
    axes[1].set_title("External Validation: Recall@k")
    axes[1].set_xscale("log")
    axes[1].set_ylim(0, 1.05)

    auc = ext_val.get("auc_roc")
    if auc:
        fig.suptitle(f"External Validation Performance (AUC-ROC = {auc:.3f})",
                     fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log(f"Saved {output_path.name}")


# =============================================================================
# SECTION 10: REPRODUCIBILITY METADATA
# =============================================================================

def hash_dataframe(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "empty"
    sample = df.head(10000)
    return hashlib.sha256(
        pd.util.hash_pandas_object(sample, index=False).values.tobytes()
    ).hexdigest()[:16]


def library_versions() -> dict:
    versions = {"python": sys.version.split()[0], "platform": platform.platform()}
    for name in ["numpy", "pandas", "sklearn", "networkx", "scipy", "matplotlib",
                 "pyod", "shap", "psutil"]:
        try:
            mod = __import__(name)
            versions[name] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[name] = "not_installed"
    return versions


def save_metadata(output_dir: Path, args: argparse.Namespace,
                  feature_df: pd.DataFrame, extra: dict):
    metadata = {
        "timestamp_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "args": vars(args),
        "library_versions": library_versions(),
        "data_hash": hash_dataframe(feature_df),
        "data_shape": list(feature_df.shape),
        **extra,
    }
    path = output_dir / "experiment_metadata.json"
    path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    log(f"Saved {path.name}")


# =============================================================================
# SECTION 11: PAPER TABLE GENERATORS
# =============================================================================

def generate_paper_tables(
    seed_df: pd.DataFrame,
    stat_results: dict,
    sens_df: pd.DataFrame,
    ext_val: dict,
    full_pred: pd.DataFrame,
    output_dir: Path,
    models_used: list[str],
):
    """Save copy-paste-ready CSVs for paper tables."""
    paper_dir = output_dir / "paper_tables"
    paper_dir.mkdir(exist_ok=True)

    # Table 1: Multi-seed summary
    rows = []
    for m in models_used:
        col = f"{m}_jaccard_mean"
        if col in seed_df.columns:
            vals = seed_df[col].dropna()
            rows.append({
                "Model": MODEL_LABELS[m],
                "Mean Jaccard": f"{vals.mean():.4f}",
                "Std Jaccard": f"{vals.std():.4f}",
                "Min": f"{vals.min():.4f}",
                "Max": f"{vals.max():.4f}",
                "n_seeds": len(vals),
            })
    pd.DataFrame(rows).to_csv(paper_dir / "table_seed_summary.csv", index=False)

    # Table 2: Friedman + Wilcoxon
    if "pairwise_wilcoxon" in stat_results:
        pw_df = pd.DataFrame(stat_results["pairwise_wilcoxon"])
        pw_df.to_csv(paper_dir / "table_friedman.csv", index=False)
        # Also save header info
        header = {
            "Friedman Chi2": stat_results.get("friedman_chi2"),
            "Friedman p-value": stat_results.get("friedman_p_value"),
            "Significant at α=0.05": stat_results.get("friedman_significant_at_alpha"),
            "Bonferroni alpha": stat_results.get("bonferroni_alpha"),
        }
        (paper_dir / "table_friedman_header.json").write_text(
            json.dumps(header, indent=2, default=str), encoding="utf-8"
        )

    # Table 3: Sensitivity (wide format) — skip jika empty (stage di-skip)
    if not sens_df.empty and "contamination" in sens_df.columns:
        sens_wide = sens_df.pivot(index="contamination", columns="model", values="jaccard_mean")
        sens_wide.to_csv(paper_dir / "table_sensitivity.csv")

    # Table 4: External validation
    if "precision_at_k" in ext_val:
        ev_rows = [{
            "k": k,
            "Precision@k": f"{ext_val['precision_at_k'][k]:.4f}",
            "Recall@k": f"{ext_val['recall_at_k'][k]:.4f}",
        } for k in sorted(ext_val["precision_at_k"].keys())]
        ev_rows.append({
            "k": "AUC-ROC",
            "Precision@k": f"{ext_val.get('auc_roc', 'N/A')}",
            "Recall@k": "—",
        })
        pd.DataFrame(ev_rows).to_csv(paper_dir / "table_external_val.csv", index=False)

    # Table 5: Top-10 anomalies
    top10 = full_pred.nlargest(10, "ensemble_score_percentile_mean")[
        ["address", "ensemble_vote_count", "ensemble_score_percentile_mean"]
    ]
    top10.to_csv(paper_dir / "table_top10_anomalies.csv", index=False)

    log(f"Paper tables saved to {paper_dir}/")


# =============================================================================
# SECTION 12: CLI & MAIN ORCHESTRATION
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Enhanced Ethereum anomaly detection pipeline (Q1/Q2 ready).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--demo", action="store_true",
                   help="Run with synthetic data (default if no --input-csv).")
    p.add_argument("--input-csv", type=str, default=None,
                   help="Path to CSV with Ethereum transactions (from BigQuery).")
    p.add_argument("--labels-csv", type=str, default=None,
                   help="Optional path to labeled addresses CSV (address,label).")
    p.add_argument("--n-nodes", type=int, default=5000,
                   help="Number of synthetic nodes (demo mode only).")
    p.add_argument("--n-tx", type=int, default=50_000,
                   help="Number of synthetic transactions (demo mode only).")
    p.add_argument("--anomaly-ratio", type=float, default=0.015,
                   help="Synthetic anomaly ratio (demo mode only).")
    p.add_argument("--contamination", type=float, default=0.02)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--top-k-ratio", type=float, default=0.05)
    p.add_argument("--seeds", type=str, default=",".join(map(str, DEFAULT_SEEDS)),
                   help="Comma-separated seeds for multi-seed eval.")
    p.add_argument("--contamination-grid", type=str,
                   default=",".join(map(str, DEFAULT_CONTAMINATION_GRID)))
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--skip", type=str, default="",
                   help="Comma-separated stages to skip: sensitivity,shap,modern,external")
    p.add_argument("--max-edges", type=int, default=100_000,
                   help="Max edges for topological feature computation.")
    p.add_argument("--feature-set", type=str, default="hybrid",
                   choices=["transactional", "graph", "hybrid"],
                   help="Feature set untuk multi-seed eval.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    profiler = PipelineProfiler()

    skip = set(s.strip() for s in args.skip.split(",") if s.strip())
    seeds = [int(s) for s in args.seeds.split(",")]
    contamination_grid = [float(c) for c in args.contamination_grid.split(",")]

    banner("ENHANCED ETHEREUM ANOMALY DETECTION PIPELINE (Q1/Q2)")
    log(f"Output directory: {output_dir.absolute()}")
    log(f"Seeds: {seeds}")
    log(f"Contamination grid: {contamination_grid}")
    log(f"Stages skipped: {skip if skip else 'none'}")

    # ========== STAGE 1: LOAD/GENERATE DATA ==========
    banner("STAGE 1/8: DATA LOADING")
    with timed_stage("data_loading", profiler):
        labels_df = None
        if args.input_csv:
            log(f"Loading from {args.input_csv}")
            tx_df = pd.read_csv(args.input_csv, parse_dates=["block_timestamp"])
            if "value_eth" not in tx_df.columns and "value" in tx_df.columns:
                tx_df["value_eth"] = tx_df["value"] / 1e18
            if "gas_fee_native" not in tx_df.columns:
                tx_df["gas_fee_native"] = (
                    tx_df.get("gas_used", tx_df["gas"]) * tx_df["gas_price"] / 1e18
                )
            if args.labels_csv and Path(args.labels_csv).exists():
                labels_df = pd.read_csv(args.labels_csv)
        else:
            tx_df, labels_df = generate_synthetic_eth_data(
                args.n_nodes, args.n_tx, args.anomaly_ratio, random_state=42,
            )

    # ========== STAGE 2: FEATURE ENGINEERING ==========
    banner("STAGE 2/8: FEATURE ENGINEERING")
    with timed_stage("feature_engineering", profiler):
        edge_df = build_edge_features(tx_df)
        node_df = build_node_features(tx_df)
        node_df = add_topological_features(node_df, edge_df, max_edges=args.max_edges)
        feature_sets = get_feature_sets(node_df)

    feature_set_name = args.feature_set
    feature_cols = feature_sets[feature_set_name]
    log(f"Using feature set '{feature_set_name}' with {len(feature_cols)} features")

    # ========== STAGE 3: MULTI-SEED EVALUATION ==========
    banner("STAGE 3/8: MULTI-SEED EVALUATION")
    include_modern = "modern" not in skip
    with timed_stage("multi_seed", profiler):
        seed_df = multi_seed_evaluation_full(
            node_df, feature_cols, seeds,
            contamination=args.contamination, n_splits=args.n_splits,
            top_k_ratio=args.top_k_ratio, include_modern=include_modern,
            feature_set_name=feature_set_name,
        )
    seed_df.to_csv(output_dir / "multi_seed_results.csv", index=False)
    log(f"Saved multi_seed_results.csv")

    models_used = ALL_MODELS if include_modern else MODELS_CLASSICAL

    # ========== STAGE 4: STATISTICAL TESTS ==========
    banner("STAGE 4/8: STATISTICAL TESTS")
    with timed_stage("statistical_tests", profiler):
        stat_results = friedman_with_posthoc(seed_df, models_used)
    (output_dir / "statistical_tests.json").write_text(
        json.dumps(stat_results, indent=2, default=str), encoding="utf-8"
    )
    if "friedman_p_value" in stat_results:
        log(f"Friedman chi2={stat_results['friedman_chi2']:.4f}, "
            f"p={stat_results['friedman_p_value']:.4e} "
            f"({'SIGNIFICANT' if stat_results['friedman_significant_at_alpha'] else 'n.s.'})")
        log("Pairwise Wilcoxon (Bonferroni-corrected):")
        for pw in stat_results.get("pairwise_wilcoxon", []):
            if "p_value" in pw:
                sig = "✓" if pw.get("significant") else "✗"
                log(f"  {sig} {pw['model_a']:25s} vs {pw['model_b']:25s}: "
                    f"p={pw['p_value']:.4e}")

    # ========== STAGE 5: SENSITIVITY ANALYSIS ==========
    sens_df = pd.DataFrame()
    if "sensitivity" not in skip:
        banner("STAGE 5/8: SENSITIVITY ANALYSIS")
        with timed_stage("sensitivity", profiler):
            sens_df = contamination_sensitivity_sweep(
                node_df, feature_cols, contamination_grid,
                n_splits=args.n_splits, seed=seeds[0], top_k_ratio=args.top_k_ratio,
            )
        sens_df.to_csv(output_dir / "contamination_sensitivity.csv", index=False)

    # ========== STAGE 6: ENSEMBLE FULL-DATASET SCORING ==========
    banner("STAGE 6/8: ENSEMBLE FULL-DATASET SCORING")
    with timed_stage("ensemble_scoring", profiler):
        full_pred = ensemble_full_dataset_scoring(
            node_df, feature_cols, contamination=args.contamination,
            seed=seeds[0], include_modern=include_modern,
        )
    full_pred.to_csv(output_dir / "full_pred_with_baselines.csv", index=False)
    log(f"Saved full_pred_with_baselines.csv ({len(full_pred)} rows)")

    # ========== STAGE 7: EXTERNAL VALIDATION ==========
    ext_val = {}
    if "external" not in skip and labels_df is not None and not labels_df.empty:
        banner("STAGE 7/8: EXTERNAL VALIDATION")
        with timed_stage("external_validation", profiler):
            ext_val = external_validation(
                full_pred, labels_df, k_values=[10, 50, 100, 200, 500],
            )
        (output_dir / "external_validation.json").write_text(
            json.dumps(ext_val, indent=2, default=str), encoding="utf-8"
        )
    else:
        log("Skipping external validation (no labels)")

    # ========== STAGE 8: FEATURE IMPORTANCE ==========
    imp_df = pd.DataFrame()
    if "shap" not in skip:
        banner("STAGE 8/8: FEATURE IMPORTANCE")
        with timed_stage("feature_importance", profiler):
            imp_df = compute_feature_importance(
                node_df, feature_cols, contamination=args.contamination, seed=seeds[0],
            )
        imp_df.to_csv(output_dir / "feature_importance.csv", index=False)

    # ========== PLOTTING ==========
    banner("GENERATING PLOTS")
    with timed_stage("plotting", profiler):
        plot_seed_distribution(seed_df, models_used, plots_dir / "fig01_seed_distribution.png")
        plot_model_comparison_bars(seed_df, models_used, plots_dir / "fig02_model_comparison.png")
        if not sens_df.empty:
            plot_sensitivity(sens_df, plots_dir / "fig03_sensitivity.png")
        if not imp_df.empty:
            plot_feature_importance(imp_df, plots_dir / "fig04_feature_importance.png")
        if ext_val and "precision_at_k" in ext_val:
            plot_external_validation(ext_val, plots_dir / "fig05_external_validation.png")

    # ========== PAPER TABLES ==========
    banner("GENERATING PAPER TABLES")
    generate_paper_tables(
        seed_df, stat_results, sens_df, ext_val, full_pred,
        output_dir, models_used,
    )

    # ========== METADATA & PROFILER ==========
    profiler_df = profiler.to_df()
    profiler_df.to_csv(output_dir / "computational_profile.csv", index=False)
    save_metadata(output_dir, args, node_df, extra={
        "n_transactions": len(tx_df),
        "n_addresses": len(node_df),
        "feature_set_used": feature_set_name,
        "n_features": len(feature_cols),
        "models_evaluated": models_used,
    })

    # ========== FINAL SUMMARY ==========
    banner("PIPELINE COMPLETE")
    log(f"Total runtime: {profiler_df['duration_sec'].sum():.1f}s")
    log(f"Output directory: {output_dir.absolute()}")
    log("Key files:")
    for f in sorted(output_dir.glob("*.csv")) + sorted(output_dir.glob("*.json")):
        log(f"  - {f.name} ({f.stat().st_size // 1024} KB)")
    log("Plots:")
    for f in sorted(plots_dir.glob("*.png")):
        log(f"  - plots/{f.name}")
    log("Paper-ready tables:")
    for f in sorted((output_dir / "paper_tables").glob("*")):
        log(f"  - paper_tables/{f.name}")

    print("\n" + "=" * 70)
    print("✓ DONE. Lihat folder output untuk semua hasil pipeline.".center(70))
    print("=" * 70)


if __name__ == "__main__":
    main()
