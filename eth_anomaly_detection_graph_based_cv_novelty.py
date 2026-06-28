"""
Contoh instalasi dependency:
    py -3 -m pip install pandas numpy matplotlib networkx scikit-learn google-cloud-bigquery pyarrow db-dtypes google-auth-oauthlib

Siapkan file OAuth Desktop App dari Google Cloud Console:
    simpan sebagai client_secrets.json di folder yang sama dengan script ini

Contoh jalan dari BigQuery:
    py -3 eth_anomaly_detection_graph_based_cv_novelty.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import textwrap
from typing import List, Tuple

import matplotlib
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_OUTPUT_DIR = "outputs_eth_anomaly"
DEFAULT_GCP_PROJECT_ID = "tesis-gaptek"
DEFAULT_OAUTH_CLIENT_SECRETS_FILES = [
    "client_secret_desktopapp.json"
]
DEFAULT_USER_TOKEN_FILE = "bigquery_user_token.json"
DEFAULT_QUERY_LIMIT = 3000000
DEFAULT_START_DATE = "2025-01-01"
DEFAULT_END_DATE = "2025-12-31"
DEFAULT_QUERY_CHUNKING = "month"
DEFAULT_QUERY_CACHE_DIR = "bq_cache_eth_transactions"
DEFAULT_MAX_EDGES_FOR_NX = 50000
DEFAULT_STABILITY_RUNS = 5
DEFAULT_STABILITY_SAMPLE_RATIO = 0.8
DEFAULT_FEATURE_GRAPH_THRESHOLD = 0.35
DEFAULT_FEATURE_GRAPH_MAX_EDGES = 40
DEFAULT_TOP_ANOMALIES_EXPORT_LIMIT = 500
BIGQUERY_SCOPES = ["https://www.googleapis.com/auth/bigquery"]
MODEL_LABELS = {
    "iforest": "Isolation Forest",
    "lof": "LOF",
    "pca": "PCA Reconstruction",
}


def build_preprocessor() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("variance", VarianceThreshold(threshold=0.0)),
            ("scaler", RobustScaler()),
        ]
    )


def build_query(start_date: str, end_date: str, limit: int) -> str:
    limit_clause = f"\nLIMIT {int(limit)}" if limit and int(limit) > 0 else ""
    return textwrap.dedent(
        f"""
        SELECT
          `hash` AS tx_hash,
          value,
          gas,
          gas_price,
          receipt_gas_used,
          receipt_cumulative_gas_used,
          max_fee_per_gas,
          max_priority_fee_per_gas,
          nonce,
          transaction_index,
          block_number,
          transaction_type,
          block_timestamp,
          from_address,
          to_address
        FROM `bigquery-public-data.crypto_ethereum.transactions`
        WHERE DATE(block_timestamp) BETWEEN '{start_date}' AND '{end_date}'
          AND from_address IS NOT NULL
          AND to_address IS NOT NULL
        ORDER BY block_timestamp ASC, tx_hash ASC{limit_clause}
        """
    ).strip()


def get_user_bigquery_credentials():
    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
    except ImportError as exc:
        raise ImportError(
            "Paket OAuth belum terpasang. Install minimal: "
            "google-auth-oauthlib"
        ) from exc

    base_dir = Path(__file__).resolve().parent
    client_secrets_path = None
    for candidate_name in DEFAULT_OAUTH_CLIENT_SECRETS_FILES:
        candidate_path = base_dir / candidate_name
        if candidate_path.exists():
            client_secrets_path = candidate_path
            break
    token_path = base_dir / DEFAULT_USER_TOKEN_FILE

    credentials = None
    if token_path.exists():
        credentials = Credentials.from_authorized_user_file(
            str(token_path),
            scopes=BIGQUERY_SCOPES,
        )

    if credentials and credentials.valid:
        return credentials

    if credentials and credentials.expired and credentials.refresh_token:
        credentials.refresh(Request())
        token_path.write_text(credentials.to_json(), encoding="utf-8")
        return credentials

    if client_secrets_path is None:
        expected_paths = ", ".join(
            str(base_dir / candidate_name) for candidate_name in DEFAULT_OAUTH_CLIENT_SECRETS_FILES
        )
        raise RuntimeError(
            "File OAuth client tidak ditemukan. Simpan file OAuth Desktop App "
            "dengan script ini agar login Google via browser bisa dijalankan. "
            f"Nama file yang didukung: {DEFAULT_OAUTH_CLIENT_SECRETS_FILES}. "
            f"Path yang dicek: {expected_paths}"
        )

    print("[INFO] Membuka login Google di browser untuk autentikasi BigQuery...")
    flow = InstalledAppFlow.from_client_secrets_file(
        str(client_secrets_path),
        scopes=BIGQUERY_SCOPES,
    )
    credentials = flow.run_local_server(port=0)
    token_path.write_text(credentials.to_json(), encoding="utf-8")
    print(f"[INFO] Token user tersimpan di: {token_path}")
    return credentials


def get_bigquery_client():
    try:
        from google.cloud import bigquery
    except ImportError as exc:
        raise ImportError(
            "Paket BigQuery belum terpasang. Install minimal: "
            "google-cloud-bigquery pyarrow db-dtypes"
        ) from exc

    credentials = get_user_bigquery_credentials()
    return bigquery.Client(project=DEFAULT_GCP_PROJECT_ID, credentials=credentials)


def dry_run_query_bytes(client, query: str) -> int:
    from google.cloud import bigquery

    job_config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
    query_job = client.query(query, job_config=job_config)
    return int(query_job.total_bytes_processed or 0)


def run_query_to_dataframe(client, query: str) -> pd.DataFrame:
    return client.query(query).to_dataframe()


def is_quota_exceeded_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "quota exceeded" in message or "quotaexceeded" in message


def iter_date_chunks(
    start_date: str,
    end_date: str,
    chunking: str,
) -> List[Tuple[str, str]]:
    start_ts = pd.Timestamp(start_date).normalize()
    end_ts = pd.Timestamp(end_date).normalize()
    if start_ts > end_ts:
        raise ValueError("start-date harus <= end-date")

    if chunking == "none":
        return [(start_ts.strftime("%Y-%m-%d"), end_ts.strftime("%Y-%m-%d"))]

    chunks: List[Tuple[str, str]] = []
    current = start_ts

    while current <= end_ts:
        if chunking == "day":
            chunk_end = current
        elif chunking == "month":
            chunk_end = min(current + pd.offsets.MonthEnd(0), end_ts)
        else:
            raise ValueError(f"chunking tidak didukung: {chunking}")

        chunks.append(
            (
                current.strftime("%Y-%m-%d"),
                pd.Timestamp(chunk_end).strftime("%Y-%m-%d"),
            )
        )
        current = pd.Timestamp(chunk_end) + pd.Timedelta(days=1)

    return chunks


def get_chunk_cache_paths(cache_dir: Path, chunk_start: str, chunk_end: str) -> Tuple[Path, Path]:
    base_name = f"transactions_{chunk_start}_{chunk_end}"
    return cache_dir / f"{base_name}.parquet", cache_dir / f"{base_name}.json"


def load_transactions_chunked(
    start_date: str,
    end_date: str,
    limit: int,
    chunking: str,
    cache_dir: Path,
    force_refresh_cache: bool = False,
) -> pd.DataFrame:
    client = get_bigquery_client()
    cache_dir.mkdir(parents=True, exist_ok=True)
    all_chunks = iter_date_chunks(start_date, end_date, chunking)
    frames: List[pd.DataFrame] = []
    total_rows = 0
    remaining_limit = int(limit) if limit and int(limit) > 0 else None

    print(
        "[INFO] Mengambil data transaksi dari BigQuery public dataset "
        f"dengan project `{DEFAULT_GCP_PROJECT_ID}` menggunakan chunk `{chunking}`."
    )
    print(f"[INFO] Cache query BigQuery: {cache_dir}")

    for chunk_index, (chunk_start, chunk_end) in enumerate(all_chunks, start=1):
        if remaining_limit is not None and remaining_limit <= 0:
            break

        data_path, meta_path = get_chunk_cache_paths(cache_dir, chunk_start, chunk_end)
        if data_path.exists() and meta_path.exists() and not force_refresh_cache:
            chunk_df = pd.read_parquet(data_path)
            cached_meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if remaining_limit is not None and len(chunk_df) > remaining_limit:
                chunk_df = chunk_df.head(remaining_limit).copy()
            frames.append(chunk_df)
            total_rows += len(chunk_df)
            if remaining_limit is not None:
                remaining_limit -= len(chunk_df)
            print(
                "[INFO] Memakai cache chunk "
                f"{chunk_index}/{len(all_chunks)}: {chunk_start} s.d. {chunk_end} "
                f"({len(chunk_df):,} rows)"
            )
            if cached_meta.get("limit_reached_inside_chunk"):
                print("[INFO] LIMIT global sudah tercapai dari cache chunk sebelumnya.")
                break
            continue

        chunk_limit = remaining_limit if remaining_limit is not None else 0
        query = build_query(chunk_start, chunk_end, chunk_limit)
        estimated_bytes = dry_run_query_bytes(client, query)
        estimated_gb = estimated_bytes / (1024 ** 3)
        print(
            "[INFO] Query chunk "
            f"{chunk_index}/{len(all_chunks)}: {chunk_start} s.d. {chunk_end} "
            f"(estimasi scan {estimated_gb:.2f} GiB)"
        )

        try:
            chunk_df = run_query_to_dataframe(client, query)
        except Exception as exc:
            if is_quota_exceeded_error(exc):
                progress_message = (
                    "Kuota BigQuery habis di tengah pengambilan chunk. "
                    f"Cache yang sudah tersimpan tetap aman di {cache_dir}. "
                    f"Progress saat berhenti: {total_rows:,} rows cached."
                )
                raise RuntimeError(progress_message) from exc
            raise

        chunk_df.to_parquet(data_path, index=False)
        limit_reached_inside_chunk = bool(chunk_limit and len(chunk_df) >= chunk_limit)
        meta = {
            "chunk_start": chunk_start,
            "chunk_end": chunk_end,
            "row_count": int(len(chunk_df)),
            "query_limit": int(chunk_limit),
            "estimated_bytes_processed": int(estimated_bytes),
            "limit_reached_inside_chunk": limit_reached_inside_chunk,
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        frames.append(chunk_df)
        total_rows += len(chunk_df)
        if remaining_limit is not None:
            remaining_limit -= len(chunk_df)

        print(
            "[INFO] Chunk tersimpan ke cache: "
            f"{chunk_start} s.d. {chunk_end} ({len(chunk_df):,} rows)"
        )

        if limit_reached_inside_chunk:
            print("[INFO] LIMIT global tercapai. Pengambilan chunk berikutnya dihentikan.")
            break

    if not frames:
        raise RuntimeError(
            "Tidak ada data transaksi yang berhasil dimuat. "
            "Cek kuota BigQuery atau cache query yang tersedia."
        )

    raw_df = pd.concat(frames, ignore_index=True)
    print(f"[INFO] Total data mentah dari cache/query: {raw_df.shape}")
    return raw_df


def load_transactions(query: str) -> pd.DataFrame:
    client = get_bigquery_client()

    print(
        "[INFO] Mengambil data transaksi dari BigQuery public dataset "
        f"dengan project `{DEFAULT_GCP_PROJECT_ID}`..."
    )
    return run_query_to_dataframe(client, query)


def preprocess_transactions(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = ["tx_hash", "block_timestamp", "from_address", "to_address"]
    missing_required = [col for col in required_cols if col not in df.columns]
    if missing_required:
        raise ValueError(f"Kolom wajib tidak ditemukan: {missing_required}")

    tx_df = df.copy()

    numeric_cols = [
        "value",
        "gas",
        "gas_price",
        "receipt_gas_used",
        "receipt_cumulative_gas_used",
        "max_fee_per_gas",
        "max_priority_fee_per_gas",
        "nonce",
        "transaction_index",
        "block_number",
    ]

    for col in numeric_cols:
        if col not in tx_df.columns:
            tx_df[col] = 0

    if "transaction_type" not in tx_df.columns:
        tx_df["transaction_type"] = -1

    tx_df["block_timestamp"] = pd.to_datetime(tx_df["block_timestamp"], errors="coerce")

    for col in numeric_cols:
        tx_df[col] = pd.to_numeric(tx_df[col], errors="coerce")

    tx_df["value_eth"] = tx_df["value"] / 1e18
    tx_df["gas_fee_native"] = (
        tx_df["gas_price"].fillna(0) * tx_df["receipt_gas_used"].fillna(0)
    ) / 1e18
    tx_df["hour"] = tx_df["block_timestamp"].dt.hour
    tx_df["dayofweek"] = tx_df["block_timestamp"].dt.dayofweek
    tx_df["transaction_type"] = tx_df["transaction_type"].fillna(-1)

    fill_cols = numeric_cols + ["value_eth", "gas_fee_native", "hour", "dayofweek"]
    tx_df[fill_cols] = tx_df[fill_cols].fillna(0)

    tx_df["from_address"] = tx_df["from_address"].astype(str)
    tx_df["to_address"] = tx_df["to_address"].astype(str)

    print(f"[INFO] Data transaksi siap diproses: {tx_df.shape}")
    return tx_df


def build_edge_features(tx_df: pd.DataFrame) -> pd.DataFrame:
    edge_df = (
        tx_df.groupby(["from_address", "to_address"], dropna=False)
        .agg(
            tx_count=("tx_hash", "count"),
            total_value_eth=("value_eth", "sum"),
            mean_value_eth=("value_eth", "mean"),
            std_value_eth=("value_eth", "std"),
            mean_gas_price=("gas_price", "mean"),
            mean_gas_fee_native=("gas_fee_native", "mean"),
            first_seen=("block_timestamp", "min"),
            last_seen=("block_timestamp", "max"),
        )
        .reset_index()
    )

    edge_df["std_value_eth"] = edge_df["std_value_eth"].fillna(0)
    edge_df["active_span_hours"] = (
        (edge_df["last_seen"] - edge_df["first_seen"]).dt.total_seconds().fillna(0) / 3600
    )

    print(f"[INFO] Edge graph terbentuk: {edge_df.shape}")
    return edge_df


def build_node_features(tx_df: pd.DataFrame) -> pd.DataFrame:
    out_feat = (
        tx_df.groupby("from_address")
        .agg(
            out_tx_count=("tx_hash", "count"),
            out_total_value_eth=("value_eth", "sum"),
            out_mean_value_eth=("value_eth", "mean"),
            out_std_value_eth=("value_eth", "std"),
            out_mean_gas_price=("gas_price", "mean"),
            out_mean_gas_fee_native=("gas_fee_native", "mean"),
            out_unique_neighbors=("to_address", "nunique"),
            first_out_time=("block_timestamp", "min"),
            last_out_time=("block_timestamp", "max"),
        )
        .reset_index()
        .rename(columns={"from_address": "address"})
    )

    in_feat = (
        tx_df.groupby("to_address")
        .agg(
            in_tx_count=("tx_hash", "count"),
            in_total_value_eth=("value_eth", "sum"),
            in_mean_value_eth=("value_eth", "mean"),
            in_std_value_eth=("value_eth", "std"),
            in_unique_neighbors=("from_address", "nunique"),
            first_in_time=("block_timestamp", "min"),
            last_in_time=("block_timestamp", "max"),
        )
        .reset_index()
        .rename(columns={"to_address": "address"})
    )

    node_df = pd.merge(out_feat, in_feat, on="address", how="outer")

    for col in node_df.columns:
        if col.startswith(("out_", "in_")) and not pd.api.types.is_datetime64_any_dtype(node_df[col]):
            node_df[col] = pd.to_numeric(node_df[col], errors="coerce").fillna(0)

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
        .groupby("from_address")
        .size()
        .rename("self_loop_count")
        .reset_index()
        .rename(columns={"from_address": "address"})
    )
    node_df = node_df.merge(self_loop, on="address", how="left")
    node_df["self_loop_count"] = node_df["self_loop_count"].fillna(0)
    node_df["self_loop_ratio"] = node_df["self_loop_count"] / (node_df["total_tx_count"] + 1)

    std_cols = [col for col in node_df.columns if "std" in col]
    for col in std_cols:
        node_df[col] = node_df[col].fillna(0)

    print(f"[INFO] Node-level features terbentuk: {node_df.shape}")
    return node_df


def add_topological_features(
    node_df: pd.DataFrame,
    edge_df: pd.DataFrame,
    max_edges_for_nx: int,
) -> pd.DataFrame:
    if edge_df.empty:
        for col in ["pagerank", "weighted_in_degree", "weighted_out_degree", "clustering_coef"]:
            node_df[col] = 0.0
        return node_df

    edge_sample = edge_df.sort_values("tx_count", ascending=False).head(max_edges_for_nx).copy()

    graph = nx.DiGraph()
    for _, row in edge_sample.iterrows():
        graph.add_edge(
            row["from_address"],
            row["to_address"],
            weight=row["tx_count"],
            value=row["total_value_eth"],
        )

    print(
        "[INFO] NX graph siap dihitung: "
        f"{graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges"
    )

    if graph.number_of_nodes() == 0:
        for col in ["pagerank", "weighted_in_degree", "weighted_out_degree", "clustering_coef"]:
            node_df[col] = 0.0
        return node_df

    pagerank = (
        pd.Series(nx.pagerank(graph, weight="weight"), name="pagerank")
        .reset_index()
        .rename(columns={"index": "address"})
    )
    in_deg = (
        pd.Series(dict(graph.in_degree(weight="weight")), name="weighted_in_degree")
        .reset_index()
        .rename(columns={"index": "address"})
    )
    out_deg = (
        pd.Series(dict(graph.out_degree(weight="weight")), name="weighted_out_degree")
        .reset_index()
        .rename(columns={"index": "address"})
    )

    undirected_graph = graph.to_undirected()
    clustering = (
        pd.Series(nx.clustering(undirected_graph), name="clustering_coef")
        .reset_index()
        .rename(columns={"index": "address"})
    )

    for topo_df in [pagerank, in_deg, out_deg, clustering]:
        node_df = node_df.merge(topo_df, on="address", how="left")

    for col in ["pagerank", "weighted_in_degree", "weighted_out_degree", "clustering_coef"]:
        node_df[col] = node_df[col].fillna(0)

    return node_df


def get_feature_sets(node_df: pd.DataFrame) -> dict[str, list[str]]:
    transaction_features = [
        "out_mean_value_eth",
        "out_std_value_eth",
        "in_mean_value_eth",
        "in_std_value_eth",
        "out_mean_gas_price",
        "out_mean_gas_fee_native",
        "activity_span_hours",
        "activity_density",
    ]

    graph_features = [
        "out_tx_count",
        "in_tx_count",
        "total_tx_count",
        "out_unique_neighbors",
        "in_unique_neighbors",
        "unique_neighbors",
        "out_total_value_eth",
        "in_total_value_eth",
        "total_value_eth",
        "in_out_tx_ratio",
        "in_out_value_ratio",
        "self_loop_ratio",
        "pagerank",
        "weighted_in_degree",
        "weighted_out_degree",
        "clustering_coef",
    ]

    hybrid_features = sorted(set(transaction_features + graph_features))

    feature_sets = {
        "transaction": [col for col in transaction_features if col in node_df.columns],
        "graph": [col for col in graph_features if col in node_df.columns],
        "hybrid": [col for col in hybrid_features if col in node_df.columns],
    }

    for feature_name, cols in feature_sets.items():
        print(f"[INFO] Feature set {feature_name}: {len(cols)} kolom")

    return feature_sets


def jaccard_similarity(a: list[str] | np.ndarray, b: list[str] | np.ndarray) -> float:
    set_a, set_b = set(a), set(b)
    if not set_a and not set_b:
        return 1.0
    return len(set_a & set_b) / max(len(set_a | set_b), 1)


def evaluate_unsupervised_cv(
    feature_df: pd.DataFrame,
    feature_cols: list[str],
    contamination: float = 0.02,
    n_splits: int = 5,
    random_state: int = 42,
    top_k_ratio: float = 0.05,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not feature_cols:
        raise ValueError("Feature columns kosong. Tidak bisa menjalankan evaluasi.")

    if len(feature_df) < 2:
        raise ValueError("Jumlah node terlalu sedikit untuk cross-validation.")

    effective_splits = min(n_splits, len(feature_df))
    if effective_splits < 2:
        raise ValueError("Minimal diperlukan 2 data untuk cross-validation.")

    x_df = feature_df[feature_cols].copy()
    x_df = x_df.replace([np.inf, -np.inf], np.nan).fillna(0)

    kf = KFold(n_splits=effective_splits, shuffle=True, random_state=random_state)

    fold_records: list[dict[str, float | int]] = []
    all_test_predictions: list[pd.DataFrame] = []
    topk_sets: dict[str, list[np.ndarray]] = {"iforest": [], "lof": [], "pca": []}

    for fold, (train_idx, test_idx) in enumerate(kf.split(x_df), start=1):
        print(
            f"[INFO] CV fold {fold}/{effective_splits} "
            f"untuk {len(feature_cols)} fitur dan {len(test_idx)} test samples..."
        )
        x_train = x_df.iloc[train_idx].copy()
        x_test = x_df.iloc[test_idx].copy()
        addr_test = feature_df.iloc[test_idx]["address"].values

        prep = build_preprocessor()

        x_train_p = prep.fit_transform(x_train)
        x_test_p = prep.transform(x_test)

        if x_train_p.shape[1] == 0:
            raise ValueError("Semua fitur terhapus setelah VarianceThreshold. Coba cek datanya.")

        if_model = IsolationForest(
            n_estimators=200,
            contamination=contamination,
            random_state=random_state,
        )
        if_model.fit(x_train_p)
        if_scores = -if_model.decision_function(x_test_p)
        if_threshold = np.quantile(if_scores, 1 - contamination)
        if_flags = (if_scores >= if_threshold).astype(int)

        n_neighbors = max(1, min(20, len(x_train_p) - 1))
        lof_model = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=contamination,
            novelty=True,
        )
        lof_model.fit(x_train_p)
        lof_scores = -lof_model.decision_function(x_test_p)
        lof_threshold = np.quantile(lof_scores, 1 - contamination)
        lof_flags = (lof_scores >= lof_threshold).astype(int)

        pca_model = PCA(n_components=0.95, random_state=random_state)
        pca_model.fit(x_train_p)
        x_test_recon = pca_model.inverse_transform(pca_model.transform(x_test_p))
        pca_scores = np.mean((x_test_p - x_test_recon) ** 2, axis=1)
        pca_threshold = np.quantile(pca_scores, 1 - contamination)
        pca_flags = (pca_scores >= pca_threshold).astype(int)

        k = max(1, int(len(addr_test) * top_k_ratio))
        topk_sets["iforest"].append(addr_test[np.argsort(if_scores)[-k:]])
        topk_sets["lof"].append(addr_test[np.argsort(lof_scores)[-k:]])
        topk_sets["pca"].append(addr_test[np.argsort(pca_scores)[-k:]])

        fold_records.append(
            {
                "fold": fold,
                "n_test": len(test_idx),
                "iforest_mean_score": float(np.mean(if_scores)),
                "iforest_std_score": float(np.std(if_scores)),
                "iforest_anomaly_ratio": float(np.mean(if_flags)),
                "lof_mean_score": float(np.mean(lof_scores)),
                "lof_std_score": float(np.std(lof_scores)),
                "lof_anomaly_ratio": float(np.mean(lof_flags)),
                "pca_mean_score": float(np.mean(pca_scores)),
                "pca_std_score": float(np.std(pca_scores)),
                "pca_anomaly_ratio": float(np.mean(pca_flags)),
            }
        )

        fold_pred = pd.DataFrame(
            {
                "address": addr_test,
                "fold": fold,
                "iforest_score": if_scores,
                "iforest_flag": if_flags,
                "lof_score": lof_scores,
                "lof_flag": lof_flags,
                "pca_score": pca_scores,
                "pca_flag": pca_flags,
            }
        )
        all_test_predictions.append(fold_pred)

    cv_df = pd.DataFrame(fold_records)
    pred_df = pd.concat(all_test_predictions, ignore_index=True)

    stability_rows = []
    for model_name, sets_list in topk_sets.items():
        pairwise_scores = []
        for idx in range(len(sets_list)):
            for jdx in range(idx + 1, len(sets_list)):
                pairwise_scores.append(jaccard_similarity(sets_list[idx], sets_list[jdx]))
        stability_rows.append(
            {
                "model": model_name,
                "mean_topk_jaccard": float(np.mean(pairwise_scores)) if pairwise_scores else np.nan,
                "std_topk_jaccard": float(np.std(pairwise_scores)) if pairwise_scores else np.nan,
            }
        )
    stability_df = pd.DataFrame(stability_rows)

    summary_df = pd.DataFrame(
        [
            {
                "model": "Isolation Forest",
                "mean_score": cv_df["iforest_mean_score"].mean(),
                "std_score_across_folds": cv_df["iforest_mean_score"].std(),
                "mean_anomaly_ratio": cv_df["iforest_anomaly_ratio"].mean(),
                "std_anomaly_ratio": cv_df["iforest_anomaly_ratio"].std(),
            },
            {
                "model": "LOF",
                "mean_score": cv_df["lof_mean_score"].mean(),
                "std_score_across_folds": cv_df["lof_mean_score"].std(),
                "mean_anomaly_ratio": cv_df["lof_anomaly_ratio"].mean(),
                "std_anomaly_ratio": cv_df["lof_anomaly_ratio"].std(),
            },
            {
                "model": "PCA Reconstruction",
                "mean_score": cv_df["pca_mean_score"].mean(),
                "std_score_across_folds": cv_df["pca_mean_score"].std(),
                "mean_anomaly_ratio": cv_df["pca_anomaly_ratio"].mean(),
                "std_anomaly_ratio": cv_df["pca_anomaly_ratio"].std(),
            },
        ]
    )

    return cv_df, pred_df, stability_df, summary_df


def fit_full_dataset_models(
    feature_df: pd.DataFrame,
    feature_cols: list[str],
    contamination: float = 0.02,
    random_state: int = 42,
) -> pd.DataFrame:
    if not feature_cols:
        raise ValueError("Feature columns kosong. Tidak bisa menjalankan scoring full dataset.")

    x_df = feature_df[feature_cols].copy()
    x_df = x_df.replace([np.inf, -np.inf], np.nan).fillna(0)

    print("[INFO] Menjalankan scoring full dataset untuk hasil akhir anomaly ranking...")
    prep = build_preprocessor()
    x_all_p = prep.fit_transform(x_df)

    if x_all_p.shape[1] == 0:
        raise ValueError("Semua fitur terhapus setelah preprocessing pada full dataset.")

    if_model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=random_state,
    )
    if_model.fit(x_all_p)
    if_scores = -if_model.decision_function(x_all_p)
    if_threshold = np.quantile(if_scores, 1 - contamination)
    if_flags = (if_scores >= if_threshold).astype(int)

    n_neighbors = max(1, min(20, len(x_all_p) - 1))
    lof_model = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination=contamination,
    )
    lof_model.fit_predict(x_all_p)
    lof_scores = -lof_model.negative_outlier_factor_
    lof_threshold = np.quantile(lof_scores, 1 - contamination)
    lof_flags = (lof_scores >= lof_threshold).astype(int)

    pca_model = PCA(n_components=0.95, random_state=random_state)
    pca_model.fit(x_all_p)
    x_recon = pca_model.inverse_transform(pca_model.transform(x_all_p))
    pca_scores = np.mean((x_all_p - x_recon) ** 2, axis=1)
    pca_threshold = np.quantile(pca_scores, 1 - contamination)
    pca_flags = (pca_scores >= pca_threshold).astype(int)

    full_pred = pd.DataFrame(
        {
            "address": feature_df["address"].values,
            "iforest_full_score": if_scores,
            "iforest_full_flag": if_flags,
            "lof_full_score": lof_scores,
            "lof_full_flag": lof_flags,
            "pca_full_score": pca_scores,
            "pca_full_flag": pca_flags,
        }
    )

    for prefix in MODEL_LABELS:
        full_pred[f"{prefix}_full_score_percentile"] = full_pred[f"{prefix}_full_score"].rank(
            method="average",
            pct=True,
        )

    full_pred["ensemble_vote_count"] = (
        full_pred["iforest_full_flag"]
        + full_pred["lof_full_flag"]
        + full_pred["pca_full_flag"]
    )
    full_pred["ensemble_score_percentile_mean"] = full_pred[
        [
            "iforest_full_score_percentile",
            "lof_full_score_percentile",
            "pca_full_score_percentile",
        ]
    ].mean(axis=1)
    full_pred["ensemble_consistent_anomaly"] = (
        full_pred["ensemble_vote_count"] >= 2
    ).astype(int)
    return full_pred


def evaluate_resampled_stability(
    feature_df: pd.DataFrame,
    feature_cols: list[str],
    contamination: float = 0.02,
    n_runs: int = DEFAULT_STABILITY_RUNS,
    sample_ratio: float = DEFAULT_STABILITY_SAMPLE_RATIO,
    random_state: int = 42,
    top_k_ratio: float = 0.05,
) -> pd.DataFrame:
    if not feature_cols:
        raise ValueError("Feature columns kosong. Tidak bisa menjalankan stability evaluation.")

    x_df = feature_df[feature_cols].copy()
    x_df = x_df.replace([np.inf, -np.inf], np.nan).fillna(0)
    addresses = feature_df["address"].to_numpy()
    n_rows = len(x_df)

    sample_size = min(n_rows, max(50, int(n_rows * sample_ratio)))
    top_k_size = max(1, int(n_rows * top_k_ratio))
    rng = np.random.default_rng(random_state)
    topk_sets: dict[str, list[np.ndarray]] = {"iforest": [], "lof": [], "pca": []}

    print(
        "[INFO] Menghitung stability dengan resampling full dataset "
        f"({n_runs} runs, sample_ratio={sample_ratio:.2f}, top_k={top_k_size})..."
    )

    for run_idx in range(1, n_runs + 1):
        print(f"[INFO] Stability run {run_idx}/{n_runs}...")
        sample_idx = rng.choice(n_rows, size=sample_size, replace=False)

        prep = build_preprocessor()
        x_sample_p = prep.fit_transform(x_df.iloc[sample_idx])
        x_all_p = prep.transform(x_df)

        if x_sample_p.shape[1] == 0:
            raise ValueError("Semua fitur terhapus setelah preprocessing pada stability evaluation.")

        run_seed = random_state + run_idx
        if_model = IsolationForest(
            n_estimators=200,
            contamination=contamination,
            random_state=run_seed,
        )
        if_model.fit(x_sample_p)
        if_scores = -if_model.decision_function(x_all_p)

        n_neighbors = max(1, min(20, len(x_sample_p) - 1))
        lof_model = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=contamination,
            novelty=True,
        )
        lof_model.fit(x_sample_p)
        lof_scores = -lof_model.decision_function(x_all_p)

        pca_model = PCA(n_components=0.95, random_state=run_seed)
        pca_model.fit(x_sample_p)
        x_recon = pca_model.inverse_transform(pca_model.transform(x_all_p))
        pca_scores = np.mean((x_all_p - x_recon) ** 2, axis=1)

        topk_sets["iforest"].append(addresses[np.argsort(if_scores)[-top_k_size:]])
        topk_sets["lof"].append(addresses[np.argsort(lof_scores)[-top_k_size:]])
        topk_sets["pca"].append(addresses[np.argsort(pca_scores)[-top_k_size:]])

    stability_rows = []
    for model_name, sets_list in topk_sets.items():
        pairwise_scores = []
        for idx in range(len(sets_list)):
            for jdx in range(idx + 1, len(sets_list)):
                pairwise_scores.append(jaccard_similarity(sets_list[idx], sets_list[jdx]))
        stability_rows.append(
            {
                "model": MODEL_LABELS[model_name],
                "mean_topk_jaccard": float(np.mean(pairwise_scores)) if pairwise_scores else np.nan,
                "std_topk_jaccard": float(np.std(pairwise_scores)) if pairwise_scores else np.nan,
                "stability_method": "full_dataset_resampling",
                "stability_runs": n_runs,
                "stability_sample_ratio": sample_ratio,
                "top_k_size": top_k_size,
            }
        )

    return pd.DataFrame(stability_rows)


def build_top_anomaly_table(
    result_node: pd.DataFrame,
    limit: int = DEFAULT_TOP_ANOMALIES_EXPORT_LIMIT,
) -> pd.DataFrame:
    sort_cols = [
        "ensemble_vote_count",
        "ensemble_score_percentile_mean",
        "iforest_full_score_percentile",
        "lof_full_score_percentile",
        "pca_full_score_percentile",
    ]
    top_df = result_node.sort_values(sort_cols, ascending=False).head(limit).copy()
    return top_df[
        [
            "address",
            "ensemble_vote_count",
            "ensemble_consistent_anomaly",
            "ensemble_score_percentile_mean",
            "iforest_full_flag",
            "lof_full_flag",
            "pca_full_flag",
            "iforest_full_score",
            "lof_full_score",
            "pca_full_score",
            "iforest_cv_score_mean",
            "lof_cv_score_mean",
            "pca_cv_score_mean",
        ]
    ]


def build_feature_relationship_graph(
    feature_df: pd.DataFrame,
    feature_cols: list[str],
    transaction_feature_cols: list[str],
    graph_feature_cols: list[str],
    output_dir: Path,
    corr_threshold: float = DEFAULT_FEATURE_GRAPH_THRESHOLD,
    max_edges: int = DEFAULT_FEATURE_GRAPH_MAX_EDGES,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(feature_cols) < 2:
        raise ValueError("Minimal diperlukan dua fitur untuk membuat feature graph.")

    print("[INFO] Membuat grafik hubungan antar fitur bergaya graph/GNN...")
    corr = (
        feature_df[feature_cols]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
        .corr(method="spearman")
    )

    candidate_edges: list[dict[str, float | int | str]] = []
    for idx, source in enumerate(feature_cols):
        for target in feature_cols[idx + 1 :]:
            corr_value = corr.loc[source, target]
            if pd.isna(corr_value):
                continue
            candidate_edges.append(
                {
                    "source": source,
                    "target": target,
                    "correlation": float(corr_value),
                    "abs_correlation": float(abs(corr_value)),
                    "sign": int(np.sign(corr_value)),
                }
            )

    edge_df = pd.DataFrame(candidate_edges).sort_values("abs_correlation", ascending=False)
    strong_edges = edge_df[edge_df["abs_correlation"] >= corr_threshold].copy()
    if strong_edges.empty:
        strong_edges = edge_df.head(max_edges).copy()
    else:
        strong_edges = strong_edges.head(max_edges).copy()

    feature_groups = {}
    for feature in feature_cols:
        if feature in transaction_feature_cols:
            feature_groups[feature] = "transaction"
        elif feature in graph_feature_cols:
            feature_groups[feature] = "graph"
        else:
            feature_groups[feature] = "hybrid"

    graph = nx.Graph()
    for feature in feature_cols:
        graph.add_node(feature, feature_group=feature_groups[feature])
    for _, row in strong_edges.iterrows():
        graph.add_edge(
            row["source"],
            row["target"],
            weight=row["abs_correlation"],
            sign=row["sign"],
            correlation=row["correlation"],
        )

    node_rows = []
    for node in graph.nodes():
        strength = sum(edge_data["weight"] for _, _, edge_data in graph.edges(node, data=True))
        node_rows.append(
            {
                "feature": node,
                "feature_group": graph.nodes[node]["feature_group"],
                "degree": graph.degree(node),
                "strength": strength,
            }
        )
    node_metrics_df = pd.DataFrame(node_rows).sort_values(
        ["strength", "degree", "feature"],
        ascending=[False, False, True],
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    strong_edges.to_csv(output_dir / "feature_relationship_edges_hybrid.csv", index=False)
    node_metrics_df.to_csv(output_dir / "feature_relationship_nodes_hybrid.csv", index=False)

    node_color_map = {
        "transaction": "#f4a261",
        "graph": "#2a9d8f",
        "hybrid": "#7a7a7a",
    }
    max_strength = max(node_metrics_df["strength"].max(), 1e-9)
    node_sizes = [
        1300 + 3200 * (row["strength"] / max_strength)
        for _, row in node_metrics_df.set_index("feature").loc[list(graph.nodes())].iterrows()
    ]
    node_colors = [node_color_map[graph.nodes[node]["feature_group"]] for node in graph.nodes()]
    edge_colors = [
        "#d1495b" if edge_data["sign"] >= 0 else "#2d6a9f"
        for _, _, edge_data in graph.edges(data=True)
    ]
    edge_widths = [
        1.5 + 5.0 * edge_data["weight"]
        for _, _, edge_data in graph.edges(data=True)
    ]

    pos = nx.spring_layout(graph, seed=42, weight="weight", k=1.2)
    plt.figure(figsize=(15, 11))
    nx.draw_networkx_edges(
        graph,
        pos,
        edge_color=edge_colors,
        width=edge_widths,
        alpha=0.75,
    )
    nx.draw_networkx_nodes(
        graph,
        pos,
        node_size=node_sizes,
        node_color=node_colors,
        linewidths=1.2,
        edgecolors="#ffffff",
    )
    nx.draw_networkx_labels(
        graph,
        pos,
        labels={node: node.replace("_", "\n") for node in graph.nodes()},
        font_size=8,
        font_weight="bold",
    )
    plt.title("GNN-Style Hybrid Feature Relationship Graph (Spearman Correlation)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_dir / "feature_relationship_graph_hybrid.png", dpi=220)
    plt.close()

    return strong_edges, node_metrics_df


def maybe_transform_for_plot(series: pd.Series) -> tuple[pd.Series, bool]:
    cleaned = (
        pd.to_numeric(series, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )
    if (cleaned >= 0).all():
        positive = cleaned[cleaned > 0]
        if positive.empty:
            return cleaned, False
        q50 = positive.quantile(0.50)
        q95 = positive.quantile(0.95)
        if q50 <= 0 or (q95 / max(q50, 1e-9)) >= 20:
            return np.log1p(cleaned), True
    return cleaned, False


def compute_feature_label_effects(
    feature_df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str = "ensemble_consistent_anomaly",
) -> pd.DataFrame:
    rows = []
    labels = feature_df[label_col].fillna(0).astype(int)

    for feature in feature_cols:
        values = (
            pd.to_numeric(feature_df[feature], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0)
        )
        values_0 = values[labels == 0]
        values_1 = values[labels == 1]
        if len(values_0) == 0 or len(values_1) == 0:
            continue

        mean_0 = float(values_0.mean())
        mean_1 = float(values_1.mean())
        median_0 = float(values_0.median())
        median_1 = float(values_1.median())

        var_0 = float(values_0.var(ddof=1)) if len(values_0) > 1 else 0.0
        var_1 = float(values_1.var(ddof=1)) if len(values_1) > 1 else 0.0
        pooled_num = max(len(values_0) - 1, 0) * var_0 + max(len(values_1) - 1, 0) * var_1
        pooled_den = max(len(values_0) + len(values_1) - 2, 1)
        pooled_std = float(np.sqrt(pooled_num / pooled_den)) if pooled_num > 0 else 0.0
        effect_size = (mean_1 - mean_0) / pooled_std if pooled_std > 0 else 0.0

        rows.append(
            {
                "feature": feature,
                "mean_label_0": mean_0,
                "mean_label_1": mean_1,
                "median_label_0": median_0,
                "median_label_1": median_1,
                "effect_size": effect_size,
                "abs_effect_size": abs(effect_size),
            }
        )

    return pd.DataFrame(rows).sort_values("abs_effect_size", ascending=False)


def create_label_relationship_outputs(
    result_node: pd.DataFrame,
    feature_cols: list[str],
    output_dir: Path,
    label_col: str = "ensemble_consistent_anomaly",
    top_n: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if label_col not in result_node.columns:
        raise ValueError(f"Kolom label `{label_col}` tidak ditemukan.")

    print("[INFO] Membuat visual hubungan fitur terhadap label anomaly...")
    output_dir.mkdir(parents=True, exist_ok=True)

    effects_df = compute_feature_label_effects(result_node, feature_cols, label_col=label_col)
    effects_df.to_csv(output_dir / "ensemble_label_feature_effects.csv", index=False)

    top_effects = effects_df.head(top_n).copy()
    labels = result_node[label_col].fillna(0).astype(int)
    label_names = {0: "Non-anomaly", 1: "Anomaly"}
    colors = {0: "#4f6d7a", 1: "#d1495b"}

    n_panels = len(top_effects)
    n_cols = 2
    n_rows = max(1, int(np.ceil(n_panels / n_cols)))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4.3 * n_rows))
    axes_array = np.atleast_1d(axes).ravel()

    for ax, (_, row) in zip(axes_array, top_effects.iterrows()):
        feature = row["feature"]
        plot_values, transformed = maybe_transform_for_plot(result_node[feature])
        group_0 = plot_values[labels == 0]
        group_1 = plot_values[labels == 1]

        box = ax.boxplot(
            [group_0, group_1],
            tick_labels=[label_names[0], label_names[1]],
            showfliers=False,
            patch_artist=True,
            widths=0.6,
        )
        for patch, group_label in zip(box["boxes"], [0, 1]):
            patch.set_facecolor(colors[group_label])
            patch.set_alpha(0.6)
        for median in box["medians"]:
            median.set_color("#111111")
            median.set_linewidth(1.8)

        title_suffix = " (log1p)" if transformed else ""
        ax.set_title(
            f"{feature}{title_suffix}\nabs effect={row['abs_effect_size']:.2f}",
            fontsize=10,
            fontweight="bold",
        )
        ax.grid(axis="y", alpha=0.2)

    for ax in axes_array[n_panels:]:
        ax.axis("off")

    plt.suptitle("Top Features vs Ensemble Anomaly Label", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "feature_vs_ensemble_anomaly_boxplots_top10.png", dpi=220)
    plt.close()

    raw_group_means = (
        result_node.groupby(label_col)[feature_cols]
        .mean()
        .rename(index=label_names)
    )
    raw_group_means.to_csv(output_dir / "ensemble_label_feature_means_raw.csv")

    feature_matrix = (
        result_node[feature_cols]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )
    medians = feature_matrix.median(axis=0)
    iqrs = feature_matrix.quantile(0.75) - feature_matrix.quantile(0.25)
    iqrs = iqrs.replace(0, 1.0)
    scaled_matrix = (feature_matrix - medians) / iqrs

    feature_order = effects_df["feature"].tolist()
    heatmap_df = (
        scaled_matrix.assign(**{label_col: labels})
        .groupby(label_col)[feature_order]
        .mean()
        .rename(index=label_names)
    )
    heatmap_df = heatmap_df.astype(float)
    heatmap_df.to_csv(output_dir / "ensemble_label_feature_means_robust_scaled.csv")

    heatmap_values = heatmap_df.to_numpy(dtype=float)
    vmax = max(np.abs(heatmap_values).max(), 1e-9)
    plt.figure(figsize=(18, 5.5))
    plt.imshow(
        heatmap_values,
        cmap="coolwarm",
        aspect="auto",
        vmin=-vmax,
        vmax=vmax,
    )
    plt.colorbar(label="Mean robust-scaled feature value")
    plt.xticks(range(len(feature_order)), [f.replace("_", "\n") for f in feature_order], rotation=0)
    row_labels = []
    for label_value, label_name in label_names.items():
        count = int((labels == label_value).sum())
        row_labels.append(f"{label_name}\n(n={count:,})")
    plt.yticks(range(len(row_labels)), row_labels)
    plt.title("Feature Mean Heatmap by Ensemble Anomaly Label")
    plt.tight_layout()
    plt.savefig(output_dir / "feature_mean_heatmap_by_ensemble_label.png", dpi=220)
    plt.close()

    return effects_df, heatmap_df


def create_paper_ready_outputs(
    summary_all: pd.DataFrame,
    stability_all: pd.DataFrame,
    result_node: pd.DataFrame,
    top_anomaly_table: pd.DataFrame,
    feature_effects_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    print("[INFO] Membuat figure paper-ready...")
    output_dir.mkdir(parents=True, exist_ok=True)

    vote_counts = (
        result_node["ensemble_vote_count"]
        .value_counts()
        .reindex([0, 1, 2, 3], fill_value=0)
        .rename_axis("ensemble_vote_count")
        .reset_index(name="count")
    )
    vote_counts["share"] = vote_counts["count"] / max(len(result_node), 1)
    vote_counts.to_csv(output_dir / "paper_ensemble_vote_distribution.csv", index=False)

    plt.figure(figsize=(9, 5))
    colors = ["#cfd8dc", "#90caf9", "#ffcc80", "#ef9a9a"]
    plt.bar(
        vote_counts["ensemble_vote_count"].astype(str),
        vote_counts["share"],
        color=colors,
        edgecolor="#263238",
        linewidth=0.8,
    )
    for _, row in vote_counts.iterrows():
        plt.text(
            x=str(int(row["ensemble_vote_count"])),
            y=row["share"] + 0.003,
            s=f"{row['count']:,}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    plt.title("Distribution of Ensemble Anomaly Votes", fontsize=15, fontweight="bold")
    plt.xlabel("Number of models voting anomaly")
    plt.ylabel("Share of addresses")
    plt.ylim(0, vote_counts["share"].max() * 1.18)
    plt.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_dir / "paper_ensemble_vote_distribution.png", dpi=240)
    plt.close()

    top_effects = feature_effects_df.head(10).copy()
    plot_df = top_effects.sort_values("effect_size")
    bar_colors = ["#2a9d8f" if value >= 0 else "#d1495b" for value in plot_df["effect_size"]]
    plt.figure(figsize=(10, 6.5))
    plt.barh(
        [feature.replace("_", " ") for feature in plot_df["feature"]],
        plot_df["effect_size"],
        color=bar_colors,
        edgecolor="#24323b",
        linewidth=0.7,
    )
    plt.axvline(0, color="#24323b", linewidth=1.0)
    plt.title("Top Feature Effects on Final Ensemble Anomaly Label", fontsize=15, fontweight="bold")
    plt.xlabel("Standardized mean difference (anomaly - non-anomaly)")
    plt.ylabel("Feature")
    plt.grid(axis="x", alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_dir / "paper_top_feature_effects.png", dpi=240)
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    stability_plot_df = stability_all.copy()
    stability_plot_df["label"] = stability_plot_df["feature_set"] + "\n" + stability_plot_df["model"]
    axes[0].barh(
        stability_plot_df["label"],
        stability_plot_df["mean_topk_jaccard"],
        color="#577590",
        edgecolor="#24323b",
        linewidth=0.7,
    )
    axes[0].set_title("Top-k Stability Across Resampling", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Mean Jaccard similarity")
    axes[0].grid(axis="x", alpha=0.2)

    top10 = top_anomaly_table.head(10).copy()
    axes[1].barh(
        top10["address"].iloc[::-1],
        top10["ensemble_score_percentile_mean"].iloc[::-1],
        color="#bc4749",
        edgecolor="#24323b",
        linewidth=0.7,
    )
    axes[1].set_title("Top 10 Final Ensemble Anomalies", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Mean ensemble percentile score")
    axes[1].grid(axis="x", alpha=0.2)

    plt.tight_layout()
    plt.savefig(output_dir / "paper_summary_panel.png", dpi=240)
    plt.close()


def write_thesis_results_summary(
    summary_all: pd.DataFrame,
    stability_all: pd.DataFrame,
    result_node: pd.DataFrame,
    top_anomaly_table: pd.DataFrame,
    feature_effects_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    print("[INFO] Menulis ringkasan hasil gaya tesis...")
    output_dir.mkdir(parents=True, exist_ok=True)

    total_nodes = len(result_node)
    total_anomalies = int(result_node["ensemble_consistent_anomaly"].sum())
    anomaly_share = total_anomalies / max(total_nodes, 1)

    vote_counts = (
        result_node["ensemble_vote_count"]
        .value_counts()
        .reindex([0, 1, 2, 3], fill_value=0)
        .to_dict()
    )

    median_by_label = result_node.groupby("ensemble_consistent_anomaly")[
        ["total_tx_count", "unique_neighbors", "activity_span_hours", "total_value_eth"]
    ].median()
    median_non = median_by_label.loc[0]
    median_anom = median_by_label.loc[1]

    top_features = feature_effects_df.head(5)
    top_features_lines = "\n".join(
        f"- `{row.feature}`: effect size {row.effect_size:.2f}, median non-anomaly {row.median_label_0:.4f}, median anomaly {row.median_label_1:.4f}"
        for row in top_features.itertuples()
    )

    top_addresses = top_anomaly_table.head(10)
    top_addresses_lines = "\n".join(
        f"- `{row.address}`: vote {int(row.ensemble_vote_count)}/3, ensemble percentile {row.ensemble_score_percentile_mean:.4f}"
        for row in top_addresses.itertuples()
    )

    best_stability = stability_all.sort_values("mean_topk_jaccard", ascending=False).head(5)
    stability_lines = "\n".join(
        f"- {row.feature_set} / {row.model}: mean Jaccard {row.mean_topk_jaccard:.4f}"
        for row in best_stability.itertuples()
    )

    text = f"""# Ringkasan Hasil Deteksi Anomali Ethereum

## 1. Ringkasan Umum
Eksperimen graph-based anomaly detection dijalankan pada **{total_nodes:,} address/node** yang dibangun dari transaksi Ethereum hasil query BigQuery. Label akhir tidak berasal dari ground truth eksternal, melainkan dari **ensemble vote** tiga model unsupervised, yaitu Isolation Forest, Local Outlier Factor (LOF), dan PCA Reconstruction. Address diberi label `ensemble_consistent_anomaly = 1` apabila memperoleh minimal **2 dari 3 vote**.

Hasil akhir menunjukkan terdapat **{total_anomalies:,} address** yang masuk kategori anomali ensemble, atau sekitar **{anomaly_share:.2%}** dari seluruh node. Distribusi vote menunjukkan:
- vote 0: {vote_counts.get(0, 0):,} address
- vote 1: {vote_counts.get(1, 0):,} address
- vote 2: {vote_counts.get(2, 0):,} address
- vote 3: {vote_counts.get(3, 0):,} address

## 2. Interpretasi Hasil Utama
Secara umum, node yang diberi label anomali memiliki perilaku yang jauh lebih ekstrem dibanding node biasa. Median `total_tx_count` pada kelompok non-anomali adalah **{median_non['total_tx_count']:.2f}**, sedangkan pada kelompok anomali mencapai **{median_anom['total_tx_count']:.2f}**. Median `unique_neighbors` meningkat dari **{median_non['unique_neighbors']:.2f}** menjadi **{median_anom['unique_neighbors']:.2f}**. Hal yang paling kontras adalah `activity_span_hours`, yang naik dari median **{median_non['activity_span_hours']:.2f} jam** pada non-anomali menjadi **{median_anom['activity_span_hours']:.2f} jam** pada anomali.

Temuan ini mengindikasikan bahwa label anomali dalam eksperimen ini terutama menangkap address dengan:
- aktivitas yang berlangsung sangat lama
- jumlah transaksi dan relasi tetangga yang jauh lebih besar
- volume nilai transaksi yang lebih tinggi
- posisi graph yang lebih sentral atau lebih ekstrem daripada mayoritas node

## 3. Stabilitas Model
Evaluasi stabilitas dilakukan menggunakan resampling full-dataset dan pengukuran top-k Jaccard similarity. Nilai yang tinggi menunjukkan bahwa address-address anomali yang dipilih model cenderung konsisten pada pengulangan yang berbeda.

Model dengan stabilitas tertinggi adalah:
{stability_lines}

Secara umum, hasil ini menunjukkan bahwa pipeline tidak hanya menghasilkan outlier, tetapi juga cukup konsisten dalam memilih kandidat anomali utama.

## 4. Fitur yang Paling Membedakan Label
Fitur-fitur berikut merupakan pembeda terkuat antara label anomali dan non-anomali berdasarkan standardized mean difference:
{top_features_lines}

Dengan kata lain, label anomali lebih banyak dipengaruhi oleh intensitas aktivitas, jangkauan koneksi graph, dan durasi keterlibatan address di jaringan daripada hanya satu ukuran transaksi tunggal.

## 5. Address Anomali Peringkat Atas
Sepuluh address teratas berdasarkan skor ensemble adalah:
{top_addresses_lines}

Perlu dicatat bahwa address pada peringkat atas tidak selalu berarti address tersebut bersifat fraud atau malicious. Dalam konteks unsupervised anomaly detection, address tersebut lebih tepat ditafsirkan sebagai **address dengan pola perilaku yang sangat tidak biasa** dibandingkan populasi lain.

## 6. Keterbatasan
- Label anomali bersifat pseudo-label dari ensemble model, bukan ground truth eksternal.
- Nilai `mean_anomaly_ratio` mendekati 2% karena mengikuti parameter `contamination=0.02`, sehingga metrik ini tidak boleh dibaca sebagai prevalensi anomali riil.
- Address besar seperti kontrak token, exchange hot wallet, atau smart contract populer dapat muncul sebagai anomali karena perilakunya memang sangat ekstrem secara statistik.

## 7. Artefak Visual yang Disarankan untuk Bab Hasil
- `paper_ensemble_vote_distribution.png`
- `paper_top_feature_effects.png`
- `paper_summary_panel.png`
- `feature_mean_heatmap_by_ensemble_label.png`
- `feature_relationship_graph_hybrid.png`
"""

    (output_dir / "thesis_results_summary.md").write_text(text, encoding="utf-8")
def build_final_result(
    node_df: pd.DataFrame,
    pred_hybrid_cv: pd.DataFrame,
    full_pred_hybrid: pd.DataFrame,
) -> pd.DataFrame:
    cv_pred = (
        pred_hybrid_cv.groupby("address")
        .agg(
            cv_test_appearances=("fold", "count"),
            iforest_cv_score_mean=("iforest_score", "mean"),
            iforest_cv_flag_sum=("iforest_flag", "sum"),
            lof_cv_score_mean=("lof_score", "mean"),
            lof_cv_flag_sum=("lof_flag", "sum"),
            pca_cv_score_mean=("pca_score", "mean"),
            pca_cv_flag_sum=("pca_flag", "sum"),
        )
        .reset_index()
    )

    result_node = node_df.merge(cv_pred, on="address", how="left")
    result_node = result_node.merge(full_pred_hybrid, on="address", how="left")

    fill_zero_cols = [
        "cv_test_appearances",
        "iforest_cv_flag_sum",
        "lof_cv_flag_sum",
        "pca_cv_flag_sum",
        "iforest_full_flag",
        "lof_full_flag",
        "pca_full_flag",
        "ensemble_vote_count",
        "ensemble_consistent_anomaly",
    ]
    for col in fill_zero_cols:
        result_node[col] = result_node[col].fillna(0)

    score_cols = [
        "iforest_cv_score_mean",
        "lof_cv_score_mean",
        "pca_cv_score_mean",
        "iforest_full_score",
        "lof_full_score",
        "pca_full_score",
        "iforest_full_score_percentile",
        "lof_full_score_percentile",
        "pca_full_score_percentile",
        "ensemble_score_percentile_mean",
    ]
    for col in score_cols:
        result_node[col] = result_node[col].fillna(0.0)

    int_cols = [
        "cv_test_appearances",
        "iforest_cv_flag_sum",
        "lof_cv_flag_sum",
        "pca_cv_flag_sum",
        "iforest_full_flag",
        "lof_full_flag",
        "pca_full_flag",
        "ensemble_vote_count",
        "ensemble_consistent_anomaly",
    ]
    result_node[int_cols] = result_node[int_cols].astype(int)
    return result_node


def save_bar_plots(summary_all: pd.DataFrame, stability_all: pd.DataFrame, output_dir: Path) -> None:
    for feature_set in summary_all["feature_set"].unique():
        sub = summary_all[summary_all["feature_set"] == feature_set]
        plt.figure(figsize=(8, 4))
        plt.bar(sub["model"], sub["mean_anomaly_ratio"])
        plt.title(f"Mean Anomaly Ratio - {feature_set}")
        plt.ylabel("Mean anomaly ratio")
        plt.xticks(rotation=20)
        plt.tight_layout()
        plt.savefig(output_dir / f"mean_anomaly_ratio_{feature_set}.png", dpi=200)
        plt.close()

    for feature_set in stability_all["feature_set"].unique():
        sub = stability_all[stability_all["feature_set"] == feature_set]
        plt.figure(figsize=(8, 4))
        plt.bar(sub["model"], sub["mean_topk_jaccard"])
        plt.title(f"Top-k Stability (Jaccard) - {feature_set}")
        plt.ylabel("Mean Jaccard")
        plt.xticks(rotation=20)
        plt.tight_layout()
        plt.savefig(output_dir / f"topk_stability_{feature_set}.png", dpi=200)
        plt.close()


def save_outputs(
    output_dir: Path,
    query: str,
    args: argparse.Namespace,
    edge_df: pd.DataFrame,
    cv_tx: pd.DataFrame,
    cv_graph: pd.DataFrame,
    cv_hybrid: pd.DataFrame,
    pred_tx: pd.DataFrame,
    pred_graph: pd.DataFrame,
    pred_hybrid: pd.DataFrame,
    full_pred_hybrid: pd.DataFrame,
    summary_all: pd.DataFrame,
    stability_all: pd.DataFrame,
    result_node: pd.DataFrame,
    top_anomaly_table: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "query_used.sql").write_text(query, encoding="utf-8")
    (output_dir / "run_config.json").write_text(
        json.dumps(vars(args), indent=2, ensure_ascii=True),
        encoding="utf-8",
    )

    summary_all.to_csv(output_dir / "cv_summary_all_models.csv", index=False)
    stability_all.to_csv(output_dir / "cv_stability_all_models.csv", index=False)
    cv_tx.to_csv(output_dir / "cv_fold_transaction.csv", index=False)
    cv_graph.to_csv(output_dir / "cv_fold_graph.csv", index=False)
    cv_hybrid.to_csv(output_dir / "cv_fold_hybrid.csv", index=False)

    pred_tx.to_csv(output_dir / "pred_transaction_cv.csv", index=False)
    pred_graph.to_csv(output_dir / "pred_graph_cv.csv", index=False)
    pred_hybrid.to_csv(output_dir / "pred_hybrid_cv.csv", index=False)
    full_pred_hybrid.to_csv(output_dir / "pred_hybrid_full_dataset.csv", index=False)

    result_node.to_csv(output_dir / "ethereum_graph_node_anomalies_cv.csv", index=False)
    edge_df.to_csv(output_dir / "ethereum_graph_edges.csv", index=False)
    top_anomaly_table.to_csv(output_dir / "top_ensemble_anomalies.csv", index=False)

    save_bar_plots(summary_all, stability_all, output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Ethereum graph-based anomaly detection dengan 5-fold cross-validation. "
            "Skrip ini langsung mengambil data dari BigQuery."
        )
    )
    parser.add_argument(
        "--start-date",
        default=DEFAULT_START_DATE,
        help=f"Tanggal awal query BigQuery. Default: {DEFAULT_START_DATE}",
    )
    parser.add_argument(
        "--end-date",
        default=DEFAULT_END_DATE,
        help=f"Tanggal akhir query BigQuery. Default: {DEFAULT_END_DATE}",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_QUERY_LIMIT,
        help=f"LIMIT query BigQuery. Gunakan 0 untuk tanpa LIMIT. Default: {DEFAULT_QUERY_LIMIT}",
    )
    parser.add_argument(
        "--query-chunking",
        choices=["none", "month", "day"],
        default=DEFAULT_QUERY_CHUNKING,
        help=(
            "Pecah query BigQuery per potongan tanggal agar bisa dicache dan di-resume. "
            f"Default: {DEFAULT_QUERY_CHUNKING}"
        ),
    )
    parser.add_argument(
        "--query-cache-dir",
        default=DEFAULT_QUERY_CACHE_DIR,
        help=(
            "Folder cache untuk hasil query BigQuery per chunk. "
            f"Default: {DEFAULT_QUERY_CACHE_DIR}"
        ),
    )
    parser.add_argument(
        "--force-refresh-cache",
        action="store_true",
        help="Abaikan cache query BigQuery yang sudah ada dan ambil ulang dari server.",
    )
    parser.add_argument(
        "--max-edges-for-nx",
        type=int,
        default=DEFAULT_MAX_EDGES_FOR_NX,
        help=(
            "Jumlah edge agregat maksimum untuk perhitungan NetworkX. "
            f"Default: {DEFAULT_MAX_EDGES_FOR_NX}"
        ),
    )
    parser.add_argument(
        "--contamination",
        type=float,
        default=0.02,
        help="Proporsi anomaly yang diasumsikan model. Default: 0.02",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Jumlah fold cross-validation. Default: 5",
    )
    parser.add_argument(
        "--top-k-ratio",
        type=float,
        default=0.05,
        help="Rasio top-k untuk perhitungan Jaccard stability. Default: 0.05",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed untuk CV dan model. Default: 42",
    )
    parser.add_argument(
        "--stability-runs",
        type=int,
        default=DEFAULT_STABILITY_RUNS,
        help=f"Jumlah resampling untuk stability metric. Default: {DEFAULT_STABILITY_RUNS}",
    )
    parser.add_argument(
        "--stability-sample-ratio",
        type=float,
        default=DEFAULT_STABILITY_SAMPLE_RATIO,
        help=(
            "Proporsi data yang dipakai per stability resampling. "
            f"Default: {DEFAULT_STABILITY_SAMPLE_RATIO}"
        ),
    )
    parser.add_argument(
        "--feature-graph-threshold",
        type=float,
        default=DEFAULT_FEATURE_GRAPH_THRESHOLD,
        help=(
            "Threshold absolute Spearman correlation untuk edge feature graph. "
            f"Default: {DEFAULT_FEATURE_GRAPH_THRESHOLD}"
        ),
    )
    parser.add_argument(
        "--feature-graph-max-edges",
        type=int,
        default=DEFAULT_FEATURE_GRAPH_MAX_EDGES,
        help=(
            "Jumlah edge maksimum pada feature graph. "
            f"Default: {DEFAULT_FEATURE_GRAPH_MAX_EDGES}"
        ),
    )
    parser.add_argument(
        "--top-anomalies-export-limit",
        type=int,
        default=DEFAULT_TOP_ANOMALIES_EXPORT_LIMIT,
        help=(
            "Jumlah address teratas yang diekspor ke top_ensemble_anomalies.csv. "
            f"Default: {DEFAULT_TOP_ANOMALIES_EXPORT_LIMIT}"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Folder output. Default: {DEFAULT_OUTPUT_DIR}",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    cache_dir = Path(args.query_cache_dir).expanduser().resolve()
    query = build_query(args.start_date, args.end_date, args.limit)

    if args.query_chunking == "none":
        raw_df = load_transactions(query)
    else:
        raw_df = load_transactions_chunked(
            start_date=args.start_date,
            end_date=args.end_date,
            limit=args.limit,
            chunking=args.query_chunking,
            cache_dir=cache_dir,
            force_refresh_cache=args.force_refresh_cache,
        )

    tx_df = preprocess_transactions(raw_df)
    edge_df = build_edge_features(tx_df)
    node_df = build_node_features(tx_df)
    node_df = add_topological_features(node_df, edge_df, args.max_edges_for_nx)

    feature_sets = get_feature_sets(node_df)

    cv_tx, pred_tx, stab_tx, sum_tx = evaluate_unsupervised_cv(
        node_df,
        feature_sets["transaction"],
        contamination=args.contamination,
        n_splits=args.n_splits,
        random_state=args.random_state,
        top_k_ratio=args.top_k_ratio,
    )
    cv_graph, pred_graph, stab_graph, sum_graph = evaluate_unsupervised_cv(
        node_df,
        feature_sets["graph"],
        contamination=args.contamination,
        n_splits=args.n_splits,
        random_state=args.random_state,
        top_k_ratio=args.top_k_ratio,
    )
    cv_hybrid, pred_hybrid, stab_hybrid, sum_hybrid = evaluate_unsupervised_cv(
        node_df,
        feature_sets["hybrid"],
        contamination=args.contamination,
        n_splits=args.n_splits,
        random_state=args.random_state,
        top_k_ratio=args.top_k_ratio,
    )

    stab_tx = evaluate_resampled_stability(
        node_df,
        feature_sets["transaction"],
        contamination=args.contamination,
        n_runs=args.stability_runs,
        sample_ratio=args.stability_sample_ratio,
        random_state=args.random_state,
        top_k_ratio=args.top_k_ratio,
    )
    stab_graph = evaluate_resampled_stability(
        node_df,
        feature_sets["graph"],
        contamination=args.contamination,
        n_runs=args.stability_runs,
        sample_ratio=args.stability_sample_ratio,
        random_state=args.random_state,
        top_k_ratio=args.top_k_ratio,
    )
    stab_hybrid = evaluate_resampled_stability(
        node_df,
        feature_sets["hybrid"],
        contamination=args.contamination,
        n_runs=args.stability_runs,
        sample_ratio=args.stability_sample_ratio,
        random_state=args.random_state,
        top_k_ratio=args.top_k_ratio,
    )

    full_pred_hybrid = fit_full_dataset_models(
        node_df,
        feature_sets["hybrid"],
        contamination=args.contamination,
        random_state=args.random_state,
    )

    sum_tx["feature_set"] = "transaction"
    sum_graph["feature_set"] = "graph"
    sum_hybrid["feature_set"] = "hybrid"
    summary_all = pd.concat([sum_tx, sum_graph, sum_hybrid], ignore_index=True)
    summary_all = summary_all[
        [
            "feature_set",
            "model",
            "mean_score",
            "std_score_across_folds",
            "mean_anomaly_ratio",
            "std_anomaly_ratio",
        ]
    ]

    stab_tx["feature_set"] = "transaction"
    stab_graph["feature_set"] = "graph"
    stab_hybrid["feature_set"] = "hybrid"
    stability_all = pd.concat([stab_tx, stab_graph, stab_hybrid], ignore_index=True)
    stability_all = stability_all[
        [
            "feature_set",
            "model",
            "mean_topk_jaccard",
            "std_topk_jaccard",
            "stability_method",
            "stability_runs",
            "stability_sample_ratio",
            "top_k_size",
        ]
    ]

    result_node = build_final_result(node_df, pred_hybrid, full_pred_hybrid)
    top_anomaly_table = build_top_anomaly_table(
        result_node,
        limit=args.top_anomalies_export_limit,
    )
    build_feature_relationship_graph(
        feature_df=node_df,
        feature_cols=feature_sets["hybrid"],
        transaction_feature_cols=feature_sets["transaction"],
        graph_feature_cols=feature_sets["graph"],
        output_dir=output_dir,
        corr_threshold=args.feature_graph_threshold,
        max_edges=args.feature_graph_max_edges,
    )
    feature_effects_df, _ = create_label_relationship_outputs(
        result_node=result_node,
        feature_cols=feature_sets["hybrid"],
        output_dir=output_dir,
        label_col="ensemble_consistent_anomaly",
        top_n=10,
    )
    write_thesis_results_summary(
        summary_all=summary_all,
        stability_all=stability_all,
        result_node=result_node,
        top_anomaly_table=top_anomaly_table,
        feature_effects_df=feature_effects_df,
        output_dir=output_dir,
    )
    create_paper_ready_outputs(
        summary_all=summary_all,
        stability_all=stability_all,
        result_node=result_node,
        top_anomaly_table=top_anomaly_table,
        feature_effects_df=feature_effects_df,
        output_dir=output_dir,
    )

    save_outputs(
        output_dir=output_dir,
        query=query,
        args=args,
        edge_df=edge_df,
        cv_tx=cv_tx,
        cv_graph=cv_graph,
        cv_hybrid=cv_hybrid,
        pred_tx=pred_tx,
        pred_graph=pred_graph,
        pred_hybrid=pred_hybrid,
        full_pred_hybrid=full_pred_hybrid,
        summary_all=summary_all,
        stability_all=stability_all,
        result_node=result_node,
        top_anomaly_table=top_anomaly_table,
    )

    print("[INFO] Semua output tersimpan di:", output_dir)
    print("[INFO] Top 10 anomaly final (berdasarkan ensemble/full-dataset score):")
    print(
        top_anomaly_table.head(10)[
            [
                "address",
                "ensemble_vote_count",
                "ensemble_consistent_anomaly",
                "ensemble_score_percentile_mean",
                "iforest_full_flag",
                "lof_full_flag",
                "pca_full_flag",
            ]
        ]
    )


if __name__ == "__main__":
    main()
