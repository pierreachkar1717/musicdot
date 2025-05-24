from __future__ import annotations

"""clustering.py — Dimensionality‑reduction + HDBSCAN clustering

This script projects high‑dimensional music‑track embeddings to a compact
manifold and discovers density‑based clusters suitable for playlist/tagging
workflows.  It targets libraries in the **5 k – 50 k** range but works on
smaller sets as well.

Pipeline
~~~~~~~~
1. **Load** L2‑normalised embeddings from DuckDB.
2. **PCA** – retain enough components to explain 95 % variance (cap 512).
3. **UMAP** (cosine metric) -> 35‑D.
4. **HDBSCAN** (leaf selection).
5. Reassign noise points via 5‑NN majority vote.
6. **Persist** ``track_id ↔ cluster_id`` into `track_clusters`.
"""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Final

import duckdb
import hdbscan
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import umap  # type: ignore
from tqdm import tqdm

# ── Logging -----------------------------------------------------------
LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
_handler = RotatingFileHandler(
    LOG_DIR / "clustering.log", maxBytes=5_000_000, backupCount=3, encoding="utf-8"
)
_formatter = logging.Formatter(
    "%(asctime)s %(levelname)8s [%(name)s] %(message)s", "%Y-%m-%d %H:%M:%S"
)
_handler.setFormatter(_formatter)
logger.addHandler(_handler)
logger.propagate = False  # keep other modules out of this file

# ── Hyper‑parameters --------------------------------------------------
BASE_DIR: Final[Path] = Path(__file__).resolve().parents[1]
DB_PATH: Final[Path] = BASE_DIR / "db" / "library.duckdb"

PCA_VARIANCE: Final[float] = 0.95  # keep 95 % explained variance
PCA_MAX_COMPONENTS: Final[int] = 512

UMAP_COMPONENTS: Final[int] = 35
UMAP_MIN_DIST: Final[float] = 0.2

HDBSCAN_K_FACTOR: Final[float] = 0.5  # min_cluster_size ≈ sqrt(N)*factor
KNN_K: Final[int] = 5

INSERT_BATCH: Final[int] = 1_000


# ── Core --------------------------------------------------------------


def run_clustering(db_path: str | Path = DB_PATH) -> None:
    """Execute full clustering pipeline and write results to DuckDB."""

    logger.info("Opening DuckDB at %s", db_path)
    con = duckdb.connect(str(db_path))

    # 1. Load embeddings ------------------------------------------------
    rows = con.execute("SELECT track_id, vector FROM embeddings").fetchall()
    if not rows:
        logger.warning("No embeddings found; aborting.")
        return

    track_ids, blobs = zip(*rows)
    vecs = np.stack([np.frombuffer(b, dtype=np.float32) for b in blobs])
    n_samples, emb_dim = vecs.shape
    logger.info("Loaded %d embeddings (%d-D)", n_samples, emb_dim)

    # 2. PCA ------------------------------------------------------------
    pca = PCA(n_components=min(PCA_MAX_COMPONENTS, n_samples), random_state=42)
    vecs_pca = pca.fit_transform(vecs)
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    k = int(np.searchsorted(cum_var, PCA_VARIANCE) + 1)
    vecs_pca = vecs_pca[:, :k]
    logger.info("PCA → %d dims (%.1f%% variance)", k, cum_var[k - 1] * 100)

    # 3. UMAP -----------------------------------------------------------
    nn = int(max(15, min(64, np.sqrt(n_samples))))
    logger.info("UMAP (n_neighbors=%d) …", nn)
    reducer = umap.UMAP(
        n_components=UMAP_COMPONENTS,
        n_neighbors=nn,
        min_dist=UMAP_MIN_DIST,
        metric="cosine",
        random_state=42,
        init="random",
        low_memory=True,
    )
    vecs_umap = reducer.fit_transform(vecs_pca)
    logger.info("UMAP done → %s", vecs_umap.shape)

    # 4. HDBSCAN --------------------------------------------------------
    min_cluster = max(5, int(np.sqrt(n_samples) * HDBSCAN_K_FACTOR))
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster,
        metric="euclidean",
        cluster_selection_method="leaf",
        prediction_data=True,
        core_dist_n_jobs=1,
    )
    labels = clusterer.fit_predict(vecs_umap)
    real_clusters = [c for c in set(labels) if c != -1]
    logger.info(
        "HDBSCAN → %d clusters, %d noise points",
        len(real_clusters),
        int(np.sum(labels == -1)),
    )

    # 5. Reassign noise -------------------------------------------------
    noise_mask = labels == -1
    if noise_mask.any() and real_clusters:
        knn = KNeighborsClassifier(n_neighbors=min(KNN_K, len(real_clusters)))
        knn.fit(vecs_umap[~noise_mask], labels[~noise_mask])
        labels[noise_mask] = knn.predict(vecs_umap[noise_mask])
        logger.info("Noise reassigned with %d-NN majority vote", KNN_K)

    # 6. Persist labels -------------------------------------------------
    logger.info("Writing labels to track_clusters …")
    con.execute(
        "CREATE TABLE IF NOT EXISTS track_clusters (track_id INTEGER, cluster_id INTEGER)"
    )
    con.execute("DELETE FROM track_clusters")

    data = list(zip(track_ids, labels.tolist()))
    for i in tqdm(range(0, len(data), INSERT_BATCH), desc="Saving clusters", ncols=80):
        con.executemany(
            "INSERT INTO track_clusters VALUES (?, ?)", data[i : i + INSERT_BATCH]
        )

    con.commit()
    con.close()
    logger.info("Clusters written to track_clusters table.")


# ---------------------------------------------------------------------
if __name__ == "__main__":
    run_clustering()
