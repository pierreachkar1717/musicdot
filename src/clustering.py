from __future__ import annotations
"""
clustering.py — Dimensionality-reduction + geometry-aware HDBSCAN clustering

Pipeline
~~~~~~~~
1. Load L2-normalised embeddings from DuckDB.
2. PCA → enough dims to explain 95 % variance (cap 512).
3. UMAP (cosine) → 35-D manifold.
4. HDBSCAN (leaf) → raw labels.
5. Reassign noise via 5-NN majority vote.
6. **Name clusters** by polar‐angle order and store centroids.
7. Persist results to `track_clusters` + `cluster_meta`.
"""

# ── stdlib
import json
import logging
from logging.handlers import RotatingFileHandler
from math import atan2, pi, sqrt
from pathlib import Path
from typing import Final

# ── 3rd-party
import duckdb
import hdbscan
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import umap        # type: ignore
from tqdm import tqdm

# ── Logging -----------------------------------------------------------
LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
_handler = RotatingFileHandler(
    LOG_DIR / "clustering.log",
    maxBytes=5_000_000,
    backupCount=3,
    encoding="utf-8",
)
_handler.setFormatter(
    logging.Formatter(
        "%(asctime)s %(levelname)8s [%(name)s] %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )
)
logger.addHandler(_handler)
logger.propagate = False

# ── Hyper-parameters --------------------------------------------------
BASE_DIR:          Final = Path(__file__).resolve().parents[1]
DB_PATH:           Final = BASE_DIR / "db" / "library.duckdb"

PCA_VARIANCE:      Final = 0.95
PCA_MAX_COMPONENTS:Final = 512

UMAP_COMPONENTS:   Final = 35
UMAP_MIN_DIST:     Final = 0.2

HDBSCAN_K_FACTOR:  Final = 0.5   # min_cluster_size ≈ √N × factor
KNN_K:             Final = 5

INSERT_BATCH:      Final = 1_000


# ── Helpers -----------------------------------------------------------
def polar_angle(x: float, y: float) -> float:
    """Return angle in radians ∈ [0, 2π)."""
    return (atan2(y, x) + 2 * pi) % (2 * pi)


# ── Core --------------------------------------------------------------
def run_clustering(db_path: str | Path = DB_PATH) -> None:
    """Execute full clustering pipeline and write results to DuckDB."""
    logger.info("Opening DuckDB at %s", db_path)
    con = duckdb.connect(str(db_path))

    # 1 ─ Load embeddings ---------------------------------------------
    rows = con.execute("SELECT track_id, vector FROM embeddings").fetchall()
    if not rows:
        logger.warning("No embeddings found; aborting.")
        return

    track_ids, blobs = zip(*rows)
    vecs = np.stack([np.frombuffer(b, dtype=np.float32) for b in blobs])
    n_samples, emb_dim = vecs.shape
    logger.info("Loaded %d embeddings (%d-D)", n_samples, emb_dim)

    # 2 ─ PCA ----------------------------------------------------------
    pca = PCA(n_components=min(PCA_MAX_COMPONENTS, n_samples), random_state=42)
    vecs_pca = pca.fit_transform(vecs)
    cum_var  = np.cumsum(pca.explained_variance_ratio_)
    k        = int(np.searchsorted(cum_var, PCA_VARIANCE) + 1)
    vecs_pca = vecs_pca[:, :k]
    logger.info("PCA → %d dims (%.1f%% variance)", k, cum_var[k - 1] * 100)

    # 3 ─ UMAP ---------------------------------------------------------
    nn = int(max(15, min(64, sqrt(n_samples))))
    logger.info("UMAP (n_neighbors=%d)…", nn)
    reducer = umap.UMAP(
        n_components     = UMAP_COMPONENTS,
        n_neighbors      = nn,
        min_dist         = UMAP_MIN_DIST,
        metric           = "cosine",
        random_state     = 42,
        init             = "random",
        low_memory       = True,
    )
    vecs_umap = reducer.fit_transform(vecs_pca)
    logger.info("UMAP done → %s", vecs_umap.shape)

    # 4 ─ HDBSCAN ------------------------------------------------------
    min_cluster = max(5, int(sqrt(n_samples) * HDBSCAN_K_FACTOR))
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size        = min_cluster,
        metric                  = "euclidean",
        cluster_selection_method= "leaf",
        prediction_data         = True,
        core_dist_n_jobs        = 1,
    )
    labels = clusterer.fit_predict(vecs_umap)
    real_clusters = [c for c in set(labels) if c != -1]
    logger.info(
        "HDBSCAN → %d clusters, %d noise points",
        len(real_clusters),
        int(np.sum(labels == -1)),
    )

    # 5 ─ Reassign noise ----------------------------------------------
    noise_mask = labels == -1
    if noise_mask.any() and real_clusters:
        knn = KNeighborsClassifier(n_neighbors=min(KNN_K, len(real_clusters)))
        knn.fit(vecs_umap[~noise_mask], labels[~noise_mask])
        labels[noise_mask] = knn.predict(vecs_umap[noise_mask])
        logger.info("Noise reassigned with %d-NN majority vote", KNN_K)

    # 6 ─ Name clusters + centroids -----------------------------------
    centroids = {
        cid: vecs_umap[labels == cid].mean(axis=0)[:2]   # take x,y
        for cid in real_clusters
    }
    ordered = sorted(
        centroids.items(),
        key=lambda kv: polar_angle(*kv[1])               # clockwise order
    )
    friendly = {cid: f"C{idx:02d}" for idx, (cid, _) in enumerate(ordered, 1)}
    logger.info("Assigned friendly IDs to clusters (%s…)", ", ".join(list(friendly.values())[:5]))

    # 7 ─ Persist to DB -----------------------------------------------
    logger.info("Writing results …")
    con.execute("CREATE TABLE IF NOT EXISTS track_clusters (track_id INT, cluster_id INT)")
    con.execute("DELETE FROM track_clusters")

    pairs = list(zip(track_ids, labels.tolist()))
    for i in tqdm(range(0, len(pairs), INSERT_BATCH), desc="track_clusters", ncols=80):
        con.executemany("INSERT INTO track_clusters VALUES (?, ?)", pairs[i:i+INSERT_BATCH])

    # -- cluster_meta table
    con.execute("""
        CREATE TABLE IF NOT EXISTS cluster_meta(
            cluster_id   INT PRIMARY KEY,
            friendly_id  TEXT,
            size         INT,
            centroid_x   REAL,
            centroid_y   REAL,
            neighbours   TEXT     -- JSON list of 3 nearest cluster_ids
        )
    """)
    con.execute("DELETE FROM cluster_meta")

    # compute 3-nearest cluster ids for neighbourhood context
    cids = np.array(real_clusters)
    centroid_mat = np.vstack([centroids[c] for c in cids])
    dists = np.linalg.norm(centroid_mat[:, None, :] - centroid_mat[None, :, :], axis=-1)
    np.fill_diagonal(dists, np.inf)
    neighbour_map = {cid: cids[idx[:3]].tolist() for cid, idx in zip(cids, np.argsort(dists, axis=1))}

    rows_meta = [
        (
            int(cid),
            friendly[cid],
            int(np.sum(labels == cid)),
            float(centroids[cid][0]),
            float(centroids[cid][1]),
            json.dumps(neighbour_map[cid]),
        )
        for cid in real_clusters
    ]
    con.executemany("INSERT INTO cluster_meta VALUES (?, ?, ?, ?, ?, ?)", rows_meta)

    con.commit()
    con.close()
    logger.info("All tables updated successfully.")


# ---------------------------------------------------------------------
if __name__ == "__main__":
    run_clustering()