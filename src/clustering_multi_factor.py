"""Cluster tracks on embeddings + musical key.

Adds a 24-dim one-hot key vector (12 semitones × major/minor) to each
track’s embedding.  The final feature vector is:

    concat(embedding (1280d), KEY_WEIGHT * one_hot_key (24d))

Adjust KEY_WEIGHT to control how strongly harmonic similarity influences
clustering.
"""
import os
import duckdb
import numpy as np
import umap
import hdbscan
from tqdm import tqdm

# --- CONFIG ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "db", "library.duckdb")
UMAP_COMPONENTS = 50      # dim after reduction
MIN_CLUSTER_SIZE = 5      # HDBSCAN min cluster size
KEY_WEIGHT = 0.5          # weight applied to one-hot key vector

# ordered list of keys -> index in 24-dim one-hot (Camelot style)
KEYS = [
    "Cmajor", "C#major", "Dmajor", "D#major", "Emajor", "Fmajor", "F#major", "Gmajor", "G#major", "Amajor", "A#major", "Bmajor",
    "Cminor", "C#minor", "Dminor", "D#minor", "Eminor", "Fminor", "F#minor", "Gminor", "G#minor", "Aminor", "A#minor", "Bminor",
]
KEY2IDX = {k: i for i, k in enumerate(KEYS)}


def one_hot_key(key: str) -> np.ndarray:
    vec = np.zeros(24, dtype=np.float32)
    idx = KEY2IDX.get(key)
    if idx is not None:
        vec[idx] = 1.0
    return vec


def run_clustering(
    db_path: str = DB_PATH,
    n_components: int = UMAP_COMPONENTS,
    min_cluster_size: int = MIN_CLUSTER_SIZE,
    key_weight: float = KEY_WEIGHT,
):
    con = duckdb.connect(db_path)

    # fetch embeddings + key in one query
    rows = con.execute(
        """
        SELECT e.track_id, e.vector, a.key
        FROM embeddings e
        JOIN audio_features a ON e.track_id = a.track_id
        """
    ).fetchall()
    if not rows:
        print("No embeddings found.")
        return

    track_ids, blobs, keys = zip(*rows)
    emb = np.stack([np.frombuffer(b, dtype=np.float32) for b in blobs])
    key_vecs = np.stack([one_hot_key(k) for k in keys]) * key_weight
    vecs = np.hstack([emb, key_vecs])

    n_samples = vecs.shape[0]
    print(f"Loaded {n_samples} tracks (vec dim={vecs.shape[1]}) for clustering.")

    if n_samples < 3:
        labels = np.zeros(n_samples, dtype=int)
        print("Too few samples; all assigned to cluster 0.")
    else:
        n_neighbors = min(15, n_samples - 1)
        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            init="random",
            random_state=42,
        )
        umap_embs = reducer.fit_transform(vecs)
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
        labels = clusterer.fit_predict(umap_embs)
        real_clusters = [c for c in set(labels) if c != -1]
        print(f"HDBSCAN found {len(real_clusters)} clusters + noise.")
        if real_clusters:
            centroids = {c: umap_embs[labels == c].mean(axis=0) for c in real_clusters}
            for i, lab in enumerate(labels):
                if lab == -1:
                    dists = [np.linalg.norm(umap_embs[i] - centroids[c]) for c in real_clusters]
                    labels[i] = real_clusters[int(np.argmin(dists))]
            print("Noise reassigned to nearest centroids.")

    # save clusters
    con.execute("CREATE TABLE IF NOT EXISTS track_clusters(track_id INT, cluster_id INT)")
    con.execute("DELETE FROM track_clusters")
    chunks = [list(zip(track_ids, labels.tolist()))[i:i+1000] for i in range(0, n_samples, 1000)]
    for chunk in tqdm(chunks, desc="Saving clusters"):
        con.executemany("INSERT INTO track_clusters VALUES (?, ?)", chunk)
    con.commit(); con.close()
    print("Cluster assignments saved.")


if __name__ == "__main__":
    run_clustering()