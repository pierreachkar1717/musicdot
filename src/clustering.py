#!/usr/bin/env python3
import os
import duckdb
import numpy as np
import umap
import hdbscan
from tqdm import tqdm

# --- CONFIG ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "db", "library.duckdb")
UMAP_COMPONENTS = 50  # dimensions to reduce to before clustering
MIN_CLUSTER_SIZE = 5  # HDBSCAN parameter: minimum cluster size


def run_clustering(
    db_path: str = DB_PATH,
    n_components: int = UMAP_COMPONENTS,
    min_cluster_size: int = MIN_CLUSTER_SIZE,
) -> None:
    """
    Perform UMAP reduction and HDBSCAN clustering on stored embeddings,
    then persist cluster assignments in DuckDB table 'track_clusters'.
    Noise points are reassigned to the nearest real-cluster centroid.
    """
    # Connect to DuckDB
    con = duckdb.connect(db_path)

    # Fetch embeddings
    rows = con.execute("SELECT track_id, vector FROM embeddings").fetchall()
    if not rows:
        print("No embeddings found in the database.")
        return

    track_ids, blobs = zip(*rows)
    vecs = np.stack([np.frombuffer(b, dtype=np.float32) for b in blobs])
    n_samples = vecs.shape[0]
    print(f"Loaded {n_samples} embeddings (dim={vecs.shape[1]}) for clustering.")

    # Handle very small collections
    if n_samples < 3:
        labels = np.zeros(n_samples, dtype=int)
        print("Too few samples for UMAP; all assigned to cluster 0.")
    else:
        # choose n_neighbors < n_samples
        n_neighbors = min(15, n_samples - 1)
        print(f"Using n_neighbors={n_neighbors} for UMAP (samples={n_samples}).")

        # UMAP reduction with random init to avoid spectral eigsh errors
        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            init="random",
            random_state=42,
        )
        umap_embs = reducer.fit_transform(vecs)
        print(f"UMAP reduction to {n_components} dims complete.")

        # HDBSCAN clustering
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
        labels = clusterer.fit_predict(umap_embs)
        real_clusters = [c for c in set(labels) if c != -1]
        num_clusters = len(real_clusters)
        print(f"HDBSCAN found {num_clusters} clusters plus noise (label -1).")

        # Reassign noise points to nearest centroid
        if real_clusters:
            centroids = {c: umap_embs[labels == c].mean(axis=0) for c in real_clusters}
            for i, lab in enumerate(labels):
                if lab == -1:
                    # compute distances to each real cluster centroid
                    dists = [
                        np.linalg.norm(umap_embs[i] - centroids[c])
                        for c in real_clusters
                    ]
                    # assign to closest cluster
                    labels[i] = real_clusters[int(np.argmin(dists))]
            print("Noise points reassigned to nearest cluster centroids.")

    # Persist cluster assignments
    con.execute(
        "CREATE TABLE IF NOT EXISTS track_clusters (track_id INTEGER, cluster_id INTEGER)"
    )
    con.execute("DELETE FROM track_clusters")
    insert_data = list(zip(track_ids, labels.tolist()))

    # batch insert
    batch_size = 1000
    for i in tqdm(range(0, len(insert_data), batch_size), desc="Saving clusters"):
        chunk = insert_data[i : i + batch_size]
        con.executemany("INSERT INTO track_clusters VALUES (?, ?)", chunk)

    con.commit()
    con.close()
    print("Cluster assignments saved to 'track_clusters'.")


if __name__ == "__main__":
    run_clustering()
