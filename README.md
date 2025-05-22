<pre>
musicdot/
├── README.md               # project overview & setup instructions
├── requirements.txt
├── models/
│   └── discogs-effnet-bs64-1.pb
├── data/
│   └── (your audio files here)
├── exports/
│   └── (playlists, clusters, reports)
└── src/
    ├── __init__.py
    ├── embedder.py          # EffNetDiscogsEmbedder class
    ├── feature_extractor.py # extract BPM, key, loudness, ID3 metadata
    ├── vector_store.py      # init FAISS + DuckDB, save/retrieve embeddings
    ├── clustering.py        # UMAP + HDBSCAN pipeline to compute clusters & tags
    ├── export_clusters.py   # SQL/DB queries + XML/CSV export of clusters/playlists
    └── data_ingestion.py    # orchestrator: runs ingest → store → cluster → export
</pre>
