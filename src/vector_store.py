import os
import duckdb
import faiss
import numpy as np


class VectorStore:
    """
    Manages track metadata and embeddings using DuckDB for metadata
    and FAISS for vector similarity search.
    """

    def __init__(
        self,
        db_path: str = "library.duckdb",
        index_path: str = "vector.index",
        embedding_dim: int = 1280,
    ):
        self.db_path = db_path
        self.index_path = index_path
        self.embedding_dim = embedding_dim
        self._conn = None
        self._index = None

    def init_store(self):
        """Initialize DuckDB schema and FAISS index."""
        # DuckDB connection
        self._conn = duckdb.connect(self.db_path)
        self._conn.execute("PRAGMA threads=4")

        # Create sequence for auto-incrementing primary key
        self._conn.execute(
            """
            CREATE SEQUENCE IF NOT EXISTS track_id_seq START 1;
        """
        )

        # create tables
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS tracks (
                track_id    INTEGER DEFAULT nextval('track_id_seq'),
                path         TEXT UNIQUE,
                title        TEXT,
                artist       TEXT,
                album        TEXT,
                year         INT,
                duration     REAL,
                bitrate      INT,
                PRIMARY KEY(track_id)
            )
        """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS audio_features (
                track_id INTEGER,
                bpm      REAL,
                key      TEXT,
                loudness REAL,
                FOREIGN KEY(track_id) REFERENCES tracks(track_id)
            )
        """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings (
                track_id INTEGER,
                vector   BLOB,
                FOREIGN KEY(track_id) REFERENCES tracks(track_id)
            )
        """
        )

        # FAISS index
        if os.path.isfile(self.index_path):
            self._index = faiss.read_index(self.index_path)
        else:
            # inner-product index on L2-normalized vectors approximates cosine
            self._index = faiss.IndexFlatIP(self.embedding_dim)

    def add_track(
        self, path: str, metadata: dict, features: dict, embedding: np.ndarray
    ) -> int:
        """
        Insert a track into the database and FAISS index.

        Returns the new track_id.
        """
        # insert into tracks (uses default sequence for track_id)
        insert_sql = (
            "INSERT INTO tracks(path, title, artist, album, year, duration, bitrate) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)"
        )
        self._conn.execute(
            insert_sql,
            [
                path,
                metadata.get("title"),
                metadata.get("artist"),
                metadata.get("album"),
                metadata.get("year"),
                metadata.get("duration"),
                metadata.get("bitrate"),
            ],
        )
        # fetch track_id
        cur = self._conn.execute("SELECT track_id FROM tracks WHERE path = ?", [path])
        track_id = cur.fetchone()[0]

        # insert audio features
        feat_sql = "INSERT INTO audio_features(track_id, bpm, key, loudness) VALUES (?, ?, ?, ?)"
        self._conn.execute(
            feat_sql,
            [
                track_id,
                features.get("bpm"),
                features.get("key"),
                features.get("loudness"),
            ],
        )

        # insert embedding
        emb_sql = "INSERT INTO embeddings(track_id, vector) VALUES (?, ?)"
        blob = embedding.astype("float32").tobytes()
        self._conn.execute(emb_sql, [track_id, blob])

        # add to FAISS index
        vec = embedding / np.linalg.norm(embedding)
        self._index.add(np.expand_dims(vec, axis=0))

        return track_id

    def save(self):
        """Persist DuckDB changes and write FAISS index to disk."""
        if self._conn:
            self._conn.commit()
        if self._index:
            faiss.write_index(self._index, self.index_path)

    def search(
        self, query_vector: np.ndarray, topk: int = 10
    ) -> list[tuple[int, float]]:
        """
        Perform a nearest-neighbor search on the FAISS index.

        Returns a list of (track_id, score) tuples.
        """
        vec = query_vector / np.linalg.norm(query_vector)
        D, I = self._index.search(np.expand_dims(vec, axis=0), topk)
        results = []
        for idx, dist in zip(I[0], D[0]):
            if idx < 0:
                continue
            cur = self._conn.execute(
                "SELECT track_id FROM tracks LIMIT 1 OFFSET ?", [int(idx)]
            )
            row = cur.fetchone()
            if row:
                results.append((row[0], float(dist)))
        return results

    def close(self):
        """Close the DuckDB connection."""
        if self._conn:
            self._conn.close()
