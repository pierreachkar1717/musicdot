from __future__ import annotations

"""
Features
--------
- ID‑stable FAISS via IndexIDMap (vectors use their *track_id* as key)
- Upserts on duplicate file paths
- File‑based logging (`logs/vector_store.log`)
- Explicit helper `delete_track()` because DuckDB (<v0.10) lacks ON DELETE CASCADE
"""

from pathlib import Path
import logging
import os
from typing import Final, Optional

import duckdb
import faiss
import numpy as np

logger = logging.getLogger("vector_store")

# ---------------------------------------------------------------------------
# Logging setup (only once)
# ---------------------------------------------------------------------------
LOG_PATH: Final[Path] = (
    Path(__file__).resolve().parent.parent / "logs" / "vector_store.log"
)
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
if not any(
    isinstance(h, logging.FileHandler) and h.baseFilename == str(LOG_PATH)
    for h in logger.handlers
):
    fh = logging.handlers.RotatingFileHandler(
        LOG_PATH, maxBytes=5_242_880, backupCount=3
    )
    fh.setFormatter(
        logging.Formatter("%(asctime)s  %(levelname)8s  %(name)s: %(message)s")
    )
    logger.addHandler(fh)
    logger.setLevel(logging.INFO)


class VectorStore:
    """Persist metadata in DuckDB and vectors in FAISS."""

    def __init__(
        self,
        db_path: str | os.PathLike[str] = "library.duckdb",
        index_path: str | os.PathLike[str] = "vector.index",
        embedding_dim: int = 1_280,
    ) -> None:
        self.db_path: str = str(db_path)
        self.index_path: str = str(index_path)
        self.embedding_dim: int = int(embedding_dim)

        self._conn: Optional[duckdb.DuckDBPyConnection] = None
        self._index: Optional[faiss.IndexIDMap] = None

    # ---------------------------------------------------------------------
    # Schema & index initialisation
    # ---------------------------------------------------------------------
    def init_store(self) -> None:
        """Create tables + FAISS index if missing."""
        logger.info("Opening DuckDB at %s", self.db_path)
        self._conn = duckdb.connect(self.db_path)
        self._conn.execute("PRAGMA threads=4")

        # sequences / tables
        self._conn.execute(
            """
            CREATE SEQUENCE IF NOT EXISTS track_id_seq START 1;
        """
        )

        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS tracks (
                track_id  INTEGER DEFAULT nextval('track_id_seq') PRIMARY KEY,
                path      TEXT UNIQUE,
                title     TEXT,
                artist    TEXT,
                album     TEXT,
                year      INT,
                duration  REAL,
                bitrate   INT
            );
        """
        )

        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS audio_features (
                track_id INTEGER PRIMARY KEY,
                bpm      REAL,
                key      TEXT,
                loudness REAL
            );
        """
        )

        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings (
                track_id INTEGER PRIMARY KEY,
                vector   BLOB
            );
        """
        )

        # indices for faster filtering
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_tracks_year  ON tracks(year);"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_feat_bpm    ON audio_features(bpm);"
        )

        # FAISS
        if os.path.isfile(self.index_path):
            logger.info("Loading existing FAISS index → %s", self.index_path)
            self._index = faiss.read_index(self.index_path)
        else:
            logger.info("Creating new FAISS index (%d dims)", self.embedding_dim)
            self._index = faiss.IndexIDMap(faiss.IndexFlatIP(self.embedding_dim))

    # ---------------------------------------------------------------------
    # CRUD helpers
    # ---------------------------------------------------------------------
    def add_track(
        self, *, path: str, metadata: dict, features: dict, embedding: np.ndarray
    ) -> int:
        """Insert or update a track and its vector; returns *track_id*."""
        if self._conn is None or self._index is None:
            raise RuntimeError("VectorStore not initialised; call init_store() first")

        # ----------------------- DB upsert ------------------------------ #
        row = self._conn.execute(
            """
            INSERT INTO tracks(path, title, artist, album, year, duration, bitrate)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (path) DO UPDATE SET
                title    = excluded.title,
                artist   = excluded.artist,
                album    = excluded.album,
                year     = excluded.year,
                duration = excluded.duration,
                bitrate  = excluded.bitrate
            RETURNING track_id;
            """,
            (
                path,
                metadata.get("title"),
                metadata.get("artist"),
                metadata.get("album"),
                metadata.get("year"),
                metadata.get("duration"),
                metadata.get("bitrate"),
            ),
        ).fetchone()
        track_id = int(row[0])

        # audio features (simple replace strategy)
        self._conn.execute("DELETE FROM audio_features WHERE track_id = ?", (track_id,))
        self._conn.execute(
            "INSERT INTO audio_features VALUES (?, ?, ?, ?)",
            (
                track_id,
                features.get("bpm"),
                features.get("key"),
                features.get("loudness"),
            ),
        )

        # embeddings table
        blob = embedding.astype("float32").tobytes()
        self._conn.execute("DELETE FROM embeddings WHERE track_id = ?", (track_id,))
        self._conn.execute("INSERT INTO embeddings VALUES (?, ?)", (track_id, blob))

        # ----------------------- FAISS upsert --------------------------- #
        vec = embedding.astype("float32")
        vec /= np.linalg.norm(vec)
        self._index.remove_ids(np.array([track_id], dtype="int64"))
        self._index.add_with_ids(
            vec[np.newaxis, :], np.array([track_id], dtype="int64")
        )

        logger.debug("Stored track_id=%d path=%s", track_id, os.path.basename(path))
        return track_id

    # ------------------------------------------------------------------ #
    def delete_track(self, track_id: int) -> None:
        """Remove track row + child rows + vector."""
        if self._conn is None or self._index is None:
            raise RuntimeError("VectorStore not initialised; call init_store() first")
        self._index.remove_ids(np.array([track_id], dtype="int64"))
        self._conn.execute("DELETE FROM embeddings     WHERE track_id = ?", (track_id,))
        self._conn.execute("DELETE FROM audio_features WHERE track_id = ?", (track_id,))
        self._conn.execute("DELETE FROM tracks         WHERE track_id = ?", (track_id,))
        logger.info("Deleted track_id=%d", track_id)

    # ------------------------------------------------------------------ #
    def search(
        self, query_vector: np.ndarray, topk: int = 10
    ) -> list[tuple[int, float]]:
        """Return ``[(track_id, score), …]`` using inner‑product similarity."""
        if self._index is None:
            raise RuntimeError("VectorStore not initialised; call init_store() first")
        vec = query_vector.astype("float32")
        vec /= np.linalg.norm(vec)
        D, I = self._index.search(vec[np.newaxis, :], topk)
        return [(int(i), float(d)) for i, d in zip(I[0], D[0]) if i != -1]

    # ------------------------------------------------------------------ #
    def save(self) -> None:
        """Flush DB transaction and persist FAISS index."""
        if self._conn:
            self._conn.commit()
        if self._index:
            faiss.write_index(self._index, self.index_path)
            logger.info("FAISS index saved → %s", self.index_path)

    def close(self) -> None:
        """Close DuckDB connection (but keep FAISS object in memory)."""
        if self._conn:
            self._conn.close()
            logger.info("DuckDB connection closed")
