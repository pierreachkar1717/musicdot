from __future__ import annotations

"""
This script scans a music folder, extracts metadata + audio features, computes
Discogs‑EffNet embeddings, and stores everything into DuckDB + FAISS.
"""

from pathlib import Path
import logging
from typing import Final, List

from tqdm import tqdm

from feature_extractor import FeatureExtractor
from embedder import EffNetDiscogsEmbedder
from vector_store import VectorStore

# ---------------------------------------------------------------------------
# Configuration – edit these paths to suit your environment
# ---------------------------------------------------------------------------
DATA_DIR: Final = Path("/Users/pierreachkar/Documents/projects/musicdot/data")
MODEL_PATH: Final = Path(
    "/Users/pierreachkar/Documents/projects/musicdot/models/discogs-effnet-bs64-1.pb"
)
DB_PATH: Final = Path(
    "/Users/pierreachkar/Documents/projects/musicdot/db/library.duckdb"
)
INDEX_PATH: Final = Path(
    "/Users/pierreachkar/Documents/projects/musicdot/db/vector.index"
)

# Supported audio extensions
AUDIO_EXTS: Final = {".mp3", ".flac", ".wav", ".aiff"}

# ---------------------------------------------------------------------------
# Logging – console + rotating file handler  (5 MB × 3 backups)
# ---------------------------------------------------------------------------
LOG_FILE = Path(__file__).resolve().parent.parent / "logs" / "data_ingestion.log"
LOG_FILE.parent.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)8s  %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def gather_audio_files(root: Path) -> List[Path]:
    """Recursively collect audio files under *root* by extension."""
    return [p for p in root.rglob("*") if p.suffix.lower() in AUDIO_EXTS]


# ---------------------------------------------------------------------------
# Main routine
# ---------------------------------------------------------------------------


def main() -> None:  # noqa: D401 – simple verb ok
    """Run the ingestion pipeline with progress and logging."""

    logger.info("Starting ingestion run → data=%s", DATA_DIR)

    # Initialise pipeline components ------------------------------------------------
    fe = FeatureExtractor()
    embedder = EffNetDiscogsEmbedder(model_path=str(MODEL_PATH))
    vs = VectorStore(
        db_path=str(DB_PATH),
        index_path=str(INDEX_PATH),
        embedding_dim=embedder.output_dim,  # lazily discovered the first time
    )
    vs.init_store()

    # Discover audio -----------------------------------------------------------------
    files = gather_audio_files(DATA_DIR)
    logger.info("Discovered %d audio files", len(files))

    # Process ------------------------------------------------------------------------
    errors = 0
    for audio_path in tqdm(files, desc="Indexing tracks", ncols=100):
        # Skip if already in DB -------------------------------------------------------
        if vs._conn.execute(
            "SELECT 1 FROM tracks WHERE path = ?", [str(audio_path)]
        ).fetchone():
            continue

        try:
            meta = fe.extract_id3(str(audio_path))
            feats = fe.extract_audio_features(str(audio_path))
            vector = embedder.embed(str(audio_path))
            vs.add_track(
                path=str(audio_path), metadata=meta, features=feats, embedding=vector
            )
        except Exception as exc:  # noqa: BLE001 – broad but logged
            errors += 1
            logger.exception("Error processing %s: %s", audio_path, exc)

    # Persist ------------------------------------------------------------------------
    vs.save()
    vs.close()

    logger.info(
        "Ingestion finished: %d files processed, %d errors • DB=%s • IDX=%s",
        len(files),
        errors,
        DB_PATH,
        INDEX_PATH,
    )


if __name__ == "__main__":
    main()
