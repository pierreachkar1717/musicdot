#!/usr/bin/env python3
import os
from tqdm import tqdm
from feature_extractor import FeatureExtractor
from embedder import EffNetDiscogsEmbedder
from vector_store import VectorStore

# Paths
data_dir = "/Users/pierreachkar/Documents/projects/musicdot/data"
model_path = (
    "/Users/pierreachkar/Documents/projects/musicdot/models/discogs-effnet-bs64-1.pb"
)
db_path = "/Users/pierreachkar/Documents/projects/musicdot/db/library.duckdb"
index_path = "/Users/pierreachkar/Documents/projects/musicdot/db/vector.index"

# Supported audio extensions
AUDIO_EXTS = {".mp3", ".flac", ".wav", ".aiff"}


def main():
    # Initialize components
    fe = FeatureExtractor()
    embedder = EffNetDiscogsEmbedder(model_path=model_path)
    # Initialize vector store
    # compute a dummy embedding_dim from a short silent array to inspect model dimension
    sample_vec = embedder.embed(os.path.join(data_dir, os.listdir(data_dir)[0]))
    vs = VectorStore(
        db_path=db_path, index_path=index_path, embedding_dim=sample_vec.shape[0]
    )
    vs.init_store()

    # Gather audio files
    files = [
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if os.path.splitext(f)[1].lower() in AUDIO_EXTS
    ]

    # Process each file with a progress bar
    for audio_path in tqdm(files, desc="Indexing tracks"):
        # skip if already indexed
        exists = vs._conn.execute(
            "SELECT 1 FROM tracks WHERE path = ?", [audio_path]
        ).fetchone()
        if exists:
            continue

        try:
            meta = fe.extract_id3(audio_path)
            feats = fe.extract_audio_features(audio_path)
            vector = embedder.embed(audio_path)
            vs.add_track(
                path=audio_path, metadata=meta, features=feats, embedding=vector
            )
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")

    # Persist changes
    vs.save()
    vs.close()
    print(f"Indexed {len(files)} tracks into {db_path} and {index_path}")


if __name__ == "__main__":
    main()
