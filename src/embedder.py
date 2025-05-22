import os
import numpy as np
from essentia.standard import MonoLoader, TensorflowPredictEffnetDiscogs


class EffNetDiscogsEmbedder:
    """
    Class to compute track embeddings using the Discogs-EffNet model.

    Example:
        embedder = EffNetDiscogsEmbedder(
            model_path="/path/to/discogs-effnet-bs64-1.pb",
            sample_rate=22050,
            window_sec=30
        )
        vector = embedder.embed("/path/to/track.mp3")
    """

    def __init__(self, model_path: str, sample_rate: int = 22050, window_sec: int = 30):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        self.model_path = model_path
        self.sample_rate = sample_rate
        self.window_sec = window_sec
        # load the TensorFlow-based extractor
        self._model = TensorflowPredictEffnetDiscogs(
            graphFilename=self.model_path, output="PartitionedCall:1"
        )

    def embed(self, audio_path: str) -> np.ndarray:
        """
        Compute a single L2-normalized embedding for the given track.
        Splits the track into three equal parts and extracts the middle
        window_sec seconds from each, then averages.

        :param audio_path: Path to the audio file
        :return: 1D numpy array of length model_dimension
        """
        if not os.path.isfile(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # load full audio
        loader = MonoLoader(filename=audio_path, sampleRate=self.sample_rate)
        audio = loader()
        total_samples = audio.size
        part_samples = total_samples // 3
        window_samples = int(self.window_sec * self.sample_rate)

        vecs = []
        for i in range(3):
            part_start = i * part_samples
            part_end = part_start + part_samples
            center = (part_start + part_end) // 2

            start = max(0, center - window_samples // 2)
            end = start + window_samples
            if end > total_samples:
                start = max(0, total_samples - window_samples)
                end = total_samples

            chunk = audio[start:end]
            if chunk.size < window_samples * 0.5:
                continue

            emb = self._model(chunk).mean(axis=0)
            emb /= np.linalg.norm(emb)
            vecs.append(emb)

        if not vecs:
            raise ValueError(f"No valid audio chunks extracted from {audio_path}")

        # average and normalize final vector
        track_vec = np.mean(vecs, axis=0)
        track_vec /= np.linalg.norm(track_vec)
        return track_vec


# # Example usage
# if __name__ == "__main__":
#     embedder = EffNetDiscogsEmbedder(
#         model_path="/Users/pierreachkar/Documents/projects/musicdot/models/discogs-effnet-bs64-1.pb",
#         sample_rate=22050,
#         window_sec=30
#     )
#     vector = embedder.embed("/Users/pierreachkar/Documents/projects/musicdot/models/discogs-effnet-bs64-1.pb")
#     print(vector)
