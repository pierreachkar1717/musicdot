from __future__ import annotations

"""
Compute fixed‑size audio embeddings for music tracks using Essentia’s
Discogs‑EffNet model.

Workflow
--------
1. Load the audio file as single‑channel (mono) at a fixed sample‑rate.
2. Crop one centred window of ``window_sec`` seconds (default 128 s).
3. Infer the embedding via the frozen TensorFlow graph.
4. Normalise (L2) the resulting vector and return it.

"""

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Final

import numpy as np
from essentia.standard import MonoLoader, TensorflowPredictEffnetDiscogs

# ---------------------------------------------------------------------------
# Logging setup – write to logs/embedder.log and echo to console
# ---------------------------------------------------------------------------
LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# file handler dedicated to this module
_fh = logging.handlers.RotatingFileHandler(
    LOG_DIR / "embedder.log", maxBytes=5_000_000, backupCount=3, encoding="utf-8"
)
_fmt = logging.Formatter(
    "%(asctime)s  %(levelname)8s  [%(name)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
_fh.setFormatter(_fmt)
_fh.setLevel(logging.INFO)

logger.addHandler(_fh)
logger.propagate = False


class EffNetDiscogsEmbedder:
    """Discogs‑EffNet audio embedder.

    Parameters
    ----------
    model_path
        Path to the frozen ``discogs‑effnet-bs64-1.pb`` TensorFlow graph.
    sample_rate
        Target sample‑rate (Hz). 22 050 Hz matches the model’s training setup.
    window_sec
        Length of the analysis window in *seconds*. If *None*, the class‑level
        :pyattr:`DEFAULT_WINDOW_SEC` (128) is used.
    pad_short
        If *True*, tracks shorter than ``window_sec`` are zero‑padded in the
        centre; otherwise a :class:`ValueError` is raised.

    Notes
    -----
    * The first call populates :pyattr:`output_dim` via a dummy forward‑pass,
      so you can query it without loading real audio.
    * Returned vectors are **unit‑norm** so that cosine similarity equals dot
      product, making integration with FAISS `IndexFlatIP` trivial.
    """

    #: Default window length (seconds) when the caller does not specify one.
    DEFAULT_WINDOW_SEC: Final[int] = 128

    def __init__(
        self,
        model_path: str | os.PathLike[str],
        *,
        sample_rate: int = 22_050,
        window_sec: int | None = None,
        pad_short: bool = False,
    ) -> None:
        # ---- Resolve & validate model path ---------------------------------
        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        self.model_path: str = str(model_path)

        # ---- Store configuration -------------------------------------------
        self.sample_rate: int = sample_rate
        self.window_sec: int = window_sec or self.DEFAULT_WINDOW_SEC
        self.pad_short: bool = pad_short

        # ---- Initialise Essentia's TF wrapper ------------------------------
        # NOTE: The output tensor name comes from the original graph definition
        # where node 1 is the embedding before the classification head.
        self._model = TensorflowPredictEffnetDiscogs(
            graphFilename=self.model_path,
            output="PartitionedCall:1",
        )

        # Embedding dimensionality will be discovered lazily at runtime.
        self._output_dim: int | None = None

        logger.info(
            "Embedder ready (sr=%d Hz, window=%ds, pad_short=%s)",
            self.sample_rate,
            self.window_sec,
            self.pad_short,
        )

    # --------------------------------------------------------------------- #
    # Public helpers
    # --------------------------------------------------------------------- #
    @property
    def output_dim(self) -> int:
        """Return the embedding length (runs a dummy forward‑pass if needed)."""
        if self._output_dim is None:
            logger.debug("Discovering output_dim via dummy inference …")
            dummy = np.zeros(self.window_sec * self.sample_rate, dtype=np.float32)
            self._output_dim = int(self._model(dummy).shape[-1])
        return self._output_dim

    # --------------------------------------------------------------------- #
    # Core method
    # --------------------------------------------------------------------- #
    def embed(
        self, audio_path: str | os.PathLike[str]
    ) -> np.ndarray:  # noqa: D401 – short imperative OK
        """Embed a single audio file.

        Parameters
        ----------
        audio_path
            Audio file to process.

        Returns
        -------
        np.ndarray
            1‑D *unit‑norm* embedding vector.
        """
        audio_path = Path(audio_path)
        if not audio_path.is_file():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # ---- Load audio ----------------------------------------------------
        # MonoLoader automatically converts stereo → mono via channel averaging.
        loader = MonoLoader(filename=str(audio_path), sampleRate=self.sample_rate)
        audio = loader()
        total_samples = audio.size
        logger.debug("Loaded %s (%d samples)", audio_path.name, total_samples)

        # ---- Compute start/end of the centred analysis window -------------
        window_samples = int(self.window_sec * self.sample_rate)

        if total_samples < window_samples:
            if not self.pad_short:
                raise ValueError(
                    f"Track < {self.window_sec}s and pad_short=False: {audio_path}"
                )
            # Zero‑pad equally left and right so the content stays centred.
            pad_width = window_samples - total_samples
            left, right = pad_width // 2, pad_width - pad_width // 2
            audio = np.pad(audio, (left, right))
            total_samples = audio.size
            logger.info(
                "Padded short track (%s) to %d samples (left=%d, right=%d)",
                audio_path.name,
                total_samples,
                left,
                right,
            )

        center = total_samples // 2
        start = center - window_samples // 2
        end = start + window_samples
        chunk = audio[start:end]
        logger.debug("Extracted window [%d:%d] from %s", start, end, audio_path.name)

        # ---- Model inference ----------------------------------------------
        try:
            emb = self._model(chunk).mean(axis=0)
        except Exception:  # pragma: no cover – let caller handle/see traceback
            logger.exception("TensorFlow inference failed on %s", audio_path)
            raise

        # ---- Normalise to unit length -------------------------------------
        emb /= np.linalg.norm(emb)

        # Cache dimensionality if not already known
        if self._output_dim is None:
            self._output_dim = emb.shape[0]
            logger.debug("Cached output_dim=%d", self._output_dim)

        logger.info("Embedded %s → dim=%d", audio_path.name, emb.shape[0])
        return emb

    # --------------------------------------------------------------------- #
    # Dunder helpers
    # --------------------------------------------------------------------- #
    def __repr__(self) -> str:  # pragma: no cover – debug convenience
        return (
            f"{self.__class__.__name__}(model='{self.model_path}', "
            f"sr={self.sample_rate}, window={self.window_sec}s, pad_short={self.pad_short})"
        )


# ---------------------------------------------------------------------------
# Smoke‑test
# ---------------------------------------------------------------------------
# if __name__ == "__main__":
#     model_path = (
#         "/Users/pierreachkar/Documents/projects/musicdot/models/discogs-effnet-bs64-1.pb"
#     )
#     track = (
#         "/Users/pierreachkar/Documents/projects/musicdot/data/01-L.U.P.O._-_Hell_Or_Heaven_(extended_mix)-320kb_s_MP3.mp3"
#     )
#     embedder = EffNetDiscogsEmbedder(model_path=model_path)
#     vec = embedder.embed(track)

#     # Show a shortened preview so the terminal does not explode
#     preview = np.array2string(vec[:10], precision=4, separator=", ")
#     print(f"Embedding shape: {vec.shape}  preview: {preview} …")
