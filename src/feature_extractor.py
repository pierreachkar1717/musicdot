import os
from mutagen import File as MutagenFile
from essentia.standard import MonoLoader, RhythmExtractor2013, KeyExtractor, ReplayGain


class FeatureExtractor:
    """
    Extract metadata and audio features from audio files.

    Usage:
        fe = FeatureExtractor(sample_rate=22050)
        meta = fe.extract_id3("/path/to/track.mp3")
        feats = fe.extract_audio_features("/path/to/track.mp3")
    """

    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate

    def extract_id3(self, audio_path: str) -> dict:
        """
        Extract ID3 metadata using mutagen.

        Returns:
            dict with keys: title, artist, album, year, duration, bitrate
        """
        if not os.path.isfile(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        audio = MutagenFile(audio_path)
        tags = audio.tags or {}
        info = audio.info

        def get_text(tag_key):
            frame = tags.get(tag_key)
            if frame and hasattr(frame, "text") and frame.text:
                return frame.text[0]
            return None

        title = get_text("TIT2")
        artist = get_text("TPE1")
        album = get_text("TALB")

        year_frame = tags.get("TDRC")
        year = None
        if year_frame:
            try:
                val = (
                    year_frame.text[0]
                    if hasattr(year_frame, "text") and year_frame.text
                    else str(year_frame)
                )
                year = int(val[:4])
            except Exception:
                year = None

        return {
            "title": title,
            "artist": artist,
            "album": album,
            "year": year,
            "duration": getattr(info, "length", None),
            "bitrate": getattr(info, "bitrate", None),
        }

    def extract_audio_features(self, audio_path: str) -> dict:
        """
        Extract audio features (bpm, key, loudness) using Essentia.

        Returns:
            dict with keys: bpm, key, loudness
        """
        if not os.path.isfile(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        loader = MonoLoader(filename=audio_path, sampleRate=self.sample_rate)
        audio = loader()

        rhythm_extractor = RhythmExtractor2013(method="multifeature")
        bpm, _, _, _, _ = rhythm_extractor(audio)

        key_extractor = KeyExtractor()
        key_results = key_extractor(audio)
        tonic = key_results[0]
        scale = key_results[1]
        key = f"{tonic}{scale}" if tonic and scale else tonic

        rg = ReplayGain()
        loudness = rg(audio)

        return {"bpm": float(bpm), "key": key, "loudness": float(loudness)}


# Example usage
if __name__ == "__main__":
    path = "/Users/pierreachkar/Documents/projects/musicdot/data/Marc Poppcke, Alex Niggemann - Berlin Down the House (Original Mix).mp3"
    fe = FeatureExtractor()
    meta = fe.extract_id3(path)
    feats = fe.extract_audio_features(path)
    print("Metadata:", meta)
    print("Audio features:", feats)
