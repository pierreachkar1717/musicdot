"""
Flatten-copy with progress bar: grab every file under SOURCE_DIR (recursively)
and drop it into DEST_DIR. Any name clash gets a numeric suffix.
"""

from pathlib import Path
import os
import shutil
from tqdm import tqdm

# --- set your paths -------------------------------------------------
SOURCE_DIR = Path("/Volumes/WEAPONS/music_archive")      
DEST_DIR   = Path("/Volumes/WEAPONS/all_music") 
# --------------------------------------------------------------------

def unique_path(dst_dir: Path, name: str) -> Path:
    """Return a file path in dst_dir that doesn’t already exist,
       appending _1, _2 … if necessary."""
    candidate = dst_dir / name
    if not candidate.exists():
        return candidate
    stem, suffix = Path(name).stem, Path(name).suffix
    i = 1
    while True:
        candidate = dst_dir / f"{stem}_{i}{suffix}"
        if not candidate.exists():
            return candidate
        i += 1

def flat_copy_with_progress(src: Path, dst: Path) -> None:
    if not src.is_dir():
        raise NotADirectoryError(f"{src} is not a directory")
    dst.mkdir(parents=True, exist_ok=True)

    # gather all files
    all_files = []
    for root, _dirs, files in os.walk(src):
        for fname in files:
            all_files.append(Path(root) / fname)

    # copy with tqdm progress bar
    for src_file in tqdm(all_files, desc="Copying files", unit="file"):
        dst_file = unique_path(dst, src_file.name)
        shutil.copy2(src_file, dst_file, follow_symlinks=True)

if __name__ == "__main__":
    flat_copy_with_progress(SOURCE_DIR.resolve(), DEST_DIR.resolve())
    print("Copy complete.")