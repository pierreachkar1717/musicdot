from __future__ import annotations

"""export_clusters.py – Export HDBSCAN clusters to Rekordbox XML and copy
files into cluster‑named sub‑folders.

Features
~~~~~~~~
* Uses **friendly_id** from `cluster_meta` (e.g. ``C01``) for playlist/folder
  names instead of raw numeric IDs.
* Preserves the polar‑angle ordering computed in `cluster_meta` so that clusters
  appear sequentially in Rekordbox under a parent folder "All Clusters".
* Writes a sidecar JSON (`cluster_nav.json`) mapping each friendly_id
  → list of its three nearest neighbour clusters (for external UIs).
* Copies audio files into `exports/{friendly_id}/`.

Re‑run this script whenever you re‑cluster.
"""

from pathlib import Path
import json
import os
import shutil
import urllib.parse as ulp
import xml.etree.ElementTree as ET

import duckdb

# ── CONFIG -----------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
DB_PATH   = BASE_DIR / "db" / "library.duckdb"
EXPORT_DIR = BASE_DIR / "exports"

# Rekordbox XML constants
PRODUCT_NAME    = "MusicDot"
PRODUCT_VERSION = "1.0"
COMPANY_NAME    = "DIY"


# --------------------------------------------------------------------

def _file_uri(path: Path) -> str:
    """Return `file://localhost/<url‑quoted path>` URI for Rekordbox."""
    return "file://localhost/" + ulp.quote(str(path))


def export_clusters(db_path: Path = DB_PATH, export_dir: Path = EXPORT_DIR) -> None:
    """Export every cluster to XML playlist + copy audio into folders."""
    export_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))

    # Fetch cluster order & meta
    meta_rows = con.execute(
        """
        SELECT cluster_id, friendly_id, size,
               centroid_x, centroid_y, neighbours
        FROM cluster_meta
        ORDER BY friendly_id
        """
    ).fetchall()
    if not meta_rows:
        print("No cluster_meta entries found. Did you run clustering.py?")
        return

    # Neighbour JSON map
    nav_map: dict[str, list[str]] = {}

    for cid, fid, size, cx, cy, neigh_json in meta_rows:
        neighbour_ids = json.loads(neigh_json)
        nav_map[fid] = [
            con.execute("SELECT friendly_id FROM cluster_meta WHERE cluster_id=?",
                          (nid,)).fetchone()[0]
            for nid in neighbour_ids
        ]

    (export_dir / "cluster_nav.json").write_text(
        json.dumps(nav_map, indent=2), encoding="utf-8"
    )

    # Build a *single* XML that nests all cluster nodes, then also write per-cluster XMLs
    root   = ET.Element("DJ_PLAYLISTS", Version="1,0,0")
    ET.SubElement(root, "PRODUCT", Name=PRODUCT_NAME, Version=PRODUCT_VERSION, Company=COMPANY_NAME)
    playlists_node = ET.SubElement(root, "PLAYLISTS")
    parent_node = ET.SubElement(playlists_node, "NODE", Type="0", Name="All Clusters")

    for cid, fid, size, *_ in meta_rows:
        # -------- fetch tracks for this cluster ----------
        rows = con.execute(
            """
            SELECT t.track_id, t.path, a.bpm, a.key
            FROM tracks t
            JOIN audio_features a  ON t.track_id = a.track_id
            JOIN track_clusters c  ON t.track_id = c.track_id
            WHERE c.cluster_id = ?
            """,
            (cid,),
        ).fetchall()
        if not rows:
            continue

        # ------------ XML for this cluster --------------
        coll = ET.SubElement(root, "COLLECTION", Entries=str(len(rows)))
        for tid, path, bpm, key in rows:
            ET.SubElement(
                coll,
                "TRACK",
                TrackID=str(tid),
                Name=Path(path).stem,
                Location=_file_uri(Path(path)),
                AverageBpm=str(round(bpm or 0, 2)),
                Tonality=key or "",
            )

        node = ET.SubElement(
            parent_node,
            "NODE",
            Type="1",           # playlist
            Name=fid,
            Entries=str(len(rows)),
        )
        for tid, *_ in rows:
            ET.SubElement(node, "TRACK", Key=str(tid))

        # write per‑cluster standalone XML
        tree_single = ET.ElementTree(root)
        tree_single.write(export_dir / f"{fid}.xml", encoding="UTF-8", xml_declaration=True)

        # ------------ copy audio -------------------------
        folder = export_dir / fid
        folder.mkdir(exist_ok=True)
        for _tid, path, *_ in rows:
            try:
                shutil.copy2(path, folder)
            except Exception as e:  # noqa: BLE001
                print(f"Failed to copy {path}: {e}")

        print(f"Exported {fid} ({len(rows)} tracks) → {folder}")

    # Write master XML (root contains nested structure)
    ET.ElementTree(root).write(export_dir / "all_clusters.xml", encoding="UTF-8", xml_declaration=True)
    print(f"Navigation map + XML exports written to {export_dir}.")


# --------------------------------------------------------------------
if __name__ == "__main__":
    export_clusters()
