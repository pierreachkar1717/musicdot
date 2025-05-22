import os
import shutil
import duckdb
import pathlib
import xml.etree.ElementTree as ET

# --- CONFIG ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "db", "library.duckdb")
EXPORT_DIR = os.path.join(BASE_DIR, "exports")

# Rekordbox XML constants
PRODUCT_NAME = "MusicDot"
PRODUCT_VERSION = "1.0"
COMPANY_NAME = "DIY"


def export_clusters(db_path: str = DB_PATH, export_dir: str = EXPORT_DIR) -> None:
    """
    Export each cluster as a Rekordbox XML playlist and copy audio files
    into corresponding cluster-named subfolders.
    """
    os.makedirs(export_dir, exist_ok=True)
    con = duckdb.connect(db_path)

    # Fetch all distinct cluster IDs
    clusters = con.execute("SELECT DISTINCT cluster_id FROM track_clusters").fetchall()
    clusters = [row[0] for row in clusters]
    print(f"Found {len(clusters)} clusters to export: {clusters}")

    for cid in clusters:
        # Query tracks in this cluster
        rows = con.execute(
            """
            SELECT t.track_id, t.path, a.bpm, a.key
            FROM tracks t
            JOIN audio_features a ON t.track_id = a.track_id
            JOIN track_clusters c ON t.track_id = c.track_id
            WHERE c.cluster_id = ?
            """,
            [cid],
        ).fetchall()
        if not rows:
            print(f"No tracks in cluster {cid}, skipping.")
            continue

        # Build Rekordbox XML
        root = ET.Element("DJ_PLAYLISTS", Version="1,0,0")
        ET.SubElement(
            root,
            "PRODUCT",
            Name=PRODUCT_NAME,
            Version=PRODUCT_VERSION,
            Company=COMPANY_NAME,
        )
        coll = ET.SubElement(root, "COLLECTION", Entries=str(len(rows)))

        # Add each track to the COLLECTION
        for track_id, path, bpm, key in rows:
            ET.SubElement(
                coll,
                "TRACK",
                TrackID=str(track_id),
                Name=pathlib.Path(path).stem,
                Location="file://localhost/" + path,
                AverageBpm=str(round(bpm, 2)),
                Tonality=key or "",
            )

        # Create PLAYLISTS node
        pls_root = ET.SubElement(root, "PLAYLISTS")
        playlist_name = f"Cluster_{cid}"
        node = ET.SubElement(
            pls_root,
            "NODE",
            Type="1",
            Name=playlist_name,
            Entries=str(len(rows)),
            KeyType="0",
        )
        for track_id, *_ in rows:
            ET.SubElement(node, "TRACK", Key=str(track_id))

        # Write the XML file
        xml_path = os.path.join(export_dir, f"{playlist_name}.xml")
        tree = ET.ElementTree(root)
        tree.write(xml_path, encoding="UTF-8", xml_declaration=True)
        print(f"Exported cluster {cid} playlist to {xml_path}")

        # Copy audio files into a cluster folder
        folder = os.path.join(export_dir, playlist_name)
        os.makedirs(folder, exist_ok=True)
        for _, path, *_ in rows:
            try:
                shutil.copy2(path, folder)
            except Exception as e:
                print(f"Failed to copy {path} to {folder}: {e}")
        print(f"Copied {len(rows)} files into folder {folder}")

    con.close()
    print(f"All clusters exported and organized in {export_dir}.")


if __name__ == "__main__":
    export_clusters()
