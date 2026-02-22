# src/scraping_transfermarkt_values.py
"""
Downloads the transfermarkt-datasets ZIP and extracts:
  - tm_players.csv          (current market values, club, position per player)
  - tm_player_valuations.csv (historical market value timeline per player)

into `output_folder` (typically Data/Raw/).
"""
from __future__ import annotations

import gzip
import io
import zipfile
from pathlib import Path

import requests

ZIP_URL = (
    "https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data/transfermarkt-datasets.zip"
)

# Map from archive basename (plain or .gz) → desired output filename (plain CSV).
_WANTED_BASENAMES = {
    "players.csv": "tm_players.csv",
    "player_valuations.csv": "tm_player_valuations.csv",
}


def run_pipeline(output_folder: Path) -> None:
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    print(f"[TM] Downloading transfermarkt dataset from {ZIP_URL} …")
    resp = requests.get(ZIP_URL, timeout=180)
    resp.raise_for_status()
    print(f"[TM] Downloaded {len(resp.content) / 1_048_576:.1f} MB")

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        names = zf.namelist()

        for basename, out_name in _WANTED_BASENAMES.items():
            # Accept both plain CSV and gzip-compressed CSV in the archive
            candidates = [basename, basename + ".gz"]
            match = None
            for candidate in candidates:
                for n in names:
                    if n.split("/")[-1] == candidate:
                        match = n
                        break
                if match:
                    break

            if match is None:
                print(
                    f"[TM WARN] '{basename}' (or .gz) not found in ZIP. "
                    f"Available entries (first 20): {names[:20]}"
                )
                continue

            dest = output_folder / out_name
            with zf.open(match) as src:
                raw = src.read()

            # Decompress if the archive entry is gzip-compressed
            if match.endswith(".gz"):
                raw = gzip.decompress(raw)

            dest.write_bytes(raw)
            print(f"[TM] Saved → {dest}  ({len(raw) / 1_048_576:.1f} MB)")

    print("[TM] Done.")


if __name__ == "__main__":
    run_pipeline(Path("Data/Raw"))
