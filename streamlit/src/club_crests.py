"""
Club crest helpers — slug, path lookup, bytes/b64 loader.
Crests are stored as PNGs in Data/Assets/club_crests/{slug}.png.
"""
import base64
import re
from pathlib import Path

CREST_DIR = Path(__file__).resolve().parent.parent / "Data" / "Assets" / "club_crests"


def _slug(name: str) -> str:
    """Convert a club name to a filesystem-safe slug."""
    return re.sub(r"[^a-z0-9_]", "", name.lower().replace(" ", "_").replace("-", "_").replace(".", ""))


def get_crest_path(club_name: str) -> Path | None:
    """Return the Path to the crest PNG if it exists, else None."""
    if not club_name:
        return None
    p = CREST_DIR / f"{_slug(club_name)}.png"
    return p if p.exists() else None


def get_crest_bytes(club_name: str) -> bytes | None:
    """Return raw PNG bytes for the club crest, or None if not found."""
    p = get_crest_path(club_name)
    return p.read_bytes() if p else None


def get_crest_b64(club_name: str) -> str | None:
    """Return a data-URI base64 string for embedding in HTML, or None."""
    b = get_crest_bytes(club_name)
    if b is None:
        return None
    return "data:image/png;base64," + base64.b64encode(b).decode()
