from pathlib import Path
from typing import Dict, List, Optional


def _parse_coord(coord: str) -> Optional[float]:
    coord = coord.strip()
    if not coord:
        return None
    hemi = coord[-1].upper()
    try:
        value = float(coord[:-1].strip())
    except ValueError:
        return None
    if hemi in ("S", "W"):
        value = -value
    return value


def load_storm_tracks(file_path: Optional[str] = None) -> Dict[str, List[List[float]]]:
    """Return {storm_id: [[lat, lon], ...]} from HURDAT2 file."""
    if file_path is None:
        file_path = str(Path(__file__).with_name("hurdat2.txt"))

    tracks: Dict[str, List[List[float]]] = {}
    current_id: Optional[str] = None

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(",")]
            first = parts[0] if parts else ""

            # Header lines contain letters in the storm id (e.g., AL192023) and a name as 2nd field
            if any(c.isalpha() for c in first) and len(parts) > 1 and not parts[1].isdigit():
                current_id = first
                tracks.setdefault(current_id, [])
                continue

            # Detail line
            if current_id is None or len(parts) < 6:
                continue
            lat = _parse_coord(parts[4])
            lon = _parse_coord(parts[5])
            if lat is None or lon is None:
                continue
            tracks[current_id].append([lat, lon])

    return tracks


# Loaded on import for convenience
STORM_TRACKS = load_storm_tracks()


