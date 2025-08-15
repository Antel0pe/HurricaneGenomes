import math
import random
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from track_comparison import (
    load_tracks,
    preprocess_tracks_to_array,
    vector_compare_single_to_many_greatcircle,
)


LatLon = List[float]
Track = List[LatLon]


def weighting(pct: float) -> float:
    x = max(0.0, min(1.0, pct / 100.0))
    return -0.5 * (x ** 0.5) + 0.5


def build_tree(
    all_tracks: Dict[str, Track],
    root_id: Optional[str],
    threshold: float,
    weight_fn: Optional[Callable[[float], float]] = None,
) -> Tuple[str, List[Tuple[str, str]], Dict[str, int]]:
    # Vectorized build using preprocessed arrays and great-circle distance
    if not all_tracks:
        return "", [], {}

    arr, ids = preprocess_tracks_to_array(all_tracks)  # (N,20,2) and order of ids
    if arr.shape[0] == 0:
        return "", [], {}
    ids_arr = np.array(ids)
    id_to_idx = {sid: i for i, sid in enumerate(ids)}

    if root_id is None:
        root_id = random.choice(ids)
    if root_id not in id_to_idx:
        raise ValueError("root_id not found")
    root_idx = id_to_idx[root_id]

    remaining_mask = np.ones(arr.shape[0], dtype=bool)
    remaining_mask[root_idx] = False  # exclude root from remaining

    queue: List[str] = [root_id]
    edges: List[Tuple[str, str]] = []
    depth: Dict[str, int] = {root_id: 0}
    last_printed_remaining: Optional[int] = None

    while queue and remaining_mask.any():
        remaining_count = int(remaining_mask.sum())
        if remaining_count % 100 == 0 and remaining_count != last_printed_remaining:
            print(f"queue={len(queue)} remaining={remaining_count}")
            last_printed_remaining = remaining_count

        parent_id = queue.pop(0)
        parent_idx = id_to_idx[parent_id]

        subset_indices = np.flatnonzero(remaining_mask)
        if subset_indices.size == 0:
            break
        dists = vector_compare_single_to_many_greatcircle(arr[parent_idx], arr[subset_indices])
        child_mask_subset = np.isfinite(dists) & (dists <= threshold)
        if not np.any(child_mask_subset):
            continue
        children_indices = subset_indices[child_mask_subset]
        children_ids = ids_arr[children_indices].tolist()

        for cid in children_ids:
            edges.append((parent_id, cid))
            depth[cid] = depth[parent_id] + 1
        remaining_mask[children_indices] = False
        queue.extend(children_ids)

    return root_id, edges, depth


def plot_tree(root_id: str, edges: List[Tuple[str, str]], depth: Dict[str, int], tracks: Dict[str, Track]) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return
    plt.style.use("dark_background")

    levels: Dict[int, List[str]] = {}
    nodes = {root_id}
    for a, b in edges:
        nodes.add(a); nodes.add(b)
    for n in nodes:
        d = depth.get(n, 0)
        levels.setdefault(d, []).append(n)

    pos: Dict[str, Tuple[float, float]] = {}
    min_y = 0
    for d in sorted(levels.keys()):
        ids = levels[d]
        for i, nid in enumerate(ids):
            x = (i + 1) / (len(ids) + 1)
            y = -float(d)
            pos[nid] = (x, y)
            min_y = min(min_y, y)

    fig = plt.figure(figsize=(9, 6))
    ax = plt.gca()
    ax.set_facecolor("#000000")
    for a, b in edges:
        xa, ya = pos[a]; xb, yb = pos[b]
        ax.plot([xa, xb], [ya, yb], color="yellow", alpha=0.5, linewidth=1.0)
    for n, (x, y) in pos.items():
        color = "red" if n == root_id else "yellow"
        size = 40 if n == root_id else 18
        ax.scatter([x], [y], color=color, s=size)
        ax.text(x, y - 0.05, n, color="white", fontsize=6, ha="center", va="top")
    ax.set_ylim(min_y - 0.5, 0.5)
    ax.set_xlim(-0.05, 1.05)
    ax.set_xticks([]); ax.set_yticks([])
    plt.title(f"Storm similarity tree (root: {root_id})")
    plt.tight_layout()
    try:
        plt.show(block=False)
        plt.pause(0.1)
    except Exception:
        plt.show()

    # Map view colored by level (only show first 3 levels: 0,1,2)
    try:
        import cartopy.crs as ccrs  # type: ignore
        import cartopy.feature as cfeature  # type: ignore
        import matplotlib.pyplot as plt  # type: ignore
        import matplotlib.cm as cm  # type: ignore
        from matplotlib.lines import Line2D  # type: ignore
    except Exception:
        return

    fig2 = plt.figure(figsize=(10, 5))
    ax2 = plt.axes(projection=ccrs.PlateCarree())
    try:
        ax2.add_feature(cfeature.LAND, facecolor="#101010")
        ax2.add_feature(cfeature.OCEAN, facecolor="#050505")
        ax2.coastlines(color="white", linewidth=0.5)
        ax2.gridlines(draw_labels=False, color="gray", alpha=0.2, linestyle="--", linewidth=0.5)
    except Exception:
        ax2.set_facecolor("#000000")

    # Limit to first 3 levels
    max_depth_overall = max(depth.values()) if depth else 0
    max_depth_shown = min(2, max_depth_overall)
    n_levels_shown = max(1, max_depth_shown + 1)
    cmap = cm.get_cmap("tab20", max(2, n_levels_shown))

    # Only include tracks at levels <= 2 for plotting
    allowed_ids = {sid for sid, d in depth.items() if d <= max_depth_shown}
    subset_tracks = {sid: tr for sid, tr in tracks.items() if sid in allowed_ids}

    all_pts = [pt for tr in subset_tracks.values() for pt in tr]
    if all_pts:
        lats = [pt[0] for pt in all_pts]
        lons = [pt[1] for pt in all_pts]
        min_lat, max_lat = min(lats), max(lats)
        min_lon, max_lon = min(lons), max(lons)
        lat_pad = max(2.0, (max_lat - min_lat) * 0.05)
        lon_pad = max(2.0, (max_lon - min_lon) * 0.05)
        ax2.set_extent([min_lon - lon_pad, max_lon + lon_pad, min_lat - lat_pad, max_lat + lat_pad], crs=ccrs.PlateCarree())

    for sid, tr in subset_tracks.items():
        if not tr:
            continue
        lats = [p[0] for p in tr]
        lons = [p[1] for p in tr]
        d = depth.get(sid, 0)
        color = cmap(d / max(1, n_levels_shown - 1))
        lw = 1.8 if sid == root_id else 1.0
        alpha = 0.95 if sid == root_id else 0.8
        ax2.plot(lons, lats, color=color, linewidth=lw, alpha=alpha, transform=ccrs.PlateCarree())

    levels_present = [d for d in sorted(set(depth.values())) if d <= max_depth_shown] if depth else [0]
    handles = [Line2D([0], [0], color=cmap(d / max(1, n_levels_shown - 1)), lw=2, label=f"level {d}") for d in levels_present]
    ax2.legend(handles=handles, title="Tree level", loc="lower left")

    plt.title("Tracks colored by tree level (<= 2)")
    plt.tight_layout()
    try:
        plt.show()
    except Exception:
        pass


def main(threshold: float = 20_000.0, root_id: Optional[str] = None) -> None:
    tracks = load_tracks()
    if not tracks:
        print("No tracks")
        return
    rid, edges, depth = build_tree(tracks, root_id, threshold, weighting)
    print(f"Root: {rid}, edges: {len(edges)}, nodes: {len(depth)}")
    plot_tree(rid, edges, depth, tracks)


if __name__ == "__main__":
    try:
        import argparse
        p = argparse.ArgumentParser()
        p.add_argument("--threshold", type=float, default=20_000.0)
        p.add_argument("--root", type=str, default=None)
        args = p.parse_args()
        main(args.threshold, args.root)
    except Exception:
        main()

