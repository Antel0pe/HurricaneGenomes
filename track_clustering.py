from typing import Any, List, Tuple
import time
import numpy as np
import matplotlib.pyplot as plt  # type: ignore
import matplotlib.cm as cm  # type: ignore
import cartopy.crs as ccrs  # type: ignore
import cartopy.feature as cfeature  # type: ignore
from scipy.cluster.hierarchy import dendrogram  # type: ignore

from hurdat2_processing import load_storm_tracks
from track_comparison import pairwise_compare_tracks_greatcircle


def average_linkage_hcl(ids: List[str], dist: np.ndarray) -> Any:
    if not ids:
        return []
    D = np.asarray(dist, dtype=float)
    n = len(ids)
    if D.shape != (n, n):
        raise ValueError("dist shape mismatch")
    np.fill_diagonal(D, np.inf)
    clusters: List[Any] = list(ids)
    sizes = np.ones(n, dtype=int)
    while len(clusters) > 1:
        m = D.shape[0]
        # get idx of min dist
        i, j = divmod(np.argmin(D), m)
        if i == j:
            break
        if i > j:
            i, j = j, i
        keep = [k for k in range(m) if k not in (i, j)]
        if keep:
            # compute size weighted average of newly merged cluster vs each other cluster
            # averaging rmse of merged tracks/clusters
            new_row = (sizes[i] * D[i, keep] + sizes[j] * D[j, keep]) / (sizes[i] + sizes[j])
            A = D[np.ix_(keep, keep)]
            D_new = np.zeros((m - 1, m - 1), dtype=float)
            if m - 1 > 1:
                D_new[:-1, :-1] = A
                D_new[-1, :-1] = new_row
                D_new[:-1, -1] = new_row
            D = D_new
        else:
            D = np.zeros((1, 1), dtype=float)
        np.fill_diagonal(D, np.inf)
        new_size = sizes[i] + sizes[j]
        clusters = [clusters[k] for k in keep] + [[clusters[i], clusters[j]]]
        sizes = np.concatenate([sizes[keep], np.array([new_size], dtype=int)])
    return clusters[0] if clusters else []


def average_linkage_hcl_linkage(ids: List[str], dist: np.ndarray) -> Tuple[Any, np.ndarray]:
    if not ids:
        return [], np.zeros((0, 4), dtype=float)
    D = np.asarray(dist, dtype=float)
    n = len(ids)
    if D.shape != (n, n):
        raise ValueError("dist shape mismatch")
    np.fill_diagonal(D, np.inf)
    clusters: List[Any] = list(ids)
    sizes = np.ones(n, dtype=int)
    id_map = list(range(n))
    next_id = n
    merges: List[List[float]] = []
    while len(clusters) > 1:
        m = D.shape[0]
        i, j = divmod(np.argmin(D), m)
        if i == j:
            break
        if i > j:
            i, j = j, i
        keep = [k for k in range(m) if k not in (i, j)]
        dij = float(D[i, j])
        if keep:
            new_row = (sizes[i] * D[i, keep] + sizes[j] * D[j, keep]) / (sizes[i] + sizes[j])
            A = D[np.ix_(keep, keep)]
            D_new = np.zeros((m - 1, m - 1), dtype=float)
            if m - 1 > 1:
                D_new[:-1, :-1] = A
                D_new[-1, :-1] = new_row
                D_new[:-1, -1] = new_row
            D = D_new
        else:
            D = np.zeros((1, 1), dtype=float)
        np.fill_diagonal(D, np.inf)
        new_size = sizes[i] + sizes[j]
        merges.append([float(id_map[i]), float(id_map[j]), dij, float(new_size)])
        clusters = [clusters[k] for k in keep] + [[clusters[i], clusters[j]]]
        sizes = np.concatenate([sizes[keep], np.array([new_size], dtype=int)])
        id_map = [id_map[k] for k in keep] + [next_id]
        next_id += 1
    Z = np.asarray(merges, dtype=float) if merges else np.zeros((0, 4), dtype=float)
    return (clusters[0] if clusters else []), Z


def _is_leaf(node: Any) -> bool:
    return isinstance(node, str)


def _collect_leaves(node: Any) -> List[str]:
    if _is_leaf(node):
        return [node]
    a, b = node
    return _collect_leaves(a) + _collect_leaves(b)


def _clusters_at_depth(node: Any, level: int, max_level: int) -> List[Any]:
    if level >= max_level or _is_leaf(node):
        return [node]
    a, b = node
    return _clusters_at_depth(a, level + 1, max_level) + _clusters_at_depth(b, level + 1, max_level)


def _collect_leaves_excluding(node: Any, exclude: set) -> List[str]:
    leaves = _collect_leaves(node)
    return [s for s in leaves if s not in exclude]


def _pick_medoid(leaves: List[str], id_to_idx: dict, D: np.ndarray) -> str:
    if len(leaves) == 1:
        return leaves[0]
    idxs = [id_to_idx[s] for s in leaves]
    sub = D[np.ix_(idxs, idxs)].astype(float, copy=True)
    # Diagonal is zero distances; row sums give sum of distances to others
    sums = sub.sum(axis=1)
    best_local = int(np.argmin(sums))
    return leaves[best_local]


def build_parent_child_tree(cluster_tree: Any, ids: List[str], dist: np.ndarray) -> dict:
    id_to_idx = {sid: i for i, sid in enumerate(ids)}

    def assign(node: Any, exclude: set) -> dict:
        leaves = _collect_leaves_excluding(node, exclude)
        if not leaves:
            return {}
        if len(leaves) <= 2:
            return {sid: {} for sid in leaves}
        parent = _pick_medoid(leaves, id_to_idx, dist)
        child_dict: dict = {}
        if _is_leaf(node):
            # Should not happen given len(leaves)>2, but guard anyway
            return {parent: {}}
        left, right = node
        # Exclude parent as we recurse into sub-clusters
        excl_next = set(exclude)
        excl_next.add(parent)
        left_children = assign(left, excl_next)
        right_children = assign(right, excl_next)
        # Merge child dicts
        child_dict.update(left_children)
        child_dict.update(right_children)
        return {parent: child_dict}

    return assign(cluster_tree, set())


def plot_top_levels_map(tracks: dict, cluster_tree: Any, max_levels: int = 4) -> None:
    groups = _clusters_at_depth(cluster_tree, 0, max_levels)
    leaves_per_group = [
        _collect_leaves(g) if not _is_leaf(g) else [g] for g in groups
    ]
    color_for_id: dict = {}
    n = max(1, len(leaves_per_group))
    cmap = cm.get_cmap("tab20", max(2, n))
    for idx, leaves in enumerate(leaves_per_group):
        for sid in leaves:
            color_for_id[sid] = cmap(idx / max(1, n - 1))

    plt.style.use("dark_background")
    fig = plt.figure(figsize=(10, 5))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor="#101010")
    ax.add_feature(cfeature.OCEAN, facecolor="#050505")
    ax.coastlines(color="white", linewidth=0.5)
    ax.gridlines(draw_labels=False, color="gray", alpha=0.2, linestyle="--", linewidth=0.5)

    all_pts = [pt for tr in tracks.values() for pt in tr]
    if all_pts:
        lats = [pt[0] for pt in all_pts]
        lons = [pt[1] for pt in all_pts]
        min_lat, max_lat = min(lats), max(lats)
        min_lon, max_lon = min(lons), max(lons)
        lat_pad = max(2.0, (max_lat - min_lat) * 0.05)
        lon_pad = max(2.0, (max_lon - min_lon) * 0.05)
        ax.set_extent([min_lon - lon_pad, max_lon + lon_pad, min_lat - lat_pad, max_lat + lat_pad], crs=ccrs.PlateCarree())

    for sid, tr in tracks.items():
        if not tr:
            continue
        col = color_for_id.get(sid, (1.0, 1.0, 1.0, 0.5))
        lats = [p[0] for p in tr]
        lons = [p[1] for p in tr]
        ax.plot(lons, lats, color=col, linewidth=1.0, alpha=0.9, transform=ccrs.PlateCarree())

    plt.title(f"Top {max_levels} levels (average-linkage)")
    plt.tight_layout()
    plt.show()


def _nodes_at_levels_from_parent_child(parent_child: dict, max_levels: int) -> List[List[str]]:
    levels: List[List[str]] = []
    if not parent_child:
        return levels
    current: List[Tuple[str, dict]] = [(rid, parent_child[rid]) for rid in parent_child.keys()]
    depth = 0
    while current and depth < max_levels:
        levels.append([nid for nid, _ in current])
        next_level: List[Tuple[str, dict]] = []
        for _, child_dict in current:
            for cid, cchildren in child_dict.items():
                next_level.append((cid, cchildren))
        current = next_level
        depth += 1
    return levels


def plot_top_levels_map_from_parent_tree(tracks: dict, parent_child: dict, max_levels: int = 4) -> None:
    levels = _nodes_at_levels_from_parent_child(parent_child, max_levels)
    ids_to_plot = {nid for lvl in levels for nid in lvl}
    if not ids_to_plot:
        return
    # Build color map per depth level
    id_to_color: dict = {}
    for d, lvl in enumerate(levels):
        if not lvl:
            continue
        if d == 0:
            for nid in lvl:
                id_to_color[nid] = (1.0, 0.2, 0.2, 1.0)  # red-ish for root level
            continue
        n = max(2, len(lvl))
        for j, nid in enumerate(lvl):
            id_to_color[nid] = cm.get_cmap("tab20", n)(j / max(1, n - 1))

    plt.style.use("dark_background")
    fig = plt.figure(figsize=(10, 5))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor="#101010")
    ax.add_feature(cfeature.OCEAN, facecolor="#050505")
    ax.coastlines(color="white", linewidth=0.5)
    ax.gridlines(draw_labels=False, color="gray", alpha=0.2, linestyle="--", linewidth=0.5)

    subset = {sid: tracks.get(sid, []) for sid in ids_to_plot}
    all_pts = [pt for tr in subset.values() for pt in tr]
    if all_pts:
        lats = [pt[0] for pt in all_pts]
        lons = [pt[1] for pt in all_pts]
        min_lat, max_lat = min(lats), max(lats)
        min_lon, max_lon = min(lons), max(lons)
        lat_pad = max(2.0, (max_lat - min_lat) * 0.05)
        lon_pad = max(2.0, (max_lon - min_lon) * 0.05)
        ax.set_extent([min_lon - lon_pad, max_lon + lon_pad, min_lat - lat_pad, max_lat + lat_pad], crs=ccrs.PlateCarree())

    for sid, tr in subset.items():
        if not tr:
            continue
        col = id_to_color.get(sid, (1.0, 1.0, 1.0, 0.8))
        lw = 2.0 if sid in (levels[0][0] if levels and levels[0] else []) else 1.0
        lats = [p[0] for p in tr]
        lons = [p[1] for p in tr]
        ax.plot(lons, lats, color=col, linewidth=lw, alpha=0.95, transform=ccrs.PlateCarree())

    plt.title(f"Parent-child top {max_levels} levels")
    plt.tight_layout()
    plt.show()

def main() -> None:
    tracks = load_storm_tracks()
    if not tracks:
        print("No tracks")
        return
    t0 = time.perf_counter()
    mat, ids = pairwise_compare_tracks_greatcircle(tracks)
    t1 = time.perf_counter()
    cluster, Z = average_linkage_hcl_linkage(ids, mat)
    t2 = time.perf_counter()
    print(f"tracks={len(ids)} pairwise_time={t1 - t0:.4f}s cluster_time={t2 - t1:.4f}s")
    # print(cluster)
    if Z.shape[0] > 0:
        plt.style.use("dark_background")
        plt.figure(figsize=(10, 5))
        dendrogram(Z, labels=ids, distance_sort="ascending")
        plt.title("Hierarchical Clustering Dendrogram")
        plt.tight_layout()
        plt.show(block=False)


    # Build parent-child assignment tree (medoid selection at each cluster)
    parent_child = build_parent_child_tree(cluster, ids, mat)
    # Optional: print root
    root = next(iter(parent_child.keys())) if parent_child else None
    if root is not None:
        print(f"root={root}")
        # print(parent_child)

    # Map from parent-child tree: show only first N levels from root
    plot_top_levels_map_from_parent_tree(tracks, parent_child, max_levels=6)

if __name__ == "__main__":
    main()


