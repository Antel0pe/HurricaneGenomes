from typing import Any, List, Tuple
import time
import numpy as np

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
        try:
            import matplotlib.pyplot as plt  # type: ignore
            from scipy.cluster.hierarchy import dendrogram  # type: ignore
        except Exception:
            return
        plt.style.use("dark_background")
        plt.figure(figsize=(10, 5))
        dendrogram(Z, labels=ids, distance_sort="ascending")
        plt.title("Hierarchical Clustering Dendrogram")
        plt.tight_layout()
        try:
            plt.show()
        except Exception:
            pass


if __name__ == "__main__":
    main()


