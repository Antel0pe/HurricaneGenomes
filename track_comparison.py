from typing import Callable, Dict, List, Optional
import math
import random
import time
import numpy as np

from hurdat2_processing import load_storm_tracks


LatLon = List[float]
Track = List[LatLon]


def _point_step(track: Track) -> float:
    return 100.0 / len(track) if track else 100.0


def _interp(a: LatLon, b: LatLon, t: float) -> LatLon:
    return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]


def coord_at_percent(track: Track, percent: float) -> LatLon:
    if not track:
        return [0.0, 0.0]
    if len(track) == 1:
        return track[0]
    p = _point_step(track)
    pos = percent / p
    i = int(math.floor(pos))
    if i >= len(track) - 1:
        return track[-1]
    f = pos - i
    return _interp(track[i], track[i + 1], f)


def rmse(a: LatLon, b: LatLon) -> float:
    return math.sqrt(((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) / 2.0)


def compare_track_to_all(
    single_track: Track,
    all_tracks: Dict[str, Track],
    weightingFunc: Optional[Callable[[float], float]] = None,
) -> Dict[str, float]:
    if weightingFunc is None:
        weightingFunc = lambda pct: 1.0  # pct in [0,100]

    scores: Dict[str, float] = {}
    step_single = _point_step(single_track)

    for storm_id, track in all_tracks.items():
        if not track:
            scores[storm_id] = float("inf")
            continue
        step_other = _point_step(track)
        step = min(step_single, step_other)
        if step <= 0:
            scores[storm_id] = float("inf")
            continue

        total = 0.0
        t = step
        while t <= 100.0 + 1e-9:
            a = coord_at_percent(single_track, t)
            b = coord_at_percent(track, t)
            w = max(0.0, min(1.0, float(weightingFunc(t))))
            total += w * rmse(a, b)
            t += step
        scores[storm_id] = total

    return scores


def load_tracks() -> Dict[str, Track]:
    return load_storm_tracks()


def resample_track_to_20_points(track: Track) -> np.ndarray:
    # 20 points at 5%,10%,...,100%
    percents = [(i + 1) * 5.0 for i in range(20)]
    pts: List[LatLon] = [coord_at_percent(track, p) for p in percents]
    return np.asarray(pts, dtype=float)


from typing import Tuple as _Tuple


def preprocess_tracks_to_array(tracks: Dict[str, Track]) -> _Tuple[np.ndarray, List[str]]:
    ids = list(tracks.keys())
    arr_list = [resample_track_to_20_points(tracks[sid]) for sid in ids]
    arr = np.stack(arr_list, axis=0) if arr_list else np.zeros((0, 20, 2), dtype=float)
    return arr, ids


def vector_compare_single_to_many_greatcircle(single_20: np.ndarray, many_20: np.ndarray, earth_radius_km: float = 6371.0) -> np.ndarray:
    # single_20: (20,2), many_20: (N,20,2)
    if many_20.size == 0:
        return np.zeros((0,), dtype=float)
    deg2rad = np.pi / 180.0
    lat1 = single_20[:, 0] * deg2rad
    lon1 = single_20[:, 1] * deg2rad
    lat2 = many_20[:, :, 0] * deg2rad
    lon2 = many_20[:, :, 1] * deg2rad
    # Broadcast single to (N,20)
    lat1b = lat1[None, :]
    lon1b = lon1[None, :]
    dlat = lat2 - lat1b
    dlon = lon2 - lon1b
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1b) * np.cos(lat2) * (np.sin(dlon / 2.0) ** 2)
    c = 2.0 * np.arcsin(np.minimum(1.0, np.sqrt(a)))
    dist = earth_radius_km * c  # (N,20)
    return dist.sum(axis=1)


def pairwise_compare_tracks_greatcircle_array(arr: np.ndarray, earth_radius_km: float = 6371.0) -> np.ndarray:
    # arr: (N,20,2) lat,lon in degrees
    if arr.size == 0:
        return np.zeros((0, 0), dtype=float)
    deg2rad = np.pi / 180.0
    lat = arr[:, :, 0] * deg2rad  # (N,20)
    lon = arr[:, :, 1] * deg2rad  # (N,20)
    lat_i = lat[:, None, :]  # (N,1,20)
    lon_i = lon[:, None, :]
    lat_j = lat[None, :, :]  # (1,N,20)
    lon_j = lon[None, :, :]
    dlat = lat_j - lat_i  # (N,N,20)
    dlon = lon_j - lon_i
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat_i) * np.cos(lat_j) * (np.sin(dlon / 2.0) ** 2)
    c = 2.0 * np.arcsin(np.minimum(1.0, np.sqrt(a)))
    dist = earth_radius_km * c  # (N,N,20)
    return dist.sum(axis=2)  # (N,N)


def pairwise_compare_tracks_greatcircle(tracks: Dict[str, Track], earth_radius_km: float = 6371.0) -> _Tuple[np.ndarray, List[str]]:
    arr, ids = preprocess_tracks_to_array(tracks)  # (N,20,2)
    mat = pairwise_compare_tracks_greatcircle_array(arr, earth_radius_km=earth_radius_km)
    return mat, ids


def main(top_n: int = 5, vectorized: bool = False, pairwise: bool = False) -> None:
    tracks = load_tracks()
    if not tracks:
        print("No tracks loaded")
        return
    if pairwise:
        arr, ids = preprocess_tracks_to_array(tracks)
        t0 = time.perf_counter()
        mat = pairwise_compare_tracks_greatcircle_array(arr)
        dt = time.perf_counter() - t0
        n = len(ids)
        print(f"Pairwise great-circle similarity: N={n}, shape={mat.shape}, compare_time={dt:.4f}s (excl. preprocess)")
        return
    storm_id = random.choice(list(tracks.keys()))
    single = tracks[storm_id]
    def weighting(pct: float) -> float:
        x = max(0.0, min(1.0, pct / 100.0))
        return -0.5 * (x ** 0.5) + 0.5

    if vectorized:
        t0 = time.perf_counter()
        arr, ids = preprocess_tracks_to_array(tracks)
        prep_dt = time.perf_counter() - t0
        try:
            idx = ids.index(storm_id)
        except ValueError:
            idx = 0
            storm_id = ids[0]
            single = tracks[storm_id]
        t1 = time.perf_counter()
        vec_scores = vector_compare_single_to_many_greatcircle(arr[idx], arr)
        comp_dt = time.perf_counter() - t1
        # Map ids to scores
        scores = {sid: float(s) for sid, s in zip(ids, vec_scores)}
        print(f"Vectorized preprocess: {prep_dt:.4f}s, compare: {comp_dt:.4f}s")
    else:
        scores = compare_track_to_all(single, tracks, weightingFunc=weighting)
        best_all = sorted(scores.items(), key=lambda kv: kv[1])
    best_all = sorted(scores.items(), key=lambda kv: kv[1])
    best = [(sid, s) for sid, s in best_all if sid != storm_id][:top_n]
    print(f"Random storm: {storm_id}")
    for i, (sid, score) in enumerate(best, 1):
        print(f"{i}. {sid}: {score:.6f}")

    vals = [v for v in scores.values() if math.isfinite(v)]
    if vals:
        try:
            import matplotlib.pyplot as plt  # type: ignore
        except Exception:
            plt = None
        if plt is not None:
            plt.style.use("dark_background")
            fig1 = plt.figure(figsize=(8, 4))
            plt.hist(vals, bins=50, color="cyan", alpha=0.9)
            plt.title(f"Similarity distribution to {storm_id}")
            plt.xlabel("Weighted RMSE")
            plt.ylabel("Count")
            plt.tight_layout()
            try:
                plt.show(block=False)
                plt.pause(0.1)
            except Exception:
                plt.show()

    # Map of tracks: closest N in yellow, chosen in red
    try:
        import matplotlib.pyplot as plt  # type: ignore
        import cartopy.crs as ccrs  # type: ignore
        import cartopy.feature as cfeature  # type: ignore
    except Exception:
        plt = None
        ccrs = None
    if plt is not None and ccrs is not None:
        plt.style.use("dark_background")
        fig = plt.figure(figsize=(9, 5))
        ax = plt.axes(projection=ccrs.PlateCarree())
        try:
            ax.add_feature(cfeature.LAND, facecolor="#101010")
            ax.add_feature(cfeature.OCEAN, facecolor="#050505")
        except Exception:
            ax.set_facecolor("#000000")
        try:
            ax.coastlines(color="white", linewidth=0.5)
            ax.gridlines(draw_labels=False, color="gray", alpha=0.2, linestyle="--", linewidth=0.5)
        except Exception:
            pass

        closest_ids = [sid for sid, _ in best]
        subset = {sid: tracks.get(sid, []) for sid in closest_ids + [storm_id]}
        all_pts = [pt for tr in subset.values() for pt in tr]
        if all_pts:
            lats = [pt[0] for pt in all_pts]
            lons = [pt[1] for pt in all_pts]
            min_lat, max_lat = min(lats), max(lats)
            min_lon, max_lon = min(lons), max(lons)
            lat_pad = max(2.0, (max_lat - min_lat) * 0.05)
            lon_pad = max(2.0, (max_lon - min_lon) * 0.05)
            ax.set_extent(
                [min_lon - lon_pad, max_lon + lon_pad, min_lat - lat_pad, max_lat + lat_pad],
                crs=ccrs.PlateCarree(),
            )

        for sid, tr in subset.items():
            if not tr:
                continue
            lats = [p[0] for p in tr]
            lons = [p[1] for p in tr]
            is_sel = sid == storm_id
            ax.plot(
                lons,
                lats,
                color="red" if is_sel else "yellow",
                alpha=0.95 if is_sel else 0.7,
                linewidth=1.8 if is_sel else 0.9,
                transform=ccrs.PlateCarree(),
            )
        plt.title(f"Storm tracks (selected: {storm_id})")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    try:
        import argparse
        p = argparse.ArgumentParser()
        p.add_argument("-v", "--vectorized", action="store_true")
        p.add_argument("-p", "--pairwise", action="store_true")
        args = p.parse_args()
        main(vectorized=args.vectorized, pairwise=args.pairwise)
    except Exception:
        main()
