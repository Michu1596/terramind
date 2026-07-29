import numpy as np
import geopandas as gpd
from libpysal.weights import KNN
from esda.moran import Moran
from scipy.optimize import minimize

def moran(points: np.ndarray, labels: np.ndarray):
    gdf = gpd.GeoDataFrame(
        {"label": labels}, geometry=gpd.points_from_xy(points[:, 0], points[:, 1])
    )
    w = KNN.from_dataframe(gdf, k=8)
    w.transform = "R"
    m = Moran(gdf["label"], w)
    return m.I, m.p_norm, m.z_norm

def circ_mean(a: np.ndarray) -> float:
    return np.arctan2(np.sin(a).sum(), np.cos(a).sum())

def circ_corr(a: np.ndarray, b: np.ndarray) -> float:
    ma, mb = circ_mean(a), circ_mean(b)
    num = (np.sin(a - ma) * np.sin(b - mb)).sum()
    den = np.sqrt((np.sin(a - ma) ** 2).sum() * (np.sin(b - mb) ** 2).sum())
    return num / den if den != 0 else 0.0

def mrl(phi: np.ndarray, theta: np.ndarray) -> float:
    d = theta - phi
    return np.sqrt(np.cos(d).mean() ** 2 + np.sin(d).mean() ** 2)

def _neg_abs_circ_corr(center, pts, lbls):
    dx, dy = pts[:, 0] - center[0], pts[:, 1] - center[1]
    if np.any((np.abs(dx) < 1e-12) & (np.abs(dy) < 1e-12)):
        return 0.0
    phi = np.arctan2(dy, dx)
    c = circ_corr(phi, lbls)
    return -abs(c) if not np.isnan(c) else 0.0

def optimal_center(pts: np.ndarray, lbls: np.ndarray, use_bbox: bool = False):
    if use_bbox:
        c = (pts.min(axis=0) + pts.max(axis=0)) / 2
        p = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
        return c, abs(circ_corr(p, lbls)), mrl(p, lbls)

    best_c, best_corr = None, -1.0
    for guess in [pts.mean(axis=0), (pts.min(axis=0) + pts.max(axis=0)) / 2]:
        res = minimize(
            _neg_abs_circ_corr,
            guess,
            args=(pts, lbls),
            method="Nelder-Mead",
            options={"maxiter": 500},
        )
        corr = -res.fun
        if corr > best_corr:
            best_corr, best_c = corr, res.x

    p = np.arctan2(pts[:, 1] - best_c[1], pts[:, 0] - best_c[0])
    return best_c, best_corr, mrl(p, lbls)

def bootstrap_ci(vals, n_boot: int = 1000, ci: float = 0.95):
    vals = np.asarray(vals)
    bm = np.sort([np.random.choice(vals, size=len(vals), replace=True).mean() for _ in range(n_boot)])
    a = (1 - ci) / 2
    return bm[int(a * n_boot)], bm[int((1 - a) * n_boot)]
