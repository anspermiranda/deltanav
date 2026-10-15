"""Visibility-aware change detection between two recordings.

The core rule: absence of evidence is not evidence of absence. A surface seen in recording A is only called
*gone* in recording B if B's cameras looked straight through the place where it was and measured something
further away (free space). If B's cameras saw something nearer in the way, it is *occluded*; if they never
pointed there, it is *unobserved*. Both stay *unknown* - never a change."""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass

from .geometry import View


@dataclass
class Evidence:
    n_free: np.ndarray      # frames that saw through the point (it is not there any more)
    n_same: np.ndarray      # frames that saw a surface at the point (it is still there)
    n_occl: np.ndarray      # frames where something nearer blocked the line of sight
    n_fov: np.ndarray       # frames whose view contains the point (any outcome)

    def classify(self, min_free: int = 3, free_ratio: float = 0.7, min_same: int = 2, free_frac_fov: float = 0.25):
        """-> per point 1 (gone), 0 (same), -1 (unknown). 'gone' needs see-through frames to dominate the frames
        that saw a surface there AND to be a real share of all frames that pointed there: a place seen through
        once and hidden thirty times stays unknown."""
        tot = self.n_free + self.n_same
        ratio = np.where(tot > 0, self.n_free / np.maximum(tot, 1), 0.0)
        gone = (self.n_free >= min_free) & (ratio >= free_ratio) & (self.n_free >= free_frac_fov * self.n_fov)
        same = (~gone) & (self.n_same >= min_same) & (ratio < 0.5)
        lab = np.full(len(tot), -1, np.int8)
        lab[gone] = 1
        lab[same] = 0
        return lab


def _sample(depth: np.ndarray, u: np.ndarray, v: np.ndarray):
    H, W = depth.shape
    ui = np.round(u).astype(np.int64); vi = np.round(v).astype(np.int64)
    ok = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    out = np.full(len(u), np.nan, np.float32)
    out[ok] = depth[vi[ok], ui[ok]]
    return out


def gather_evidence(points: np.ndarray, views: list[View], depths: dict, edge_masks: dict | None = None,
                    tau0: float = 0.06, tau_rel: float = 0.06, free0: float = 0.10, free_rel: float = 0.10,
                    near: float = 0.25, far: float = 6.0, margin: int = 4, chunk: int = 200_000) -> Evidence:
    """Test every point (N,3, room frame, metres) against the depth maps of `views`. Asymmetric thresholds:
    'a surface is here' within max(6 cm, 6%), 'seen clearly past it' beyond max(10 cm, 10%)."""
    N = len(points)
    n_free = np.zeros(N, np.int32); n_same = np.zeros(N, np.int32)
    n_occl = np.zeros(N, np.int32); n_fov = np.zeros(N, np.int32)
    for v in views:
        D = depths.get(v.name)
        if D is None:
            continue
        H, W = D.shape
        E = None if edge_masks is None else edge_masks.get(v.name)
        for s in range(0, N, chunk):
            P = points[s:s + chunk]
            uv, z = v.project(P)
            inside = (z > near) & (z < far) & (uv[:, 0] >= margin) & (uv[:, 0] < W - margin) & \
                     (uv[:, 1] >= margin) & (uv[:, 1] < H - margin)
            idx = np.nonzero(inside)[0]
            if len(idx) == 0:
                continue
            d = _sample(D, uv[idx, 0], uv[idx, 1])
            valid = np.isfinite(d)
            if E is not None:
                valid &= ~(_sample(E.astype(np.float32), uv[idx, 0], uv[idx, 1]) > 0.5)
            idx, d, zz = idx[valid], d[valid], z[idx][valid]
            tau = np.maximum(tau0, tau_rel * zz)
            tfree = np.maximum(free0, free_rel * zz)
            r = d - zz
            g = s + idx
            n_fov[g] += 1
            n_free[g[r > tfree]] += 1
            n_same[g[np.abs(r) <= tau]] += 1
            n_occl[g[r < -tau]] += 1
    return Evidence(n_free, n_same, n_occl, n_fov)


def witness_rays(Q: np.ndarray, views: list[View], depths: dict, edge_masks: dict | None = None,
                 tau0: float = 0.06, tau_rel: float = 0.06, free0: float = 0.10, free_rel: float = 0.10,
                 near: float = 0.25, far: float = 6.0, margin: int = 4, min_hits: int = 5, max_views: int = 7,
                 per_view: int = 2, min_sep: float = 0.25, seed: int = 0):
    """Object-level evidence: what did each frame of the other recording see at the object's place?
    Each frame that looked there gets one verdict: saw *through* it, saw a *surface* there, had it *occluded*
    by something nearer, or unclear. Returns (rays, stats); rays are [camera, point on the object, surface hit
    behind it] triples from see-through frames at least `min_sep` apart, for the viewers."""
    rng = np.random.default_rng(seed)
    st = dict(views_looked=0, views_saw_through=0, views_saw_surface=0, views_occluded=0, rays_through=0)
    through = []
    for v in views:
        D = depths.get(v.name)
        if D is None:
            continue
        H, W = D.shape
        uv, z = v.project(Q)
        inside = (z > near) & (z < far) & (uv[:, 0] >= margin) & (uv[:, 0] < W - margin) & \
                 (uv[:, 1] >= margin) & (uv[:, 1] < H - margin)
        idx = np.nonzero(inside)[0]
        if len(idx) < min_hits:
            continue
        d = _sample(D, uv[idx, 0], uv[idx, 1])
        ok = np.isfinite(d)
        if edge_masks is not None and edge_masks.get(v.name) is not None:
            ok &= ~(_sample(edge_masks[v.name].astype(np.float32), uv[idx, 0], uv[idx, 1]) > 0.5)
        if ok.sum() < min_hits:
            continue
        st['views_looked'] += 1
        idx, d, zz = idx[ok], d[ok], z[idx][ok]
        r = d - zz
        free = r > np.maximum(free0, free_rel * zz)
        same = np.abs(r) <= np.maximum(tau0, tau_rel * zz)
        occl = r < -np.maximum(tau0, tau_rel * zz)
        nf, ns = int(free.sum()), int(same.sum())
        if nf >= min_hits and nf >= 0.5 * (nf + ns) and nf >= 0.3 * len(idx):
            st['views_saw_through'] += 1; st['rays_through'] += nf
            through.append((nf, v, idx[free], zz[free], d[free]))
        elif ns >= min_hits and ns >= 0.5 * (nf + ns):
            st['views_saw_surface'] += 1
        elif occl.sum() >= 0.5 * len(idx):
            st['views_occluded'] += 1
    through.sort(key=lambda t: -t[0])
    rays, centres = [], []
    for cnt, v, idx, zz, d in through:
        C = v.C
        if any(np.linalg.norm(C - c) < min_sep for c in centres):
            continue
        centres.append(C)
        for k in rng.choice(len(idx), min(per_view, len(idx)), replace=False):
            q = Q[idx[k]]
            hit = C + (q - C) * (d[k] / zz[k])          # z-depth scales linearly along the ray
            rays.append([np.round(C, 3).tolist(), np.round(q, 3).tolist(), np.round(hit, 3).tolist()])
        if len(centres) >= max_views:
            break
    return rays, st


def object_verdict(st: dict, min_views: int = 5, max_surface_share: float = 0.1) -> tuple[bool, str]:
    """Report an object only if enough frames of the other recording saw through its place and almost none saw
    a surface there. Thin or see-through structures fool single points, rarely whole frames."""
    t, s = st['views_saw_through'], st['views_saw_surface']
    if t < min_views:
        return False, f'only {t} frames of the other video saw through its place (at least {min_views} needed)'
    if s > max_surface_share * (t + s):
        return False, f'{s} of {t + s} frames that saw the place clearly saw a surface there'
    return True, ''


def naive_diff(points_a: np.ndarray, points_b: np.ndarray, radius: float = 0.08) -> np.ndarray:
    """Baseline: a point 'changed' if the other cloud has no point within `radius`. It ignores visibility, so
    hidden or unfilmed places come out as changes."""
    from scipy.spatial import cKDTree
    d, _ = cKDTree(points_b).query(points_a, k=1, distance_upper_bound=radius)
    return np.isinf(d)
