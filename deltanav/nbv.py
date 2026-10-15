"""Where to look next.

The surfaces the after video could not check are exactly where a robot should look next. Candidate viewpoints
on floor the robot can reach are scored by how many of those surfaces they would see with nothing nearer in
the way. Occlusion comes from a 360-degree depth panorama of the room as it is now (the after video's
surfaces), so one panorama per position serves every viewing direction. The best views are picked greedily,
each one counting only what the previous ones have not covered yet."""
from __future__ import annotations
import numpy as np


def targets_from(P, L, S, Hgt, max_height: float = 2.0, min_height: float = -0.05):
    """Before-video surfaces that could not be re-checked, and are worth checking: seen consistently by the
    before video (S), not on the ceiling."""
    return (L == -1) & S & (Hgt < max_height) & (Hgt > min_height)


def panorama(occluders, C, az_bins: int = 180, el_bins: int = 90):
    d = occluders - C
    r = np.linalg.norm(d, axis=1)
    ok = r > 0.05
    d, r = d[ok], r[ok]
    az = np.arctan2(d[:, 1], d[:, 0]); el = np.arcsin(np.clip(d[:, 2] / r, -1, 1))
    ia = ((az + np.pi) / (2 * np.pi) * az_bins).astype(int) % az_bins
    ie = np.clip(((el + np.pi / 2) / np.pi * el_bins).astype(int), 0, el_bins - 1)
    Z = np.full((el_bins, az_bins), np.inf)
    np.minimum.at(Z, (ie, ia), r)
    from scipy.ndimage import minimum_filter
    return minimum_filter(Z, size=3, mode='wrap')     # close gaps between samples (occlusion errs on the safe side)


def visible_from(T, C, Z, near: float = 0.3, far: float = 4.5):
    """Targets T visible from position C (any direction): in range and not behind a nearer surface."""
    az_bins, el_bins = Z.shape[1], Z.shape[0]
    d = T - C
    r = np.linalg.norm(d, axis=1)
    az = np.arctan2(d[:, 1], d[:, 0]); el = np.arcsin(np.clip(d[:, 2] / np.maximum(r, 1e-6), -1, 1))
    ia = ((az + np.pi) / (2 * np.pi) * az_bins).astype(int) % az_bins
    ie = np.clip(((el + np.pi / 2) / np.pi * el_bins).astype(int), 0, el_bins - 1)
    vis = (r > near) & (r < far) & (r <= Z[ie, ia] + 0.08 + 0.03 * r)
    return vis, az, el


def best_views(T, w, occluders, positions, cam_height: float, k: int = 3, hfov: float = 70.0, vfov: float = 50.0,
               n_yaw: int = 24, pitches=(-10.0, -35.0), min_sep: float = 0.5):
    """Greedy k best (position, yaw, pitch) for a camera at `cam_height` above the floor at each 2D position.
    T: target points (room frame, z = height above floor); w: weights. -> list of dicts."""
    if len(T) == 0 or len(positions) == 0:
        return []
    w = w.astype(float).copy()
    cand = []
    for p in positions:
        C = np.array([p[0], p[1], cam_height])
        Z = panorama(occluders, C)
        vis, az, el = visible_from(T, C, Z)
        cand.append((C, vis, az, el))
    yaws = np.radians(np.arange(n_yaw) * 360.0 / n_yaw)
    out = []
    for _ in range(k):
        best = (0.0, None)
        for ci, (C, vis, az, el) in enumerate(cand):
            if any(np.linalg.norm(C[:2] - o['position'][:2]) < min_sep for o in out):
                continue
            if not vis.any():
                continue
            wv = w * vis
            if wv.sum() <= best[0]:
                continue
            for yaw in yaws:
                da = np.abs((az - yaw + np.pi) % (2 * np.pi) - np.pi) < np.radians(hfov / 2)
                for pitch in pitches:
                    m = da & (np.abs(el - np.radians(pitch)) < np.radians(vfov / 2))
                    s = float((wv * m).sum())
                    if s > best[0]:
                        best = (s, (ci, yaw, pitch, vis & m))
        if best[1] is None or best[0] <= 0:
            break
        ci, yaw, pitch, seen = best[1]
        C = cand[ci][0]
        out.append(dict(position=C.round(3).tolist(), yaw_deg=round(float(np.degrees(yaw)), 1), pitch_deg=pitch,
                        hfov_deg=hfov, n_points=int(seen.sum()), score=round(best[0], 1), seen=seen))
        w[seen] = 0.0
    return out


def plan_next_views(Pb, Lb, Sb, Hb, Pa, Ha, trav_reach, spec, robot, landmarks, bounds, voxel: float = 0.025,
                    k: int = 3, step: int = 3, max_targets: int = 6000, seed: int = 0):
    """Next-best views for one robot on its updated map. Returns (views, info)."""
    from .places import where
    rng = np.random.default_rng(seed)
    m = targets_from(Pb, Lb, Sb, Hb)
    idx = np.nonzero(m)[0]
    total = len(idx)
    if total == 0:
        return [], dict(unknown_points=0)
    if len(idx) > max_targets:
        idx = rng.choice(idx, max_targets, replace=False)
    T = Pb[idx].copy(); T[:, 2] = Hb[idx]                     # heights above the floor
    k_occ = rng.choice(len(Pa), min(len(Pa), 60000), replace=False)
    occ = Pa[k_occ].copy(); occ[:, 2] = Ha[k_occ]              # the room as it is now, same height convention
    ys, xs = np.nonzero(trav_reach)
    sel = (xs % step == 0) & (ys % step == 0)
    if sel.sum() < 5:
        sel = np.ones(len(xs), bool)
    pos = spec.centre(xs[sel], ys[sel])
    pri = np.where(T[:, 2] < 1.0, 1.0, 0.25)          # floor-level unknowns matter most to a robot
    views = best_views(T, pri, occ, pos, robot.camera_height, k=k)
    scale = total / len(idx)
    for v in views:
        S = T[v.pop('seen')]
        v['unknown_area_m2'] = round(float(len(S) * scale * voxel * voxel), 2)
        v['share_of_unknown'] = round(float(len(S) / len(idx)), 3)
        c = np.median(S, axis=0)
        v['looks_at'] = c.round(3).tolist()
        v['looks_at_words'] = where(c, landmarks, bounds) if landmarks else ''
        v['robot'] = robot.name
    return views, dict(unknown_points=int(total), unknown_area_m2=round(float(total * voxel * voxel), 2))


# ---------------------------------------------------------------- choosing among real frames (verification loop)
def zbuffer(view, P, H: int, W: int):
    """Depth image of points P seen from `view` at (H, W), nearest point per pixel, gaps closed by a 3x3 min."""
    v = view.scaled(W, H)
    uv, z = v.project(P)
    ok = (z > 0.1) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    Zb = np.full((H, W), np.inf)
    np.minimum.at(Zb, (uv[ok, 1].astype(int), uv[ok, 0].astype(int)), z[ok])
    from scipy.ndimage import minimum_filter
    return minimum_filter(Zb, size=3)


def predicted_gain(view, T, occluders, H: int = 72, W: int = 128, far: float = 5.0):
    """How many targets this camera would see, judged only from the current model (not from its image)."""
    v = view.scaled(W, H)
    uv, z = v.project(T)
    ok = (z > 0.2) & (z < far) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    if not ok.any():
        return 0, ok
    Zb = zbuffer(view, occluders, H, W)
    d = Zb[uv[ok, 1].astype(int), uv[ok, 0].astype(int)]
    vis = np.zeros(len(T), bool)
    vis[np.nonzero(ok)[0]] = z[ok] <= d + 0.08 + 0.03 * z[ok]
    return int(vis.sum()), vis
