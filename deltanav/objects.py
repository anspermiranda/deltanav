"""From changed points to objects: cluster, measure, name, colour, and pair disappearances with appearances to
recognise moves (with the rotation)."""
from __future__ import annotations
import os
import numpy as np
import cv2
from dataclasses import dataclass, field, asdict

from .semantics import family, plausible


@dataclass
class ChangeObject:
    id: int
    epoch: str                  # the recording it exists in: 'before' (left) or 'after' (arrived)
    n_points: int
    centroid: list              # room frame, metres (z = height of the centroid above the floor)
    footprint: dict             # rectangle on the floor: center, size (w, d), angle (deg)
    z_range: list               # [bottom, top] height above the floor
    points_idx: np.ndarray = field(repr=False, default=None)
    label: str = 'object'
    label_conf: float = 0.0
    label_votes: dict = field(default_factory=dict)
    color_name: str = ''
    evidence: dict = field(default_factory=dict)
    sigma_m: float = 0.0        # 1-sigma position uncertainty

    def size(self):
        w, d = self.footprint['size']
        return float(max(w, d)), float(min(w, d)), float(self.z_range[1] - self.z_range[0])

    def to_dict(self):
        d = asdict(self); d.pop('points_idx', None)
        return d


def _trim(Q: np.ndarray, k: int = 10, nsd: float = 2.0) -> np.ndarray:
    """Statistical outlier removal: drop points far from their neighbours (stray depth pixels)."""
    from scipy.spatial import cKDTree
    if len(Q) < k + 2:
        return np.ones(len(Q), bool)
    d, _ = cKDTree(Q).query(Q, k=k + 1)
    m = d[:, 1:].mean(1)
    return m <= m.mean() + nsd * m.std()


def _measure(P, Hh, idx):
    Q = P[idx]
    lo, hi = np.percentile(Q[:, :2], [3, 97], axis=0)
    core = np.all((Q[:, :2] >= lo) & (Q[:, :2] <= hi), axis=1)
    (cx, cy), (w, d), ang = cv2.minAreaRect(Q[core, :2].astype(np.float32))
    hq = Hh[idx]
    zr = [float(np.percentile(hq, 2)), float(np.percentile(hq, 98))]
    c = Q.mean(0); c[2] = float(np.mean(hq))
    return dict(center=[float(cx), float(cy)], size=[float(w), float(d)], angle=float(ang)), zr, c


def cluster_points(P: np.ndarray, mask: np.ndarray, epoch: str, eps: float = 0.06, min_samples: int = 8,
                   min_points: int = 150, floor_z: float = 0.06, min_extent: float = 0.12, min_height: float = 0.12,
                   start_id: int = 0, heights: np.ndarray | None = None):
    from sklearn.cluster import DBSCAN
    Hh = P[:, 2] if heights is None else heights
    sel = np.nonzero(mask & (Hh > floor_z))[0]
    if len(sel) == 0:
        return []
    lab = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(P[sel])
    groups = []
    for k in sorted(set(lab) - {-1}):
        idx = sel[lab == k]
        if len(idx) >= min_points // 3:
            groups.append(idx[_trim(P[idx])])
    groups = _merge_stacked(P, Hh, groups)
    objs = []
    for idx in groups:
        if len(idx) < min_points:
            continue
        fp, zr, c = _measure(P, Hh, idx)
        if max(fp['size']) < min_extent or (zr[1] - zr[0]) < min_height:
            continue
        objs.append(ChangeObject(id=start_id + len(objs), epoch=epoch, n_points=int(len(idx)),
                                 centroid=np.round(c, 4).tolist(), footprint=fp, z_range=zr, points_idx=idx))
    return objs


def _merge_stacked(P, Hh, groups, overlap: float = 0.5, gap: float = 0.15):
    """Join parts stacked on top of each other (a chair's seat and its base): footprints overlap and the
    height ranges touch."""
    boxes = []
    for idx in groups:
        Q = P[idx]
        boxes.append((np.percentile(Q[:, 0], [5, 95]), np.percentile(Q[:, 1], [5, 95]), np.percentile(Hh[idx], [2, 98])))
    parent = list(range(len(groups)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            (xi, yi, zi), (xj, yj, zj) = boxes[i], boxes[j]
            ix = max(0, min(xi[1], xj[1]) - max(xi[0], xj[0])); iy = max(0, min(yi[1], yj[1]) - max(yi[0], yj[0]))
            ai = (xi[1] - xi[0]) * (yi[1] - yi[0]); aj = (xj[1] - xj[0]) * (yj[1] - yj[0])
            if ix * iy > overlap * min(ai, aj) and max(zi[0], zj[0]) - min(zi[1], zj[1]) < gap:
                parent[find(i)] = find(j)
    merged = {}
    for i, idx in enumerate(groups):
        merged.setdefault(find(i), []).append(idx)
    return [np.concatenate(v) for v in merged.values()]


# ---------------------------------------------------------------- colour
def color_stats(bgr_pixels: np.ndarray, frame_bgr: np.ndarray):
    """Lightness, chroma and hue (CIELAB) of an object's pixels after grey-world white balance of the frame, so
    warm room light does not turn a blue suitcase black or a black chair orange."""
    gains = frame_bgr.reshape(-1, 3).astype(float).mean(0); gains = gains.mean() / np.maximum(gains, 1)
    q = np.clip(bgr_pixels.astype(float) * gains, 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(q.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(float)
    L = np.median(lab[:, 0]) * 100 / 255; a = np.median(lab[:, 1] - 128); b = np.median(lab[:, 2] - 128)
    return L, float(np.hypot(a, b)), float((np.degrees(np.arctan2(b, a)) + 360) % 360)


def color_name_lab(L: float, chroma: float, hue: float) -> str:
    if chroma < 8:
        return 'black' if L < 30 else ('white' if L > 75 else 'grey')
    for lim, n in [(40, 'red'), (75, 'orange'), (105, 'yellow'), (180, 'green'), (235, 'teal'), (300, 'blue'),
                   (335, 'purple'), (361, 'pink')]:
        if hue < lim:
            return ('dark ' if L < 25 and n not in ('blue', 'purple') else '') + n
    return 'red'


# ---------------------------------------------------------------- naming
def visible_views(Q: np.ndarray, views: list, depths: dict, tau: float = 0.08, top: int = 6):
    """Frames of the object's own recording that actually see it (depth test), best first:
    [(score, view, uv of the visible points in depth-map pixels)]."""
    scored = []
    for v in views:
        D = depths.get(v.name)
        if D is None:
            continue
        H, W = D.shape
        uv, z = v.project(Q)
        ok = (z > 0.2) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
        if ok.sum() < 20:
            continue
        d = D[np.round(uv[ok, 1]).astype(int).clip(0, H - 1), np.round(uv[ok, 0]).astype(int).clip(0, W - 1)]
        vis = np.abs(d - z[ok]) < np.maximum(tau, 0.05 * z[ok])
        scored.append((vis.sum() * (0.5 + 0.5 * ok.mean()), v, uv[ok][vis]))
    scored.sort(key=lambda s: -s[0])
    return scored[:top]


def name_objects(objs: list, P_by_epoch: dict, views_by_epoch: dict, depths: dict, dets: dict, img_size: dict,
                 image_dir: str | None = None,
                 ignore=('bed', 'table', 'couch', 'television', 'refrigerator', 'desk', 'wardrobe', 'nightstand', 'radiator',
                         'window', 'door', 'shelf', 'bookshelf', 'drawers', 'person', 'socket', 'switch', 'hanger', 'cabinet',
                         'fridge', 'oven', 'stove', 'sink', 'counter', 'kitchen island', 'washing machine', 'toilet',
                         'bathtub', 'shower', 'fireplace', 'stairs', 'man', 'woman', 'child', 'dog', 'cat')):
    """Name each object by voting over instance masks in the frames that see it best (votes pooled by family;
    the name must fit the measured size), and read its colour from those frames. img_size: name -> (w, h)."""
    for o in objs:
        Q = P_by_epoch[o.epoch][o.points_idx]
        vis = visible_views(Q, views_by_epoch[o.epoch], depths)
        st = []
        for _, v0, uv0 in vis[:3]:
            img = cv2.imread(os.path.join(image_dir, v0.name)) if image_dir else None
            if img is None or len(uv0) < 20:
                continue
            sc = np.array([img_size[v0.name][0] / v0.width, img_size[v0.name][1] / v0.height], np.float32)
            xy = (uv0 * sc).astype(int)
            xy[:, 0] = xy[:, 0].clip(0, img.shape[1] - 1); xy[:, 1] = xy[:, 1].clip(0, img.shape[0] - 1)
            st.append(color_stats(img[xy[:, 1], xy[:, 0]], img))
        if st:
            L, ch, hu = np.median(np.array(st), 0)
            o.color_name = color_name_lab(L, ch, hu)
        votes = {}
        for score, v, uvs in vis:
            if len(uvs) < 15:
                continue
            sc = np.array([img_size[v.name][0] / v.width, img_size[v.name][1] / v.height], np.float32)
            uv_img = uvs * sc
            best = None
            for det in dets.get(v.name, []):
                if det['poly'] is None or det['label'] in ignore:
                    continue
                poly = det['poly'].reshape(-1, 1, 2).astype(np.float32)
                inside = np.array([cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0 for x, y in uv_img[::3]])
                frac = inside.mean() if len(inside) else 0.0
                if frac >= 0.3:
                    s = frac * det['conf']
                    if best is None or s > best[0]:
                        best = (s, det['label'])
            if best:
                votes[best[1]] = votes.get(best[1], 0.0) + best[0]
        o.label_votes = {k: round(v, 3) for k, v in sorted(votes.items(), key=lambda kv: -kv[1])}
        if votes:
            fam = {}
            for k, v in votes.items():
                fam[family(k)] = fam.get(family(k), 0.0) + v
            w, d, h = o.size()
            for best_f, val in sorted(fam.items(), key=lambda kv: -kv[1]):
                members = {k: v for k, v in votes.items() if family(k) == best_f}
                top = max(members.items(), key=lambda kv: kv[1] * (1.0 if kv[0] == best_f else 1.25))
                # its own height: from the floor if it stands on it, else its extent (a laptop on a desk is short)
                own_h = o.z_range[1] if o.z_range[0] < 0.2 else o.z_range[1] - o.z_range[0]
                if plausible(top[0], own_h, w):
                    o.label, o.label_conf = top[0], float(val / sum(votes.values()))
                    break
    return objs


def match_moves(gone: list, new: list, min_score: float = 0.55):
    """Pair objects that disappeared with objects that appeared (Hungarian assignment on name, size, colour)."""
    from scipy.optimize import linear_sum_assignment
    if not gone or not new:
        return []
    S = np.zeros((len(gone), len(new)))
    for i, g in enumerate(gone):
        for j, a in enumerate(new):
            fg, fa = family(g.label), family(a.label)
            if g.label != 'object' and a.label != 'object' and fg != fa:
                continue
            lab = 1.0 if (fg == fa and g.label != 'object') else 0.4
            sg, sa = np.array(g.size()), np.array(a.size())
            ratios = np.minimum(sg, sa) / np.maximum(np.maximum(sg, sa), 1e-3)
            size = float(np.sqrt(ratios.mean() * ratios.min()))
            col = 1.0 if (g.color_name and g.color_name == a.color_name) else 0.5
            S[i, j] = 0.5 * lab + 0.3 * size + 0.2 * col
            if lab < 1.0 and size < 0.6:
                S[i, j] = 0.0
    r, c = linear_sum_assignment(-S)
    return [(gone[i], new[j], float(S[i, j])) for i, j in zip(r, c) if S[i, j] >= min_score]


# ---------------------------------------------------------------- rotation
def _yaw_icp(A, B, th0, iters=25, max_d=0.05):
    """ICP restricted to a rotation about the vertical + translation. A, B centred (N,3). -> (yaw deg, fitness)."""
    from scipy.spatial import cKDTree
    tree = cKDTree(B)

    def apply(th, t):
        c, s_ = np.cos(th), np.sin(th)
        X = A.copy(); X[:, 0] = c * A[:, 0] - s_ * A[:, 1]; X[:, 1] = s_ * A[:, 0] + c * A[:, 1]
        return X + t
    th, t = th0, np.zeros(3)
    for _ in range(iters):
        d, j = tree.query(apply(th, t), distance_upper_bound=max_d)
        ok = np.isfinite(d)
        if ok.sum() < 10:
            break
        src, dst = A[ok], B[j[ok]]
        ms, md = src[:, :2].mean(0), dst[:, :2].mean(0)
        a, b = src[:, :2] - ms, dst[:, :2] - md
        th_new = np.arctan2((a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]).sum(), (a * b).sum())
        c, s_ = np.cos(th_new), np.sin(th_new)
        t_new = np.r_[md - np.array([c * ms[0] - s_ * ms[1], s_ * ms[0] + c * ms[1]]), (dst[:, 2] - src[:, 2]).mean()]
        done = abs(th_new - th) < 1e-4 and np.linalg.norm(t_new - t) < 1e-4
        th, t = th_new, t_new
        if done:
            break
    d, _ = tree.query(apply(th, t), distance_upper_bound=max_d)
    return float(np.degrees(th)), float(np.isfinite(d).mean())


def _adiff(a, b):
    return abs((a - b + 180) % 360 - 180)


def estimate_yaw(Pg: np.ndarray, Pa: np.ndarray, step_deg: int = 15, margin: float = 0.04):
    """Geometry-only rotation: ICP from a start yaw every `step_deg`, best overlap wins. Partial, blob-like
    point clouds often fit almost equally well at several angles, so the result says whether to trust it."""
    A = (Pg - Pg.mean(0)).astype(np.float64); B = (Pa - Pa.mean(0)).astype(np.float64)
    prof = [_yaw_icp(A, B, np.radians(d)) for d in range(0, 360, step_deg)]
    prof = [((y + 180) % 360 - 180, f) for y, f in prof]
    yaw, fit = max(prof, key=lambda p: p[1])
    alt = max([f for y, f in prof if _adiff(y, yaw) > 45] or [0.0])
    ident = max([f for y, f in prof if _adiff(y, 0) < 15] or [0.0])
    return dict(yaw_deg=round(yaw, 1), fitness=round(fit, 3), confident=bool(fit - alt >= margin),
                turned=bool(fit - ident >= margin and _adiff(yaw, 0) >= 20), source='geometry (ICP)',
                profile=[[round(y, 1), round(f, 3)] for y, f in prof])


def feature_yaw(g, a, views_by_epoch: dict, depths: dict, image_dir: str, img_size: dict, P_by_epoch: dict,
                n_views: int = 10, inlier_m: float = 0.06, min_inliers: int = 20, seed: int = 0):
    """Rotation from appearance: SIFT matches between the frames that see the object best in each recording,
    lifted to 3D with the depth maps, then a 2-point RANSAC for yaw + translation."""
    from scipy.spatial import cKDTree
    sift = cv2.SIFT_create(nfeatures=3000)
    feats = {}
    for o in (g, a):
        Q = P_by_epoch[o.epoch][o.points_idx]
        tree = cKDTree(Q)
        out = []
        for _, v, _uv in visible_views(Q, views_by_epoch[o.epoch], depths, top=n_views):
            img = cv2.imread(os.path.join(image_dir, v.name), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            sx, sy = img_size[v.name][0] / v.width, img_size[v.name][1] / v.height
            uv, z = v.project(Q)
            ok = z > 0.2
            x0, y0 = np.percentile(uv[ok], 2, axis=0) * [sx, sy] - 25
            x1, y1 = np.percentile(uv[ok], 98, axis=0) * [sx, sy] + 25
            mask = np.zeros_like(img)
            mask[int(max(0, y0)):int(min(img.shape[0], y1)), int(max(0, x0)):int(min(img.shape[1], x1))] = 255
            kp, des = sift.detectAndCompute(img, mask)
            if des is None or len(kp) < 10:
                continue
            D = depths[v.name]; H, W = D.shape
            pts = np.array([k.pt for k in kp], np.float32) / [sx, sy]
            ui = np.clip(np.round(pts[:, 0]).astype(int), 0, W - 1); vi = np.clip(np.round(pts[:, 1]).astype(int), 0, H - 1)
            d = D[vi, ui]
            good = np.isfinite(d)
            Xc = np.stack([(pts[:, 0] - v.K[0, 2]) / v.K[0, 0] * d, (pts[:, 1] - v.K[1, 2]) / v.K[1, 1] * d, d], 1)
            Xw = (Xc - v.t) @ v.R
            near = np.zeros(len(Xw), bool)
            if good.any():
                dd, _ = tree.query(Xw[good], distance_upper_bound=0.06)
                near[np.nonzero(good)[0]] = np.isfinite(dd)
            good &= near
            if good.sum() >= 8:
                out.append((des[good], Xw[good]))
        feats[o.epoch] = out
    if not feats.get(g.epoch) or not feats.get(a.epoch):
        return None
    bf = cv2.BFMatcher(cv2.NORM_L2)
    Xg, Xa = [], []
    for dg, xg in feats[g.epoch]:
        for da, xa in feats[a.epoch]:
            if len(dg) < 2 or len(da) < 2:
                continue
            for m in bf.knnMatch(dg, da, k=2):
                if len(m) == 2 and m[0].distance < 0.75 * m[1].distance:
                    Xg.append(xg[m[0].queryIdx]); Xa.append(xa[m[0].trainIdx])
    if len(Xg) < min_inliers:
        return dict(yaw_deg=None, inliers=0, matches=len(Xg), confident=False, source='appearance (SIFT)')
    Xg, Xa = np.array(Xg), np.array(Xa)
    rng = np.random.default_rng(seed)
    best = (0, None)
    for _ in range(2000):
        i, j = rng.choice(len(Xg), 2, replace=False)
        vg, va = Xg[j, :2] - Xg[i, :2], Xa[j, :2] - Xa[i, :2]
        if np.linalg.norm(vg) < 0.08 or abs(np.linalg.norm(vg) - np.linalg.norm(va)) > 0.05:
            continue
        th = np.arctan2(va[1], va[0]) - np.arctan2(vg[1], vg[0])
        c, s_ = np.cos(th), np.sin(th)
        Rg = Xg[:, :2] @ np.array([[c, s_], [-s_, c]])
        t = Xa[i, :2] - Rg[i]
        inl = np.linalg.norm(Rg + t - Xa[:, :2], axis=1) < inlier_m
        if inl.sum() > best[0]:
            best = (int(inl.sum()), inl)
    if best[1] is None or best[0] < min_inliers:
        return dict(yaw_deg=None, inliers=best[0], matches=len(Xg), confident=False, source='appearance (SIFT)')
    inl = best[1]
    a_, b_ = Xg[inl, :2] - Xg[inl, :2].mean(0), Xa[inl, :2] - Xa[inl, :2].mean(0)
    th = np.degrees(np.arctan2((a_[:, 0] * b_[:, 1] - a_[:, 1] * b_[:, 0]).sum(), (a_ * b_).sum()))
    return dict(yaw_deg=round(float((th + 180) % 360 - 180), 1), inliers=int(inl.sum()), matches=len(Xg),
                confident=bool(inl.sum() >= min_inliers and inl.mean() >= 0.25), source='appearance (SIFT)')
