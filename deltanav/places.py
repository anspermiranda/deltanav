"""Where things are, in words and relations.

Furniture that does not move (bed, desk, wardrobe, ...) is lifted from instance masks to 3D footprints and used
as landmarks. A changed object's position is described with a few spatial relations: 'at the desk', 'between
the bed and the wardrobe', 'at the foot of the bed', 'next to the radiator', 'against the wall'."""
from __future__ import annotations
import numpy as np
import cv2

from .semantics import LANDMARKS, CORE_LANDMARKS, KITCHEN, BATHROOM, NESTED, landmark_check

PREFERENCE = ['bed', 'desk', 'wardrobe', 'nightstand', 'radiator', 'door', 'window', 'bookshelf', 'shelf', 'drawers',
              'couch', 'television', 'mirror', 'cabinet', 'table',
              'sink', 'stove', 'oven', 'fridge', 'washing machine', 'toilet', 'bathtub', 'shower', 'fireplace', 'piano',
              'kitchen island', 'counter', 'stairs']


def lift_landmarks(views: dict, depths: dict, dets: dict, img_size: dict, floor_c=None, classes=LANDMARKS,
                   min_conf: float = 0.45, voxel: float = 0.05, min_support: int = 3, eps: float = 0.16):
    """-> list of dict(label, xy, footprint{center,size,angle}, z, support). A 5 cm voxel counts towards a
    landmark only if at least `min_support` separate detections put a surface there (mask borders bleed onto
    the floor and walls behind; consensus removes that)."""
    from sklearn.cluster import DBSCAN
    from .depth import backproject
    from .geometry import height_above_floor
    acc = {}
    classes = set(classes)
    # the core furniture first, then the wider vocabulary with its own random stream, so that adding names to the
    # vocabulary never moves the landmarks found for the others
    for group, seed in ((classes & CORE_LANDMARKS, 0), (classes - CORE_LANDMARKS, 1)):
        rng = np.random.default_rng(seed)
        for n, v in views.items():
            D = depths.get(n)
            if D is None:
                continue
            H, W = D.shape
            sc = np.array([img_size[n][0] / W, img_size[n][1] / H], np.float32)
            for d in dets.get(n, []):
                if d['label'] not in group or d['poly'] is None or d['conf'] < min_conf:
                    continue
                m = np.zeros((H, W), np.uint8)
                cv2.fillPoly(m, [(d['poly'] / sc).astype(np.int32)], 1)
                m = cv2.erode(m, np.ones((5, 5), np.uint8))
                X, _ = backproject(np.where(m > 0, D, np.nan), v, step=3)
                if floor_c is not None and len(X):
                    h = height_above_floor(X, floor_c)
                    X = X[(h > 0.04) & (h < 2.3)]
                if len(X) < 20:
                    continue
                if len(X) > 500:
                    X = X[rng.choice(len(X), 500, replace=False)]
                acc.setdefault(d['label'], []).append(np.unique(np.floor(X / voxel).astype(np.int64), axis=0))
    out = []
    for lab, keys in acc.items():
        uk, cnt = np.unique(np.vstack(keys), axis=0, return_counts=True)
        sel = cnt >= min_support
        if sel.sum() < 10:
            continue
        P = (uk[sel] + 0.5) * voxel; w = cnt[sel]
        cl = DBSCAN(eps=eps, min_samples=4).fit_predict(P)
        for c in sorted(set(cl) - {-1}):
            Q = P[cl == c]
            if len(Q) < 15:
                continue
            lo, hi = np.percentile(Q[:, :2], [2, 98], axis=0)
            core = np.all((Q[:, :2] >= lo) & (Q[:, :2] <= hi), 1)
            (cx, cy), (fw, fd), ang = cv2.minAreaRect(Q[core, :2].astype(np.float32))
            hz = height_above_floor(Q, floor_c) if floor_c is not None else Q[:, 2]
            name = landmark_check(lab, float(np.percentile(hz, 98)), max(fw, fd), min(fw, fd))
            if name is None:
                continue
            out.append(dict(label=name, xy=[round(float(cx), 3), round(float(cy), 3)],
                            footprint=dict(center=[float(cx), float(cy)], size=[float(fw), float(fd)], angle=float(ang)),
                            z=[round(float(np.percentile(hz, 2)), 3), round(float(np.percentile(hz, 98)), 3)],
                            support=int(w[cl == c].sum()), detections=len(keys)))
    return _dedupe(_in_context(out))


def _in_context(lms: list) -> list:
    """A fridge alone in a bedroom is a wardrobe the detector misread. Kitchen and bathroom fixtures are kept
    only when at least two different ones were found (fridge + sink, toilet + bathtub, ...)."""
    labels = {l['label'] for l in lms}
    k, b = labels & KITCHEN, labels & BATHROOM
    ok = (k if len(k) >= 2 else set()) | (b if len(b) >= 2 else set())
    return [l for l in lms if l['label'] not in KITCHEN | BATHROOM or l['label'] in ok]


def _overlap(a, b) -> float:
    pa = cv2.boxPoints((tuple(a['footprint']['center']), tuple(a['footprint']['size']), a['footprint']['angle']))
    pb = cv2.boxPoints((tuple(b['footprint']['center']), tuple(b['footprint']['size']), b['footprint']['angle']))
    inter, _ = cv2.intersectConvexConvex(pa.astype(np.float32), pb.astype(np.float32))
    small = min(cv2.contourArea(pa), cv2.contourArea(pb))
    return float(inter / small) if small > 1e-6 else 0.0


def _rank(lm):
    return (PREFERENCE.index(lm['label']) if lm['label'] in PREFERENCE else len(PREFERENCE), -lm['support'])


def _nested(a, b) -> bool:
    """A sink or a hob inside a counter, an oven or a washing machine under one: both are kept."""
    la, lb = a['label'], b['label']
    return la != lb and ((la in NESTED and lb in ('counter', 'kitchen island', 'cabinet'))
                         or (lb in NESTED and la in ('counter', 'kitchen island', 'cabinet')))


def _dedupe(lms: list) -> list:
    keep = []
    for lm in sorted(lms, key=lambda l: -l['support']):
        clash = [k for k in keep if _overlap(lm, k) > 0.5 and not _nested(lm, k)]
        if not clash:
            keep.append(lm)
            continue
        k = clash[0]
        if lm['label'] != k['label'] and _rank(lm) < _rank(k):
            keep[keep.index(k)] = dict(lm, footprint=k['footprint'] if k['support'] > lm['support'] else lm['footprint'],
                                       xy=k['xy'] if k['support'] > lm['support'] else lm['xy'])
    out = []
    for lm in keep:
        big = max(l['support'] for l in keep if l['label'] == lm['label'])
        if lm['support'] >= 0.15 * big:
            out.append(lm)
    return out


# ---------------------------------------------------------------- relations
def _local(p, fp):
    a = np.radians(fp['angle']); c = np.array(fp['center'], float)
    ux, uy = np.array([np.cos(a), np.sin(a)]), np.array([-np.sin(a), np.cos(a)])
    d = np.asarray(p, float)[:2] - c
    return np.array([d @ ux, d @ uy]), ux, uy, c


def dist_to(p, lm) -> float:
    q, _, _, _ = _local(p, lm['footprint'])
    half = np.array(lm['footprint']['size']) / 2
    return float(np.hypot(*np.maximum(np.abs(q) - half, 0)))


def closest_point(p, lm):
    q, ux, uy, c = _local(p, lm['footprint'])
    half = np.array(lm['footprint']['size']) / 2
    q = np.clip(q, -half, half)
    return c + q[0] * ux + q[1] * uy


def _wall_dist(q, bounds):
    (x0, y0), (x1, y1) = bounds
    return min(q[0] - x0, x1 - q[0], q[1] - y0, y1 - q[1])


def at_foot_of(p, bed, bounds) -> bool:
    """Beyond the end of the bed away from the wall (beds stand with their head to a wall)."""
    q, ux, uy, c = _local(p, bed['footprint'])
    w, d = bed['footprint']['size']
    axis, along, across = (0, w, d) if w >= d else (1, d, w)
    u = ux if axis == 0 else uy
    e1, e2 = c + u * along / 2, c - u * along / 2
    sign = 1.0 if _wall_dist(e1, bounds) > _wall_dist(e2, bounds) else -1.0
    beyond = sign * q[axis] - along / 2
    return -0.15 < beyond < 0.9 and abs(q[1 - axis]) < across / 2 + 0.2


def between(p, a, b, reach: float = 0.75) -> bool:
    qa, qb = closest_point(p, a), closest_point(p, b)
    pa, pb = qa - np.asarray(p, float)[:2], qb - np.asarray(p, float)[:2]
    da, db = np.linalg.norm(pa), np.linalg.norm(pb)
    if da < 1e-3 or db < 1e-3 or da > reach or db > reach:
        return False
    return float(pa @ pb / (da * db)) < -0.5                 # on opposite sides (> 120 degrees apart)


def nice(label: str) -> str:
    return {'dining table': 'table', 'sports ball': 'ball', 'cell phone': 'phone', 'potted plant': 'plant'}.get(label, label)


def locate(p, landmarks: list, bounds, label: str = ''):
    """-> (relation, target) for a floor position p = (x, y), e.g. ('between', 'the bed and the wardrobe')."""
    p = np.asarray(p, float)[:2]
    for bed in [l for l in landmarks if l['label'] == 'bed']:
        if dist_to(p, bed) < 0.9 and at_foot_of(p, bed, bounds):
            return ('at', 'the foot of the bed, against the wall') if _wall_dist(p, bounds) < 0.35 else ('at', 'the foot of the bed')
    near = sorted(((dist_to(p, l), l) for l in landmarks), key=lambda t: t[0])
    if 'chair' in label and near:
        desks = [(d, l) for d, l in near if l['label'] in ('desk', 'table') and d < 0.45]
        if desks and desks[0][0] <= near[0][0] + 0.05:          # a chair is 'at the desk' if the desk is what it is next to
            return 'at', f"the {nice(desks[0][1]['label'])}"
    pairs = [(da + db, a, b) for i, (da, a) in enumerate(near) for db, b in near[i + 1:]
             if a['label'] != b['label'] and between(p, a, b)]
    if pairs:
        _, a, b = min(pairs, key=lambda t: t[0])
        a, b = sorted([a, b], key=_rank)              # 'the bed and the wardrobe', whichever is nearer
        return 'between', f"the {nice(a['label'])} and the {nice(b['label'])}"
    if near and near[0][0] < 0.5:
        return 'next to', f"the {nice(near[0][1]['label'])}"
    # no furniture close by: say where in the room it is
    (x0, y0), (x1, y1) = bounds
    dx, dy = min(p[0] - x0, x1 - p[0]), min(p[1] - y0, y1 - p[1])
    if max(dx, dy) < 0.6:
        return 'in', 'a corner of the room'
    if min(dx, dy) < 0.3:
        return 'against', 'the wall'
    if dx > (x1 - x0) / 3 and dy > (y1 - y0) / 3:
        return 'in', 'the middle of the room'
    return 'in', 'the open floor area'


def nearest_wall(p, bounds) -> int:
    (x0, y0), (x1, y1) = bounds
    return int(np.argmin([p[0] - x0, x1 - p[0], p[1] - y0, y1 - p[1]]))


def where(p, landmarks: list, bounds, label: str = '') -> str:
    rel, target = locate(p, landmarks, bounds, label)
    return f'{rel} {target}'


def where_from_to(p, landmarks: list, bounds, label: str = '') -> str:
    """Form that reads after 'from' / 'to': 'the desk', 'between the bed and the wardrobe', 'beside the bed'."""
    rel, target = locate(p, landmarks, bounds, label)
    return {'next to': f'beside {target}', 'between': f'between {target}'}.get(rel, target)


def relations(p, z_bottom: float, landmarks: list, bounds, label: str = '') -> list:
    """Scene-graph edges for an object: its main spatial relation plus what it stands on."""
    rel, target = locate(p, landmarks, bounds, label)
    out = [dict(relation=rel, target=target.replace('the ', ''))]
    out.append(dict(relation='on', target='floor' if z_bottom < 0.2 else 'furniture'))
    near = sorted(((dist_to(p, l), l['label']) for l in landmarks), key=lambda t: t[0])[:2]
    out += [dict(relation='near', target=lab, distance_m=round(d, 2)) for d, lab in near if d < 1.0]
    return out
