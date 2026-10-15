"""Synthetic check of the core rule: absence of evidence is not evidence of absence.

A box room is rendered from two sets of cameras. Between the two 'recordings' one cube is removed, and a new
cube hides part of the back wall. The second set of cameras never looks at the left wall. The see-through test
must call the removed cube 'gone', the hidden wall and the never-filmed wall 'unknown', and the rest 'same'. A
naive nearest-neighbour diff gets the last two wrong."""
import os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from deltanav.geometry import View
from deltanav.change import gather_evidence, naive_diff

W, H, F = 160, 120, 110.0
ROOM = (0.0, 4.0, 0.0, 3.0, 0.0, 2.5)


def look_at(C, target):
    f = np.asarray(target, float) - C; f /= np.linalg.norm(f)
    right = np.cross(f, [0, 0, 1.0]); right /= np.linalg.norm(right)
    down = np.cross(f, right)
    R = np.stack([right, down, f])
    return R, -R @ C


def cam(C, target, name, epoch):
    R, t = look_at(np.asarray(C, float), target)
    return View(name, epoch, 0, np.array([[F, 0, W / 2], [0, F, H / 2], [0, 0, 1.0]]), R, t, W, H)


def ray_box(o, d, b):
    lo = np.array([b[0], b[2], b[4]]); hi = np.array([b[1], b[3], b[5]])
    with np.errstate(divide='ignore', invalid='ignore'):
        t1 = (lo - o) / d; t2 = (hi - o) / d
    tn = np.nanmax(np.minimum(t1, t2), axis=1); tf = np.nanmin(np.maximum(t1, t2), axis=1)
    return np.where((tf >= tn) & (tf > 0), np.where(tn > 0, tn, tf), np.inf)


def render(v, boxes):
    vs, us = np.mgrid[0:H, 0:W]
    dc = np.stack([(us.ravel() - W / 2) / F, (vs.ravel() - H / 2) / F, np.ones(W * H)], 1)
    d = dc @ v.R
    o = np.repeat(v.C[None], len(d), 0)
    lo = np.array([ROOM[0], ROOM[2], ROOM[4]]); hi = np.array([ROOM[1], ROOM[3], ROOM[5]])
    with np.errstate(divide='ignore', invalid='ignore'):
        t1 = (lo - o) / d; t2 = (hi - o) / d
    t = np.nanmin(np.maximum(t1, t2), axis=1)
    for b in boxes:
        t = np.minimum(t, ray_box(o, d, b))
    return (t * dc[:, 2]).reshape(H, W).astype(np.float32)


def surface_points(views, depths, step=2):
    from deltanav.depth import backproject
    P = np.vstack([backproject(depths[v.name], v, step)[0] for v in views])
    return P[np.unique(np.floor(P / 0.05).astype(int), axis=0, return_index=True)[1]]


def build():
    gone_cube = (1.8, 2.3, 1.2, 1.7, 0.0, 0.5)
    new_cube = (1.4, 2.0, 2.2, 2.6, 0.0, 0.9)
    A = [cam((x, 0.4, 1.3), (2.0, 2.4, 0.6), f'a{i}', 'before') for i, x in enumerate(np.linspace(0.5, 3.5, 9))]
    B = [cam((x, 0.3, 1.2), (x, 3.0, 0.4), f'b{i}', 'after') for i, x in enumerate(np.linspace(2.1, 2.9, 6))]
    B += [cam((x, 0.5, 1.2), (x + 0.6, 3.0, 0.4), f'c{i}', 'after') for i, x in enumerate(np.linspace(2.2, 3.4, 6))]
    depths = {v.name: render(v, [gone_cube]) for v in A}
    depths.update({v.name: render(v, [new_cube]) for v in B})
    return A, B, depths, gone_cube, new_cube


def inside(P, b, pad=0.03):
    return ((P[:, 0] > b[0] - pad) & (P[:, 0] < b[1] + pad) & (P[:, 1] > b[2] - pad) & (P[:, 1] < b[3] + pad)
            & (P[:, 2] > b[4] - pad) & (P[:, 2] < b[5] + pad))


def regions(PA, gone_cube):
    return {'removed cube': inside(PA, gone_cube) & (PA[:, 2] > 0.05),
            'wall hidden behind the new cube': (PA[:, 1] > 2.95) & (PA[:, 0] > 1.4) & (PA[:, 0] < 2.0) & (PA[:, 2] < 0.85),
            'wall never filmed the second time': PA[:, 0] < 0.05,
            'walls seen by both': ((PA[:, 1] > 2.95) & (PA[:, 0] > 2.4)) | (PA[:, 0] > 3.95)}


def test_visibility_states():
    A, B, depths, gone_cube, _ = build()
    PA = surface_points(A, depths)
    lab = gather_evidence(PA, B, depths).classify()
    r = regions(PA, gone_cube)
    assert all(m.sum() > 10 for m in r.values())
    assert np.mean(lab[r['removed cube']] == 1) > 0.65, 'removed cube must be seen through (gone)'
    assert np.mean(lab[r['removed cube']] == 0) < 0.05
    assert np.mean(lab[r['wall hidden behind the new cube']] == 1) < 0.05, 'occluded wall must not be called gone'
    assert np.mean(lab[r['wall never filmed the second time']] == 1) == 0.0, 'unobserved wall must not be called gone'
    assert np.mean(lab[r['wall never filmed the second time']] == -1) > 0.95, 'unobserved wall must be unknown'
    assert np.mean(lab[r['walls seen by both']] == 0) > 0.8, 'unchanged walls must be the same'


def test_naive_diff_fails_where_visibility_matters():
    A, B, depths, gone_cube, _ = build()
    PA = surface_points(A, depths); PB = surface_points(B, depths)
    naive = naive_diff(PA, PB, radius=0.08)
    assert naive[regions(PA, gone_cube)['wall never filmed the second time']].mean() > 0.9


if __name__ == '__main__':
    A, B, depths, gone_cube, _ = build()
    PA = surface_points(A, depths); PB = surface_points(B, depths)
    lab = gather_evidence(PA, B, depths).classify(); nv = naive_diff(PA, PB)
    for k, m in regions(PA, gone_cube).items():
        print(f'{k:36s} gone {np.mean(lab[m] == 1):4.0%}  same {np.mean(lab[m] == 0):4.0%}  '
              f'unknown {np.mean(lab[m] == -1):4.0%}  | naive diff says changed {np.mean(nv[m]):4.0%}')
