"""Coloured 3D surfaces of the room by TSDF fusion (truncated signed distance) of the metric depth maps, then
marching cubes. Three surfaces: one per recording, and a *shared* one built from both recordings with the
changed objects cut out (the part of the room both videos agree on), plus a small mesh of every changed object
in each of its versions. Runs on the GPU when there is one."""
from __future__ import annotations
import os, time
import numpy as np
import cv2




def _torch():
    import torch
    return torch, ('cuda' if torch.cuda.is_available() else 'cpu')


def fuse_tsdf(items, lo, hi, voxel: float = 0.015, trunc: float = 0.05, rel_trunc: float = 0.04, log=print):
    """items: iterable of (View at depth resolution, depth HxW metres (nan = skip), colour image HxWx3 RGB).
    Returns tsdf, weight and colour grids (numpy) over the box lo..hi."""
    torch, dev = _torch()
    dims = np.ceil((np.asarray(hi) - np.asarray(lo)) / voxel).astype(int)
    nx, ny, nz = map(int, dims)
    gx = torch.arange(nx, device=dev, dtype=torch.float32) * voxel + lo[0] + voxel / 2
    gy = torch.arange(ny, device=dev, dtype=torch.float32) * voxel + lo[1] + voxel / 2
    gz = torch.arange(nz, device=dev, dtype=torch.float32) * voxel + lo[2] + voxel / 2
    X, Y, Z = torch.meshgrid(gx, gy, gz, indexing='ij')
    pts = torch.stack([X.reshape(-1), Y.reshape(-1), Z.reshape(-1)], 1)
    del X, Y, Z
    N = pts.shape[0]
    tsdf = torch.ones(N, device=dev); wt = torch.zeros(N, device=dev); col = torch.zeros(N, 3, device=dev)
    cw_sum = torch.zeros(N, device=dev); nfree = torch.zeros(N, device=dev)
    t0 = time.time()
    for k, (v, D, rgb) in enumerate(items):
        H, W = D.shape
        R = torch.tensor(v.R, device=dev, dtype=torch.float32); t = torch.tensor(v.t, device=dev, dtype=torch.float32)
        Dt = torch.tensor(np.nan_to_num(D, nan=0.0), device=dev, dtype=torch.float32)
        It = torch.tensor(rgb, device=dev, dtype=torch.float32)
        for s in range(0, N, 4_000_000):
            P = pts[s:s + 4_000_000]
            Xc = P @ R.T + t
            z = Xc[:, 2]
            front = z > 0.15
            u = torch.round(v.K[0, 0] * Xc[:, 0] / z.clamp(min=1e-3) + v.K[0, 2]).long()
            w_ = torch.round(v.K[1, 1] * Xc[:, 1] / z.clamp(min=1e-3) + v.K[1, 2]).long()
            ok = front & (u >= 0) & (u < W) & (w_ >= 0) & (w_ < H)
            idx = torch.nonzero(ok).squeeze(1)
            if idx.numel() == 0:
                continue
            d = Dt[w_[idx], u[idx]]
            sdf = d - z[idx]
            tr = (rel_trunc * d).clamp(min=trunc)        # monocular depth errs in proportion to distance
            ahead = (d > 0) & (sdf >= tr)                  # seen through: free space (counted, not fused)
            if ahead.any():
                nfree[idx[ahead] + s] += 1.0
            m = (d > 0) & (sdf > -tr) & (sdf < tr)        # only the band around the surface is fused: carving
            idx, sdf, d, tr = idx[m], sdf[m], d[m], tr[m]  # with frames that over-estimate depth would put a
            g = idx + s                                    # second, outward-facing skin behind every wall
            wn = 1.0 / d.clamp(min=0.5)
            val = (sdf / tr).clamp(-1.0, 1.0)
            wo = wt[g]
            tsdf[g] = (tsdf[g] * wo + val * wn) / (wo + wn)
            wt[g] = wo + wn
            near = sdf.abs() < tr
            if near.any():                            # colour: near views dominate (sharper texture)
                gn = g[near]; cn = It[w_[idx[near]], u[idx[near]]]
                wc = wn[near] ** 2
                col[gn] += cn * wc[:, None]; cw_sum[gn] += wc
        if k % 60 == 0:
            log(f'    surface fusion {k}: {time.time() - t0:.0f}s')
    shape = (nx, ny, nz)
    cw = col / cw_sum.clamp(min=1e-9)[:, None]
    return (tsdf.reshape(shape).cpu().numpy(), wt.reshape(shape).cpu().numpy(),
            cw.reshape(nx, ny, nz, 3).clamp(0, 255).cpu().numpy().astype(np.uint8),
            nfree.reshape(shape).cpu().numpy())


def occupancy3d(tsdf, wt, nfree, factor: int = 3, min_weight: float = 1.0, min_free: int = 2):
    """3D occupancy at `factor` x the TSDF voxel: 1 occupied (a surface), 0 free (seen through by at least
    `min_free` frames), -1 unknown (never observed, or inside solid things). A voxel is free only if most of it
    was seen through and no surface is in it."""
    f = factor
    nx, ny, nz = (np.array(tsdf.shape) // f) * f
    t = tsdf[:nx, :ny, :nz]; w = wt[:nx, :ny, :nz]
    obs = w >= min_weight
    occ = obs & (np.abs(t) < 0.5)
    free = (nfree[:nx, :ny, :nz] >= min_free) & ~occ
    blk = lambda m: m.reshape(nx // f, f, ny // f, f, nz // f, f)
    out = np.full((nx // f, ny // f, nz // f), -1, np.int8)
    out[blk(free).sum((1, 3, 5)) >= 0.5 * f ** 3] = 0
    out[blk(occ).sum((1, 3, 5)) >= 2] = 1
    return out


def extract_mesh(tsdf, wt, col, lo, voxel, min_weight: float = 1.5):
    from skimage.measure import marching_cubes
    obs = wt >= min_weight
    if obs.sum() < 1000:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int32), np.zeros((0, 3), np.uint8)
    # a cube is meshed only if all 8 of its corners were observed; otherwise the edge of the observed band
    # behind every wall would become a second, outward-facing surface
    valid = obs.copy()
    for ax in range(3):
        sl_a = [slice(None)] * 3; sl_b = [slice(None)] * 3; sl_last = [slice(None)] * 3
        sl_a[ax] = slice(0, -1); sl_b[ax] = slice(1, None); sl_last[ax] = -1
        valid[tuple(sl_a)] &= valid[tuple(sl_b)]
        valid[tuple(sl_last)] = False
    vol = np.where(obs, tsdf, 1.0).astype(np.float32)
    V, F, _, _ = marching_cubes(vol, level=0.0, mask=valid, allow_degenerate=False)
    ii = np.clip(np.round(V).astype(int), 0, np.array(tsdf.shape) - 1)
    C = col[ii[:, 0], ii[:, 1], ii[:, 2]]
    V = (np.asarray(lo) + (V + 0.5) * voxel).astype(np.float32)
    return V, F.astype(np.int32), C


def clean_mesh(V, F, C, min_faces: int = 400):
    """Drop small floating pieces (connected components with fewer than `min_faces` triangles)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    if len(F) == 0:
        return V, F, C
    n = len(V)
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    A = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n))
    _, lab = connected_components(A, directed=False)
    fl = lab[F[:, 0]]
    counts = np.bincount(fl, minlength=lab.max() + 1)
    keepF = counts[fl] >= min_faces
    return _compact(V, F[keepF], C)


def taubin(V, F, iters: int = 6, lam: float = 0.5, mu: float = -0.53):
    """Taubin smoothing: removes the small bumps monocular depth leaves on flat surfaces without shrinking them."""
    from scipy.sparse import coo_matrix
    if len(F) == 0:
        return V
    n = len(V)
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]); e = np.vstack([e, e[:, ::-1]])
    A = coo_matrix((np.ones(len(e), np.float32), (e[:, 0], e[:, 1])), shape=(n, n)).tocsr()
    A.data[:] = 1.0
    dinv = (1.0 / np.maximum(np.asarray(A.sum(1)).ravel(), 1))[:, None]
    X = V.astype(np.float64)
    for _ in range(iters):
        for f in (lam, mu):
            X = X + f * (dinv * (A @ X) - X)
    return X.astype(np.float32)


def orient_faces(V, F, free_points):
    """Wind every triangle so that its front faces free space (where the cameras were). Viewers that cull back
    faces then show the room like a doll's house from outside: the near walls disappear, the far ones stay."""
    from scipy.spatial import cKDTree
    if len(F) == 0:
        return F
    k = np.random.default_rng(0).choice(len(F), min(len(F), 20000), replace=False)
    a, b, c = V[F[k, 0]], V[F[k, 1]], V[F[k, 2]]
    n = np.cross(b - a, c - a); m = (a + b + c) / 3
    _, j = cKDTree(free_points).query(m)
    if np.mean(np.sum(n * (free_points[j] - m), axis=1) > 0) < 0.5:
        F = F[:, [0, 2, 1]]
    return np.ascontiguousarray(F)


def drop_unseen_side(V, F, C, cams, margin: float = 0.05):
    """Remove surface that faces away from every camera. Frames whose depth disagrees put a second skin
    behind a wall, facing outwards; no camera was ever in front of it (they were all in the room), so it cannot
    have been seen. Needs faces oriented towards free space."""
    if len(F) == 0:
        return V, F, C
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    N = np.zeros_like(V, dtype=np.float64)
    for k in range(3):
        np.add.at(N, F[:, k], fn)
    N /= np.linalg.norm(N, axis=1, keepdims=True) + 1e-12
    off = np.sum(N * V, axis=1)
    seen = np.zeros(len(V), bool)
    for s in range(0, len(V), 200000):
        seen[s:s + 200000] = ((N[s:s + 200000] @ cams.T - off[s:s + 200000, None]) > margin).any(1)
    keepF = seen[F].all(1)
    return _compact(V, F[keepF], C) if keepF.any() else (V[:0], F[:0], C[:0])


def _compact(V, F, C):
    used = np.unique(F)
    remap = -np.ones(len(V), np.int64); remap[used] = np.arange(len(used))
    return V[used], remap[F].astype(np.int32), C[used]


def crop_mesh(V, F, C, keep_vertex: np.ndarray):
    keepF = keep_vertex[F].all(1)
    if not keepF.any():
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int32), np.zeros((0, 3), np.uint8)
    return _compact(V, F[keepF], C)


def decimate(V, F, C, target_faces: int):
    """Quadric decimation (Open3D) down to about `target_faces` triangles; vertex clustering if Open3D is missing."""
    if len(F) <= target_faces:
        return V, F, C
    try:                                              # quadric decimation, small pip wheel
        import fast_simplification as fs
        _, _, coll = fs.simplify(V.astype(np.float32), F.astype(np.int32), target_reduction=1 - target_faces / len(F),
                                 return_collapses=True)
        Vn, Fn, mapping = fs.replay_simplification(V.astype(np.float32), F.astype(np.int32), coll)
        cnt = np.maximum(np.bincount(mapping, minlength=len(Vn)), 1)
        Cn = np.stack([np.bincount(mapping, C[:, k].astype(float), minlength=len(Vn)) / cnt for k in range(3)], 1)
        return Vn.astype(np.float32), Fn.astype(np.int32), Cn.clip(0, 255).astype(np.uint8)
    except Exception:
        pass
    try:
        import open3d as o3d
        m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V.astype(np.float64)),
                                      o3d.utility.Vector3iVector(F.astype(np.int32)))
        m.vertex_colors = o3d.utility.Vector3dVector(C.astype(np.float64) / 255.0)
        m = m.simplify_quadric_decimation(target_number_of_triangles=int(target_faces))
        m.remove_degenerate_triangles(); m.remove_unreferenced_vertices()
        return (np.asarray(m.vertices, np.float32), np.asarray(m.triangles, np.int32),
                (np.asarray(m.vertex_colors) * 255).clip(0, 255).astype(np.uint8))
    except Exception:
        cell = 0.02 * np.sqrt(len(F) / target_faces)
        key = np.floor(V / cell).astype(np.int64)
        _, inv = np.unique(key, axis=0, return_inverse=True); inv = inv.ravel()
        cnt = np.bincount(inv)
        Vn = np.stack([np.bincount(inv, V[:, k]) / cnt for k in range(3)], 1).astype(np.float32)
        Cn = np.stack([np.bincount(inv, C[:, k].astype(float)) / cnt for k in range(3)], 1).astype(np.uint8)
        Fn = inv[F]
        Fn = Fn[(Fn[:, 0] != Fn[:, 1]) & (Fn[:, 1] != Fn[:, 2]) & (Fn[:, 0] != Fn[:, 2])]
        return Vn, Fn.astype(np.int32), Cn


def write_ply(path: str, V, F, C):
    """Binary PLY with vertex colours (opens in MeshLab, Blender, CloudCompare)."""
    vert = np.zeros(len(V), dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    vert['x'], vert['y'], vert['z'] = V[:, 0], V[:, 1], V[:, 2]
    vert['red'], vert['green'], vert['blue'] = C[:, 0], C[:, 1], C[:, 2]
    face = np.zeros(len(F), dtype=[('n', 'u1'), ('a', '<i4'), ('b', '<i4'), ('c', '<i4')])
    face['n'] = 3; face['a'], face['b'], face['c'] = F[:, 0], F[:, 1], F[:, 2]
    with open(path, 'wb') as f:
        f.write((f'ply\nformat binary_little_endian 1.0\ncomment deltanav room frame (metres, z up)\n'
                 f'element vertex {len(V)}\nproperty float x\nproperty float y\nproperty float z\n'
                 f'property uchar red\nproperty uchar green\nproperty uchar blue\n'
                 f'element face {len(F)}\nproperty list uchar int vertex_indices\nend_header\n').encode())
        f.write(vert.tobytes()); f.write(face.tobytes())


def write_points_ply(path: str, P, C):
    vert = np.zeros(len(P), dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    vert['x'], vert['y'], vert['z'] = P[:, 0], P[:, 1], P[:, 2]
    vert['red'], vert['green'], vert['blue'] = C[:, 0], C[:, 1], C[:, 2]
    with open(path, 'wb') as f:
        f.write((f'ply\nformat binary_little_endian 1.0\nelement vertex {len(P)}\nproperty float x\nproperty float y\n'
                 f'property float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n').encode())
        f.write(vert.tobytes())


# ---------------------------------------------------------------- the three room surfaces
def object_box(o, pad: float = 0.04):
    """Oriented box of a change object: (center xy, half sizes xy, angle rad, z0, z1)."""
    fp = o['footprint']
    return (np.array(fp['center']), np.array(fp['size']) / 2 + pad, np.radians(fp['angle']),
            max(0.0, o['z_range'][0]) - pad, o['z_range'][1] + pad)


def in_box(P, box, Hh):
    c, half, a, z0, z1 = box
    ux, uy = np.array([np.cos(a), np.sin(a)]), np.array([-np.sin(a), np.cos(a)])
    d = P[:, :2] - c
    return (np.abs(d @ ux) <= half[0]) & (np.abs(d @ uy) <= half[1]) & (Hh >= z0) & (Hh <= z1)


def build_surfaces(views: dict, depths: dict, image_dir: str, changes: list, floor_c, out_dir: str,
                   voxel: float = 0.015, trunc: float = 0.05, log=print, cache_key: str = ''):
    """Writes meshes/{shared,before,after}.ply and one PLY per changed object version; returns the meshes.
    Cached in meshes/surfaces.npz (reused while `cache_key` is unchanged)."""
    import json, pickle
    from .depth import backproject
    from .geometry import height_above_floor
    os.makedirs(out_dir, exist_ok=True)
    cache = os.path.join(out_dir, 'surfaces.pkl')
    cache_id = json.dumps(dict(k=cache_key, voxel=voxel, trunc=trunc, ch=[(c['id'], c['type']) for c in changes]))
    if os.path.exists(cache):
        try:
            c = pickle.load(open(cache, 'rb'))
            if c.get('key') == cache_id:
                log('  3D surfaces: reusing the cached ones')
                return c['out']
        except Exception:
            pass
    rgb_cache = {}

    def rgb(n, H, W):
        if n not in rgb_cache:
            img = cv2.imread(os.path.join(image_dir, n))
            rgb_cache[n] = cv2.cvtColor(cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        return rgb_cache[n]
    # bounds from the depth points
    pts = []
    for n, D in list(depths.items())[::4]:
        X, _ = backproject(D, views[n], step=6)
        pts.append(X)
    pts = np.vstack(pts)
    lo = np.percentile(pts, 0.5, axis=0) - 0.1; hi = np.percentile(pts, 99.5, axis=0) + 0.1
    boxes = {'before': [], 'after': []}
    for c in changes:
        if c['type'] == 'moved':
            boxes['before'].append(object_box(c['from_obj'])); boxes['after'].append(object_box(c['to_obj']))
        else:
            boxes['before' if c['type'] == 'removed' else 'after'].append(object_box(c['obj']))

    def cut(n):
        """Depth map with the changed objects of this frame's recording removed (for the shared surface)."""
        D = depths[n]; v = views[n]
        bx = boxes[v.epoch]
        if not bx:
            return D
        X, (vi, ui) = backproject(D, v)
        Hh = height_above_floor(X, floor_c)
        m = np.zeros(len(X), bool)
        for b in bx:
            m |= in_box(X, b, Hh)
        Dc = D.copy(); Dc[vi[m], ui[m]] = np.nan
        return Dc
    out = {}
    cams = np.array([v.C for v in views.values()])
    plans = [('before', [n for n in sorted(depths) if views[n].epoch == 'before'], False),
             ('after', [n for n in sorted(depths) if views[n].epoch == 'after'], False),
             ('shared', sorted(depths), True)]
    for tag, names, masked in plans:
        t0 = time.time()
        items = ((views[n], cut(n) if masked else depths[n], rgb(n, *depths[n].shape)) for n in names)
        tsdf, wt, col, nfree = fuse_tsdf(items, lo, hi, voxel, trunc, log=log)
        if not masked:
            out.setdefault('occ3d', {})[tag] = occupancy3d(tsdf, wt, nfree)
        V, F, C = extract_mesh(tsdf, wt, col, lo, voxel, min_weight=1.5 if not masked else 2.0)
        del tsdf, wt, col, nfree
        V, F, C = clean_mesh(V, F, C)
        V = taubin(V, F)
        F = orient_faces(V, F, cams)
        V, F, C = clean_mesh(*drop_unseen_side(V, F, C, cams), min_faces=1500)
        write_ply(os.path.join(out_dir, f'{tag}.ply'), V, F, C)
        out[tag] = (V, F, C)
        log(f'  {tag} surface: {len(F):,} triangles ({time.time() - t0:.0f}s)')
    # each changed object, in each of its versions, cut from its recording's surface
    objs = {}
    for c in changes:
        parts = [('before', c['from_obj']), ('after', c['to_obj'])] if c['type'] == 'moved' else \
                [('before' if c['type'] == 'removed' else 'after', c['obj'])]
        for ep, o in parts:
            V, F, C = out[ep]
            keep = in_box(V, object_box(o, pad=0.03), height_above_floor(V, floor_c))
            Vo, Fo, Co = crop_mesh(V, F, C, keep)
            key = f"{c['id']}_{c['type']}_{c['label'].replace(' ', '_')}_{ep}"
            write_ply(os.path.join(out_dir, f'{key}.ply'), Vo, Fo, Co)
            objs[(c['id'], ep)] = (Vo, Fo, Co)
    out['objects'] = objs
    out['bounds'] = (lo, hi)
    out['occ3d_spec'] = dict(origin=np.asarray(lo, float).tolist(), voxel=voxel * 3)
    pickle.dump(dict(key=cache_id, out=out), open(cache + '.tmp', 'wb')); os.replace(cache + '.tmp', cache)
    return out
