"""Stage 3: a metric, room-aligned scene: cameras in the room frame and a cleaned depth map per keyframe."""
from __future__ import annotations
import json, os, time
import numpy as np
import cv2

from .geometry import load_colmap, sparse_observations, estimate_room_frame, View
from .depth import DepthNet, fit_field, consistency_filter, pick_neighbours, lens_correction

WORK_LONG = 640            # depth maps: long side in pixels


def work_size(w: int, h: int) -> tuple[int, int]:
    s = WORK_LONG / max(w, h)
    return int(round(h * s)), int(round(w * s))          # (H, W)


def build_scene(sfm_path: str, image_dir: str, out_dir: str, depth_model: str | None = None, log=print):
    os.makedirs(out_dir, exist_ok=True)
    views, xyz, rgb, ids, err, tl = load_colmap(sfm_path)
    keep = (tl >= 3) & (err < 1.5)
    rf = estimate_room_frame(views, xyz, keep)
    obs = sparse_observations(sfm_path, ids)
    names = sorted(views)
    net = DepthNet(depth_model)
    log(f'  depth network on {net.device}; {len(names)} keyframes')
    cache = os.path.join(out_dir, 'pred_cache'); os.makedirs(cache, exist_ok=True)     # resumable
    preds, sparse, ratios = {}, {}, []
    t0 = time.time()
    for i, n in enumerate(names):
        v1 = rf.view(views[n])
        H, W = work_size(v1.width, v1.height)
        cp = os.path.join(cache, n.replace('/', '__').replace('.jpg', '.npy'))
        if os.path.exists(cp):
            preds[n] = np.load(cp).astype(np.float32)
        else:
            preds[n] = net(cv2.imread(os.path.join(image_dir, n)), (H, W))
            np.save(cp, preds[n].astype(np.float16))
        uv, idx = obs[n]
        sel = keep[idx]; uv, idx = uv[sel], idx[sel]
        z = (rf.apply(xyz[idx]) @ v1.R.T + v1.t)[:, 2]
        uvs = uv * np.array([W / v1.width, H / v1.height], np.float32)
        sparse[n] = (uvs, z)
        if len(z) > 20:
            ui = np.clip(np.round(uvs[:, 0]).astype(int), 0, W - 1); vi = np.clip(np.round(uvs[:, 1]).astype(int), 0, H - 1)
            r = preds[n][vi, ui] / np.maximum(z, 1e-6)
            ratios.append(np.median(r[(z > 0) & np.isfinite(r)]))
        if i % 40 == 0:
            log(f'    depth {i}/{len(names)}  {time.time() - t0:.0f}s')
    v0 = views[names[0]]
    hfov = float(np.degrees(2 * np.arctan(max(v0.width, v0.height) / 2 / v0.K[0, 0])))
    k_fov = lens_correction(hfov)
    S = float(np.median(ratios)) * k_fov                                   # metres per reconstruction unit
    rf.s = S
    scale_src = f'depth network, corrected for the lens ({hfov:.0f} deg vs {60} deg in training: x{k_fov:.3f})'
    log(f'  scale: {S:.4f} m per reconstruction unit; {scale_src}')
    depths, info, mviews = {}, {}, {}
    for n in names:
        v1 = rf.view(views[n]); H, W = work_size(v1.width, v1.height)
        mviews[n] = v1.scaled(W, H)
        uvs, z = sparse[n]
        D, inf = fit_field(preds[n], uvs, z * S)
        if D is not None:
            depths[n] = D; info[n] = inf
    errs = [i['rel_err'] for i in info.values()]; errs0 = [i['rel_err_scale_only'] for i in info.values()]
    log(f'  per-frame correction: median error vs the 3D points {np.median(errs0) * 100:.1f}% -> {np.median(errs) * 100:.1f}%')
    out = {}
    for ep in ('before', 'after'):
        nm = [n for n in names if mviews[n].epoch == ep and n in depths]
        out.update(consistency_filter(nm, mviews, depths, pick_neighbours(mviews, nm, k=6), rel_tol=0.05, min_agree=2))
    log(f'  multi-view consistency kept {np.mean([np.isfinite(out[n]).mean() for n in out]) * 100:.0f}% of pixels')
    # walls along x and y, from the dense points (much better than the SfM points on plain walls)
    from .depth import backproject
    from .geometry import dense_wall_alignment
    Pd = np.vstack([backproject(out[n], mviews[n], step=4)[0] for n in list(out)[::3]])
    th, oxy = dense_wall_alignment(Pd)
    Rz = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1.0]])
    shift = np.array([oxy[0], oxy[1], 0.0])
    for n, v in mviews.items():
        Rn = v.R @ Rz.T
        mviews[n] = View(v.name, v.epoch, v.frame, v.K, Rn, v.t + Rn @ shift, v.width, v.height, v.dist)
    rf.R = Rz @ rf.R; rf.o = rf.o + (rf.R.T @ shift) / rf.s
    log(f'  walls aligned with x and y from the dense points (turned {np.degrees(th):+.1f} deg)')
    np.savez_compressed(os.path.join(out_dir, 'depths.npz'), **{n.replace('/', '__'): out[n].astype(np.float16) for n in out})
    cams = {n: dict(epoch=v.epoch, frame=v.frame, K=v.K.tolist(), R=v.R.tolist(), t=v.t.tolist(), width=v.width,
                    height=v.height, img_width=views[n].width, img_height=views[n].height) for n, v in mviews.items()}
    P = rf.apply(xyz[keep])
    np.savez_compressed(os.path.join(out_dir, 'sparse.npz'), xyz=P.astype(np.float32), rgb=rgb[keep])
    json.dump(dict(cameras=cams, scale=S, scale_source=scale_src, hfov_deg=hfov, focal_px=float(v0.K[0, 0]),
                   room_R=rf.R.tolist(), room_o=rf.o.tolist(), fit=info,
                   depth_error=dict(scale_only=float(np.median(errs0)), corrected=float(np.median(errs)))),
              open(os.path.join(out_dir, 'scene.json'), 'w'))
    return out_dir


def load_scene(out_dir: str):
    """-> meta, views (depth-map resolution, metric room frame), depths (name -> HxW metres, nan = unknown),
    sparse points, colours."""
    meta = json.load(open(os.path.join(out_dir, 'scene.json')))
    views = {}
    for n, c in meta['cameras'].items():
        views[n] = View(n, c['epoch'], c['frame'], np.array(c['K']), np.array(c['R']), np.array(c['t']),
                        c['width'], c['height'])
    dz = np.load(os.path.join(out_dir, 'depths.npz'))
    depths = {k.replace('__', '/'): dz[k].astype(np.float32) for k in dz.files}
    sp = np.load(os.path.join(out_dir, 'sparse.npz'))
    return meta, views, depths, sp['xyz'], sp['rgb']


def image_view(meta, v: View) -> View:
    """The same camera at the keyframe image's full resolution (for drawing on the frames)."""
    c = meta['cameras'][v.name]
    return v.scaled(c['img_width'], c['img_height'])
