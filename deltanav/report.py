"""Stage 6: everything a person or a robot reads from a run. Furniture and words, the ledger, the coloured 3D
surfaces, the robot maps and routes, where to look next, the figures and 3D renders, the annotated videos, the
browser viewer, the 3D window recording and the report."""
from __future__ import annotations
import datetime, json, os, pickle, time
import numpy as np

from . import __version__


def _timer(log):
    t0 = [time.time()]

    def lap(msg):
        log(f'  {msg} ({time.time() - t0[0]:.0f}s)'); t0[0] = time.time()
    return lap


def room_bounds(Z):
    P = np.vstack([Z['P_before'], Z['P_after']]); H = np.concatenate([Z['H_before'], Z['H_after']])
    band = (H > 0.3) & (H < 2.0)
    lo = np.percentile(P[band, :2], 1, axis=0); hi = np.percentile(P[band, :2], 99, axis=0)
    return (tuple(float(v) for v in lo.round(3)), tuple(float(v) for v in hi.round(3)))


def _np(o):
    """json.dump default for numpy scalars and arrays."""
    return o.item() if hasattr(o, 'item') and getattr(o, 'ndim', 0) == 0 else o.tolist()


def make_outputs(out: str, navigation: bool = True, log=print, videos: bool = True, rerun: bool = True):
    from .scene import load_scene
    from .places import lift_landmarks
    from .describe import describe
    from .detect import mask_people
    from .ledger import build_ledger
    from .mesh import build_surfaces
    from .html_report import write_report, write_markdown
    from . import viz
    res = os.path.join(out, 'result'); frames = os.path.join(out, 'frames')
    img_dir = os.path.join(res, 'images'); data_dir = os.path.join(res, 'data')
    for d in (img_dir, data_dir):
        os.makedirs(d, exist_ok=True)
    lap = _timer(log)
    meta, views, depths, sparse_xyz, _ = load_scene(os.path.join(out, 'scene'))
    R = json.load(open(os.path.join(res, 'changes.json')))
    changes = R['changes']
    Z = dict(np.load(os.path.join(res, 'data', 'points.npz')))
    floor_c = Z['floor_c']
    OI = dict(np.load(os.path.join(res, 'data', 'object_points.npz')))
    dets_p = os.path.join(out, 'detections.pkl')
    dets = pickle.load(open(dets_p, 'rb')) if os.path.exists(dets_p) else {}
    img_size = {n: (c['img_width'], c['img_height']) for n, c in meta['cameras'].items()}
    man = json.load(open(os.path.join(out, 'keyframes.json')))
    info = json.load(open(os.path.join(out, 'run_info.json'))) if os.path.exists(os.path.join(out, 'run_info.json')) else {}
    times = {f"{ep}/{k['name']}": k['t'] for ep in ('before', 'after') for k in man[ep]['keyframes']}
    mask_people(depths, dets, img_size)

    # furniture, words ----------------------------------------------------------------------------------
    landmarks = lift_landmarks(views, depths, dets, img_size, floor_c) if dets else []
    bounds = room_bounds(Z)
    sentences = describe(changes, landmarks, bounds)
    json.dump(dict(sentences=sentences, landmarks=landmarks, bounds=bounds), open(os.path.join(data_dir, 'descriptions.json'), 'w'),
              indent=1, default=_np)
    for s in sentences:
        log('  ' + s)
    lap(f"furniture: {', '.join(sorted(set(l['label'] for l in landmarks))) or 'none found'}")

    # surfaces -------------------------------------------------------------------------------------------
    key = f"{os.path.getmtime(os.path.join(out, 'scene', 'scene.json'))}:{os.path.getmtime(os.path.join(res, 'changes.json'))}"
    surf = build_surfaces(views, depths, frames, changes, floor_c, os.path.join(res, 'meshes'), log=log, cache_key=key)
    lap('3D surfaces')

    # robot ----------------------------------------------------------------------------------------------
    nav, grids, next_views = None, None, []
    if navigation:
        from .navigation import navigation as plan_nav, ROBOTS, GridSpec
        from .nbv import plan_next_views
        nav = plan_nav({ep: Z[f'P_{ep}'] for ep in ('before', 'after')}, {ep: Z[f'H_{ep}'] for ep in ('before', 'after')},
                       views, changes, landmarks, os.path.join(res, 'maps'), log=log,
                       stable={ep: Z[f'S_{ep}'] for ep in ('before', 'after')})
        grids = {}
        for rb in ROBOTS:
            G = dict(np.load(os.path.join(res, 'maps', f'{rb.name}_grids.npz')))
            G['spec'] = tuple(G['spec'].tolist()); grids[rb.name] = G
        for rb in ROBOTS:
            G = grids[rb.name]; x0, y0, nx, ny, r_ = G['spec']
            spec = GridSpec(x0, y0, int(nx), int(ny), r_)
            nv, nv_info = plan_next_views(Z['P_before'], Z['L_before'], Z['S_before'], Z['H_before'], Z['P_after'],
                                          Z['H_after'], G['trav_updated'] & G['reach_updated'], spec, rb, landmarks, bounds,
                                          k=2 if rb.name != 'humanoid' else 3)
            next_views += nv
        json.dump(next_views, open(os.path.join(data_dir, 'next_views.json'), 'w'), indent=1, default=_np)
        if surf.get('occ3d'):
            from .navigation import occupancy3d_updated, export_occupancy3d
            o3, sp3 = surf['occ3d'], surf['occ3d_spec']
            o3['updated'] = occupancy3d_updated(o3['before'], o3['after'], sp3['origin'], sp3['voxel'], changes, floor_c)
            nav['occupancy3d'] = export_occupancy3d(os.path.join(res, 'maps', 'occupancy3d'), o3, sp3['origin'], sp3['voxel'])
            nav['occupancy3d']['files'] = [os.path.relpath(f, res) for f in nav['occupancy3d']['files']]
            json.dump(nav, open(os.path.join(res, 'maps', 'routes.json'), 'w'), indent=1, default=_np)
        lap(f'robot maps, routes and {len(next_views)} next-best views')

    # ledger ---------------------------------------------------------------------------------------------
    sec = info.get('seconds', {})
    run = dict(version=__version__, date=datetime.date.today().isoformat(),
               before_video=os.path.basename(info.get('inputs', {}).get('before', 'before')),
               after_video=os.path.basename(info.get('inputs', {}).get('after', 'after')),
               frames_before=R['frames']['before'], frames_after=R['frames']['after'],
               depth_error_pct=round(100 * meta['depth_error']['corrected'], 1),
               depth_error_scale_only_pct=round(100 * meta['depth_error']['scale_only'], 1),
               focal_px=round(meta['focal_px'], 1), hfov_deg=round(meta['hfov_deg'], 1),
               hfov_source=man.get('hfov', {}).get('source', ''), scale_source=R.get('scale_source', ''),
               sfm_points=len(sparse_xyz), device=info.get('device', ''),
               total_minutes=round(sum(sec.values()) / 60) if sec else None,
               stage_seconds=', '.join(f'{k} {v}s' for k, v in sec.items()))
    ledger = build_ledger(changes, sentences, landmarks, bounds, R['coverage'], R.get('rejected', []), nav, next_views, run)
    json.dump(ledger, open(os.path.join(res, 'ledger.json'), 'w'), indent=1, default=_np)
    lap('ledger.json')

    # pictures -------------------------------------------------------------------------------------------
    images = {}
    lo, hi = surf['bounds']
    cams = np.array([v.C for v in views.values()])
    V, F, C = viz.ceiling_cut(*surf['shared'], floor_c)
    obj_parts = [o for o in surf['objects'].values() if len(o[1])]
    if obj_parts:
        Vs = [V] + [o[0] for o in obj_parts]; off = np.cumsum([0] + [len(v) for v in Vs[:-1]])
        V = np.vstack(Vs); F = np.vstack([F] + [o[1] + k for o, k in zip(obj_parts, off[1:])]); C = np.vstack([C] + [o[2] for o in obj_parts])
    N = viz.shading_normals(V, F, cams)
    sp = 1.2 * viz.edge_len(V, F)
    bg, ext = viz.topdown(V, C, N, lo, hi, spacing=sp)
    unk = Z['P_before'][(Z['L_before'] == -1) & Z['S_before']][:, :2]
    unk = unk[np.random.default_rng(0).choice(len(unk), min(len(unk), 40000), replace=False)] if len(unk) else unk
    traj = {ep: np.array([views[n].C for n in sorted(views, key=lambda n: views[n].frame) if views[n].epoch == ep])
            for ep in ('before', 'after')}
    start = nav['robots']['humanoid']['start'] if nav and nav['robots'].get('humanoid', {}).get('start') else None
    images['floor_plan'] = os.path.join(img_dir, 'floor_plan.png')
    viz.floor_plan(images['floor_plan'], bg, ext, landmarks, changes, unk, traj, next_views, start)
    obj_pts = [o[0] for o in obj_parts]
    cam = viz.best_dollhouse(V, C, N, changes, floor_c, lo, hi, obj_pts)
    import cv2
    img, _ = viz.render_ss(V, C, N, cam, spacing=sp)
    img = viz.change_overlay(img, cam, changes, floor_c)
    images['3d_overview'] = os.path.join(img_dir, '3d_overview.jpg')
    cv2.imwrite(images['3d_overview'], cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
    for ep in ('before', 'after'):
        Ve, Fe, Ce = viz.ceiling_cut(*surf[ep], floor_c)
        Ne = viz.shading_normals(Ve, Fe, cams)
        img, _ = viz.render_ss(Ve, Ce, Ne, cam, spacing=1.2 * viz.edge_len(Ve, Fe))
        img = viz.change_overlay(img, cam, changes, floor_c, which=(ep,))
        images[f'3d_{ep}'] = os.path.join(img_dir, f'3d_{ep}.jpg')
        cv2.imwrite(images[f'3d_{ep}'], cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
    if nav:
        images['robot_routes'] = os.path.join(img_dir, 'robot_routes.png')
        viz.robot_figure(images['robot_routes'], nav, grids, changes, landmarks)
    from .objects import cluster_points
    naive = sum(len(cluster_points(Z[f'P_{ep}'], Z[f'naive_{ep}'], ep, heights=Z[f'H_{ep}'])) for ep in ('before', 'after'))
    images['naive_vs_ours'] = os.path.join(img_dir, 'naive_vs_ours.png')
    viz.naive_vs_ours(images['naive_vs_ours'], Z, naive, len(changes))
    images['coverage'] = os.path.join(img_dir, 'coverage.png')
    viz.coverage_chart(images['coverage'], R['coverage'])
    lap('floor plan, 3D renders, robot and evaluation figures')

    # frames and videos ----------------------------------------------------------------------------------
    from .annotate import Annotator, pair_image, annotated_videos, places
    probes = {}
    rng = np.random.default_rng(0)
    for c in changes:
        for o, ep, role in places(c):
            idx = OI.get(f"obj{o['id']}")
            P = Z[f'P_{ep}'][idx] if idx is not None and len(idx) else np.zeros((0, 3))
            probes[(c['id'], role)] = P[rng.choice(len(P), min(len(P), 400), replace=False)] if len(P) else P
    ann = Annotator(meta, views, depths, changes, floor_c, probes, frames, times)
    for k, (c, s) in enumerate(zip(changes, sentences)):
        p = os.path.join(img_dir, f"change_{k + 1}_{c['type']}_{c['label'].replace(' ', '_')}.jpg")
        images[f"change_{c['id']}"] = pair_image(ann, c, s, p)
    if videos:
        annotated_videos(ann, os.path.join(res, 'video'), log=log)
    lap('annotated frames and videos')

    # viewers --------------------------------------------------------------------------------------------
    from .viewer import build_viewer
    sub = f"{run['before_video']} → {run['after_video']} · {len(changes)} change{'s' if len(changes) != 1 else ''}"
    notes = (('Solid box: the object is there in that video. Dashed: its place is empty in that video. ' if changes else '')
             + f"{100 * R['coverage']['before']['unknown']:.0f}% of the before surfaces could not be re-checked (purple layer).")
    tgt = np.r_[(np.asarray(lo[:2]) + np.asarray(hi[:2])) / 2, 0.35]
    home = dict(pos=(tgt + (cam.pos - tgt) * 0.95).round(3).tolist(), target=tgt.round(3).tolist())   # the overview's corner
    build_viewer(os.path.join(res, 'viewer.html'), surf, changes, sentences, Z, floor_c, views, nav, grids, next_views,
                 landmarks, subtitle=sub, notes=notes, home=home)
    lap('viewer.html')
    if rerun:
        try:
            from .rerun_export import export_rerun
            from .mesh import decimate
            sr = dict(surf)
            for k in ('shared', 'before', 'after'):
                sr[k] = decimate(*surf[k], 400000 if k == 'shared' else 250000)
            md = '# What changed\n\n' + '\n'.join(f"{k + 1}. **{c['type']}** {s}" for k, (c, s) in enumerate(zip(changes, sentences)))
            if nav:
                md += '\n\n## Robots\n\n' + '\n'.join(f"- **{n}**: " + ' '.join(v) for n, v in nav['sentences'].items())
            md += ('\n\n## Layers\n\nSwitch layers on and off in the left panel (eye icon): `world/room/shared`, '
                   '`world/room/before`, `world/room/after`, `world/changes`, `world/could_not_check`, `world/evidence`, '
                   '`world/robot`, `world/next_views`, `world/cameras`. Scrub the timeline to walk through both videos.')
            export_rerun(os.path.join(res, 'scene.rrd'), sr, changes, sentences, Z, floor_c, views, meta, ann, nav, grids,
                         next_views, landmarks, md, image_dir=frames, log=log, jpeg_width=800,
                         occ3=(surf['occ3d'].get('updated'), surf['occ3d_spec']) if surf.get('occ3d') else None)
            lap('scene.rrd (3D window)')
        except Exception as e:                       # the rest of the outputs do not depend on it
            log(f'  could not write the Rerun recording: {e}')
    write_report(res, ledger, nav, images, info)
    write_markdown(res, ledger, nav, images)
    lap('report.html, report.md')
    out_s = list(sentences)
    if nav:
        for name, s in nav['sentences'].items():
            out_s += [f'{name}: {x}' for x in s if 'reach' not in x or 'Floor' not in x][:2]
    return out_s
