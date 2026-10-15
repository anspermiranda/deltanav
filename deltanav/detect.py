"""Stage 5: fuse each recording, run the see-through test both ways, and turn the result into verified changes."""
from __future__ import annotations
import json, os, pickle, time
import numpy as np

from .scene import load_scene
from .fusion import fuse_epoch, voxel_downsample
from .geometry import fit_floor_surface, height_above_floor
from .change import gather_evidence, naive_diff, witness_rays, object_verdict
from .depth import edge_mask
from .objects import cluster_points, name_objects, match_moves, estimate_yaw, feature_yaw

PARAMS = dict(tau0=0.06, tau_rel=0.06, min_free=3, free_ratio=0.7, free_frac_fov=0.25, self_free_ratio=0.25, eps=0.06,
              min_points=150, min_views=5, max_surface_share=0.1, voxel=0.025)
TYPE_ORDER = {'moved': 0, 'removed': 1, 'added': 2}


def mask_people(depths: dict, dets: dict, img_size: dict, labels=('person', 'dog', 'cat')) -> int:
    """The person holding the phone sometimes appears (a hand, a foot, a reflection), and people and pets move
    between the two recordings without being a change to the room. Their pixels are removed from the depth maps
    before fusion and the visibility tests."""
    import cv2
    n = 0
    for name, D in depths.items():
        H, W = D.shape
        sx, sy = W / img_size[name][0], H / img_size[name][1]
        hit = False
        for d in dets.get(name, []):
            if d['label'] in labels and d['conf'] > 0.3 and d['poly'] is not None:
                m = np.zeros((H, W), np.uint8)
                cv2.fillPoly(m, [(d['poly'] * np.array([sx, sy], np.float32)).astype(np.int32)], 1)
                D[cv2.dilate(m, np.ones((9, 9), np.uint8)) > 0] = np.nan
                hit = True
        n += hit
    return n


def split_passes(views: dict, epoch: str):
    """Split one video's frames into its two laps (chest height, then waist height) at the best two-segment
    change point of the camera height."""
    nm = sorted([n for n, v in views.items() if v.epoch == epoch], key=lambda n: views[n].frame)
    h = np.array([views[n].C[2] for n in nm])
    m = max(5, int(0.2 * len(h)))
    best = (np.inf, None)
    for k in range(m, len(h) - m):
        cost = ((h[:k] - h[:k].mean()) ** 2).sum() + ((h[k:] - h[k:].mean()) ** 2).sum()
        if cost < best[0]:
            best = (cost, k)
    return nm[:best[1]], nm[best[1]:]


def _sigma(o, views, depth_err, voxel):
    """1-sigma position uncertainty: depth error at the typical viewing distance, plus voxel quantisation."""
    c = np.array(o.centroid)
    d = np.median([np.linalg.norm(v.C - c) for v in views]) if views else 2.0
    return float(np.hypot(depth_err * d, voxel / np.sqrt(12)))


def detect_changes(scene_dir: str, image_dir: str, dets_path: str | None, out_dir: str, log=print,
                   params: dict | None = None, groups: tuple | None = None):
    p = dict(PARAMS); p.update(params or {})
    os.makedirs(os.path.join(out_dir, 'data'), exist_ok=True)
    meta, views, depths, _, _ = load_scene(scene_dir)
    img_size = {n: (c['img_width'], c['img_height']) for n, c in meta['cameras'].items()}
    if groups is not None:                          # custom comparison, e.g. chest lap vs waist lap
        ga, gb = set(groups[0]), set(groups[1])
        for n in list(views):
            if n in ga:
                views[n].epoch = 'before'
            elif n in gb:
                views[n].epoch = 'after'
            else:
                views.pop(n); depths.pop(n, None)
    names = sorted(views)
    dets = pickle.load(open(dets_path, 'rb')) if dets_path and os.path.exists(dets_path) else None
    if dets is not None:
        log(f'  masked the camera operator in {mask_people(depths, dets, img_size)} frames')
    by_ep = {ep: [n for n in names if views[n].epoch == ep and n in depths] for ep in ('before', 'after')}
    t0 = time.time()
    P, C = {}, {}
    for ep in ('before', 'after'):
        Pf, Cf, _ = fuse_epoch(by_ep[ep], views, depths, image_dir, voxel=0.02)
        P[ep], C[ep] = voxel_downsample(Pf, Cf, p['voxel'])
        log(f'  {ep}: {len(by_ep[ep])} frames fused into {len(P[ep])} surface points ({p["voxel"] * 100:.1f} cm)')
    floor_c = fit_floor_surface(np.vstack([P['before'], P['after']]))
    Hgt = {ep: height_above_floor(P[ep], floor_c) for ep in P}
    edges = {n: edge_mask(depths[n]) for n in depths}
    ev, lab, stable = {}, {}, {}
    for ep, other in (('before', 'after'), ('after', 'before')):
        own = [views[n] for n in by_ep[ep]]
        se = gather_evidence(P[ep], own, depths, edges, tau0=p['tau0'], tau_rel=p['tau_rel'])
        stable[ep] = (se.n_same >= 3) & (se.n_free <= p['self_free_ratio'] * np.maximum(se.n_free + se.n_same, 1))
        ov = [views[n] for n in by_ep[other]]
        ev[ep] = gather_evidence(P[ep], ov, depths, edges, tau0=p['tau0'], tau_rel=p['tau_rel'])
        lab[ep] = ev[ep].classify(min_free=p['min_free'], free_ratio=p['free_ratio'], free_frac_fov=p['free_frac_fov'])
        lab[ep][~stable[ep]] = -1
        log(f'  {ep} surfaces vs the {other} video: {np.mean(lab[ep] == 0) * 100:.0f}% same, '
            f'{np.mean(lab[ep] == 1) * 100:.1f}% seen through, {np.mean(lab[ep] == -1) * 100:.0f}% could not be checked')
    coverage = {}
    for ep in ('before', 'after'):
        e, L = ev[ep], lab[ep]
        unstable = ~stable[ep]
        unseen = ~unstable & (e.n_fov == 0)
        occluded = ~unstable & ~unseen & (L == -1) & (e.n_occl >= e.n_same + e.n_free)
        coverage[ep] = dict(points=int(len(L)), same=round(float(np.mean(L == 0)), 4), gone=round(float(np.mean(L == 1)), 4),
                            unknown=round(float(np.mean(L == -1)), 4),
                            unknown_self_inconsistent=round(float(np.mean(unstable)), 4),
                            unknown_never_in_view=round(float(np.mean(unseen)), 4),
                            unknown_occluded=round(float(np.mean(occluded)), 4),
                            unknown_too_few_views=round(float(np.mean((L == -1) & ~unstable & ~unseen & ~occluded)), 4))
    gone = cluster_points(P['before'], lab['before'] == 1, 'before', eps=p['eps'], min_points=p['min_points'],
                          heights=Hgt['before'])
    new = cluster_points(P['after'], lab['after'] == 1, 'after', eps=p['eps'], min_points=p['min_points'],
                         start_id=len(gone), heights=Hgt['after'])
    vb = {ep: [views[n] for n in by_ep[ep]] for ep in by_ep}
    if dets is not None:
        name_objects(gone + new, P, vb, depths, dets, img_size, image_dir=image_dir)
    rejected = []
    for o in gone + new:
        other = 'after' if o.epoch == 'before' else 'before'
        rays, st = witness_rays(P[o.epoch][o.points_idx], vb[other], depths, edges, tau0=p['tau0'], tau_rel=p['tau_rel'])
        o.evidence = dict(st, rays=rays)
        o.sigma_m = round(_sigma(o, vb[o.epoch], meta['depth_error']['corrected'], p['voxel']), 3)
        ok, why = object_verdict(st, p['min_views'], p['max_surface_share'])
        if ok and o.label == 'object' and (o.n_points < 500 or o.z_range[1] - o.z_range[0] < 0.25):
            ok, why = False, 'small, and no detector recognises it as an object in any frame'
        if not ok:
            rejected.append(dict(o.to_dict(), reason=why))
    bad = {r['id'] for r in rejected}
    gone = [o for o in gone if o.id not in bad]; new = [o for o in new if o.id not in bad]
    if rejected:
        log('  rejected: ' + '; '.join(f"{r['label']} ({r['reason']})" for r in rejected))
    changes, used = [], set()
    for g, a, s in match_moves(gone, new):
        geo = estimate_yaw(P['before'][g.points_idx], P['after'][a.points_idx])
        app = feature_yaw(g, a, vb, depths, image_dir, img_size, P)
        rot = app if (app and app.get('confident')) else geo
        d = np.array(a.centroid) - np.array(g.centroid)
        name = a.label if a.label != 'object' else g.label
        if g.label != 'object' and a.label != 'object' and g.label != a.label:
            name = max((g, a), key=lambda o: (len(o.label.split()), o.label_conf)).label
        changes.append(dict(type='moved', label=name, score=round(s, 3), from_obj=g.to_dict(), to_obj=a.to_dict(),
                            displacement_m=round(float(np.linalg.norm(d[:2])), 3),
                            displacement_sigma_m=round(float(np.hypot(g.sigma_m, a.sigma_m)), 3),
                            delta=[round(float(x), 3) for x in d],
                            yaw_deg=rot['yaw_deg'], yaw_confident=bool(rot['confident']), yaw_source=rot['source'],
                            turned=bool(rot['confident'] and abs(rot['yaw_deg'] or 0) >= 20) or geo['turned'],
                            rotation=dict(appearance=app, geometry={k: v for k, v in geo.items() if k != 'profile'},
                                          geometry_profile=geo['profile'])))
        used |= {g.id, a.id}
    changes += [dict(type='removed', label=g.label, obj=g.to_dict()) for g in gone if g.id not in used]
    changes += [dict(type='added', label=a.label, obj=a.to_dict()) for a in new if a.id not in used]
    changes.sort(key=lambda c: (TYPE_ORDER[c['type']], -(c.get('to_obj') or c.get('obj'))['n_points']))
    for k, c in enumerate(changes):
        c['id'] = f'C{k + 1}'
    nb = {ep: naive_diff(P[ep], P[other]) for ep, other in (('before', 'after'), ('after', 'before'))}
    np.savez_compressed(os.path.join(out_dir, 'data', 'points.npz'),
                        P_before=P['before'], C_before=C['before'], L_before=lab['before'], S_before=stable['before'],
                        H_before=Hgt['before'], P_after=P['after'], C_after=C['after'], L_after=lab['after'],
                        S_after=stable['after'], H_after=Hgt['after'], naive_before=nb['before'], naive_after=nb['after'],
                        floor_c=floor_c,
                        **{f'ev_{ep}_{k}': getattr(ev[ep], k) for ep in ev for k in ('n_free', 'n_same', 'n_occl', 'n_fov')})
    np.savez_compressed(os.path.join(out_dir, 'data', 'object_points.npz'),
                        **{f'obj{o.id}': o.points_idx for o in gone + new})
    json.dump(dict(changes=changes, params=p, scale=meta['scale'], scale_source=meta['scale_source'],
                   coverage=coverage, frames=dict(before=len(by_ep['before']), after=len(by_ep['after'])),
                   objects=[o.to_dict() for o in gone + new], rejected=rejected),
              open(os.path.join(out_dir, 'changes.json'), 'w'), indent=1)
    log(f'  {len(changes)} change(s) in {time.time() - t0:.0f}s: ' +
        ', '.join(f"{c['id']} {c['type']} {c['label']}" for c in changes))
    return changes
