"""Experiments on a finished run (ours vs a naive nearest-neighbour point-cloud diff).

  1. no-change controls: the two laps (chest / waist height) of the SAME video compared with each other;
  2. cross-height: before at one height vs after at the other (large viewpoint change + the real changes);
  3. never filmed: drop every after-video frame that had the removed object's place in view;
  4. only occluded: drop only the after frames that actually saw that place;
  5. verification loop: start from (3) and give the held-out frames back one at a time, in the order a
     next-best-view planner picks them (from the current model only), or in random order, until the removal
     is confirmed.

usage:  python tools/run_experiments.py outputs/my_room [--only name1,name2] [--random-orders 20]
writes: outputs/my_room/experiments/experiments.json and experiments.md"""
import argparse, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from deltanav.scene import load_scene
from deltanav.detect import detect_changes, split_passes, PARAMS
from deltanav.objects import cluster_points
from deltanav.change import gather_evidence, witness_rays, object_verdict
from deltanav.depth import edge_mask, backproject
from deltanav.nbv import predicted_gain


def naive_objects(out_dir):
    Z = np.load(os.path.join(out_dir, 'data', 'points.npz'))
    return int(sum(len(cluster_points(Z[f'P_{ep}'], Z[f'naive_{ep}'], ep, heights=Z[f'H_{ep}'])) for ep in ('before', 'after')))


def place_box(o, pad=0.15):
    fp = o['footprint']; c = np.array(fp['center']); half = np.array(fp['size']) / 2 + pad
    a = np.radians(fp['angle']); ux, uy = np.array([np.cos(a), np.sin(a)]), np.array([-np.sin(a), np.cos(a)])
    return c, half, ux, uy


def in_place(P, Hgt, o, pad=0.0):
    c, half, ux, uy = place_box(o, pad)
    d = P[:, :2] - c
    return (np.abs(d @ ux) < half[0]) & (np.abs(d @ uy) < half[1]) & (Hgt > o['z_range'][0] + 0.05) & (Hgt < o['z_range'][1])


def looked_at(views, names, depths, o, floor_c):
    """After frames whose image contains the object's place (any part of its box)."""
    from deltanav.geometry import floor_z
    c, half, ux, uy = place_box(o)
    g = np.stack(np.meshgrid(np.linspace(-1, 1, 7), np.linspace(-1, 1, 7), np.linspace(0, 1, 5)), -1).reshape(-1, 3)
    xy = np.array([c + x * half[0] * ux + y * half[1] * uy for x, y, _ in g])
    z = floor_z(xy, floor_c) + o['z_range'][0] + g[:, 2] * (o['z_range'][1] - o['z_range'][0])
    Q = np.c_[xy, z]
    out = []
    for n in names:
        v = views[n]; H, W = depths[n].shape
        uv, zz = v.project(Q)
        if ((zz > 0.1) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)).any():
            out.append(n)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out'); ap.add_argument('--only', default=None); ap.add_argument('--random-orders', type=int, default=20)
    a = ap.parse_args()
    out = os.path.abspath(a.out)
    scene, frames, dets = os.path.join(out, 'scene'), os.path.join(out, 'frames'), os.path.join(out, 'detections.pkl')
    exp_dir = os.path.join(out, 'experiments'); os.makedirs(exp_dir, exist_ok=True)
    only = a.only.split(',') if a.only else None
    meta, views, depths, _, _ = load_scene(scene)
    b1, b2 = split_passes(views, 'before'); a1, a2 = split_passes(views, 'after')
    hb = lambda g: round(float(np.median([views[n].C[2] for n in g])), 2)
    info = dict(passes=dict(before=[len(b1), len(b2)], after=[len(a1), len(a2)],
                            camera_height_m=dict(before=[hb(b1), hb(b2)], after=[hb(a1), hb(a2)])))
    print('laps', info['passes'], flush=True)
    allb = sorted(n for n in views if views[n].epoch == 'before'); alla = sorted(n for n in views if views[n].epoch == 'after')
    full = json.load(open(os.path.join(out, 'result', 'changes.json')))
    Zf = np.load(os.path.join(out, 'result', 'data', 'points.npz'))
    floor_c = Zf['floor_c']
    removed = [c for c in full['changes'] if c['type'] == 'removed']
    exps = {'control_before_chest_vs_waist': (b1, b2), 'control_after_chest_vs_waist': (a1, a2),
            'cross_before_chest_vs_after_waist': (b1, a2), 'cross_before_waist_vs_after_chest': (b2, a1),
            'full': (allb, alla)}
    o = None
    if removed:
        o = removed[0]['obj']
        seen = looked_at(views, alla, depths, o, floor_c)
        keep = [n for n in alla if n not in set(seen)]
        exps['never_filmed'] = (allb, keep)
        # only occluded: drop the frames that actually saw the place (through it or a surface there)
        Pb = Zf['P_before']; idx = np.load(os.path.join(out, 'result', 'data', 'object_points.npz'))[f"obj{o['id']}"]
        edges = {n: edge_mask(depths[n]) for n in alla}
        saw = []
        for n in seen:
            _, st = witness_rays(Pb[idx], [views[n]], depths, edges)
            if st['views_saw_through'] or st['views_saw_surface']:
                saw.append(n)
        exps['only_occluded'] = (allb, [n for n in alla if n not in set(saw)])
        info['hidden'] = dict(object=removed[0]['label'], after_frames=len(alla), in_view=len(seen), saw_it=len(saw))
        print('hidden tests:', info['hidden'], flush=True)
    res = json.load(open(os.path.join(exp_dir, 'experiments.json'))) if os.path.exists(os.path.join(exp_dir, 'experiments.json')) else {}
    for name, grp in exps.items():
        if only and name not in only:
            continue
        t = time.time()
        d = os.path.join(exp_dir, name)
        ch = detect_changes(scene, frames, dets, d, log=lambda m: None, groups=grp)
        Z = np.load(os.path.join(d, 'data', 'points.npz'))
        R = dict(ours=[f"{c['type']} {c['label']}" for c in ch], ours_n=len(ch), naive_objects=naive_objects(d),
                 frames=[len(grp[0]), len(grp[1])], seconds=round(time.time() - t),
                 rejected=[f"{r['label']}: {r['reason']}" for r in json.load(open(os.path.join(d, 'changes.json')))['rejected']])
        if name in ('never_filmed', 'only_occluded') and o is not None:
            loc = in_place(Z['P_before'], Z['H_before'], o)
            L = Z['L_before']
            R['place_points'] = int(loc.sum())
            R['ours_on_place'] = dict(gone=round(float(np.mean(L[loc] == 1)), 3), same=round(float(np.mean(L[loc] == 0)), 3),
                                      unknown=round(float(np.mean(L[loc] == -1)), 3))
            R['naive_on_place'] = round(float(np.mean(Z['naive_before'][loc])), 3)
        res[name] = R
        print(f"{name:36s} ours: {R['ours']}  naive: {R['naive_objects']} objects  ({R['seconds']}s)", flush=True)
    if o is not None and (not only or 'verification_loop' in only):
        res['verification_loop'] = verification_loop(views, depths, allb, alla, exps['never_filmed'][1], o, out, a.random_orders)
    res['info'] = info
    json.dump(res, open(os.path.join(exp_dir, 'experiments.json'), 'w'), indent=1)
    write_md(res, os.path.join(exp_dir, 'experiments.md'))


def verification_loop(views, depths, allb, alla, kept, o, out, n_random, n_init=25, n_steps=30, n_points=20000):
    """Active verification. The robot starts with a partial look at the room: n_init after-video frames, none
    of which saw the removed object's place. Then it gets one more frame at a time from all the other after
    frames, either in the order the next-best-view planner picks them (predicted from the current model only:
    how much still-unknown before-video surface each camera would see, with a z-buffer for occlusion) or in
    random order. Measured: how much of the before surface is checked after k frames, and after how many
    frames the removal is confirmed by the pipeline's own rules (a see-through cluster + the frame verdict)."""
    p = PARAMS
    Zf = np.load(os.path.join(out, 'result', 'data', 'points.npz'))
    Pb, Sb, Hb = Zf['P_before'], Zf['S_before'], Zf['H_before']
    idx = np.load(os.path.join(out, 'result', 'data', 'object_points.npz'))[f"obj{o['id']}"]
    edges = {n: edge_mask(depths[n]) for n in alla}
    Q = Pb[idx]
    rng = np.random.default_rng(0)
    spot = set(alla) - set(kept)                         # frames that had the place in view
    init = list(rng.choice(sorted(set(kept)), min(n_init, len(kept)), replace=False))
    pool = [n for n in alla if n not in set(init)]
    sample = rng.choice(np.nonzero(Sb)[0], min(n_points, int(Sb.sum())), replace=False)
    T = Pb[sample]

    def checked(names):
        e = gather_evidence(T, [views[n] for n in names], depths, edges, tau0=p['tau0'], tau_rel=p['tau_rel'])
        lab = e.classify(min_free=p['min_free'], free_ratio=p['free_ratio'], free_frac_fov=p['free_frac_fov'])
        return lab

    def confirmed(names):
        vs = [views[n] for n in names]
        e = gather_evidence(Q, vs, depths, edges, tau0=p['tau0'], tau_rel=p['tau_rel'])
        lab = e.classify(min_free=p['min_free'], free_ratio=p['free_ratio'], free_frac_fov=p['free_frac_fov'])
        lab[~Sb[idx]] = -1
        if not cluster_points(Q, lab == 1, 'before', eps=p['eps'], min_points=p['min_points'], heights=Hb[idx]):
            return False
        _, st = witness_rays(Q, vs, depths, edges, tau0=p['tau0'], tau_rel=p['tau_rel'])
        return object_verdict(st, p['min_views'], p['max_surface_share'])[0]

    def run(order):
        cur, cov, conf = list(init), [], None
        cov.append(float(np.mean(checked(cur) != -1)))
        for k, n in enumerate(order[:n_steps]):
            cur.append(n)
            cov.append(float(np.mean(checked(cur) != -1)))
            if conf is None and confirmed(cur):
                conf = k + 1
        return cov, conf

    # the planner, in a closed loop: after every frame it re-checks what is still unknown (with the real depth of
    # the frames it has), rebuilds the occluders, and picks the next frame by predicted gain only (it never sees
    # a candidate's image before choosing it). Floor-level surfaces count more: that is where a change matters
    # to a robot.
    pri = np.where(Hb[sample] < 1.0, 1.0, 0.25)
    order, left, cur = [], list(pool), list(init)
    occ_parts = [backproject(depths[n], views[n], step=10)[0] for n in init]
    lab = checked(cur)
    while left and len(order) < n_steps:
        occ = np.vstack(occ_parts)
        occ = occ[rng.choice(len(occ), min(len(occ), 150000), replace=False)]
        w = (lab == -1) * pri
        best = max(left, key=lambda n: (w * predicted_gain(views[n], T, occ)[1]).sum())
        order.append(best); left.remove(best); cur.append(best)
        occ_parts.append(backproject(depths[best], views[best], step=10)[0])
        lab = checked(cur)
    g_cov, g_conf = run(order)
    r_cov, r_conf = [], []
    for s in range(n_random):
        r = list(pool); np.random.default_rng(100 + s).shuffle(r)
        c, k = run(r); r_cov.append(c); r_conf.append(k)
    ok = [x for x in r_conf if x is not None]
    R = dict(initial_frames=len(init), candidate_frames=len(pool), candidates_seeing_the_place=len(spot & set(pool)),
             guided=dict(confirmed_after=g_conf, checked=[round(x, 4) for x in g_cov],
                         picked_seeing_the_place=[bool(n in spot) for n in order]),
             random=dict(confirmed_after=r_conf, median=float(np.median(ok)) if ok else None,
                         never=sum(x is None for x in r_conf),
                         checked_mean=[round(float(x), 4) for x in np.mean(r_cov, axis=0)],
                         checked_p25=[round(float(x), 4) for x in np.percentile(r_cov, 25, axis=0)],
                         checked_p75=[round(float(x), 4) for x in np.percentile(r_cov, 75, axis=0)]))
    print('verification loop: guided confirmed after', g_conf, '| random', r_conf, flush=True)
    plot_loop(R, os.path.join(out, 'experiments', 'verification_loop.png'), o.get('label', 'object') if isinstance(o, dict) else 'object')
    return R


def plot_loop(R, path, label):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    g = np.array(R['guided']['checked']) * 100; m = np.array(R['random']['checked_mean']) * 100
    lo, hi = np.array(R['random']['checked_p25']) * 100, np.array(R['random']['checked_p75']) * 100
    k = np.arange(len(g))
    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=130)
    ax.fill_between(k, lo, hi, color=(0.6, 0.6, 0.6), alpha=0.3, lw=0, label='random order (25-75%)')
    ax.plot(k, m, color=(0.45, 0.45, 0.45), lw=2, label='random order (mean)')
    ax.plot(k, g, color=(0.08, 0.45, 0.86), lw=2.5, label='next-best-view order')
    if R['guided']['confirmed_after']:
        x = R['guided']['confirmed_after']
        ax.axvline(x, color=(0.08, 0.45, 0.86), ls=':', lw=1.5)
        ax.text(x + 0.3, g.min() + 2, f'{label} removal\nconfirmed (planner)', fontsize=8, color=(0.08, 0.45, 0.86))
    if R['random']['median']:
        x = R['random']['median']
        ax.axvline(x, color=(0.45, 0.45, 0.45), ls=':', lw=1.5)
        ax.text(x + 0.3, g.min() + 10, 'confirmed (random,\nmedian)', fontsize=8, color=(0.4, 0.4, 0.4))
    ax.set_xlabel('after-video frames added to the initial partial look')
    ax.set_ylabel('% of before-video surface checked')
    ax.set_title('Verification loop: which frame to look at next')
    ax.grid(alpha=0.25); ax.legend(fontsize=8, loc='lower right')
    fig.tight_layout(); fig.savefig(path, bbox_inches='tight'); plt.close(fig)


def write_md(res, path):
    rows = ['| Test | Frames (before / after) | This pipeline | Naive point-cloud diff |', '|---|---|---|---|']
    for k, r in res.items():
        if k in ('info', 'verification_loop'):
            continue
        ours = ', '.join(r['ours']) or 'nothing'
        extra = f" (place: {100 * r['ours_on_place']['unknown']:.1f}% could not be checked)" if 'ours_on_place' in r else ''
        rows.append(f"| {k} | {r['frames'][0]} / {r['frames'][1]} | {ours}{extra} | {r['naive_objects']} objects |")
    v = res.get('verification_loop')
    if v:
        g, r = v['guided'], v['random']
        rows += ['', f"Verification loop: start from {v['initial_frames']} after frames that never saw the place; "
                     f"{v['candidate_frames']} candidate frames ({v['candidates_seeing_the_place']} of them see it). "
                     f"Planner order: removal confirmed after {g['confirmed_after']} frame(s), "
                     f"{100 * g['checked'][10]:.0f}% of the surface checked after 10. Random order: confirmed after a median "
                     f"of {r['median']} (never within {len(g['checked']) - 1} frames in {r['never']} of {len(r['confirmed_after'])} orders), "
                     f"{100 * r['checked_mean'][10]:.0f}% checked after 10."]
    open(path, 'w').write('\n'.join(rows) + '\n')


if __name__ == '__main__':
    main()
