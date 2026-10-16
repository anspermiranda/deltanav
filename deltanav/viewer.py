"""The browser viewer: one self-contained HTML page (three.js) with the room surfaces, the changes, the robot
maps and the places that could not be checked. Opens from disk or from GitHub Pages; no install."""
from __future__ import annotations
import base64, gzip, json, os
import numpy as np

from .annotate import box_corners, places
from .geometry import floor_z
from .ledger import height


def _b64(a):
    return base64.b64encode(gzip.compress(np.ascontiguousarray(a).tobytes(), 6)).decode()


def _mesh(V, F, C, q):
    Vq = np.clip(np.round((V - q['lo']) / q['step']), 0, 65535).astype(np.uint16)
    return dict(v=_b64(Vq), c=_b64(C.astype(np.uint8)), f=_b64(F.astype(np.uint32)))


def _png_data_url(rgba):
    import cv2
    ok, buf = cv2.imencode('.png', cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
    return 'data:image/png;base64,' + base64.b64encode(buf.tobytes()).decode()


def _map_rgba(G, key):
    g = G[f'map_{key}']
    img = np.zeros(g.shape + (4,), np.uint8)
    img[:] = (170, 170, 185, 90)
    img[g == 0] = (250, 250, 250, 170); img[g == 100] = (35, 35, 40, 235)
    img[(g == 0) & ~G[f'trav_{key}']] = (250, 200, 195, 180)
    img[G[f'reach_{key}']] = (195, 240, 205, 190)
    return img[::-1]                                 # row 0 = top (max y) for the texture


def _focus(pts):
    P = np.asarray(pts, float)
    c = P.mean(0); span = max(0.8, float(np.ptp(P[:, :2], axis=0).max()))
    d = 1.6 + 1.2 * span
    return dict(target=c.round(3).tolist(), pos=(c + np.array([-0.55 * d, -0.75 * d, 0.75 * d])).round(3).tolist())


def build_viewer(path, surfaces, changes, sentences, Z, floor_c, views, nav=None, grids=None, next_views=None,
                 landmarks=(), subtitle='', notes='', template=None, max_faces=(320000, 220000), home=None):
    from .mesh import decimate
    lo, hi = surfaces['bounds']
    q = dict(lo=np.asarray(lo, float) - 0.05, step=0.001)
    meshes = {}
    for k in ('shared', 'before', 'after'):
        if k in surfaces and len(surfaces[k][1]):
            V, F, C = decimate(*surfaces[k], max_faces[0] if k == 'shared' else max_faces[1])
            meshes[k] = _mesh(V, F, C, q)
    objects = []
    for (cid, ep), (V, F, C) in surfaces.get('objects', {}).items():
        if len(F):
            objects.append(dict(id=cid, epoch=ep, mesh=_mesh(*decimate(V, F, C, 30000), q)))
    ch = []
    for c, s in zip(changes, sentences):
        pl = []
        for o, ep, role in places(c):
            lab = {'from': f"{c['id']} {c['label']} (before)", 'to': f"{c['id']} {c['label']} (after)"}.get(
                role, f"{c['id']} {c['label']}")
            pl.append(dict(epoch=ep, role=role, label=lab, corners=box_corners(o, floor_c).round(3).tolist()))
        arrow = None
        if c['type'] == 'moved':
            a = np.array(c['from_obj']['centroid'], float); b = np.array(c['to_obj']['centroid'], float)
            a[2] += floor_z(a[None, :2], floor_c)[0]; b[2] += floor_z(b[None, :2], floor_c)[0]
            arrow = [a.round(3).tolist(), b.round(3).tolist()]
        objs = [c['from_obj'], c['to_obj']] if c['type'] == 'moved' else [c['obj']]
        rays = [r for o in objs for r in o.get('evidence', {}).get('rays', [])]
        rows = []
        for o, ep, role in places(c):
            x, y, z = o['centroid']
            sig = f" ± {o['sigma_m'] * 100:.0f} cm" if o.get('sigma_m') else ''
            name = {'from': 'before', 'to': 'after'}.get(role, 'position')
            rows.append([name, f'x {x:.2f}, y {y:.2f}, z {z:.2f} m{sig}'])
        if c['type'] == 'moved':
            rows.append(['moved', f"{c['displacement_m']:.2f} m (Δx {c['delta'][0]:+.2f}, Δy {c['delta'][1]:+.2f})"])
            if c.get('yaw_deg') is not None:
                rows.append(['turned', f"{c['yaw_deg']:+.0f}° ({c.get('yaw_source')}{'' if c.get('yaw_confident') else ', uncertain'})"])
        o = c.get('to_obj') or c['obj']
        w, d = o['footprint']['size']
        rows.append(['size', f"{max(w, d):.2f} × {min(w, d):.2f} × {height(o):.2f} m"])
        e = (c.get('from_obj') or c['obj']).get('evidence', {})
        if e.get('views_looked'):
            rows.append(['evidence', f"{e['views_saw_through']} frames saw through, {e.get('views_saw_surface', 0)} saw a surface"])
        ch.append(dict(id=c['id'], type=c['type'], label=c['label'], sentence=s, places=pl, arrow=arrow, rays=rays,
                       rows=rows, focus=_focus([p for x in pl for p in x['corners']])))
    unknown = None
    m = (Z['L_before'] == -1) & Z['S_before']
    if m.any():
        U = Z['P_before'][m]
        U = U[np.random.default_rng(0).choice(len(U), min(len(U), 70000), replace=False)]
        unknown = _b64(np.clip(np.round((U - q['lo']) / q['step']), 0, 65535).astype(np.uint16))
    robots = []
    if nav and grids:
        for name, r in nav['robots'].items():
            G = grids[name]; x0, y0, nx, ny, res = G['spec']
            robots.append(dict(name=name, extent=[x0, x0 + nx * res, y0, y0 + ny * res],
                               map_png=_png_data_url(_map_rgba(G, 'updated')), start=r.get('start'),
                               routes_before=[rt['waypoints'] for rt in r['plans']['before'].values() if rt.get('reachable')],
                               routes_updated=[rt['waypoints'] for rt in r['plans']['updated'].values() if rt.get('reachable')],
                               goals=[dict(name=g, x=rt['goal'][0], y=rt['goal'][1], yaw=float(np.radians(rt['yaw_deg'])))
                                      for g, rt in r['plans']['updated'].items() if rt.get('reachable')],
                               checks=[dict(name='check ' + ' '.join(g.split('_')[1:]), x=v['goal'][0], y=v['goal'][1],
                                            yaw=float(np.radians(v['yaw_deg'])), waypoints=v.get('waypoints'))
                                       for g, v in r.get('inspect_changes', {}).items() if v.get('reachable')]))
    nv = []
    for v in next_views or []:
        p = np.array(v['position'], float); p[2] += floor_z(p[None, :2], floor_c)[0]
        nv.append(dict(pos=p.round(3).tolist(), yaw_deg=v['yaw_deg'], pitch_deg=v['pitch_deg'], hfov_deg=v['hfov_deg']))
    cams = {}
    for ep in ('before', 'after'):
        nm = sorted([n for n in views if views[n].epoch == ep], key=lambda n: views[n].frame)
        cams[ep] = [views[n].C.round(3).tolist() for n in nm]
    ctr = (np.asarray(lo) + np.asarray(hi)) / 2
    span = float(max(hi[0] - lo[0], hi[1] - lo[1]))
    home = home or dict(target=[float(ctr[0]), float(ctr[1]), 0.4],
                        pos=[float(ctr[0] - 0.75 * span), float(ctr[1] - 1.05 * span), 0.4 + 1.25 * span])
    data = dict(quant=dict(lo=q['lo'].tolist(), step=q['step']), meshes=meshes, objects=objects, changes=ch,
                unknown=unknown, robots=robots, next_views=nv, cams=cams, subtitle=subtitle, notes=notes,
                furniture=[dict(label=l['label'], pos=[l['xy'][0], l['xy'][1], l['z'][1] + 0.1]) for l in landmarks],
                extent=[float(lo[0]), float(hi[0]), float(lo[1]), float(hi[1])], home=home)
    tpl = open(template or os.path.join(os.path.dirname(__file__), 'viewer_template.html')).read()
    html = tpl.replace('/*DATA*/null', json.dumps(data, separators=(',', ':')).replace('</', '<\\/'))
    open(path, 'w').write(inline_scripts(html))
    return path


CDN = {'https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js': 'three.min.js',
       'https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js': 'OrbitControls.js'}


def inline_scripts(html: str) -> str:
    """Put three.js (r128, MIT, shipped in deltanav/vendor) into the page, so the viewer opens without
    internet; the CDN links stay if the files are missing."""
    for url, name in CDN.items():
        p = os.path.join(os.path.dirname(__file__), 'vendor', name)
        tag = f'<script src="{url}"></script>'
        if tag in html and os.path.exists(p):
            js = open(p, encoding='utf-8').read().replace('</script', '<\\/script')
            html = html.replace(tag, f'<script>/* {name}: three.js r128, MIT licence (deltanav/vendor) */\n{js}\n</script>')
    return html
