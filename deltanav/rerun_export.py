"""The 3D window: a Rerun recording (scene.rrd) of the whole result.

Open it with `deltanav view <out>` (or `rerun scene.rrd`): rotate with the left mouse button, pan with the
right, zoom with the wheel, double-click to focus. Every layer is an entity in the left panel and can be
switched on and off with its eye icon: the shared room model, the before-only and after-only surfaces, the
changed objects (both versions of a move), the places that could not be checked, the evidence rays, each
robot's map with its routes and goals, where to look next, and both camera paths. The timeline scrubs through
both walk-throughs, with the annotated keyframe shown in each camera."""
from __future__ import annotations
import os
import numpy as np
import cv2

from .annotate import COL, box_corners, places


def _quat_z(deg):
    a = np.radians(deg) / 2
    return [0.0, 0.0, float(np.sin(a)), float(np.cos(a))]


def _rot_to_quat(R):
    """3x3 rotation -> quaternion (x, y, z, w)."""
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(R).as_quat().tolist()


def grid_mesh(g, spec, trav=None, reach=None, z=0.01):
    """An occupancy map as a flat coloured mesh (two triangles per cell) lying on the floor."""
    x0, y0, nx, ny, res = spec
    ys, xs = np.nonzero(np.ones_like(g, bool))
    col = np.zeros((len(xs), 4), np.uint8)
    v = g[ys, xs]
    col[v == 0] = (245, 245, 245, 150); col[v == 100] = (40, 40, 45, 230); col[v == -1] = (170, 170, 185, 90)
    if trav is not None:
        m = (v == 0) & ~trav[ys, xs]; col[m] = (250, 200, 195, 160)
    if reach is not None:
        m = reach[ys, xs]; col[m] = (200, 240, 205, 170)
    X0 = x0 + xs * res; Y0 = y0 + ys * res
    V = np.stack([np.c_[X0, Y0], np.c_[X0 + res, Y0], np.c_[X0 + res, Y0 + res], np.c_[X0, Y0 + res]], 1).reshape(-1, 2)
    V = np.c_[V, np.full(len(V), z)].astype(np.float32)
    b = np.arange(len(xs)) * 4
    F = np.concatenate([np.stack([b, b + 1, b + 2], 1), np.stack([b, b + 2, b + 3], 1)]).astype(np.uint32)
    C = np.repeat(col, 4, axis=0)
    return V, F, C


def export_rerun(path, surfaces, changes, sentences, Z, floor_c, views, meta, ann=None, nav=None, grids=None,
                 next_views=None, landmarks=(), summary_md='', jpeg_width=960, image_dir='.', log=print, occ3=None):
    import rerun as rr
    import rerun.blueprint as rrb
    from .geometry import floor_z
    rec = rr.RecordingStream('deltanav')
    with rec:
        hidden = ['world/room/before', 'world/room/after', 'world/points', 'world/evidence', 'world/robot',
                  'world/could_not_check', 'world/furniture']
        lo, hi = surfaces['bounds']
        ctr = (np.asarray(lo) + np.asarray(hi)) / 2
        eye = rrb.EyeControls3D(position=[float(lo[0]) - 0.6, float(lo[1]) - 0.8, 3.4],
                                look_target=[float(ctr[0]), float(ctr[1]), 0.3], eye_up=[0, 0, 1])
        bp = rrb.Blueprint(
            rrb.Horizontal(
                rrb.Spatial3DView(origin='/world', name='Room in 3D (switch layers on/off in the left panel)',
                                  overrides={e: rrb.EntityBehavior(visible=False) for e in hidden}, eye_controls=eye,
                                  background=[245, 245, 248]),
                rrb.Vertical(
                    rrb.Spatial2DView(origin='/world/cameras/before/cam', name='Before video (annotated)'),
                    rrb.Spatial2DView(origin='/world/cameras/after/cam', name='After video (annotated)'),
                    rrb.TextDocumentView(origin='/report', name='What changed'),
                    row_shares=[1, 1, 1.2]),
                column_shares=[2.6, 1]),
            rrb.TimePanel(timeline='keyframe', expanded=False), collapse_panels=False)
        rr.save(path, default_blueprint=bp)
        rr.log('world', rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        rr.log('world/axes', rr.Arrows3D(origins=[[0, 0, 0]] * 3, vectors=[[0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]],
                                         colors=[[230, 50, 50], [50, 190, 60], [60, 90, 230]],
                                         labels=['x', 'y', 'z'], show_labels=True), static=True)
        for name in ('shared', 'before', 'after'):
            if name in surfaces and len(surfaces[name][1]):
                V, F, C = surfaces[name]
                rr.log(f'world/room/{name}', rr.Mesh3D(vertex_positions=V, triangle_indices=F, vertex_colors=C,
                                                       face_rendering=rr.components.MeshFaceRendering.Front), static=True)
        for ep in ('before', 'after'):
            P, Cc = Z[f'P_{ep}'], Z[f'C_{ep}']
            k = np.random.default_rng(0).choice(len(P), min(len(P), 400000), replace=False)
            rr.log(f'world/points/{ep}', rr.Points3D(P[k], colors=Cc[k], radii=0.006), static=True)
        # could not be checked
        L, S = Z['L_before'], Z['S_before']
        U = Z['P_before'][(L == -1) & S]
        if len(U):
            k = np.random.default_rng(1).choice(len(U), min(len(U), 150000), replace=False)
            rr.log('world/could_not_check', rr.Points3D(U[k], colors=[133, 92, 199], radii=0.008), static=True)
        # changes
        for c, s in zip(changes, sentences):
            base = f"world/changes/{c['id']}_{c['type']}_{c['label'].replace(' ', '_')}"
            col = COL[c['type']]
            for o, ep, role in places(c):
                K = box_corners(o, floor_c)
                ctr = K.mean(0); half = np.array(o['footprint']['size']) / 2 + 0.03
                hz = (K[4:, 2].mean() - K[:4, 2].mean()) / 2
                lab = {'from': f"{c['id']} {c['label']} (before)", 'to': f"{c['id']} {c['label']} (after)"}.get(
                    role, f"{c['id']} {c['type']}: {c['label']}")
                rr.log(f'{base}/{ep}_box', rr.Boxes3D(centers=[ctr], half_sizes=[[half[0], half[1], hz]],
                                                      quaternions=[_quat_z(o['footprint']['angle'])], colors=[col],
                                                      radii=0.008, labels=[lab], show_labels=True), static=True)
                ob = surfaces.get('objects', {}).get((c['id'], ep))
                if ob is not None and len(ob[1]):
                    rr.log(f'{base}/{ep}_surface', rr.Mesh3D(vertex_positions=ob[0], triangle_indices=ob[1],
                                                             vertex_colors=ob[2]), static=True)
            if c['type'] == 'moved':
                a = np.array(c['from_obj']['centroid'], float); b = np.array(c['to_obj']['centroid'], float)
                a[2] += floor_z(a[None, :2], floor_c)[0]; b[2] += floor_z(b[None, :2], floor_c)[0]
                rr.log(f'{base}/move', rr.Arrows3D(origins=[a], vectors=[b - a], colors=[col], radii=0.015,
                                                   labels=[f"{c['displacement_m']:.2f} m"
                                                           + (f", {c['yaw_deg']:+.0f}°" if c.get('yaw_deg') is not None
                                                              and c.get('yaw_confident') else '')], show_labels=True),
                       static=True)
            rr.log(f'{base}/description', rr.TextDocument(s), static=True)
            objs = [c['from_obj'], c['to_obj']] if c['type'] == 'moved' else [c['obj']]
            rays = [r for o in objs for r in o.get('evidence', {}).get('rays', [])]
            if rays:
                rr.log(f"world/evidence/{c['id']}", rr.LineStrips3D([[r[0], r[1], r[2]] for r in rays],
                                                                    colors=[[255, 210, 60]], radii=0.003), static=True)
        for lm in landmarks:
            fp = lm['footprint']; z0, z1 = lm['z']
            rr.log(f"world/furniture/{lm['label'].replace(' ', '_')}_{int(fp['center'][0] * 100)}",
                   rr.Boxes3D(centers=[[fp['center'][0], fp['center'][1], (z0 + z1) / 2]],
                              half_sizes=[[fp['size'][0] / 2, fp['size'][1] / 2, max(0.01, (z1 - z0) / 2)]],
                              quaternions=[_quat_z(fp['angle'])], colors=[[150, 150, 160]], radii=0.004,
                              labels=[lm['label']], show_labels=True), static=True)
        # robots
        if nav and grids:
            for name, r in nav['robots'].items():
                G = grids[name]
                for key in ('before', 'updated'):
                    V, F, C = grid_mesh(G[f'map_{key}'], G['spec'], G[f'trav_{key}'], G[f'reach_{key}'])
                    rr.log(f'world/robot/{name}/map_{key}', rr.Mesh3D(vertex_positions=V, triangle_indices=F,
                                                                      vertex_colors=C), static=True)
                for key, colr, zz in (('before', [140, 140, 140], 0.03), ('updated', [20, 115, 220], 0.04)):
                    strips, labels, goals, dirs = [], [], [], []
                    for gname, rt in r['plans'][key].items():
                        if rt.get('reachable'):
                            W = np.array(rt['waypoints']); strips.append(np.c_[W, np.full(len(W), zz)])
                            labels.append(f"{gname} {rt['length_m']:.1f} m")
                            yaw = np.radians(rt['yaw_deg']); goals.append([*rt['goal'], zz])
                            dirs.append([0.25 * np.cos(yaw), 0.25 * np.sin(yaw), 0])
                    if strips:
                        rr.log(f'world/robot/{name}/routes_{key}', rr.LineStrips3D(strips, colors=[colr], radii=0.012,
                                                                                    labels=labels), static=True)
                        rr.log(f'world/robot/{name}/goals_{key}', rr.Arrows3D(origins=goals, vectors=dirs, colors=[colr],
                                                                               radii=0.02), static=True)
                if r.get('start'):
                    rr.log(f'world/robot/{name}/start', rr.Points3D([[*r['start'], 0.05]], colors=[[255, 200, 0]],
                                                                    radii=0.06, labels=['start'], show_labels=True),
                           static=True)
        if occ3 is not None and occ3[0] is not None:
            g, sp = occ3
            idx = np.argwhere(g == 1)
            P = np.asarray(sp['origin']) + (idx + 0.5) * sp['voxel']
            t = np.clip((P[:, 2] - P[:, 2].min()) / max(1e-6, np.ptp(P[:, 2])), 0, 1)
            col = np.stack([255 * t, 80 + 120 * (1 - np.abs(2 * t - 1)), 255 * (1 - t)], 1).astype(np.uint8)
            rr.log('world/robot/occupancy3d_updated', rr.Points3D(P, colors=col, radii=sp['voxel'] * 0.45), static=True)
        # where to look next
        for k, v in enumerate(next_views or []):
            p = np.array(v['position'], float); yaw, pitch = np.radians(v['yaw_deg']), np.radians(v['pitch_deg'])
            p[2] += floor_z(p[None, :2], floor_c)[0]
            fwd = np.array([np.cos(yaw) * np.cos(pitch), np.sin(yaw) * np.cos(pitch), np.sin(pitch)])
            right = np.array([np.sin(yaw), -np.cos(yaw), 0.0]); down = np.cross(fwd, right)
            R = np.stack([right, down, fwd], 1)                      # camera -> world (RDF)
            base = f"world/next_views/{v.get('robot', 'robot')}_{k + 1}"
            rr.log(base, rr.Transform3D(translation=p, quaternion=_rot_to_quat(R)), static=True)
            f = float(320 / np.tan(np.radians(v['hfov_deg']) / 2))
            rr.log(base + '/camera', rr.Pinhole(focal_length=f, width=640, height=480, image_plane_distance=0.4,
                                                camera_xyz=rr.ViewCoordinates.RDF, color=[30, 140, 230]), static=True)
            rr.log(base + '/label', rr.Points3D([[0, 0, 0]], radii=0.03, colors=[[30, 140, 230]],
                                                labels=[f"look here next #{k + 1} ({v.get('robot', '')})"], show_labels=True),
                   static=True)
        # both walk-throughs, scrubbable
        for ep in ('before', 'after'):
            nm = sorted([n for n in views if views[n].epoch == ep], key=lambda n: views[n].frame)
            if not nm:
                continue
            Cs = np.array([views[n].C for n in nm])
            rr.log(f'world/cameras/{ep}/path', rr.LineStrips3D([Cs], colors=[[40, 90, 205] if ep == 'before' else [140, 50, 165]],
                                                               radii=0.005), static=True)
            for i, n in enumerate(nm):
                v = views[n]
                rr.set_time('keyframe', sequence=i)
                rr.log(f'world/cameras/{ep}/cam', rr.Transform3D(translation=v.C, mat3x3=v.R.T))
                if ann is not None:
                    img, _ = ann.draw(n)
                else:
                    img = cv2.cvtColor(cv2.imread(os.path.join(image_dir, n)), cv2.COLOR_BGR2RGB)
                h = int(round(jpeg_width * img.shape[0] / img.shape[1]))
                img = cv2.resize(img, (jpeg_width, h), interpolation=cv2.INTER_AREA)
                K = v.K.copy(); sx = jpeg_width / v.width; K[0] *= sx; K[1] *= h / v.height
                rr.log(f'world/cameras/{ep}/cam', rr.Pinhole(image_from_camera=K, width=jpeg_width, height=h,
                                                             camera_xyz=rr.ViewCoordinates.RDF, image_plane_distance=0.25))
                ok, buf = cv2.imencode('.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 80])
                rr.log(f'world/cameras/{ep}/cam/frame', rr.EncodedImage(contents=buf.tobytes(), media_type='image/jpeg'))
        rr.log('report', rr.TextDocument(summary_md, media_type=rr.MediaType.MARKDOWN), static=True)
        rec.flush()
    return path

