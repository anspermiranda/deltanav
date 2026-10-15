"""Pictures: 3D renders of the reconstructed room (no GPU or display needed), the floor plan, the robot's
routes, the naive-diff comparison and the coverage chart. Everything is drawn in the room frame."""
from __future__ import annotations
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyArrowPatch, Rectangle, Wedge
from matplotlib.lines import Line2D

from .annotate import COL, box_corners, font, EDGES
from .geometry import floor_z

RGB = {k: tuple(c / 255 for c in v) for k, v in COL.items()}
UNKNOWN = (0.52, 0.36, 0.78)


# ---------------------------------------------------------------- software renderer (point splats of the mesh)
def vertex_normals(V, F):
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    N = np.zeros_like(V, dtype=np.float64)
    for k in range(3):
        np.add.at(N, F[:, k], fn)
    N /= np.linalg.norm(N, axis=1, keepdims=True) + 1e-12
    return N.astype(np.float32)


def shading_normals(V, F, free_points, iters: int = 20):
    """Normals for shading, from a well-smoothed copy of the mesh (monocular depth leaves small bumps that would
    otherwise make flat walls look pebbly); the geometry itself is not changed."""
    from .mesh import taubin
    return orient_normals(V, vertex_normals(taubin(V, F, iters=iters), F), free_points)


def orient_normals(V, N, free_points):
    """Make normals point into free space (towards where the cameras were)."""
    from scipy.spatial import cKDTree
    idx = np.random.default_rng(0).choice(len(V), min(20000, len(V)), replace=False)
    _, j = cKDTree(free_points).query(V[idx])
    s = np.sum(N[idx] * (free_points[j] - V[idx]), axis=1)
    return N if np.mean(s > 0) >= 0.5 else -N


def edge_len(V, F):
    """Median edge length: how far apart the vertices are, so the splats close the gaps between them."""
    if len(F) == 0:
        return 0.015
    k = np.random.default_rng(0).choice(len(F), min(len(F), 50000), replace=False)
    return float(np.median(np.linalg.norm(V[F[k, 0]] - V[F[k, 1]], axis=1)))


class Camera:
    def __init__(self, pos, look_at, W=1600, H=1000, fov=50.0, up=(0, 0, 1.0), ortho_scale=None):
        pos, look_at = np.asarray(pos, float), np.asarray(look_at, float)
        f = look_at - pos; f /= np.linalg.norm(f)
        r = np.cross(f, up)
        if np.linalg.norm(r) < 1e-6:
            r = np.cross(f, [0, 1.0, 0])
        r /= np.linalg.norm(r); u = np.cross(r, f)
        self.R = np.stack([r, -u, f]); self.pos = pos; self.W, self.H = W, H
        self.f = (W / 2) / np.tan(np.radians(fov) / 2); self.ortho = ortho_scale; self.fwd = f

    def project(self, X):
        Xc = (np.asarray(X) - self.pos) @ self.R.T
        if self.ortho:
            return np.stack([Xc[:, 0] * self.ortho + self.W / 2, Xc[:, 1] * self.ortho + self.H / 2], 1), Xc[:, 2]
        z = Xc[:, 2]
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.stack([self.f * Xc[:, 0] / z + self.W / 2, self.f * Xc[:, 1] / z + self.H / 2], 1), z


def render(V, C, N, cam: Camera, cull=True, spacing=0.016, bg=(255, 255, 255), light=None, max_r=5):
    """Splat every vertex as a small disc, nearest wins, Lambert shading. Returns (RGB image, depth)."""
    uv, z = cam.project(V)
    ok = (z > 0.05) & np.isfinite(uv).all(1)
    if cull:
        vd = (cam.pos - V) if not cam.ortho else -cam.fwd[None]
        ok &= np.sum(N * vd, axis=1) > 0
    uv, z, col, nrm = uv[ok], z[ok], C[ok].astype(np.float32), N[ok]
    L = np.asarray(light if light is not None else (-cam.fwd + np.array([0, 0, 1.2])), float)
    L /= np.linalg.norm(L)
    shade = 0.62 + 0.38 * np.clip(np.abs(nrm @ L), 0, 1)
    col = np.clip(col * shade[:, None] * 1.05, 0, 255)
    s = np.broadcast_to(cam.ortho * spacing if cam.ortho else cam.f * spacing / z, z.shape)
    rad = np.clip(np.ceil(s * 0.75), 1, max_r).astype(int)
    W, H = cam.W, cam.H
    zb = np.full(H * W, np.inf, np.float32); img = np.empty((H * W, 3), np.float32); img[:] = bg
    px, py = np.round(uv[:, 0]).astype(int), np.round(uv[:, 1]).astype(int)
    for r in np.unique(rad):
        m = rad == r
        offs = [(dx, dy) for dx in range(-r + 1, r) for dy in range(-r + 1, r) if dx * dx + dy * dy < r * r] or [(0, 0)]
        for dx, dy in offs:
            x, y = px[m] + dx, py[m] + dy
            inb = (x >= 0) & (x < W) & (y >= 0) & (y < H)
            idx = y[inb] * W + x[inb]
            np.minimum.at(zb, idx, z[m][inb])
    for r in np.unique(rad):
        m = rad == r
        offs = [(dx, dy) for dx in range(-r + 1, r) for dy in range(-r + 1, r) if dx * dx + dy * dy < r * r] or [(0, 0)]
        for dx, dy in offs:
            x, y = px[m] + dx, py[m] + dy
            inb = (x >= 0) & (x < W) & (y >= 0) & (y < H)
            idx = y[inb] * W + x[inb]
            win = z[m][inb] <= zb[idx] + 1e-4
            img[idx[win]] = col[m][inb][win]
    return img.reshape(H, W, 3).astype(np.uint8), zb.reshape(H, W)


def render_ss(V, C, N, cam: Camera, factor: int = 2, **kw):
    """Supersampled render: draw at `factor` x the size, then shrink (smoother edges, fewer splat artefacts)."""
    big = Camera(cam.pos, cam.pos + cam.fwd, cam.W * factor, cam.H * factor,
                 fov=float(np.degrees(2 * np.arctan(cam.W / 2 / cam.f))) if not cam.ortho else 50.0,
                 ortho_scale=cam.ortho * factor if cam.ortho else None)
    if cam.ortho:
        big.R = cam.R
    kw['max_r'] = kw.get('max_r', 5) * factor
    img, zb = render(V, C, N, big, **kw)
    return cv2.resize(img, (cam.W, cam.H), interpolation=cv2.INTER_AREA), cv2.resize(zb, (cam.W, cam.H), interpolation=cv2.INTER_NEAREST)


def project_lines(img, cam: Camera, segs, col, th=3, dashed=False):
    from .annotate import _dashed
    for a, b in segs:
        uv, z = cam.project(np.array([a, b]))
        if (z < 0.05).any():
            continue
        p, q = tuple(np.round(uv[0]).astype(int)), tuple(np.round(uv[1]).astype(int))
        if dashed:
            _dashed(img, p, q, col, th)
        else:
            cv2.line(img, p, q, col, th, cv2.LINE_AA)


def change_overlay(img, cam: Camera, changes, floor_c, labels=True, which=('before', 'after')):
    """Draw each change's 3D box (both versions of a move, with an arrow) and a tag on a render."""
    from PIL import Image, ImageDraw
    from .annotate import places, tag
    tags = []
    for c in changes:
        col = COL[c['type']]
        for o, ep, role in places(c):
            if ep not in which:
                continue
            K = box_corners(o, floor_c)
            project_lines(img, cam, [(K[i], K[j]) for i, j in EDGES], col, th=3)
            uv, z = cam.project(K)
            if (z > 0).all():
                tags.append(((int(uv[:, 0].min()), int(uv[:, 1].min())), o, c, role))
        if c['type'] == 'moved' and len(which) == 2:
            a = np.array(c['from_obj']['centroid'], float); b = np.array(c['to_obj']['centroid'], float)
            a[2] += floor_z(a[None, :2], floor_c)[0] + 0.35; b[2] += floor_z(b[None, :2], floor_c)[0] + 0.35
            mid = (a + b) / 2 + np.array([0, 0, 0.35])
            pts = [(1 - t) ** 2 * a + 2 * (1 - t) * t * mid + t ** 2 * b for t in np.linspace(0, 1, 24)]
            project_lines(img, cam, list(zip(pts[:-1], pts[1:])), col, th=4)
            uv, z = cam.project(np.array([pts[-3], pts[-1]]))
            if (z > 0).all():
                cv2.arrowedLine(img, tuple(np.round(uv[0]).astype(int)), tuple(np.round(uv[1]).astype(int)), col, 4,
                                cv2.LINE_AA, tipLength=1.2)
    if labels:
        pil = Image.fromarray(img); d = ImageDraw.Draw(pil); f = font(max(14, img.shape[1] // 70))
        seen = set()
        for xy, o, c, role in tags:
            key = (c['id'], role)
            if key in seen:
                continue
            seen.add(key)
            txt = {'from': f"{c['id']} {c['label']} (before)", 'to': f"{c['id']} {c['label']} (after)"}.get(
                role, f"{c['id']} {c['type']}: {c['label']}")
            tag(d, xy, txt, COL[c['type']], f, img.shape[1], img.shape[0])
        img[:] = np.asarray(pil)
    return img


def dollhouse_cameras(lo, hi, floor=0.0, W=1600, H=1000):
    """Four elevated corner views looking into the room (the near walls are seen from behind and culled)."""
    c = (np.asarray(lo[:2]) + np.asarray(hi[:2])) / 2
    span = np.asarray(hi[:2]) - np.asarray(lo[:2])
    out = []
    for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
        pos = np.r_[c + np.array([sx, sy]) * span * 0.95, floor + 3.6]
        out.append(Camera(pos, np.r_[c, floor + 0.35], W, H, fov=55))
    return out


def best_dollhouse(V, C, N, changes, floor_c, lo, hi, obj_pts):
    """The corner view that shows the changed objects best (most of their surface unoccluded)."""
    best = (-1, None)
    for cam in dollhouse_cameras(lo, hi, W=400, H=250):
        _, zb = render(V, C, N, cam, max_r=2)
        score = 0
        for P in obj_pts:
            uv, z = cam.project(P)
            ok = (z > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < 400) & (uv[:, 1] >= 0) & (uv[:, 1] < 250)
            if ok.any():
                d = zb[uv[ok, 1].astype(int), uv[ok, 0].astype(int)]
                score += np.mean(z[ok] <= d + 0.06) * ok.mean()
        if score > best[0]:
            best = (score, cam)
    cam = best[1]
    return Camera(cam.pos, cam.pos + cam.fwd, 1600, 1000, fov=55)


def ceiling_cut(V, F, C, floor_c, h=1.95):
    from .mesh import crop_mesh
    keep = (V[:, 2] - floor_z(V[:, :2], floor_c)) < h
    return crop_mesh(V, F, C, keep)


# ---------------------------------------------------------------- floor plan
def topdown(V, C, N, lo, hi, px_per_m=200, spacing=0.016):
    """Orthographic render from above: the highest surface below the ceiling cut, with relief shading."""
    W = int((hi[0] - lo[0]) * px_per_m); H = int((hi[1] - lo[1]) * px_per_m)
    c = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, 10.0])
    cam = Camera(c, c - np.array([0, 0, 1.0]), W, H, up=(0, 1.0, 0), ortho_scale=px_per_m)
    img, zb = render(V, C, N, cam, cull=False, light=(0.4, -0.5, 1.0), spacing=spacing, max_r=3)
    # orthographic camera looking down with up=+y: image x = +x, image y = -y; flip so row 0 = max y
    return img, [lo[0], lo[0] + W / px_per_m, lo[1], lo[1] + H / px_per_m]


def _rect(fp, **kw):
    a = np.radians(fp['angle']); c = np.array(fp['center']); hw, hd = np.array(fp['size']) / 2
    ux, uy = np.array([np.cos(a), np.sin(a)]), np.array([-np.sin(a), np.cos(a)])
    return Polygon([c - hw * ux - hd * uy, c + hw * ux - hd * uy, c + hw * ux + hd * uy, c - hw * ux + hd * uy], closed=True, **kw)


def floor_plan(path, bg, extent, landmarks, changes, unknown_xy=None, traj=None, next_views=None, start=None,
               title='Floor plan (room frame, metres)'):
    fig, ax = plt.subplots(figsize=(10, 10 * (extent[3] - extent[2]) / (extent[1] - extent[0]) + 0.8), dpi=130)
    ax.imshow(bg, extent=extent, origin='upper', interpolation='bilinear')
    if unknown_xy is not None and len(unknown_xy):
        ax.scatter(unknown_xy[:, 0], unknown_xy[:, 1], s=4, color=UNKNOWN, alpha=0.35, lw=0, zorder=2)
    if traj:
        for ep, P in traj.items():
            ax.plot(P[:, 0], P[:, 1], '-', color=(0.15, 0.35, 0.8) if ep == 'before' else (0.55, 0.2, 0.65), lw=0.7,
                    alpha=0.35, zorder=3)
    for lm in landmarks:
        ax.add_patch(_rect(lm['footprint'], fill=False, ec='white', lw=1.4, ls='-', zorder=4))
        ax.add_patch(_rect(lm['footprint'], fill=False, ec='black', lw=0.6, ls='-', zorder=4))
        ax.text(*lm['xy'], lm['label'], ha='center', va='center', fontsize=9, color='black', zorder=5,
                bbox=dict(boxstyle='round,pad=0.2', fc=(1, 1, 1, 0.75), ec='none'))
    for c in changes:
        col = RGB[c['type']]
        objs = [c['from_obj'], c['to_obj']] if c['type'] == 'moved' else [c['obj']]
        for k, o in enumerate(objs):
            ls = '--' if (c['type'] == 'moved' and k == 0) or c['type'] == 'removed' else '-'
            ax.add_patch(_rect(o['footprint'], fill=True, fc=(*col, 0.25), ec=col, lw=2.2, ls=ls, zorder=6))
        if c['type'] == 'moved':
            a, b = c['from_obj']['centroid'][:2], c['to_obj']['centroid'][:2]
            ax.add_patch(FancyArrowPatch(a, b, arrowstyle='-|>', mutation_scale=22, color=col, lw=2.5,
                                         connectionstyle='arc3,rad=0.25', zorder=7))
        o = c.get('to_obj') or c['obj']
        if c['type'] == 'moved':
            txt = f"{c['id']} {c['label']} moved {c['displacement_m']:.2f} m" + (
                f", {c['yaw_deg']:+.0f}°" if c.get('yaw_confident') and c.get('yaw_deg') is not None else '')
        else:
            txt = f"{c['id']} {c['label']} {c['type']}"
        ax.annotate(txt, o['centroid'][:2], xytext=(12, -18), textcoords='offset points', fontsize=10, color='white',
                    fontweight='bold', zorder=9, bbox=dict(boxstyle='round,pad=0.3', fc=col, ec='none'))
    if next_views:
        for k, v in enumerate(next_views[:3]):
            p = v['position'][:2]
            ax.add_patch(Wedge(p, 0.45, v['yaw_deg'] - v['hfov_deg'] / 2, v['yaw_deg'] + v['hfov_deg'] / 2,
                               fc=(0.1, 0.6, 0.9, 0.25), ec=(0.1, 0.45, 0.8), lw=1.2, zorder=8))
            ax.plot(*p, 'o', color=(0.1, 0.45, 0.8), ms=6, zorder=9)
            ax.text(p[0], p[1] - 0.12, f'look here next #{k + 1}', fontsize=8, color=(0.05, 0.3, 0.6), ha='center',
                    va='top', zorder=9, bbox=dict(boxstyle='round,pad=0.15', fc=(1, 1, 1, 0.8), ec='none'))
    if start is not None:
        ax.plot(*start[:2], marker='*', color='gold', mec='black', ms=16, zorder=10)
    ax.plot(0, 0, 'k+', ms=12, mew=2, zorder=10)
    ax.annotate('origin (0, 0)', (0, 0), xytext=(6, 6), textcoords='offset points', fontsize=8)
    ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3]); ax.set_aspect('equal')
    ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)'); ax.grid(alpha=0.25, lw=0.5); ax.set_title(title)
    handles = [Line2D([], [], color=RGB['moved'], lw=3, label='moved (dashed: old place)'),
               Line2D([], [], color=RGB['removed'], lw=3, ls='--', label='removed'),
               Line2D([], [], color=RGB['added'], lw=3, label='added'),
               Line2D([], [], color=UNKNOWN, marker='s', ls='', alpha=0.6, label='could not be checked'),
               Line2D([], [], color=(0.15, 0.35, 0.8), lw=1, label='before walk-through'),
               Line2D([], [], color=(0.55, 0.2, 0.65), lw=1, label='after walk-through')]
    if next_views:
        handles.append(Line2D([], [], color=(0.1, 0.45, 0.8), marker='o', ls='', label='where to look next'))
    if start is not None:
        handles.append(Line2D([], [], color='gold', marker='*', mec='black', ms=12, ls='', label='door / start'))
    ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.07), ncol=3, fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(path, bbox_inches='tight'); plt.close(fig)


# ---------------------------------------------------------------- robot maps and routes
def robot_figure(path, nav, grids, changes, landmarks):
    names = list(nav['robots'])
    fig, axes = plt.subplots(1, len(names), figsize=(7.2 * len(names), 7.6), dpi=120)
    axes = np.atleast_1d(axes)
    for ax, name in zip(axes, names):
        r = nav['robots'][name]; G = grids[name]
        spec = G['spec']; x0, y0, nx, ny, res = spec
        ext = [x0, x0 + nx * res, y0, y0 + ny * res]
        g = G['map_updated']
        img = np.zeros(g.shape + (3,)); img[:] = (0.86, 0.86, 0.88)
        img[g == 0] = (1, 1, 1); img[g == 100] = (0.2, 0.2, 0.22)
        trav = G['trav_updated']; img[(g == 0) & ~trav] = (1.0, 0.86, 0.84)
        img[G['reach_updated']] = (0.86, 0.95, 0.87)
        ax.imshow(img, extent=ext, origin='lower', interpolation='nearest')
        for c in changes:
            objs = [c['from_obj'], c['to_obj']] if c['type'] == 'moved' else [c['obj']]
            for k, o in enumerate(objs):
                ls = '--' if (c['type'] == 'moved' and k == 0) or c['type'] == 'removed' else '-'
                ax.add_patch(_rect(o['footprint'], fill=False, ec=RGB[c['type']], lw=2, ls=ls, zorder=5))
        for lm in landmarks:
            ax.text(*lm['xy'], lm['label'], ha='center', va='center', fontsize=7, color=(0.3, 0.3, 0.3), zorder=4)
        for key in ('before', 'updated'):
            for gname, rt in r['plans'][key].items():
                if not rt.get('reachable'):
                    continue
                W = np.array(rt['waypoints'])
                if key == 'before':
                    ax.plot(W[:, 0], W[:, 1], '--', color=(0.45, 0.45, 0.45), lw=1.6, zorder=6)
                else:
                    ax.plot(W[:, 0], W[:, 1], '-', color=(0.05, 0.45, 0.85), lw=2.2, zorder=7)
                    gx, gy = rt['goal']; yaw = np.radians(rt['yaw_deg'])
                    ax.arrow(gx, gy, 0.18 * np.cos(yaw), 0.18 * np.sin(yaw), width=0.015, color=(0.05, 0.45, 0.85), zorder=8)
        if r.get('start'):
            ax.plot(*r['start'], marker='*', color='gold', mec='black', ms=16, zorder=9)
        a = r['reachable_area_m2']
        lines = [f"{r['profile']['label'] or name}", f"reachable floor: {a['before']:.2f} → {a['updated']:.2f} m²"]
        for row in r['routes']:
            if row['status'] not in ('unchanged', 'unreachable'):
                lines.append(f"{row['goal']}: {row['status']}" + (f" ({', '.join(row['causes'])})" if row['causes'] else ''))
        ax.set_title('\n'.join(lines[:2]), fontsize=11)
        ax.text(0.01, 0.01, '\n'.join(lines[2:]) or 'no route changes', transform=ax.transAxes, fontsize=9, va='bottom',
                bbox=dict(fc=(1, 1, 1, 0.85), ec='none'), zorder=10)
        ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3]); ax.set_aspect('equal'); ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
    handles = [Rectangle((0, 0), 1, 1, fc='white', ec='grey', label='free'),
               Rectangle((0, 0), 1, 1, fc=(0.2, 0.2, 0.22), label='occupied'),
               Rectangle((0, 0), 1, 1, fc=(0.86, 0.86, 0.88), label='unknown'),
               Rectangle((0, 0), 1, 1, fc=(1.0, 0.86, 0.84), label='too close to an obstacle'),
               Rectangle((0, 0), 1, 1, fc=(0.86, 0.95, 0.87), label='reachable from the door'),
               Line2D([], [], color=(0.45, 0.45, 0.45), ls='--', label="yesterday's route"),
               Line2D([], [], color=(0.05, 0.45, 0.85), label='route on the updated map')]
    fig.legend(handles=handles, loc='lower center', ncol=7, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 1)); fig.savefig(path, bbox_inches='tight'); plt.close(fig)


# ---------------------------------------------------------------- evaluation pictures
def naive_vs_ours(path, Z, n_naive, n_ours):
    P, L, nv = Z['P_before'], Z['L_before'], Z['naive_before']
    rng = np.random.default_rng(0); k = rng.choice(len(P), min(len(P), 120000), replace=False)
    P, L, nv = P[k], L[k], nv[k]
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.2), dpi=120)
    ax = axes[0]
    ax.scatter(P[~nv, 0], P[~nv, 1], s=0.5, color=(0.75, 0.75, 0.75), lw=0)
    ax.scatter(P[nv, 0], P[nv, 1], s=0.8, color=RGB['removed'], lw=0)
    ax.set_title(f'Naive point-cloud diff\nred = "no matching point in the after video" → {n_naive} objects reported')
    ax = axes[1]
    for lab, col, s in ((0, (0.75, 0.75, 0.75), 0.5), (-1, UNKNOWN, 0.6), (1, RGB['removed'], 1.2)):
        m = L == lab
        ax.scatter(P[m, 0], P[m, 1], s=s, color=col, lw=0)
    ax.set_title(f'See-through test\nred = seen through (gone), purple = could not be checked → {n_ours} changes')
    for ax in axes:
        ax.set_aspect('equal'); ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)'); ax.grid(alpha=0.2)
    fig.suptitle('The same before-video surfaces, judged two ways (top view)', fontsize=12)
    fig.tight_layout(); fig.savefig(path, bbox_inches='tight'); plt.close(fig)


def coverage_chart(path, coverage):
    fig, ax = plt.subplots(figsize=(9, 2.6), dpi=130)
    parts = [('same', 'same', (0.7, 0.7, 0.7)), ('gone', 'seen through (gone)', RGB['removed']),
             ('unknown_never_in_view', 'never in view', (0.62, 0.48, 0.86)),
             ('unknown_occluded', 'hidden behind something', (0.52, 0.36, 0.78)),
             ('unknown_too_few_views', 'too few views', (0.42, 0.26, 0.66)),
             ('unknown_self_inconsistent', "own video inconsistent", (0.32, 0.18, 0.5))]
    rows = [('before', 'before surfaces,\nchecked by the after video'), ('after', 'after surfaces,\nchecked by the before video')]
    for i, (ep, name) in enumerate(rows):
        left = 0
        for key, lab, col in parts:
            v = coverage[ep].get(key, 0) * 100
            ax.barh(i, v, left=left, color=col, label=lab if i == 0 else None, edgecolor='white', lw=0.5)
            if v > 4:
                ax.text(left + v / 2, i, f'{v:.0f}%', ha='center', va='center', fontsize=8, color='white')
            left += v
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[1] for r in rows], fontsize=8); ax.invert_yaxis()
    ax.set_xlim(0, 100); ax.set_xlabel('% of surface points')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.35), ncol=3, fontsize=7.5, frameon=False)
    fig.tight_layout(); fig.savefig(path, bbox_inches='tight'); plt.close(fig)
