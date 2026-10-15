"""Changes drawn into the real video frames.

Every keyframe of each walk-through gets the 3D boxes of the changes projected into it. A solid box is an
object that is there in this video; a dashed box is a place that is empty in this video (where something was,
or will appear). Each box is checked against the frame's depth: '(hidden)' when something nearer is in the
way. The same boxes give the before/after comparison images of each change, from frames that look at the same
place from similar viewpoints."""
from __future__ import annotations
import os
import numpy as np
import cv2

from .geometry import floor_z

COL = {'removed': (220, 60, 60), 'added': (40, 170, 70), 'moved': (245, 150, 20)}      # RGB
EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]


def font(size: int):
    from PIL import ImageFont
    try:
        from matplotlib import font_manager
        return ImageFont.truetype(font_manager.findfont('DejaVu Sans'), size)
    except Exception:
        return ImageFont.load_default()


def box_corners(o, floor_c, pad: float = 0.03):
    fp = o['footprint']; a = np.radians(fp['angle'])
    ux, uy = np.array([np.cos(a), np.sin(a)]), np.array([-np.sin(a), np.cos(a)])
    c = np.array(fp['center']); hw, hd = np.array(fp['size']) / 2 + pad
    xy = np.array([c - hw * ux - hd * uy, c + hw * ux - hd * uy, c + hw * ux + hd * uy, c - hw * ux + hd * uy])
    fz = floor_z(xy, floor_c)
    z0, z1 = max(0.0, o['z_range'][0]) - pad, o['z_range'][1] + pad
    return np.vstack([np.c_[xy, fz + z0], np.c_[xy, fz + z1]])


def places(c):
    """The places a change involves: (object dict, epoch it is present in, role)."""
    if c['type'] == 'moved':
        return [(c['from_obj'], 'before', 'from'), (c['to_obj'], 'after', 'to')]
    return [(c['obj'], 'before' if c['type'] == 'removed' else 'after', c['type'])]


def frame_state(v, D, Q, present: bool):
    """What a frame (view at depth resolution, depth D) shows at probe points Q (room coordinates)."""
    H, W = D.shape
    uv, z = v.project(Q)
    ins = (z > 0.2) & (uv[:, 0] >= 0) & (uv[:, 0] < W - 1) & (uv[:, 1] >= 0) & (uv[:, 1] < H - 1)
    f_in = float(ins.mean()) if len(Q) else 0.0
    if ins.sum() < 5:
        return dict(in_view=f_in, state='out of view', score=0.0)
    d = D[uv[ins, 1].astype(int), uv[ins, 0].astype(int)]
    zz = z[ins]
    ok = np.isfinite(d)
    if ok.sum() < 5:
        return dict(in_view=f_in, state='unclear', score=0.0)
    r = d[ok] - zz[ok]
    tol = np.maximum(0.06, 0.06 * zz[ok])
    same = np.mean(np.abs(r) <= tol); free = np.mean(r > np.maximum(0.10, 0.10 * zz[ok])); occl = np.mean(r < -tol)
    if occl >= 0.5:
        st = 'hidden'
    elif present and same >= 0.3:
        st = 'visible'
    elif not present and free >= 0.5:
        st = 'empty'
    else:
        st = 'unclear'
    good = same if present else free
    return dict(in_view=f_in, state=st, score=float(f_in * good * (1 - occl)), dist=float(np.median(zz)))


def _clip_project(v, A, B, near=0.15):
    """Project a 3D segment, clipped to the near plane. -> two pixel points or None."""
    ca, cb = v.R @ A + v.t, v.R @ B + v.t
    if ca[2] < near and cb[2] < near:
        return None
    if ca[2] < near:
        ca = cb + (ca - cb) * (cb[2] - near) / (cb[2] - ca[2])
    elif cb[2] < near:
        cb = ca + (cb - ca) * (ca[2] - near) / (ca[2] - cb[2])
    out = []
    for p in (ca, cb):
        x, y = p[0] / p[2], p[1] / p[2]
        if v.dist:
            d = 1 + v.dist * (x * x + y * y); x, y = x * d, y * d
        out.append((v.K[0, 0] * x + v.K[0, 2], v.K[1, 1] * y + v.K[1, 2]))
    return out


def _dashed(img, p, q, col, th, dash=14):
    p, q = np.array(p, float), np.array(q, float)
    L = np.linalg.norm(q - p)
    n = max(1, int(L / dash))
    for k in range(0, n, 2):
        a = p + (q - p) * k / n; b = p + (q - p) * min(k + 1, n) / n
        cv2.line(img, tuple(np.round(a).astype(int)), tuple(np.round(b).astype(int)), col, th, cv2.LINE_AA)


def draw_box(img, v, corners, col, dashed=False, th=3):
    """img: RGB at the view's resolution. Returns the 2D bounding box of what was drawn, or None."""
    H, W = img.shape[:2]
    pts = []
    for i, j in EDGES:
        s = _clip_project(v, corners[i], corners[j])
        if s is None:
            continue
        ok, p, q = cv2.clipLine((0, 0, W, H), tuple(int(round(c)) for c in s[0]), tuple(int(round(c)) for c in s[1]))
        if not ok:
            continue
        (_dashed if dashed else (lambda im, a, b, c, t: cv2.line(im, a, b, c, t, cv2.LINE_AA)))(img, p, q, col, th)
        pts += [p, q]
    if not pts:
        return None
    P = np.array(pts)
    return [int(P[:, 0].min()), int(P[:, 1].min()), int(P[:, 0].max()), int(P[:, 1].max())]


def tag(draw, xy, text, col, fnt, W, H, pad=5, avoid=()):
    """A filled label above `xy`, kept inside the image and moved down past labels already placed."""
    x, y = xy
    l, t, r, b = draw.textbbox((0, 0), text, font=fnt)
    w, h = r - l + 2 * pad, b - t + 2 * pad
    x = int(np.clip(x, 0, W - w)); y = int(np.clip(y - h - 2, 0, H - h))
    for _ in range(12):
        hit = [q for q in avoid if q and x < q[2] and q[0] < x + w and y < q[3] and q[1] < y + h]
        if not hit:
            break
        y = int(min(H - h, max(q[3] for q in hit) + 2))
    draw.rectangle([x, y, x + w, y + h], fill=tuple(int(c) for c in col))
    draw.text((x + pad - l, y + pad - t), text, fill=(255, 255, 255), font=fnt)
    return (x, y, x + w, y + h)


def label_for(c, role, present, state):
    lab = c['label']
    if c['type'] == 'moved':
        if present:
            txt = f"{c['id']} {lab}" + (' (will move)' if role == 'from' else f" moved here, {c['displacement_m']:.1f} m"
                                         + (f", turned {abs(c['yaw_deg']):.0f}°" if c.get('yaw_confident') and c.get('yaw_deg') else ''))
        else:
            txt = f"{c['id']} {lab} " + ('was here' if role == 'from' else 'will be here')
    elif c['type'] == 'removed':
        txt = f"{c['id']} REMOVED: {lab}" if present else f"{c['id']} {lab} was here (gone)"
    else:
        txt = f"{c['id']} ADDED: {lab}" if present else f"{c['id']} {lab} will appear here"
    if state == 'hidden':
        txt += ' (hidden)'
    return txt


class Annotator:
    def __init__(self, meta, views, depths, changes, floor_c, probes, image_dir, times=None):
        """probes: (change id, role) -> probe points (room frame) on the object's surface; times: name -> seconds."""
        from .scene import image_view
        self.meta, self.views, self.depths, self.changes = meta, views, depths, changes
        self.times = times or {}
        self.floor_c, self.probes, self.image_dir = floor_c, probes, image_dir
        self.iview = {n: image_view(meta, v) for n, v in views.items()}
        self.corners = {(c['id'], role): box_corners(o, floor_c) for c in changes for o, _, role in places(c)}
        self._f = {}

    def fnt(self, size):
        if size not in self._f:
            self._f[size] = font(size)
        return self._f[size]

    def states(self, n):
        v, D = self.views[n], self.depths.get(n)
        out = {}
        for c in self.changes:
            for o, ep, role in places(c):
                present = ep == v.epoch
                Q = self.probes[(c['id'], role)]
                out[(c['id'], role)] = frame_state(v, D, Q, present) if D is not None else dict(in_view=0, state='out of view', score=0)
        return out

    def draw(self, n, header=True, only=None, scale_text=1.0, roles=None):
        from PIL import Image, ImageDraw
        v = self.iview[n]
        img = cv2.cvtColor(cv2.imread(os.path.join(self.image_dir, n)), cv2.COLOR_BGR2RGB)
        H, W = img.shape[:2]
        st = self.states(n)
        th = max(2, int(round(W / 400)))
        tags = []
        for c in self.changes:
            if only is not None and c['id'] not in only:
                continue
            for o, ep, role in places(c):
                if roles is not None and role not in roles:
                    continue
                s = st[(c['id'], role)]
                if s['state'] == 'out of view' or s['in_view'] < 0.15:
                    continue
                present = ep == v.epoch
                col = COL[c['type']]
                bb = draw_box(img, v, self.corners[(c['id'], role)], col, dashed=not present or s['state'] == 'hidden',
                              th=th if present else max(2, th - 1))
                if bb:
                    tags.append(((bb[0], bb[1]), label_for(c, role, present, s['state']), col))
        pil = Image.fromarray(img); d = ImageDraw.Draw(pil)
        f = self.fnt(int(W / 55 * scale_text))
        placed = []
        for xy, text, col in tags:
            placed.append(tag(d, xy, text, col, f, W, H, avoid=placed))
        if header:
            t = self.times.get(n)
            txt = f"{v.epoch.upper()}   frame {v.frame}" + (f"  ({int(t // 60)}:{int(t % 60):02d})" if t is not None else '')
            hf = self.fnt(int(W / 45))
            l, tp, r, b = d.textbbox((0, 0), txt, font=hf)
            d.rectangle([0, 0, r - l + 24, b - tp + 16], fill=(25, 25, 35) if v.epoch == 'before' else (60, 30, 90))
            d.text((12 - l, 8 - tp), txt, fill=(255, 255, 255), font=hf)
            leg = [('solid: there in this video', (235, 235, 235)), ('dashed: empty here in this video', (235, 235, 235))]
            lf = self.fnt(int(W / 70)); y = H - 10
            for text, col in leg[::-1]:
                l, tp, r, b = d.textbbox((0, 0), text, font=lf)
                y -= (b - tp) + 10
                d.rectangle([8, y - 4, 8 + r - l + 12, y + b - tp + 4], fill=(25, 25, 35))
                d.text((14 - l, y - tp), text, fill=col, font=lf)
        return np.asarray(pil), st


def best_pairs(ann: Annotator, c, max_angle: float = 50.0):
    """For each place of a change: the before frame and the after frame that see it best, from similar views."""
    out = []
    names = {ep: [n for n in sorted(ann.views) if ann.views[n].epoch == ep and n in ann.depths] for ep in ('before', 'after')}
    for o, ep_present, role in places(c):
        cand = {}
        for ep in ('before', 'after'):
            present = ep == ep_present
            lst = []
            for n in names[ep]:
                s = frame_state(ann.views[n], ann.depths[n], ann.probes[(c['id'], role)], present)
                if s['state'] in ('visible', 'empty') and s['in_view'] > 0.6:
                    lst.append((s['score'], n))
            cand[ep] = sorted(lst, reverse=True)[:25]
        if not cand['before'] or not cand['after']:
            out.append((role, cand['before'][0][1] if cand['before'] else None, cand['after'][0][1] if cand['after'] else None))
            continue
        best = (-1e9, None, None)
        for sb, nb in cand['before']:
            vb = ann.views[nb]
            for sa, na in cand['after']:
                va = ann.views[na]
                ang = np.degrees(np.arccos(np.clip(vb.forward @ va.forward, -1, 1)))
                score = sb + sa - 0.004 * ang - 0.15 * np.linalg.norm(vb.C - va.C)
                if ang < max_angle and score > best[0]:
                    best = (score, nb, na)
        if best[1] is None:
            best = (0, cand['before'][0][1], cand['after'][0][1])
        out.append((role, best[1], best[2]))
    return out


def crop_to(img, bb, aspect=16 / 9, min_frac=0.55):
    H, W = img.shape[:2]
    if bb is None:
        return img
    cx, cy = (bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2
    w = max(bb[2] - bb[0], (bb[3] - bb[1]) * aspect) * 1.8
    w = float(np.clip(w, min_frac * W, W)); h = w / aspect
    x0 = int(np.clip(cx - w / 2, 0, W - w)); y0 = int(np.clip(cy - h / 2, 0, H - h))
    return img[y0:y0 + int(h), x0:x0 + int(w)]


def pair_image(ann: Annotator, c, sentence: str, path: str, width: int = 1600):
    """Before | after for each place of the change (two rows for a move), titled with the sentence."""
    from PIL import Image, ImageDraw
    rows = []
    for role, nb, na in best_pairs(ann, c):
        tiles = []
        for n, ep in ((nb, 'before'), (na, 'after')):
            if n is None:
                tiles.append(np.full((450, 800, 3), 40, np.uint8)); continue
            img, _ = ann.draw(n, header=False, only=[c['id']], scale_text=1.3, roles=[role])
            v = ann.iview[n]
            bb = draw_box(img.copy(), v, ann.corners[(c['id'], role)], (0, 0, 0))
            t = cv2.resize(crop_to(img, bb), (800, 450), interpolation=cv2.INTER_AREA)
            pil = Image.fromarray(t); d = ImageDraw.Draw(pil); f = ann.fnt(26)
            lab = f"{ep.upper()}  ·  frame {ann.views[n].frame}"
            l, tp, r, b = d.textbbox((0, 0), lab, font=f)
            d.rectangle([0, 0, r - l + 20, b - tp + 14], fill=(25, 25, 35) if ep == 'before' else (60, 30, 90))
            d.text((10 - l, 7 - tp), lab, fill=(255, 255, 255), font=f)
            tiles.append(np.asarray(pil))
        rows.append(np.hstack([tiles[0], np.full((450, 8, 3), 255, np.uint8), tiles[1]]))
    body = np.vstack([np.vstack([r, np.full((8, r.shape[1], 3), 255, np.uint8)]) for r in rows])[:-8]
    f = font(24); col = COL[c['type']]
    probe = ImageDraw.Draw(Image.new('RGB', (10, 10)))
    lines, cur = [], ''
    for word in sentence.split():                    # wrap the sentence to the image width
        test = (cur + ' ' + word).strip()
        if probe.textlength(test, font=f) > body.shape[1] - 48 and cur:
            lines.append(cur); cur = word
        else:
            cur = test
    lines.append(cur)
    th = 22 + 32 * len(lines)
    title = Image.new('RGB', (body.shape[1], th), (255, 255, 255)); d = ImageDraw.Draw(title)
    d.rectangle([0, 0, 12, th], fill=col)
    for i, ln in enumerate(lines):
        d.text((24, 12 + 32 * i), ln, fill=(20, 20, 20), font=f)
    out = np.vstack([np.asarray(title), body])
    if out.shape[1] != width:
        out = cv2.resize(out, (width, int(out.shape[0] * width / out.shape[1])), interpolation=cv2.INTER_AREA)
    cv2.imwrite(path, cv2.cvtColor(out, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
    return path


def write_video(frames_iter, path: str, fps: float = 3.0, size=(1280, 720)):
    """H.264 MP4 (plays in browsers and on GitHub) through the ffmpeg that ships with imageio-ffmpeg."""
    import imageio_ffmpeg
    w = imageio_ffmpeg.write_frames(path, size, fps=fps, codec='libx264', pix_fmt_out='yuv420p', macro_block_size=8, quality=None,
                                    output_params=['-crf', '27', '-preset', 'medium', '-movflags', '+faststart'])
    w.send(None)
    n = 0
    for f in frames_iter:
        if (f.shape[1], f.shape[0]) != size:
            f = cv2.resize(f, size, interpolation=cv2.INTER_AREA)
        w.send(np.ascontiguousarray(f)); n += 1
    w.close()
    return n


def annotated_videos(ann: Annotator, out_dir: str, fps: float = 3.0, width: int = 1280, log=print):
    """video/<epoch>_annotated.mp4 for both walk-throughs, plus per-frame states (for the report and viewers)."""
    os.makedirs(out_dir, exist_ok=True)
    states = {}
    for ep in ('before', 'after'):
        names = sorted([n for n in ann.views if ann.views[n].epoch == ep], key=lambda n: ann.views[n].frame)
        if not names:
            continue
        v0 = ann.iview[names[0]]
        size = (width, int(round(width * v0.height / v0.width / 2)) * 2)

        def gen():
            for n in names:
                img, st = ann.draw(n)
                states[n] = {f'{k[0]}:{k[1]}': s['state'] for k, s in st.items()}
                yield img
        k = write_video(gen(), os.path.join(out_dir, f'{ep}_annotated.mp4'), fps=fps, size=size)
        log(f'  video/{ep}_annotated.mp4: {k} keyframes')
    return states
