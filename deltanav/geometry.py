"""Cameras, reconstruction loading, and the room frame (gravity, walls, floor, metric scale)."""
from __future__ import annotations
import os
import numpy as np
from dataclasses import dataclass, field


@dataclass
class View:
    name: str            # image path relative to the frames folder, e.g. before/b_01234.jpg
    epoch: str           # 'before' | 'after'
    frame: int           # frame index in the source video
    K: np.ndarray        # 3x3 intrinsics (pixels, for width x height below)
    R: np.ndarray        # 3x3 world->camera rotation
    t: np.ndarray        # 3   world->camera translation
    width: int
    height: int
    dist: float = 0.0    # SIMPLE_RADIAL k1

    @property
    def C(self) -> np.ndarray:            # camera centre in world
        return -self.R.T @ self.t

    @property
    def forward(self) -> np.ndarray:      # viewing direction in world
        return self.R.T @ np.array([0.0, 0.0, 1.0])

    @property
    def up(self) -> np.ndarray:           # image 'up' direction in world
        return -(self.R.T @ np.array([0.0, 1.0, 0.0]))

    def project(self, X: np.ndarray):
        """World points (N,3) -> pixel coords (N,2) and depth along the optical axis (N,)."""
        Xc = X @ self.R.T + self.t
        z = Xc[:, 2]
        with np.errstate(divide='ignore', invalid='ignore'):
            x = Xc[:, 0] / z; y = Xc[:, 1] / z
        if self.dist:
            r2 = x * x + y * y; d = 1 + self.dist * r2; x = x * d; y = y * d
        u = self.K[0, 0] * x + self.K[0, 2]; v = self.K[1, 1] * y + self.K[1, 2]
        return np.stack([u, v], 1), z

    def scaled(self, width: int, height: int) -> "View":
        sx, sy = width / self.width, height / self.height
        K = self.K.copy(); K[0] *= sx; K[1] *= sy
        return View(self.name, self.epoch, self.frame, K, self.R, self.t, width, height, self.dist)


def epoch_of(name: str) -> str:
    return 'before' if os.path.basename(name).startswith('b_') else 'after'


def frame_of(name: str) -> int:
    return int(os.path.basename(name).split('_')[1].split('.')[0])


def load_colmap(path: str):
    """pycolmap reconstruction -> (views, points xyz (N,3), rgb (N,3) uint8, point ids, reprojection error, track length)."""
    import pycolmap
    rec = pycolmap.Reconstruction(path)
    views = {}
    for im in rec.images.values():
        if hasattr(im, 'has_pose') and not im.has_pose:
            continue
        cam = rec.cameras[im.camera_id]
        p = cam.params
        if cam.model.name in ('SIMPLE_RADIAL', 'SIMPLE_PINHOLE'):
            f, cx, cy = p[0], p[1], p[2]; k = p[3] if len(p) > 3 else 0.0
            K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
        else:
            fx, fy, cx, cy = p[0], p[1], p[2], p[3]; k = p[4] if len(p) > 4 else 0.0
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
        cfw = im.cam_from_world() if callable(im.cam_from_world) else im.cam_from_world
        R = np.asarray(cfw.rotation.matrix()); t = np.asarray(cfw.translation)
        views[im.name] = View(im.name, epoch_of(im.name), frame_of(im.name), K, R, t, cam.width, cam.height, float(k))
    ids = np.array(list(rec.points3D.keys()))
    xyz = np.array([rec.points3D[i].xyz for i in ids])
    rgb = np.array([rec.points3D[i].color for i in ids], dtype=np.uint8)
    err = np.array([rec.points3D[i].error for i in ids])
    tl = np.array([rec.points3D[i].track.length() for i in ids])
    return views, xyz, rgb, ids, err, tl


def sparse_observations(path: str, ids: np.ndarray) -> dict:
    """name -> (uv (M,2) pixels at the reconstruction image size, idx (M,) indices into `ids`)."""
    import pycolmap
    rec = pycolmap.Reconstruction(path)
    index = {int(i): k for k, i in enumerate(ids)}
    out = {}
    for im in rec.images.values():
        uv, idx = [], []
        for p2 in im.points2D:
            if p2.has_point3D():
                k = index.get(int(p2.point3D_id))
                if k is not None:
                    uv.append(p2.xy); idx.append(k)
        out[im.name] = (np.array(uv, np.float32).reshape(-1, 2), np.array(idx, np.int64))
    return out


@dataclass
class RoomFrame:
    """Similarity transform reconstruction -> room:  X_room = s * R @ (X - o)."""
    R: np.ndarray
    o: np.ndarray
    s: float = 1.0
    notes: dict = field(default_factory=dict)

    def apply(self, X):
        return self.s * (np.asarray(X) - self.o) @ self.R.T

    def view(self, v: View) -> View:
        Rn = v.R @ self.R.T
        Cn = self.apply(v.C[None])[0]
        return View(v.name, v.epoch, v.frame, v.K, Rn, -Rn @ Cn, v.width, v.height, v.dist)


def _manhattan_yaw(xy: np.ndarray) -> float:
    best = None
    for th in np.radians(np.arange(0, 90, 0.25)):
        c, s = np.cos(th), np.sin(th)
        r = xy @ np.array([[c, -s], [s, c]]).T
        H = 0.0
        for k in range(2):
            h, _ = np.histogram(r[:, k], bins=256); p = h / max(1, h.sum()); p = p[p > 0]
            H += -(p * np.log(p)).sum()
        if best is None or H < best[0]:
            best = (H, th)
    return best[1]


def estimate_room_frame(views: dict, xyz: np.ndarray, keep: np.ndarray | None = None) -> RoomFrame:
    """Gravity: a phone held without roll keeps its image x-axis horizontal whatever the pitch, so 'up' is the
    direction most perpendicular to every camera's x-axis (smallest eigenvector), refined by the floor plane.
    Yaw: walls axis-aligned by a histogram-entropy (Manhattan) search. Origin: the room corner on the floor
    (x and y along the walls, z up)."""
    Rx = np.array([v.R.T @ np.array([1.0, 0, 0]) for v in views.values()])
    w, V = np.linalg.eigh(Rx.T @ Rx)
    up = V[:, 0]
    if up @ np.array([v.up for v in views.values()]).mean(0) < 0:
        up = -up
    P = xyz if keep is None else xyz[keep]
    h = P @ up
    low = P[h < np.percentile(h, 12)]
    rng = np.random.default_rng(0); best = (0, up, 0.0)
    if len(low) > 50:
        for _ in range(400):
            s = low[rng.choice(len(low), 3, replace=False)]
            n = np.cross(s[1] - s[0], s[2] - s[0]); nn = np.linalg.norm(n)
            if nn < 1e-9:
                continue
            n /= nn
            if n @ up < 0:
                n = -n
            if n @ up < np.cos(np.radians(6)):
                continue
            inl = np.abs(low @ n - n @ s[0]) < 0.02 * np.ptp(h)
            if inl.sum() > best[0]:
                best = (inl.sum(), n, n @ s[0])
    if best[0] > 0.25 * len(low):
        up = best[1]
    x0 = np.cross(up, [1.0, 0, 0])
    if np.linalg.norm(x0) < 0.1:
        x0 = np.cross(up, [0, 1.0, 0])
    x0 /= np.linalg.norm(x0); y0 = np.cross(up, x0)
    B = np.stack([x0, y0, up])
    Pc = P @ B.T
    lo_z, hi_z = np.percentile(Pc[:, 2], [2, 98])
    mid = (Pc[:, 2] > lo_z + 0.15 * (hi_z - lo_z)) & (Pc[:, 2] < lo_z + 0.85 * (hi_z - lo_z))
    th = _manhattan_yaw(Pc[mid, :2])
    c, s = np.cos(th), np.sin(th)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]]) @ B
    Pr = P @ R.T
    zr = Pr[:, 2]; band = (zr > np.percentile(zr, 15)) & (zr < np.percentile(zr, 85))   # wall height band
    lo = np.percentile(Pr[band, :2], 1, axis=0)
    o_room = np.array([lo[0], lo[1], np.percentile(zr, 0.7)])
    return RoomFrame(R=R, o=R.T @ o_room, s=1.0, notes=dict(yaw_deg=float(np.degrees(th)), floor_inliers=int(best[0])))


def fit_floor_surface(P: np.ndarray, cell: float = 0.1, iters: int = 10):
    """Smooth floor z = f(x, y) (quadratic), robustly fitted to the lowest point of every 10 cm cell; heights
    are measured from it because monocular depth bends large planes by a few centimetres."""
    k = np.floor(P[:, :2] / cell).astype(np.int64)
    key = k[:, 0] * 100000 + k[:, 1]
    order = np.lexsort((P[:, 2], key))
    _, first = np.unique(key[order], return_index=True)
    low = P[order[first]]
    Q = low[low[:, 2] < np.percentile(low[:, 2], 60)]          # cells under furniture never see floor
    A = lambda X: np.stack([np.ones(len(X)), X[:, 0], X[:, 1], X[:, 0] ** 2, X[:, 1] ** 2, X[:, 0] * X[:, 1]], 1)
    w = np.ones(len(Q)); c = np.zeros(6)
    for _ in range(iters):
        c, *_ = np.linalg.lstsq(A(Q) * w[:, None], Q[:, 2] * w, rcond=None)
        r = Q[:, 2] - A(Q) @ c
        s = 1.4826 * np.median(np.abs(r)) + 1e-4
        w = np.where(r > 2.5 * s, 0.05, 1.0 / np.maximum(1.0, np.abs(r) / (2.0 * s)))
    return c


def floor_z(xy: np.ndarray, c) -> np.ndarray:
    x, y = xy[:, 0], xy[:, 1]
    return c[0] + c[1] * x + c[2] * y + c[3] * x ** 2 + c[4] * y ** 2 + c[5] * x * y


def height_above_floor(P: np.ndarray, c) -> np.ndarray:
    return (P[:, 2] - floor_z(P[:, :2], c)).astype(np.float32)


def dense_wall_alignment(P: np.ndarray):
    """Yaw (radians) and corner origin that put the walls along x and y, from DENSE points. SfM points gather on
    textured clutter, and plain painted walls give almost none, so the yaw found from them can be off by ten
    degrees or more; dense depth sees the walls fully. -> (theta, origin_xy) with X_new = Rz(theta) X - [ox, oy, 0]."""
    z = P[:, 2]
    band = (z > np.percentile(z, 25)) & (z < np.percentile(z, 85))
    Q = P[band, :2]
    if len(Q) > 300000:
        Q = Q[np.random.default_rng(0).choice(len(Q), 300000, replace=False)]
    th = _manhattan_yaw(Q)
    if th > np.pi / 4:
        th -= np.pi / 2
    c, s = np.cos(th), np.sin(th)
    Qr = Q @ np.array([[c, -s], [s, c]]).T
    return float(th), np.percentile(Qr, 1, axis=0)
