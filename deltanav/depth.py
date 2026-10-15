"""Dense metric depth per keyframe: Depth Anything V2 (metric, indoor), corrected for the lens and per frame
against the reconstruction's own 3D points, then cleaned by multi-view consistency."""
from __future__ import annotations
import numpy as np
import cv2

from .geometry import View

TRAIN_HFOV = 60.0          # Hypersim cameras, on which the metric-indoor checkpoints were trained


class DepthNet:
    def __init__(self, model: str | None = None, device: str | None = None):
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        self.torch = torch
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        model = model or 'depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf'
        self.proc = AutoImageProcessor.from_pretrained(model)
        self.net = AutoModelForDepthEstimation.from_pretrained(model).to(self.device).eval()
        if self.device == 'cuda':
            self.net = self.net.half()

    def __call__(self, bgr: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
        torch = self.torch
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        with torch.inference_mode():
            inp = self.proc(images=rgb, return_tensors='pt').to(self.device)
            if self.device == 'cuda':
                inp['pixel_values'] = inp['pixel_values'].half()
            pred = self.net(**inp).predicted_depth.float()          # (1, h, w) metres
            pred = torch.nn.functional.interpolate(pred[:, None], size=out_hw, mode='bilinear', align_corners=False)[0, 0]
        return pred.cpu().numpy().astype(np.float32)


def lens_correction(hfov_long_deg: float) -> float:
    """Metric depth networks learn the focal length of their training camera. Seen through a wider lens, the
    room looks further away by tan(hfov/2) / tan(train_hfov/2); this undoes it (0.527 for a 95 deg lens)."""
    return float(np.tan(np.radians(TRAIN_HFOV / 2)) / np.tan(np.radians(hfov_long_deg / 2)))


def _field_basis(u, v, W, H):
    x = (u - W / 2) / (W / 2); y = (v - H / 2) / (H / 2); r2 = x * x + y * y
    return np.stack([np.ones_like(x), x, y, x * x, y * y, x * y, r2 * r2], 1)


def fit_field(pred: np.ndarray, uv: np.ndarray, z: np.ndarray, iters: int = 8):
    """Smooth multiplicative correction log(z / d) = poly(u, v), robustly fitted (IRLS) to the reconstruction's
    3D points seen in this frame. A network fed an ultra-wide image distorts depth unevenly across the frame,
    which one scale factor cannot undo."""
    H, W = pred.shape
    ui = np.clip(np.round(uv[:, 0]).astype(int), 0, W - 1); vi = np.clip(np.round(uv[:, 1]).astype(int), 0, H - 1)
    d = pred[vi, ui]
    ok = np.isfinite(d) & (d > 0.05) & (z > 0.1)
    if ok.sum() < 30:
        if ok.sum() >= 8:                                     # too few points for a field: one robust scale
            s = float(np.exp(np.median(np.log(z[ok] / d[ok]))))
            return (pred * s).astype(np.float32), dict(n=int(ok.sum()), rel_err=float(np.median(np.abs(np.log(z[ok] / (d[ok] * s))))),
                                                       rel_err_scale_only=float(np.median(np.abs(np.log(z[ok] / (d[ok] * s))))))
        return None, dict(n=int(ok.sum()))
    t = np.log(z[ok] / d[ok]); A = _field_basis(uv[ok, 0], uv[ok, 1], W, H)
    w = np.ones(len(t))
    for _ in range(iters):
        c, *_ = np.linalg.lstsq(A * w[:, None], t * w, rcond=None)
        r = A @ c - t; s_ = 1.4826 * np.median(np.abs(r)) + 1e-6
        w = 1.0 / np.maximum(1.0, np.abs(r) / (2.0 * s_))
    vv, uu = np.mgrid[0:H, 0:W]
    F = np.exp(_field_basis(uu.ravel().astype(np.float32), vv.ravel().astype(np.float32), W, H) @ c).reshape(H, W)
    return (pred * F).astype(np.float32), dict(n=int(ok.sum()), rel_err=float(np.median(np.abs(r))),
                                               rel_err_scale_only=float(np.median(np.abs(t - np.median(t)))))


def edge_mask(D: np.ndarray, thresh: float = 0.08) -> np.ndarray:
    """Pixels at depth discontinuities (mixed foreground/background) are unreliable for free-space tests."""
    L = np.log(np.clip(np.nan_to_num(D, nan=10.0), 0.05, 50))
    gx = cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3) / 8; gy = cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3) / 8
    m = (np.hypot(gx, gy) > thresh).astype(np.uint8)
    return cv2.dilate(m, np.ones((5, 5), np.uint8)).astype(bool)


def backproject(D: np.ndarray, v: View, step: int = 1):
    """Depth map -> world points (M,3) and the pixel grid used."""
    H, W = D.shape
    vs, us = np.mgrid[0:H:step, 0:W:step]
    d = D[vs, us]
    ok = np.isfinite(d) & (d > 0)
    u, vv, d = us[ok].astype(np.float32), vs[ok].astype(np.float32), d[ok]
    x = (u - v.K[0, 2]) / v.K[0, 0]; y = (vv - v.K[1, 2]) / v.K[1, 1]
    Xc = np.stack([x * d, y * d, d], 1)
    return (Xc - v.t) @ v.R, (vv.astype(int), u.astype(int))


def consistency_filter(names: list[str], views: dict, depths: dict, neighbours: dict,
                       rel_tol: float = 0.05, min_agree: int = 2) -> dict:
    """Keep a pixel only if at least `min_agree` neighbouring frames see a surface at the same depth: removes
    flying pixels, mirror and window artefacts and frame-specific hallucinations."""
    out = {}
    for n in names:
        D = depths[n]; v = views[n]
        X, (vi, ui) = backproject(D, v)
        agree = np.zeros(len(X), np.int16)
        for m in neighbours[n]:
            Dm = depths[m]; vm = views[m]
            uv, z = vm.project(X)
            Hm, Wm = Dm.shape
            ok = (z > 0.1) & (uv[:, 0] >= 0) & (uv[:, 0] < Wm - 1) & (uv[:, 1] >= 0) & (uv[:, 1] < Hm - 1)
            dm = np.full(len(X), np.nan, np.float32)
            dm[ok] = Dm[np.round(uv[ok, 1]).astype(int), np.round(uv[ok, 0]).astype(int)]
            agree += (np.abs(dm - z) < rel_tol * z).astype(np.int16)
        keep = agree >= min_agree
        F = np.full_like(D, np.nan)
        F[vi[keep], ui[keep]] = D[vi[keep], ui[keep]]
        out[n] = F
    return out


def pick_neighbours(views: dict, names: list[str], k: int = 6, min_baseline: float = 0.08) -> dict:
    """The k nearest frames of the same video that look in a similar direction from at least `min_baseline`
    metres away (so they add parallax)."""
    C = np.array([views[n].C for n in names]); F = np.array([views[n].forward for n in names])
    out = {}
    for i, n in enumerate(names):
        d = np.linalg.norm(C - C[i], axis=1); cosang = F @ F[i]
        score = np.where((d > min_baseline) & (cosang > 0.6), d / np.maximum(cosang, 1e-3), np.inf)
        order = np.argsort(score)[:k]
        out[n] = [names[j] for j in order if np.isfinite(score[j])]
    return out
