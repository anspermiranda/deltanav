"""Consensus point fusion: each recording's depth maps become one cleaned, coloured point cloud."""
from __future__ import annotations
import os
import numpy as np
import cv2

from .depth import backproject


def fuse_epoch(names: list[str], views: dict, depths: dict, image_dir: str, voxel: float = 0.02,
               min_views: int = 3, step: int = 2, max_depth: float = 5.0):
    """Back-project every (consistency-filtered) depth pixel and keep a voxel only if at least `min_views`
    different frames put a surface there. Returns (points (N,3), colours (N,3) uint8, frame count per point)."""
    keys_all, pos_all, col_all = [], [], []
    for n in names:
        D = depths.get(n)
        if D is None:
            continue
        D = np.where(D < max_depth, D, np.nan)
        X, (vi, ui) = backproject(D, views[n], step)
        if len(X) == 0:
            continue
        H, W = D.shape
        img = cv2.resize(cv2.imread(os.path.join(image_dir, n)), (W, H), interpolation=cv2.INTER_AREA)
        C = img[vi, ui][:, ::-1]
        k = np.floor(X / voxel).astype(np.int64)
        kk = (k[:, 0] + 4096) * (1 << 26) + (k[:, 1] + 4096) * (1 << 13) + (k[:, 2] + 4096)
        _, first = np.unique(kk, return_index=True)          # one vote per voxel per frame
        keys_all.append(kk[first]); pos_all.append(X[first]); col_all.append(C[first])
    kk = np.concatenate(keys_all); X = np.concatenate(pos_all); C = np.concatenate(col_all).astype(np.float64)
    _, inv, cnt = np.unique(kk, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    P = np.stack([np.bincount(inv, X[:, j]) / cnt for j in range(3)], 1).astype(np.float32)
    Cm = np.stack([np.bincount(inv, C[:, j]) / cnt for j in range(3)], 1).astype(np.uint8)
    keep = cnt >= min_views
    return P[keep], Cm[keep], cnt[keep]


def voxel_downsample(P: np.ndarray, C: np.ndarray | None, size: float):
    keys = np.floor(P / size).astype(np.int64)
    _, inv = np.unique(keys, axis=0, return_inverse=True)
    inv = inv.ravel()
    cnt = np.bincount(inv)
    Pm = np.stack([np.bincount(inv, P[:, k]) / cnt for k in range(3)], 1).astype(np.float32)
    Cm = None
    if C is not None:
        Cm = np.stack([np.bincount(inv, C[:, k].astype(np.float64)) / cnt for k in range(3)], 1).astype(np.uint8)
    return Pm, Cm
