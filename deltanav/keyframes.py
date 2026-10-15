"""Stage 1: sharp keyframes from each video, and the lens's field of view from the file's metadata.

Only OpenCV is needed to read the videos (no ffmpeg). Portrait videos are rotated upright by OpenCV."""
from __future__ import annotations
import math, os, struct
import numpy as np
import cv2


def _open(video: str):
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise RuntimeError(f'cannot open the video {video}')
    return cap


def video_info(video: str) -> dict:
    cap = _open(video)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    ok, fr = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f'cannot decode the video {video}')
    h, w = fr.shape[:2]
    return dict(frames=n, fps=fps, width=w, height=h, duration_s=round(n / fps, 1))


def select_keyframes(video: str, prefix: str, outdir: str, target: int = 180, long_side: int = 1280,
                     analysis_long: int = 640, log=print) -> dict:
    """One pass over the video. Every 2nd frame is scored for sharpness (variance of the Laplacian), penalised
    by image motion (median optical flow, which causes blur); the best frame of each of `target` equal windows
    is saved at `long_side` px. Returns the keyframe list and capture-quality statistics."""
    info = video_info(video)
    n = max(info['frames'], 1)
    win = max(2, int(round(n / target)))
    os.makedirs(outdir, exist_ok=True)
    cap = _open(video)
    lk = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))
    prev, best, saved, sharp_all, flow_all = None, None, [], [], []
    i = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        w_idx = i // win
        if best is not None and w_idx != best[0]:
            saved.append(_save(best, prefix, outdir, long_side, info['fps']))
            best = None
        if i % 2 == 0:
            h, w = fr.shape[:2]
            s = analysis_long / max(h, w)
            g = cv2.cvtColor(cv2.resize(fr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
            sharp = float(cv2.Laplacian(g, cv2.CV_32F).var())
            flow = 0.0
            if prev is not None:
                pts = cv2.goodFeaturesToTrack(prev, 250, 0.01, 8)
                if pts is not None and len(pts) > 20:
                    nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev, g, pts, None, **lk)
                    st = st.ravel() == 1
                    if st.sum() > 10:
                        flow = float(np.median(np.linalg.norm((nxt[st] - pts[st]).reshape(-1, 2), axis=1)))
            prev = g
            sharp_all.append(sharp); flow_all.append(flow)
            score = sharp / (1.0 + 0.08 * flow)
            if best is None or score > best[1]:
                best = (w_idx, score, i, fr.copy(), sharp, flow)
        i += 1
    if best is not None:
        saved.append(_save(best, prefix, outdir, long_side, info['fps']))
    cap.release()
    s, f = np.array(sharp_all), np.array(flow_all)
    stats = dict(frames_read=i, median_sharpness=round(float(np.median(s)), 1),
                 fast_motion_share=round(float(np.mean(f > 8.0)), 3),
                 keyframe_median_sharpness=round(float(np.median([k['sharpness'] for k in saved])), 1))
    log(f'  {os.path.basename(video)}: {i} frames ({i / info["fps"] / 60:.1f} min) -> {len(saved)} keyframes')
    return dict(video=os.path.abspath(video), info=info, stats=stats, keyframes=saved)


def _save(best, prefix, outdir, long_side, fps):
    _, score, i, fr, sharp, flow = best
    h, w = fr.shape[:2]
    s = long_side / max(h, w)
    W, H = int(round(w * s / 2) * 2), int(round(h * s / 2) * 2)
    name = f'{prefix}_{i:05d}.jpg'
    cv2.imwrite(os.path.join(outdir, name), cv2.resize(fr, (W, H), interpolation=cv2.INTER_AREA),
                [cv2.IMWRITE_JPEG_QUALITY, 93])
    return dict(name=name, frame=i, t=round(i / fps, 3), sharpness=round(sharp, 1), flow=round(flow, 2),
                width=W, height=H)


# ---------------------------------------------------------------- lens metadata (QuickTime / MP4 'meta' keys)
_CONTAINERS = {b'moov', b'trak', b'udta', b'mdia', b'minf', b'meta', b'edts'}


def _boxes(buf: bytes, start: int, end: int):
    p = start
    while p + 8 <= end:
        size, typ = struct.unpack('>I4s', buf[p:p + 8])
        hdr = 8
        if size == 1:
            size = struct.unpack('>Q', buf[p + 8:p + 16])[0]; hdr = 16
        elif size == 0:
            size = end - p
        if size < hdr or p + size > end:
            return
        yield typ, p + hdr, p + size
        p += size


def _read_moov(path: str) -> bytes | None:
    with open(path, 'rb') as f:
        f.seek(0, 2); total = f.tell(); p = 0
        while p + 8 <= total:
            f.seek(p)
            hdr = f.read(16)
            size, typ = struct.unpack('>I4s', hdr[:8])
            if size == 1:
                size = struct.unpack('>Q', hdr[8:16])[0]
            elif size == 0:
                size = total - p
            if size < 8:
                return None
            if typ == b'moov':
                f.seek(p)
                return f.read(min(size, 64 << 20))
            p += size
    return None


def _meta_items(buf: bytes, start: int, end: int, out: dict):
    """Collect key -> value from a QuickTime 'meta' box ('keys' + 'ilst')."""
    # QuickTime meta has no version/flags; ISO meta does (4 bytes). Detect which.
    for off in (0, 4):
        kids = list(_boxes(buf, start + off, end))
        if kids and all(t.isalnum() or t in (b'\xa9too',) for t, _, _ in kids[:3]):
            break
    keys = []
    for t, s, e in kids:
        if t == b'keys':
            count = struct.unpack('>I', buf[s + 4:s + 8])[0]; p = s + 8
            for _ in range(count):
                ksize = struct.unpack('>I', buf[p:p + 4])[0]
                keys.append(buf[p + 8:p + ksize].decode('utf-8', 'replace')); p += ksize
    for t, s, e in kids:
        if t == b'ilst':
            for it, is_, ie in _boxes(buf, s, e):
                idx = struct.unpack('>I', it)[0]
                for dt, ds, de in _boxes(buf, is_, ie):
                    if dt != b'data' or not (1 <= idx <= len(keys)):
                        continue
                    typ = struct.unpack('>I', buf[ds:ds + 4])[0] & 0xFFFFFF
                    raw = buf[ds + 8:de]
                    val = None
                    if typ == 1:
                        val = raw.decode('utf-8', 'replace')
                    elif typ == 23 and len(raw) == 4:
                        val = struct.unpack('>f', raw)[0]
                    elif typ == 24 and len(raw) == 8:
                        val = struct.unpack('>d', raw)[0]
                    elif typ in (21, 22, 65, 66, 67, 74, 75, 76, 77, 78) and len(raw) in (1, 2, 4, 8):
                        val = int.from_bytes(raw, 'big', signed=typ in (21, 65, 66, 67, 74))
                    if val is not None:
                        out[keys[idx - 1]] = val


def _walk(buf, start, end, out):
    for t, s, e in _boxes(buf, start, end):
        if t == b'meta':
            _meta_items(buf, s, e, out)
        elif t in _CONTAINERS:
            _walk(buf, s, e, out)


def video_metadata(path: str) -> dict:
    """QuickTime metadata keys (e.g. com.apple.quicktime.camera.focal_length.35mm_equivalent, .model)."""
    try:
        moov = _read_moov(path)
        out = {}
        if moov:
            _walk(moov, 8, len(moov), out)
        return out
    except Exception:
        return {}


def hfov_from_metadata(path: str):
    """Horizontal field of view (degrees, across the long side) from the 35 mm-equivalent focal length.
    Phones quote it for the full 4:3 sensor diagonal; video uses the full sensor width, so the long-side
    field of view is the 4:3 one. Returns (hfov, source) or (None, reason)."""
    meta = video_metadata(path)
    f35 = None
    for k, v in meta.items():
        if 'focal_length.35mm' in k or k.lower().endswith('focallengthin35mmformat'):
            try:
                f35 = float(v)
            except (TypeError, ValueError):
                pass
    if not f35:
        return None, 'no focal length in the file metadata'
    half_diag = 43.27 / 2                         # 35 mm film diagonal / 2
    half_w = half_diag * 0.8                       # 4:3 sensor: width = 0.8 x diagonal
    half_w *= 0.9                                  # video stabilisation crops about 10% of the sensor
    hfov = math.degrees(2 * math.atan(half_w / f35))
    model = next((str(v) for k, v in meta.items() if k.endswith('.model')), '')
    return hfov, f'{f35:g} mm (35 mm equivalent) in the file metadata{" of " + model if model else ""}'
