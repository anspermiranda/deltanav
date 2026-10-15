"""Stage 2: one joint structure-from-motion reconstruction of both videos' keyframes, so that both
recordings live in the same coordinate frame from the start (no point-cloud alignment step)."""
from __future__ import annotations
import os, time, urllib.request
from pathlib import Path

VOCAB_URL = 'https://github.com/colmap/colmap/releases/download/3.11.1/vocab_tree_faiss_flickr100K_words32K.bin'


def vocab_tree(cache_dir: str | None = None) -> str:
    cache = Path(cache_dir or os.path.expanduser('~/.cache/deltanav')); cache.mkdir(parents=True, exist_ok=True)
    p = cache / 'vocab_tree_faiss_flickr100K_words32K.bin'
    if not p.exists():
        tmp = str(p) + '.part'
        urllib.request.urlretrieve(VOCAB_URL, tmp)
        os.replace(tmp, p)
    return str(p)


def run_sfm(image_dir: str, out_dir: str, focal_px: float, same_size: bool = True, max_features: int = 4096,
            log=print) -> str:
    """Images live in image_dir/before/b_*.jpg and image_dir/after/a_*.jpg. One camera model is shared by all
    frames when both videos have the same frame size (same phone, same lens); otherwise one per video.
    Matching: sequential inside each video (+ loop detection) and vocabulary-tree retrieval across both, so the
    unchanged room ties the two recordings into one model. Returns the path of the reconstruction."""
    import pycolmap
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    db = out / 'database.db'
    if db.exists():
        db.unlink()
    t0 = time.time()
    from PIL import Image
    first = sorted((Path(image_dir) / 'before').glob('*.jpg'))[0]
    w, h = Image.open(first).size
    ro = pycolmap.ImageReaderOptions(); ro.camera_model = 'SIMPLE_RADIAL'
    ro.camera_params = f'{focal_px:.2f},{w / 2:.1f},{h / 2:.1f},0'
    eo = pycolmap.FeatureExtractionOptions(); eo.sift.max_num_features = max_features; eo.max_image_size = 1280
    mode = pycolmap.CameraMode.SINGLE if same_size else pycolmap.CameraMode.PER_FOLDER
    pycolmap.extract_features(db, image_dir, camera_mode=mode, reader_options=ro, extraction_options=eo)
    log(f'  features: {time.time() - t0:.0f}s')
    vt = vocab_tree()
    so = pycolmap.SequentialPairingOptions(); so.overlap = 8; so.quadratic_overlap = True
    so.loop_detection = True; so.vocab_tree_path = vt; so.loop_detection_period = 4; so.loop_detection_num_images = 25
    pycolmap.match_sequential(db, pairing_options=so)
    vo = pycolmap.VocabTreePairingOptions(); vo.vocab_tree_path = vt; vo.num_images = 20
    pycolmap.match_vocabtree(db, pairing_options=vo)
    log(f'  matching: {time.time() - t0:.0f}s')
    sp = out / 'sparse'; sp.mkdir(exist_ok=True)
    recs = pycolmap.global_mapping(db, image_dir, sp)
    if not recs:
        raise RuntimeError('3D reconstruction failed: the two videos could not be reconstructed together')

    def registered(r):
        return [im.name for im in r.images.values() if getattr(im, 'has_pose', True)]
    best = max(recs, key=lambda k: len(registered(recs[k])))
    r = recs[best]
    names = registered(r)
    nb = sum(os.path.basename(n).startswith('b_') for n in names); na = len(names) - nb
    cam = next(iter(r.cameras.values()))
    log(f'  mapping: {time.time() - t0:.0f}s; registered {nb} before + {na} after keyframes, '
        f'{r.num_points3D()} points, focal {cam.params[0]:.1f} px (prior {focal_px:.1f})')
    tb = len(os.listdir(os.path.join(image_dir, 'before'))); ta = len(os.listdir(os.path.join(image_dir, 'after')))
    if nb < 0.3 * tb or na < 0.3 * ta:
        raise RuntimeError(f'only {nb} of {tb} before and {na} of {ta} after keyframes could be put in one 3D model. '
                           'The two videos need more views of the same unchanged parts of the room (walk the whole '
                           'room, point at every wall, use the same lens both times)')
    if nb < 0.7 * tb or na < 0.7 * ta:
        log(f'  warning: only {nb}/{tb} before and {na}/{ta} after keyframes registered; parts of the room the '
            'missing frames saw will be reported as "could not be checked"')
    return str(sp / str(best))
