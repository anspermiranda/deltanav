"""Command line.

    deltanav run  --before before.mov --after after.mov --out outputs/my_room
    deltanav view outputs/my_room          # reopen the 3D viewer and the report (--rerun: the Rerun window too)
    deltanav demo                          # download the example videos and run on them
"""
from __future__ import annotations
import argparse, json, math, os, pickle, platform, sys, time

from . import __version__

STAGES = ['keyframes', 'reconstruction', 'depth', 'detections', 'changes', 'outputs']
DEMO_URL = 'https://github.com/anspermiranda/deltanav/releases/download/v1.0/{name}'
DEMO_HFOV = 96.1        # iPhone 16 Pro 0.5x lens; the re-encoded release videos no longer carry the lens metadata


def main(argv=None):
    ap = argparse.ArgumentParser(prog='deltanav', description='DeltaNav: what changed in the room? Visibility-aware 3D '
                                 'change detection from two phone videos, with a 3D window, a report and robot maps.')
    ap.add_argument('--version', action='version', version=f'DeltaNav {__version__}')
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run', help='two videos -> changes, report, 3D window, robot maps')
    _run_args(r)
    r.add_argument('--before', required=True, help='video recorded before the changes')
    r.add_argument('--after', required=True, help='video recorded after the changes')
    r.add_argument('--out', required=True, help='output folder (every stage is cached here; re-running resumes)')
    v = sub.add_parser('view', help='open the 3D viewer and the report of a finished run')
    v.add_argument('out')
    v.add_argument('--no-report', action='store_true', help='only the 3D viewer')
    v.add_argument('--rerun', action='store_true', help='also open the Rerun 3D window (timeline through both videos)')
    d = sub.add_parser('demo', help='download the example videos (my bedroom) and run on them')
    _run_args(d)
    d.add_argument('--out', default='outputs/demo')
    d.add_argument('--videos', default=None, help='folder that already has before.mp4 and after.mp4 (skips the download)')
    a = ap.parse_args(argv)
    if a.cmd == 'run':
        run(a)
    elif a.cmd == 'view':
        view(a.out, report=not a.no_report, rerun=a.rerun)
    else:
        demo(a)


def _run_args(p):
    p.add_argument('--hfov', type=float, default=None,
                   help='field of view across the long side of the video, degrees (iPhone 0.5x: about 95, 1x: about 70). '
                        'Read from the file when possible; only needs to be roughly right')
    p.add_argument('--keyframes', type=int, default=180, help='keyframes taken from each video (default 180)')
    p.add_argument('--quick', action='store_true', help='faster, rougher: 90 keyframes per video and the small depth model')
    p.add_argument('--depth-model', default=None,
                   help='Hugging Face id or local folder of a Depth Anything V2 metric checkpoint '
                        '(default: depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf)')
    p.add_argument('--yolo', default='yolo11m-seg.pt', help='COCO instance-segmentation weights (ultralytics)')
    p.add_argument('--yoloe', default='yoloe-11m-seg-pf.pt', help="open-vocabulary weights; '' to skip")
    p.add_argument('--no-navigation', action='store_true', help='skip the robot maps, routes and next views')
    p.add_argument('--no-viewer', action='store_true', help='do not open the 3D viewer and the report at the end')
    p.add_argument('--force', action='store_true', help='recompute every stage')
    p.add_argument('--stop-after', choices=STAGES, help=argparse.SUPPRESS)


def log(m):
    print(m, flush=True)


def run(a):
    from .keyframes import select_keyframes, hfov_from_metadata
    out = os.path.abspath(a.out); os.makedirs(out, exist_ok=True)
    for f in (a.before, a.after):
        if not os.path.exists(f):
            sys.exit(f'error: {f} does not exist')
    t0 = time.time()
    info_p = os.path.join(out, 'run_info.json')
    info = json.load(open(info_p)) if os.path.exists(info_p) and not a.force else {}
    info.setdefault('seconds', {})
    try:
        import torch
        info['device'] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else f'CPU, {os.cpu_count()} cores'
    except Exception:
        info['device'] = 'CPU'
    n_key = 90 if a.quick else a.keyframes
    depth_model = a.depth_model or ('depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf' if a.quick
                                    else 'depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf')

    def done(stage, t):
        info['seconds'][stage] = round(time.time() - t)
        info.update(version=__version__, python=sys.version.split()[0], platform=platform.platform(),
                    inputs=dict(before=os.path.abspath(a.before), after=os.path.abspath(a.after)))
        json.dump(info, open(info_p, 'w'), indent=1)
        if a.stop_after == stage:
            log(f'stopped after {stage}'); sys.exit(0)

    # 1 -------------------------------------------------------------------------------------------------
    frames = os.path.join(out, 'frames'); manifest = os.path.join(out, 'keyframes.json')
    if a.force or not os.path.exists(manifest):
        log('[1/6] picking sharp keyframes from both videos')
        t = time.time()
        man = {}
        for ep, tag, video in (('before', 'b', a.before), ('after', 'a', a.after)):
            man[ep] = select_keyframes(video, tag, os.path.join(frames, ep), target=n_key, log=log)
        hfov, src = (a.hfov, '--hfov') if a.hfov else hfov_from_metadata(a.before)
        if hfov is None:
            hfov, src = 70.0, f'default 70 deg ({src}; pass --hfov for a wide or ultra-wide lens)'
        man['hfov'] = dict(degrees=round(float(hfov), 2), source=src)
        json.dump(man, open(manifest, 'w'), indent=1)
        log(f'  lens: {hfov:.1f} deg across the long side ({src})')
        done('keyframes', t)
    man = json.load(open(manifest))
    # 2 -------------------------------------------------------------------------------------------------
    sfm_dir = os.path.join(out, 'sfm'); model_txt = os.path.join(sfm_dir, 'model.txt')
    if a.force or not os.path.exists(model_txt):
        from .sfm import run_sfm
        log('[2/6] one 3D reconstruction for both videos (structure from motion)')
        t = time.time()
        kb, ka = man['before']['keyframes'][0], man['after']['keyframes'][0]
        long_px = max(kb['width'], kb['height'])
        focal = (long_px / 2) / math.tan(math.radians(man['hfov']['degrees']) / 2)
        same = (kb['width'], kb['height']) == (ka['width'], ka['height'])
        model = run_sfm(frames, sfm_dir, focal, same_size=same, log=log)
        open(model_txt, 'w').write(model)
        done('reconstruction', t)
    model = open(model_txt).read().strip()
    # 3 -------------------------------------------------------------------------------------------------
    scene = os.path.join(out, 'scene')
    if a.force or not os.path.exists(os.path.join(scene, 'scene.json')):
        from .scene import build_scene
        log('[3/6] metric depth for every keyframe, aligned to the reconstruction')
        t = time.time()
        build_scene(model, frames, scene, depth_model=depth_model, log=log)
        done('depth', t)
    # 4 -------------------------------------------------------------------------------------------------
    dets_path = os.path.join(out, 'detections.pkl')
    if a.force or not os.path.exists(dets_path):
        from .semantics import run_yolo, merge_detections
        log('[4/6] naming things: instance segmentation (COCO + open vocabulary)')
        t = time.time()
        names = sorted(os.path.join(ep, k['name']) for ep in ('before', 'after') for k in man[ep]['keyframes'])
        srcs = []
        for w in [a.yolo] + ([a.yoloe] if a.yoloe else []):
            part = os.path.join(out, f'detections_{os.path.splitext(os.path.basename(w))[0]}.pkl')   # resumable
            if a.force or not os.path.exists(part):
                det = run_yolo(frames, names, w, log=log, partial=part + '.partial')
                pickle.dump(det, open(part + '.tmp', 'wb')); os.replace(part + '.tmp', part)
                if os.path.exists(part + '.partial'):
                    os.remove(part + '.partial')
            srcs.append(pickle.load(open(part, 'rb')))
        pickle.dump(merge_detections(*srcs), open(dets_path, 'wb'))
        done('detections', t)
    # 5 -------------------------------------------------------------------------------------------------
    res = os.path.join(out, 'result')
    ch_p = os.path.join(res, 'changes.json')
    if a.force or not os.path.exists(ch_p) or os.path.getmtime(ch_p) < max(os.path.getmtime(dets_path),
                                                                          os.path.getmtime(os.path.join(scene, 'scene.json'))):
        from .detect import detect_changes
        log('[5/6] visibility-aware change detection')
        t = time.time()
        detect_changes(scene, frames, dets_path, res, log=log)
        done('changes', t)
    # 6 -------------------------------------------------------------------------------------------------
    from .report import make_outputs
    log('[6/6] 3D surfaces, robot maps, report, 3D window')
    t = time.time()
    summary = make_outputs(out, navigation=not a.no_navigation, log=log)
    done('outputs', t)
    info['total_seconds'] = round(sum(info['seconds'].values()))
    json.dump(info, open(info_p, 'w'), indent=1)
    log(f'\ndone in {time.time() - t0:.0f}s')
    for s in summary:
        log('  - ' + s)
    log(f'\n3D viewer:    {os.path.join(res, "viewer.html")}   (any browser, works offline)'
        f'\nreport:       {os.path.join(res, "report.html")}'
        f'\nRerun window: deltanav view {a.out} --rerun'
        + (f'\nrobot maps:   {os.path.join(res, "maps")}   (ROS map_server .pgm/.yaml, routes.json)'
           if os.path.isdir(os.path.join(res, 'maps')) else ''))
    if not a.no_viewer:
        view(out)


def _browse(path: str) -> bool:
    import webbrowser
    from pathlib import Path
    try:
        return webbrowser.open(Path(path).resolve().as_uri())       # file:///... on Linux, macOS and Windows
    except Exception:
        return False


def view(out: str, report: bool = True, rerun: bool = False):
    """The browser 3D viewer (Shared / Before / After / Blink, layers, a card per change) and the report;
    with rerun=True also the Rerun window."""
    res = os.path.join(os.path.abspath(out), 'result')
    if not os.path.isdir(res):
        sys.exit(f'error: {res} does not exist (give the folder passed to --out)')
    for name, wanted in (('viewer.html', True), ('report.html', report)):
        p = os.path.join(res, name)
        if wanted and os.path.exists(p):
            log(f'opening {p}' if _browse(p) else f'open {p} in a browser')
    if not rerun:
        return
    rrd = os.path.join(res, 'scene.rrd')
    if not os.path.exists(rrd):
        log('no Rerun recording found (scene.rrd)')
        return
    try:
        import importlib.util, subprocess
        if importlib.util.find_spec('rerun') is None:      # the 3D window ships with the rerun-sdk package
            raise RuntimeError('rerun-sdk is not installed')
        subprocess.Popen([sys.executable, '-m', 'rerun', rrd])
        log('opening the Rerun window ...')
    except Exception as e:
        log(f'could not open the Rerun window ({e}); run `python -m rerun {rrd}`')


def demo(a):
    import urllib.request
    vids = a.videos or os.path.join(a.out, 'videos'); os.makedirs(vids, exist_ok=True)
    for name in ('before.mp4', 'after.mp4'):
        dst = os.path.join(vids, name)
        if not os.path.exists(dst):
            url = DEMO_URL.format(name=name)
            log(f'downloading {url}')
            urllib.request.urlretrieve(url, dst + '.part'); os.replace(dst + '.part', dst)
    a.before, a.after = os.path.join(vids, 'before.mp4'), os.path.join(vids, 'after.mp4')
    a.hfov = a.hfov or DEMO_HFOV
    run(a)


if __name__ == '__main__':
    main()
