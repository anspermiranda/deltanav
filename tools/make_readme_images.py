"""README images from a finished run: a grid of annotated video frames, a screenshot of report.html, and a
screenshot of the 3D window (Rerun's web viewer, in headless Chromium).

usage: python tools/make_readme_images.py outputs/my_room docs/img [--local-three DIR] [--no-rerun]
needs: pip install playwright && playwright install chromium"""
import argparse, json, os, pickle, socket, subprocess, sys, time
import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def annotated_grid(run, path, width=1800):
    from deltanav.scene import load_scene
    from deltanav.detect import mask_people
    from deltanav.annotate import Annotator, places
    res = os.path.join(run, 'result')
    meta, views, depths, _, _ = load_scene(os.path.join(run, 'scene'))
    R = json.load(open(os.path.join(res, 'changes.json'))); changes = R['changes']
    Z = np.load(os.path.join(res, 'data', 'points.npz')); OI = np.load(os.path.join(res, 'data', 'object_points.npz'))
    dets = pickle.load(open(os.path.join(run, 'detections.pkl'), 'rb'))
    mask_people(depths, dets, {n: (c['img_width'], c['img_height']) for n, c in meta['cameras'].items()})
    man = json.load(open(os.path.join(run, 'keyframes.json')))
    times = {f"{ep}/{k['name']}": k['t'] for ep in ('before', 'after') for k in man[ep]['keyframes']}
    probes = {}
    for c in changes:
        for o, ep, role in places(c):
            P = Z[f'P_{ep}'][OI[f"obj{o['id']}"]]
            probes[(c['id'], role)] = P[np.random.default_rng(0).choice(len(P), min(len(P), 400), replace=False)]
    ann = Annotator(meta, views, depths, changes, Z['floor_c'], probes, os.path.join(run, 'frames'), times)
    tiles = []
    for ep in ('before', 'after'):
        best = []
        for n in [n for n in views if views[n].epoch == ep]:
            st = ann.states(n)
            good = [s for s in st.values() if s['state'] in ('visible', 'empty') and s['in_view'] > 0.5]
            best.append((len(good), sum(s['score'] for s in good), n))
        best.sort(reverse=True)
        tiles.append(ann.draw(best[0][2])[0])
    h = min(t.shape[0] for t in tiles)
    row = np.hstack([cv2.resize(tiles[0], (int(tiles[0].shape[1] * h / tiles[0].shape[0]), h)),
                     np.full((h, 10, 3), 255, np.uint8),
                     cv2.resize(tiles[1], (int(tiles[1].shape[1] * h / tiles[1].shape[0]), h))])
    row = cv2.resize(row, (width, int(row.shape[0] * width / row.shape[1])), interpolation=cv2.INTER_AREA)
    cv2.imwrite(path, cv2.cvtColor(row, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])


def report_shot(run, path, local_three=None):
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport=dict(width=1100, height=1500), device_scale_factor=1)
        pg.emulate_media(color_scheme='light')
        pg.goto('file://' + os.path.abspath(os.path.join(run, 'result', 'report.html')))
        time.sleep(2)
        pg.screenshot(path=path)
        b.close()


def free_port():
    s = socket.socket(); s.bind(('127.0.0.1', 0)); p = s.getsockname()[1]; s.close(); return p


def rerun_shot(run, path, wait=60):
    """The 3D window as Rerun's web viewer shows it (same viewer, same blueprint as the native window)."""
    rrd = os.path.abspath(os.path.join(run, 'result', 'scene.rrd'))
    try:                                                   # native viewer, offscreen, if a GPU adapter exists
        subprocess.run([sys.executable, '-m', 'rerun', rrd, '--headless', '--window-size', '1600x900',
                        '--screenshot-to', path], timeout=180, check=True, capture_output=True)
        if os.path.exists(path):
            return 'native'
    except Exception as e:
        print('native screenshot not available:', str(e)[:200])
    from playwright.sync_api import sync_playwright
    wp, gp = free_port(), free_port()
    srv = subprocess.Popen([sys.executable, '-m', 'rerun', '--serve-web', rrd, '--web-viewer-port', str(wp), '--port', str(gp)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        time.sleep(8)
        with sync_playwright() as p:
            b = p.chromium.launch(args=['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader',
                                        '--ignore-gpu-blocklist'])
            pg = b.new_page(viewport=dict(width=1600, height=900))
            pg.goto(f'http://127.0.0.1:{wp}/?url=rerun%2Bhttp%3A%2F%2F127.0.0.1%3A{gp}%2Fproxy&renderer=webgl')
            time.sleep(wait)
            pg.screenshot(path=path)
            b.close()
    finally:
        srv.terminate()
    return 'web'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run'); ap.add_argument('out'); ap.add_argument('--no-rerun', action='store_true')
    ap.add_argument('--wait', type=int, default=60)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    annotated_grid(a.run, os.path.join(a.out, 'annotated_frames.jpg')); print('annotated_frames.jpg')
    report_shot(a.run, os.path.join(a.out, 'report.png')); print('report.png')
    if not a.no_rerun:
        print('rerun_window.png via', rerun_shot(a.run, os.path.join(a.out, 'rerun_window.png'), a.wait))


if __name__ == '__main__':
    main()
