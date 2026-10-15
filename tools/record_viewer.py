"""Screenshots and the README animation, taken from a run's viewer.html with headless Chromium (Playwright).

usage: python tools/record_viewer.py outputs/my_room/result/viewer.html docs/img [--gif-frames 72]
needs: pip install playwright && playwright install chromium"""
import argparse, io, os, time
import numpy as np
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('viewer'); ap.add_argument('out'); ap.add_argument('--gif-frames', type=int, default=72)
    ap.add_argument('--width', type=int, default=1400); ap.add_argument('--height', type=int, default=860)
    ap.add_argument('--local-three', default=None, help='folder with three.min.js and OrbitControls.js (offline machines)')
    a = ap.parse_args()
    from playwright.sync_api import sync_playwright
    os.makedirs(a.out, exist_ok=True)
    url = 'file://' + os.path.abspath(a.viewer)
    with sync_playwright() as p:
        b = p.chromium.launch(args=['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader',
                                    '--ignore-gpu-blocklist'])
        pg = b.new_page(viewport=dict(width=a.width, height=a.height), device_scale_factor=1)
        pg.emulate_media(color_scheme='light')
        if a.local_three:
            pg.route('**/three.min.js', lambda r: r.fulfill(path=os.path.join(a.local_three, 'three.min.js')))
            pg.route('**/OrbitControls.js', lambda r: r.fulfill(path=os.path.join(a.local_three, 'OrbitControls.js')))
        pg.goto(url)
        pg.wait_for_function('window.viewerReady === true', timeout=300000)
        time.sleep(2)
        shot = lambda name: pg.screenshot(path=os.path.join(a.out, name))
        js = lambda s: pg.evaluate(s)
        D = js('window.viewer.data()')
        ext = D['extent']; cx, cy = (ext[0] + ext[1]) / 2, (ext[2] + ext[3]) / 2
        span = max(ext[1] - ext[0], ext[3] - ext[2])
        orbit = lambda az, el, dist: js(f'window.viewer.orbit({az}, {el}, {dist}, [{cx}, {cy}, 0.35])')
        orbit(np.radians(235), np.radians(48), 1.55 * span); time.sleep(1.0)
        shot('viewer_shared.png')
        js('window.viewer.hidePanels(true)')
        for m in ('before', 'after', 'shared'):
            js(f"window.viewer.setMode('{m}')"); time.sleep(0.8); shot(f'viewer_{m}_clean.png')
        js("window.viewer.setLayer('unknown', true)"); time.sleep(0.6); shot('viewer_unknown.png')
        js("window.viewer.setLayer('unknown', false)")
        js("window.viewer.setLayer('robot', true)"); js("window.viewer.setRobot('humanoid')")
        js("window.viewer.setLayer('nbv', true)")
        orbit(np.radians(270), np.radians(75), 1.45 * span); time.sleep(1.0); shot('viewer_robot.png')
        js("window.viewer.setLayer('robot', false)"); js("window.viewer.setLayer('nbv', false)")
        js("window.viewer.setLayer('rays', true)"); orbit(np.radians(235), np.radians(40), 1.4 * span); time.sleep(0.8)
        shot('viewer_evidence.png'); js("window.viewer.setLayer('rays', false)")
        # the animation: orbit the shared model, then blink before/after
        frames = []
        n = a.gif_frames
        for k in range(n):
            t = k / n
            if t < 0.6:
                js("window.viewer.setMode('shared')")
            else:
                js(f"window.viewer.setMode('{'before' if int((t - 0.6) * n / 6) % 2 == 0 else 'after'}')")
            orbit(np.radians(200 + 120 * min(t / 0.6, 1.0)), np.radians(50), 1.5 * span)
            time.sleep(0.15)
            img = Image.open(io.BytesIO(pg.screenshot())).convert('RGB')
            w = 860; frames.append(img.resize((w, int(img.height * w / img.width)), Image.LANCZOS))
        pal = frames[len(frames) // 2].quantize(colors=160, method=Image.Quantize.MEDIANCUT)
        q = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]
        q[0].save(os.path.join(a.out, 'hero.gif'), save_all=True, append_images=q[1:], duration=110, loop=0, optimize=True)
        b.close()
    for f in sorted(os.listdir(a.out)):
        print(f, os.path.getsize(os.path.join(a.out, f)) // 1024, 'KB')


if __name__ == '__main__':
    main()
