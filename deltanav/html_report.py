"""report.html (one self-contained page, images inlined) and report.md (the same for GitHub), written from the
ledger and the figures of a run."""
from __future__ import annotations
import base64, html, os

TYPE_COL = {'moved': '#f59614', 'removed': '#dc3c3c', 'added': '#28aa46'}

CSS = """
:root { --bg:#ffffff; --fg:#1d1f24; --muted:#5d6370; --line:#e3e5ea; --card:#f7f8fa; --accent:#1473dc; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --bg:#15171c; --fg:#e8eaef; --muted:#9aa1ad; --line:#2d313a; --card:#1d2027; color-scheme: dark; } }
:root[data-theme="dark"] { --bg:#15171c; --fg:#e8eaef; --muted:#9aa1ad; --line:#2d313a; --card:#1d2027; color-scheme: dark; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--fg); font:15px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
main { max-width:1080px; margin:0 auto; padding:28px 16px 60px; }
h1 { font-size:28px; margin:0 0 4px; } h2 { font-size:20px; margin:38px 0 10px; padding-top:10px; border-top:1px solid var(--line); }
h3 { font-size:16px; margin:0 0 6px; } .muted { color:var(--muted); } .small { font-size:13px; }
.summary { background:var(--card); border:1px solid var(--line); border-radius:12px; padding:14px 18px; margin:18px 0; }
.summary ol { margin:6px 0 0; padding-left:20px; } .summary li { margin:5px 0; }
.stats { display:grid; grid-template-columns:repeat(auto-fit, minmax(150px, 1fr)); gap:10px; margin:14px 0; }
.stat { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:10px 12px; }
.stat b { display:block; font-size:20px; } .stat span { font-size:12.5px; color:var(--muted); }
.badge { display:inline-block; font-size:11px; font-weight:700; color:#fff; padding:1px 8px; border-radius:10px; text-transform:uppercase; margin-right:6px; vertical-align:2px; }
.change { border:1px solid var(--line); border-left:6px solid; border-radius:12px; padding:14px 16px; margin:16px 0; background:var(--card); }
.change .sentence { font-size:16px; margin:6px 0 10px; }
img { max-width:100%; border-radius:8px; border:1px solid var(--line); display:block; }
table { border-collapse:collapse; width:100%; font-size:13.5px; margin:8px 0; }
th, td { text-align:left; padding:5px 8px; border-bottom:1px solid var(--line); vertical-align:top; }
th { color:var(--muted); font-weight:600; font-size:12.5px; }
.grid2 { display:grid; grid-template-columns:1fr 1fr; gap:12px; } .grid2 img { width:100%; }
@media (max-width:760px) { .grid2 { grid-template-columns:1fr; } }
code, pre { font-family:ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size:12.5px; }
pre { background:var(--card); border:1px solid var(--line); border-radius:8px; padding:10px 12px; overflow-x:auto; }
details { margin:8px 0; } summary { cursor:pointer; color:var(--accent); }
a { color:var(--accent); }
.cap { font-size:12.5px; color:var(--muted); margin:6px 0 0; }
.callout { border-left:4px solid var(--accent); background:var(--card); padding:10px 14px; border-radius:6px; margin:12px 0; }
"""


def _img(path, inline: bool, rel_to: str):
    if not path or not os.path.exists(path):
        return ''
    if inline:
        ext = os.path.splitext(path)[1].lower().strip('.')
        mime = {'jpg': 'jpeg', 'jpeg': 'jpeg', 'png': 'png', 'gif': 'gif'}.get(ext, 'png')
        return f'data:image/{mime};base64,' + base64.b64encode(open(path, 'rb').read()).decode()
    return os.path.relpath(path, rel_to)


def _e(s):
    return html.escape(str(s))


def _m(v):
    return '' if v is None else f'{v:.2f} m'


def _pos(p):
    if not p:
        return ''
    s = f" ± {p['sigma_m'] * 100:.0f} cm" if p.get('sigma_m') else ''
    return f"x {p['x']:.2f}, y {p['y']:.2f}, z {p['z']:.2f} m{s}"


def _size(p):
    s = p['size_m']
    return f"{s['length']:.2f} × {s['width']:.2f} × {s['height']:.2f} m"


def _rels(p):
    return ', '.join(f"{r['relation']} {r['target']}" + (f" ({r['distance_m']:.2f} m)" if r.get('distance_m') else '')
                     for r in p.get('relations', []))


def change_rows(r):
    rows = []
    if r['type'] == 'moved':
        b, a, d = r['before'], r['after'], r['delta']
        rows += [('before', _pos(b)), ('after', _pos(a)),
                 ('moved', f"{d['distance_m']:.2f} m" + (f" ± {d['distance_sigma_m'] * 100:.0f} cm" if d.get('distance_sigma_m') else '')
                  + f" (Δx {d['dx']:+.2f}, Δy {d['dy']:+.2f}, Δz {d['dz']:+.2f} m)")]
        if d.get('rotation_deg') is not None:
            sup = d.get('rotation_support') or {}
            rows.append(('turned', f"{d['rotation_deg']:+.1f}° about z" + (' (confident' if d.get('rotation_confident') else ' (uncertain')
                         + f"; {d.get('rotation_source')}" + (f"; {sup['inliers']} of {sup['matches']} matches agree" if sup.get('inliers') else '') + ')'))
        rows += [('size', _size(a)), ('orientation', f"long side at {b['orientation_deg']:.0f}° → {a['orientation_deg']:.0f}° (yaw, mod 180°)"),
                 ('relations', f"before: {_rels(b)}<br>after: {_rels(a)}")]
        ev = r['evidence']
        rows.append(('evidence', f"old place: {ev['old_place'].get('views_saw_through')} after-video frames saw through it, "
                     f"{ev['old_place'].get('views_saw_surface')} saw a surface; new place: {ev['new_place'].get('views_saw_through')} "
                     f"before-video frames saw through it, {ev['new_place'].get('views_saw_surface')} saw a surface"))
    else:
        key = 'before' if r['type'] == 'removed' else 'after'
        p = r[key]
        rows += [('position' if r['type'] == 'added' else 'was at', _pos(p)), ('size', _size(p)),
                 ('orientation', f"long side at {p['orientation_deg']:.0f}° (yaw, mod 180°)"), ('relations', _rels(p))]
        e = r['evidence']['place']
        other = 'after' if r['type'] == 'removed' else 'before'
        rows.append(('evidence', f"{e.get('views_saw_through')} {other}-video frames saw straight through its place, "
                     f"{e.get('views_saw_surface')} saw a surface there, {e.get('views_occluded')} had it hidden"))
    for name, n in (r.get('navigation') or {}).items():
        aff = '; '.join(f"route to the {x['goal']}: {x['status']}" + (
            f" (yesterday's route passes {x['yesterdays_route_clearance_m']:.2f} m from it)"
            if x.get('yesterdays_route_clearance_m') is not None and x['status'] == 'unchanged' else '')
            for x in n['routes_affected']) or 'no route affected'
        obs = '' if n.get('obstacle') is None else ('an obstacle for it; ' if n['obstacle'] else 'not an obstacle for it; ')
        rows.append((f'{name} robot', obs + aff))
    return rows


def write_report(res, ledger, nav, images, info, inline=True, filename='report.html'):
    """images: dict of named figure paths (absolute)."""
    I = lambda k: _img(images.get(k), inline, res)
    ch = ledger['changes']
    run = ledger.get('run', {})
    cov = ledger['could_not_check']
    h = [f'<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
         f'<title>DeltaNav · Room Change Report</title><style>{CSS}</style></head><body><main>']
    h.append('<h1>What changed in the room?</h1>')
    h.append(f"<div class='muted'>{_e(run.get('before_video', 'before'))} → {_e(run.get('after_video', 'after'))}"
             f" · {_e(run.get('date', ''))} · DeltaNav {_e(run.get('version', ''))}</div>")
    h.append("<div class='summary'><b>" + (f"{len(ch)} change{'s' if len(ch) != 1 else ''} found" if ch else 'No changes found')
             + "</b><ol>" + ''.join(f"<li><span class='badge' style='background:{TYPE_COL[r['type']]}'>{r['type']}</span>{_e(r['sentence'])}</li>"
                                    for r in ch) + '</ol></div>')
    unk = 100 * cov['before']['unknown']
    stats = [(f"{run.get('frames_before', '?')} / {run.get('frames_after', '?')}", 'keyframes (before / after), all in one 3D model'),
             (f"{run.get('depth_error_pct', '?')}%", 'median depth error vs the 3D model'),
             (f"{unk:.0f}%", 'of the before surfaces could not be re-checked'),
             (f"{run.get('total_minutes', '?')} min", f"run time ({_e(run.get('device', ''))})")]
    h.append("<div class='stats'>" + ''.join(f"<div class='stat'><b>{a}</b><span>{b}</span></div>" for a, b in stats) + '</div>')
    # 3D
    h.append('<h2>The room in 3D</h2>')
    h.append("<p>One 3D model built from both videos together. Open the <a href='viewer.html'><b>3D viewer</b></a> "
             "(<code>viewer.html</code>, any browser, works offline; <code>deltanav view &lt;out&gt;</code> opens it) to "
             "rotate, zoom, switch between Shared, Before, After and Blink, and turn the layers on and off. "
             "<code>deltanav view &lt;out&gt; --rerun</code> opens the same scene in the Rerun window, with a timeline "
             "through both videos.</p>")
    if images.get('3d_overview'):
        h.append(f"<img src='{I('3d_overview')}' alt='3D overview'><p class='cap'>The shared room model (from both videos) "
                 "with every change: both versions of a moved object, its path, and the boxes in the room frame.</p>")
    if images.get('3d_before') and images.get('3d_after'):
        h.append(f"<div class='grid2'><div><img src='{I('3d_before')}' alt='before'><p class='cap'>Before video only</p></div>"
                 f"<div><img src='{I('3d_after')}' alt='after'><p class='cap'>After video only</p></div></div>")
    # changes
    h.append('<h2>Changes</h2>')
    h.append("<p class='small muted'>Positions are in the room frame: x and y along the walls, z up from the floor, "
             "metres, origin on the floor in a corner (see the floor plan). ± is the 1-sigma position uncertainty.</p>")
    for r in ch:
        col = TYPE_COL[r['type']]
        h.append(f"<div class='change' style='border-left-color:{col}'><h3><span class='badge' style='background:{col}'>"
                 f"{r['type']}</span>{_e(r['id'])} · {_e(r['label'])}" + (f" ({_e(r['colour'])})" if r.get('colour') else '') + '</h3>')
        h.append(f"<div class='sentence'>{_e(r['sentence'])}</div>")
        img = images.get(f"change_{r['id']}")
        if img:
            h.append(f"<img src='{I('change_' + r['id'])}' alt='{_e(r['label'])}'><p class='cap'>Real frames from each video "
                     "with the same 3D box projected into them. Solid: the object is there in that video. Dashed: the "
                     "place is empty in that video.</p>")
        h.append('<table>' + ''.join(f'<tr><th>{_e(k)}</th><td>{v}</td></tr>' for k, v in change_rows(r)) + '</table>')
        if r['type'] == 'moved' and r['delta'].get('transform_before_to_after'):
            T = r['delta']['transform_before_to_after']
            h.append('<details><summary>Rigid transform, before → after (room frame, 4×4)</summary><pre>' +
                     '\n'.join('  '.join(f'{v:+.4f}' for v in row) for row in T) + '</pre></details>')
        h.append('</div>')
    if images.get('floor_plan'):
        h.append('<h2>Floor plan</h2>')
        h.append(f"<img src='{I('floor_plan')}' alt='floor plan'><p class='cap'>Top view of the shared 3D model. Furniture "
                 "found by the detectors, the changes, the places that could not be checked (purple), both camera paths, and "
                 "where to look next.</p>")
    # robot
    if nav:
        h.append('<h2>For a robot</h2>')
        h.append("<p>Yesterday's map (from the before video) is <b>edited</b> with the verified changes only, so depth noise "
                 "and the corners the after video skipped never change it. Routes are re-planned from the door to each piece "
                 "of furniture, and every route change is traced back to the change that caused it.</p>")
        if images.get('robot_routes'):
            h.append(f"<img src='{I('robot_routes')}' alt='robot routes'>")
        for name, rb in nav['robots'].items():
            h.append(f"<h3 style='margin-top:16px'>{_e(rb['profile'].get('label') or name)}</h3><ul>" +
                     ''.join(f'<li>{_e(s)}</li>' for s in nav['sentences'][name]) + '</ul>')
            lab = {c['id']: c['label'] for c in ch}
            h.append('<table><tr><th>goal</th><th>route</th><th>before</th><th>now</th><th>because of</th></tr>' +
                     ''.join(f"<tr><td>{_e(x['goal'])}</td><td>{_e(x['status'])}</td><td>{_m(x['before_m'])}</td>"
                             f"<td>{_m(x['updated_m'])}</td><td>{_e(', '.join(lab.get(c, c) for c in x['causes']))}</td></tr>"
                             for x in rb['routes']) + '</table>')
            e = rb['edits']
            h.append(f"<p class='small muted'>Map edits: {e['vacated_m2']:.2f} m² freed, {e['new_obstacle_m2']:.2f} m² newly blocked, "
                     f"{e['now_unknown_m2']:.2f} m² now unknown. Re-mapping from the after video alone would have changed "
                     f"{e['cells_changed_by_remapping_m2']:.2f} m² of the map ({e['remapping_changes_where_both_known_m2']:.2f} m² "
                     f"even where both videos saw the floor), against {e['cells_changed_by_edits_m2']:.2f} m² changed by the edits.</p>")
        h.append("<p>Files in <code>maps/</code>: occupancy maps for ROS <code>map_server</code> (<code>&lt;robot&gt;_map_before</code>, "
                 "<code>_updated</code>, <code>_after</code>; .pgm + .yaml) and named goal poses (<code>&lt;robot&gt;_goals.yaml</code>) "
                 "in the room frame.</p><pre>ros2 run nav2_map_server map_server --ros-args -p yaml_filename:=maps/humanoid_map_updated.yaml</pre>")
    nv = ledger.get('next_views') or []
    if nv:
        h.append('<h2>Where to look next</h2>')
        h.append("<p>The surfaces the after video could not check are where a robot should look next. These viewpoints, on "
                 "floor the robot can reach, would see the most of them with nothing nearer in the way.</p>")
        h.append('<table><tr><th>#</th><th>robot</th><th>stand at (x, y)</th><th>camera</th><th>looks at</th><th>would check</th></tr>' +
                 ''.join(f"<tr><td>{k + 1}</td><td>{_e(v['robot'])}</td><td>{v['position'][0]:.2f}, {v['position'][1]:.2f} m</td>"
                         f"<td>{v['position'][2]:.2f} m high, yaw {v['yaw_deg']:.0f}°, pitch {v['pitch_deg']:.0f}°</td>"
                         f"<td>{_e(v.get('looks_at_words', ''))}</td><td>{v['unknown_area_m2']:.2f} m² ({100 * v['share_of_unknown']:.0f}%)</td></tr>"
                         for k, v in enumerate(nv)) + '</table>')
    h.append('<h2>What could not be checked</h2>')
    h.append("<p>A change is reported only when the other video looked <i>through</i> the place. Surfaces that were hidden, "
             "never filmed again, or seen inconsistently are <b>could not be checked</b>, never <i>changed</i>.</p>")
    if images.get('coverage'):
        h.append(f"<img src='{I('coverage')}' alt='coverage'>")
    rej = ledger.get('rejected_candidates') or []
    if rej:
        h.append('<p>Candidates that failed the checks (not reported as changes):</p><table><tr><th>candidate</th><th>video</th><th>why rejected</th></tr>' +
                 ''.join(f"<tr><td>{_e(x['label'])} at ({x['centroid'][0]:.2f}, {x['centroid'][1]:.2f})</td><td>{_e(x['epoch'])}</td><td>{_e(x['reason'])}</td></tr>"
                         for x in rej) + '</table>')
    h.append('<h2>How this was measured</h2>')
    h.append("<ol class='small'><li>Sharp keyframes from both videos go into <b>one</b> structure-from-motion reconstruction "
             "(COLMAP), so both videos share one coordinate frame from the start.</li><li>Metric depth for every keyframe "
             "(Depth Anything V2, corrected for the lens and aligned to the 3D points), kept only where neighbouring frames agree.</li>"
             "<li><b>See-through test</b>: every surface point of one video is projected into every frame of the other. It counts "
             "as gone only if those frames saw <i>through</i> it, and an object only if at least 5 frames saw through its place "
             "and fewer than 10% saw a surface there.</li><li>Names from instance segmentation (YOLO11 + open-vocabulary YOLOE) "
             "voted over the frames that see the object; colours from white-balanced pixels; rotation from SIFT matches lifted to 3D.</li>"
             "<li>Sentences are generated from the numbers, so the words and the geometry cannot disagree.</li></ol>")
    if images.get('naive_vs_ours'):
        h.append(f"<img src='{I('naive_vs_ours')}' alt='naive vs ours'><p class='cap'>The obvious method, a point-cloud diff, "
                 "flags everything the after video did not see again.</p>")
    tech = [('reconstruction', f"{run.get('frames_before')} + {run.get('frames_after')} keyframes registered, {run.get('sfm_points', '?')} 3D points, "
             f"focal {run.get('focal_px', '?')} px ({run.get('hfov_deg', '?')}° across; lens from {_e(run.get('hfov_source', '?'))})"),
            ('metric scale', _e(run.get('scale_source', ''))),
            ('depth', f"median error against the 3D points {run.get('depth_error_scale_only_pct', '?')}% with a single scale, "
                      f"{run.get('depth_error_pct', '?')}% after the per-frame correction"),
            ('run time', _e(run.get('stage_seconds', '')))]
    h.append('<table>' + ''.join(f'<tr><th>{a}</th><td>{b}</td></tr>' for a, b in tech) + '</table>')
    h.append("<h2>Files</h2><ul class='small'><li><code>ledger.json</code>: every change, in the room frame (this report is "
             "generated from it)</li><li><code>viewer.html</code>: the 3D viewer (any browser, offline)</li><li><code>scene.rrd</code>: the "
             "Rerun window (<code>deltanav view &lt;out&gt; --rerun</code> or <code>rerun scene.rrd</code>)</li><li><code>meshes/</code>: coloured "
             "surfaces (PLY): shared, before, after, each changed object</li><li><code>video/</code>: annotated walk-throughs</li>"
             "<li><code>maps/</code>: occupancy maps, goal poses, routes</li><li><code>images/</code>: every figure in this report</li></ul>")
    h.append('</main></body></html>')
    p = os.path.join(res, filename)
    open(p, 'w').write('\n'.join(h))
    return p


def write_markdown(res, ledger, nav, images):
    R = lambda k: os.path.relpath(images[k], res) if images.get(k) else None
    ch = ledger['changes']; run = ledger.get('run', {})
    md = ['# What changed in the room?', '',
          f"`{run.get('before_video', 'before')}` → `{run.get('after_video', 'after')}` · {run.get('date', '')}", '']
    md += [f"**{len(ch)} change{'s' if len(ch) != 1 else ''} found**" if ch else '**No changes found**', '']
    md += [f"{k + 1}. **{r['type']}** · {r['sentence']}" for k, r in enumerate(ch)] + ['']
    if R('3d_overview'):
        md += ['## The room in 3D', '', f"![3D overview]({R('3d_overview')})", '',
               'Open `viewer.html` (the 3D viewer) in any browser, or run `deltanav view <out>`; add `--rerun` for the Rerun window.', '']
        if R('3d_before') and R('3d_after'):
            md += ['| Before video only | After video only |', '|---|---|', f"| ![before]({R('3d_before')}) | ![after]({R('3d_after')}) |", '']
    md += ['## Changes', '', 'Room frame: x and y along the walls, z up from the floor, metres; ± is 1 sigma.', '']
    for r in ch:
        md += [f"### {r['id']} · {r['type']} · {r['label']}", '', r['sentence'], '']
        if R(f"change_{r['id']}"):
            md += [f"![{r['label']}]({R('change_' + r['id'])})", '']
        md += ['| | |', '|---|---|'] + [f"| {k} | {str(v).replace('<br>', '; ')} |" for k, v in change_rows(r)] + ['']
    if R('floor_plan'):
        md += ['## Floor plan', '', f"![floor plan]({R('floor_plan')})", '']
    if nav:
        md += ['## For a robot', '']
        if R('robot_routes'):
            md += [f"![routes]({R('robot_routes')})", '']
        for name, rb in nav['robots'].items():
            md += [f"**{rb['profile'].get('label') or name}**", ''] + [f'- {s}' for s in nav['sentences'][name]] + ['']
    nv = ledger.get('next_views') or []
    if nv:
        md += ['## Where to look next', '', '| # | robot | stand at (x, y) | yaw | looks at | would check |', '|---|---|---|---|---|---|']
        md += [f"| {k + 1} | {v['robot']} | {v['position'][0]:.2f}, {v['position'][1]:.2f} m | {v['yaw_deg']:.0f}° | "
               f"{v.get('looks_at_words', '')} | {v['unknown_area_m2']:.2f} m² |" for k, v in enumerate(nv)] + ['']
    md += ['## What could not be checked', '']
    if R('coverage'):
        md += [f"![coverage]({R('coverage')})", '']
    rej = ledger.get('rejected_candidates') or []
    if rej:
        md += ['| rejected candidate | video | why |', '|---|---|---|']
        md += [f"| {x['label']} at ({x['centroid'][0]:.2f}, {x['centroid'][1]:.2f}) | {x['epoch']} | {x['reason']} |" for x in rej] + ['']
    if R('naive_vs_ours'):
        md += ['## Naive diff vs the see-through test', '', f"![naive vs ours]({R('naive_vs_ours')})", '']
    p = os.path.join(res, 'report.md')
    open(p, 'w').write('\n'.join(md) + '\n')
    return p
