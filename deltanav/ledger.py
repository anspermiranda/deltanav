"""The change ledger: one machine-readable record per verified change, in the room frame, with the numbers, the
words, the evidence and what it means for each robot. Everything in the report is read from here."""
from __future__ import annotations
import math
import numpy as np

from .places import relations, nice

FRAME = dict(
    name='room', units='metres, degrees',
    origin='on the floor in a corner of the room (the corner with the smallest x and y)',
    x='along one wall', y='along the other wall (right-handed)', z='up, measured from the fitted floor surface',
    yaw='counter-clockwise about z, seen from above, 0 = +x',
    note='the same frame as the maps (map frame), the meshes and the 3D views')


def _pose(o, landmarks, bounds, label):
    w, d = o['footprint']['size']
    return dict(x=round(o['centroid'][0], 3), y=round(o['centroid'][1], 3), z=round(o['centroid'][2], 3),
                sigma_m=o.get('sigma_m'),
                size_m=dict(length=round(max(w, d), 2), width=round(min(w, d), 2), height=round(height(o), 2)),
                footprint=dict(center=[round(v, 3) for v in o['footprint']['center']],
                               size=[round(v, 3) for v in o['footprint']['size']],
                               angle_deg=round(o['footprint']['angle'], 1)),
                orientation_deg=round(_long_axis(o['footprint']), 1),
                z_range=[round(v, 3) for v in o['z_range']],
                relations=relations(o['centroid'], o['z_range'][0], landmarks, bounds, label))


def height(o):
    """Height of an object: from the floor if it stands on the floor (its lowest few centimetres touch the floor
    and are never seen as changed), else its own extent."""
    z0, z1 = o['z_range']
    return z1 if z0 < 0.2 else z1 - z0


def _long_axis(fp):
    """Direction of the footprint's long side (yaw, degrees in [0, 180))."""
    a = fp['angle'] if fp['size'][0] >= fp['size'][1] else fp['angle'] + 90
    return a % 180


def _evidence(o):
    e = o.get('evidence', {})
    return {k: e.get(k) for k in ('views_looked', 'views_saw_through', 'views_saw_surface', 'views_occluded')}


def transform(c):
    """4x4 rigid transform (room frame) taking the object's before pose to its after pose: rotate by the
    measured yaw about the before centroid, then move the centroid."""
    yaw = math.radians(c.get('yaw_deg') or 0.0)
    R = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1.0]])
    cb, ca = np.array(c['from_obj']['centroid']), np.array(c['to_obj']['centroid'])
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = ca - R @ cb
    return T.round(4).tolist()


def build_ledger(changes, sentences, landmarks, bounds, coverage, rejected, nav=None, next_views=None, meta=None):
    recs = []
    for c, s in zip(changes, sentences):
        lab = nice(c['label'])
        r = dict(id=c['id'], type=c['type'], label=lab, colour=(c.get('to_obj') or c.get('obj'))['color_name'] or None,
                 sentence=s)
        if c['type'] == 'moved':
            f, t = c['from_obj'], c['to_obj']
            r['before'] = _pose(f, landmarks, bounds, lab); r['after'] = _pose(t, landmarks, bounds, lab)
            d = c['delta']
            r['delta'] = dict(dx=d[0], dy=d[1], dz=d[2], distance_m=c['displacement_m'],
                              distance_sigma_m=c.get('displacement_sigma_m'),
                              rotation_deg=None if c.get('yaw_deg') is None else round(c['yaw_deg'], 1),
                              rotation_source={'appearance': 'appearance (SIFT matches lifted to 3D, RANSAC)',
                                               'geometry': 'geometry (yaw-only ICP)'}.get(c.get('yaw_source'),
                                                                                          c.get('yaw_source')),
                              rotation_confident=c.get('yaw_confident'),
                              transform_before_to_after=transform(c))
            if c.get('rotation', {}).get('appearance'):
                a = c['rotation']['appearance']
                r['delta']['rotation_support'] = dict(inliers=a.get('inliers'), matches=a.get('matches'))
            r['evidence'] = dict(old_place=_evidence(f), new_place=_evidence(t))
            r['label_votes'] = dict(before=f.get('label_votes'), after=t.get('label_votes'))
        else:
            o = c['obj']
            key = 'before' if c['type'] == 'removed' else 'after'
            r[key] = _pose(o, landmarks, bounds, lab)
            r['evidence'] = dict(place=_evidence(o))
            r['label_votes'] = o.get('label_votes')
        if nav:
            r['navigation'] = {}
            for name, rb in nav['robots'].items():
                hit = [x for x in rb['routes'] if c['id'] in x.get('causes', []) or c['id'] in x.get('old_route_hits', [])]
                obj = c.get('to_obj') or c.get('obj')
                tall = obj['z_range'][1] > rb['profile']['step'] and obj['z_range'][0] < rb['profile']['height']
                r['navigation'][name] = dict(
                    obstacle=bool(tall) if c['type'] != 'removed' else None,
                    routes_affected=[dict(goal=x['goal'], status=x['status'], before_m=x['before_m'], after_m=x['updated_m'],
                                          yesterdays_route_clearance_m=x.get('old_route_clearance_m')
                                          if c['id'] in x.get('old_route_hits', []) else None)
                                     for x in hit])
        recs.append(r)
    out = dict(frame=FRAME, changes=recs,
               furniture=[dict(label=l['label'], x=l['xy'][0], y=l['xy'][1],
                               size_m=[round(v, 2) for v in l['footprint']['size']],
                               angle_deg=round(l['footprint']['angle'], 1), height_m=l['z'][1]) for l in landmarks],
               could_not_check=coverage, rejected_candidates=[dict(label=x['label'], epoch=x['epoch'],
                                                                   centroid=x['centroid'], reason=x['reason'])
                                                              for x in rejected])
    if next_views is not None:
        out['next_views'] = next_views
    if meta:
        out['run'] = meta
    return out
