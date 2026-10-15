"""Synthetic check of the robot layer: height-aware occupancy, map edits that only touch the changed places,
route re-planning with the cause traced to the right change, and the ROS map export."""
import os, sys, tempfile
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from deltanav.navigation import (ROBOTS, FREE, OCC, grid_spec, occupancy, change_regions, apply_edits, navigation,
                                    export_ros)

S = 0.025


def box_points(x0, x1, y0, y1, z0, z1, top=True, sides=True):
    pts = []
    xs, ys, zs = np.arange(x0, x1 + 1e-9, S), np.arange(y0, y1 + 1e-9, S), np.arange(z0, z1 + 1e-9, S)
    if top:
        X, Y = np.meshgrid(xs, ys); pts.append(np.c_[X.ravel(), Y.ravel(), np.full(X.size, z1)])
    if sides:
        for x in (x0, x1):
            Y, Zz = np.meshgrid(ys, zs); pts.append(np.c_[np.full(Y.size, x), Y.ravel(), Zz.ravel()])
        for y in (y0, y1):
            X, Zz = np.meshgrid(xs, zs); pts.append(np.c_[X.ravel(), np.full(X.size, y), Zz.ravel()])
    return np.vstack(pts)


def floor(exclude=()):
    X, Y = np.meshgrid(np.arange(0, 4.0, S), np.arange(0, 3.0, S))
    P = np.c_[X.ravel(), Y.ravel(), np.zeros(X.size)]
    keep = np.ones(len(P), bool)
    for x0, x1, y0, y1 in exclude:
        keep &= ~((P[:, 0] > x0) & (P[:, 0] < x1) & (P[:, 1] > y0) & (P[:, 1] < y1))
    return P[keep]


BED = (2.6, 4.0, 1.6, 3.0, 0.0, 0.5)
DESK_TOP = (0.0, 1.2, 2.4, 3.0, 0.72, 0.75)
CHAIR_BEFORE = (0.5, 1.0, 1.9, 2.4, 0.0, 0.9)
CHAIR_AFTER = (0.2, 0.7, 0.8, 1.3, 0.0, 0.9)
BACKPACK = (1.9, 2.4, 0.85, 1.15, 0.0, 0.45)


def room(objs, hidden_floor):
    walls = [box_points(0, 0, 0, 3, 0, 2.5, top=False), box_points(4, 4, 0, 3, 0, 2.5, top=False),
             box_points(0, 4, 0, 0, 0, 2.5, top=False), box_points(0, 4, 3, 3, 0, 2.5, top=False)]
    legs = [box_points(x, x + 0.05, y, y + 0.05, 0, 0.72) for x in (0.0, 1.15) for y in (2.4, 2.95)]
    P = np.vstack([floor(hidden_floor)] + walls + legs + [box_points(*DESK_TOP, sides=False), box_points(*BED)] +
                  [box_points(*o) for o in objs])
    return P, P[:, 2].astype(np.float32)


def obj(b, i, ep):
    x0, x1, y0, y1, z0, z1 = b
    return dict(id=i, epoch=ep, centroid=[(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2],
                footprint=dict(center=[(x0 + x1) / 2, (y0 + y1) / 2], size=[x1 - x0, y1 - y0], angle=0.0), z_range=[z0, z1])


def scenario():
    Pb, Hb = room([CHAIR_BEFORE], [BED[:4], CHAIR_BEFORE[:4]])
    Pa, Ha = room([CHAIR_AFTER, BACKPACK], [BED[:4], CHAIR_AFTER[:4], BACKPACK[:4]])
    changes = [dict(id='C1', type='moved', label='chair', from_obj=obj(CHAIR_BEFORE, 0, 'before'), to_obj=obj(CHAIR_AFTER, 1, 'after')),
               dict(id='C3', type='added', label='backpack', obj=obj(BACKPACK, 2, 'after'))]
    lms = [dict(label='bed', xy=[3.3, 2.3], footprint=dict(center=[3.3, 2.3], size=[1.4, 1.4], angle=0.0), z=[0, 0.5]),
           dict(label='desk', xy=[0.6, 2.7], footprint=dict(center=[0.6, 2.7], size=[1.2, 0.6], angle=0.0), z=[0, 0.75]),
           dict(label='door', xy=[2.15, 0.05], footprint=dict(center=[2.15, 0.05], size=[0.8, 0.1], angle=0.0), z=[0, 2.0])]
    return dict(before=Pb, after=Pa), dict(before=Hb, after=Ha), changes, lms


def test_height_aware_occupancy():
    P, H, _, _ = scenario()
    spec = grid_spec([P['before']])
    hum, comp = ROBOTS
    gh = occupancy(P['before'], H['before'], spec, hum); gc = occupancy(P['before'], H['before'], spec, comp)
    ix, iy = spec.cell([0.6, 2.7])
    assert gh[iy[0], ix[0]] == OCC, 'the desk top blocks a 1.8 m humanoid'
    assert gc[iy[0], ix[0]] == FREE, 'a 0.5 m robot fits under the desk'


def test_edits_only_touch_changed_places():
    P, H, changes, _ = scenario()
    spec = grid_spec([P['before'], P['after']])
    rb = ROBOTS[0]
    gb = occupancy(P['before'], H['before'], spec, rb); ga = occupancy(P['after'], H['after'], spec, rb)
    reg = change_regions(changes, spec)
    gu = apply_edits(gb, ga, reg)
    inside = np.zeros_like(gb, bool)
    for m in reg.values():
        inside |= m
    assert (gu[~inside] == gb[~inside]).all()
    ix, iy = spec.cell([2.15, 1.0])
    assert gb[iy[0], ix[0]] == FREE and gu[iy[0], ix[0]] == OCC, 'the new backpack is an obstacle on the updated map'


def test_route_change_traced_to_its_cause():
    P, H, changes, lms = scenario()
    with tempfile.TemporaryDirectory() as d:
        nav = navigation(P, H, {}, changes, lms, os.path.join(d, 'maps'), log=lambda m: None)
        rows = {r['goal']: r for r in nav['robots']['humanoid']['routes']}
        bed = rows['bed']
        assert bed['status'] != 'unchanged' or bed.get('old_route_hits'), bed
        assert 'C3' in bed['causes'] + bed.get('old_route_hits', []), 'the backpack is what changed the route to the bed'
        assert 'C1' not in bed['causes'], 'the chair move does not affect the route to the bed'
        for f in ('humanoid_map_updated.pgm', 'humanoid_map_updated.yaml', 'humanoid_goals.yaml', 'routes.json'):
            assert os.path.exists(os.path.join(d, 'maps', f))


def test_ros_export_roundtrip():
    P, H, _, _ = scenario()
    spec = grid_spec([P['before']])
    g = occupancy(P['before'], H['before'], spec, ROBOTS[0])
    with tempfile.TemporaryDirectory() as d:
        export_ros(g, spec, os.path.join(d, 'm'))
        raw = open(os.path.join(d, 'm.pgm'), 'rb').read()
        header = raw.split(b'\n', 4)
        assert header[0] == b'P5'
        w, h = map(int, header[2].split())
        img = np.frombuffer(header[4], np.uint8).reshape(h, w)[::-1]
        assert (img[g == OCC] == 0).all() and (img[g == FREE] == 254).all()
        y = open(os.path.join(d, 'm.yaml')).read()
        assert 'resolution: 0.05' in y and f'origin: [{spec.x0:.3f}, {spec.y0:.3f}, 0.0]' in y
