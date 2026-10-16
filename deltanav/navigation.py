"""The robot's view of the change: occupancy maps, map edits, routes, goal poses, ROS export.

Yesterday's map (from the before video) is not rebuilt from the after video. It is *edited*: every verified
change patches its own footprint with what the after video saw there, and nothing else in the map moves. So
depth noise and the parts of the room the after video skipped never change the robot's map, while every
real change does.

Two robot profiles show that the same edit matters differently to different robots: a compact robot
(radius 0.2 m, 0.5 m tall) that fits under the desk, and a humanoid (radius 0.3 m, 1.8 m tall).
Routes run from the door (or from where the before video started) to every piece of furniture, with A* on the
inflated map, and each change in a route is traced back to the change(s) that caused it by re-planning with
that one edit left out."""
from __future__ import annotations
import heapq, json, math, os
from dataclasses import dataclass, asdict
import numpy as np

FREE, OCC, UNK = 0, 100, -1


@dataclass
class Robot:
    name: str
    radius: float          # footprint radius, m
    height: float          # obstacles between `step` and `height` above the floor block it
    camera_height: float   # for the next-best views
    step: float = 0.12     # lower things are ignored: monocular depth cannot tell them from a bump in the floor
    label: str = ''


ROBOTS = [Robot('humanoid', 0.25, 1.80, 1.60, label='humanoid (Ø 0.5 m, 1.8 m tall)'),
          Robot('compact', 0.20, 0.50, 0.40, label='compact robot (Ø 0.4 m, 0.5 m tall)')]
GOAL_CLASSES = ['bed', 'desk', 'wardrobe', 'nightstand', 'radiator', 'window', 'bookshelf', 'shelf', 'drawers',
                'couch', 'table', 'cabinet', 'television', 'mirror',
                'fridge', 'oven', 'stove', 'sink', 'counter', 'kitchen island', 'washing machine', 'toilet', 'bathtub',
                'shower', 'fireplace', 'piano']


@dataclass
class GridSpec:
    x0: float
    y0: float
    nx: int
    ny: int
    res: float = 0.05

    def cell(self, xy):
        xy = np.atleast_2d(np.asarray(xy, float))
        return (np.floor((xy[:, 0] - self.x0) / self.res).astype(int),
                np.floor((xy[:, 1] - self.y0) / self.res).astype(int))

    def centre(self, ix, iy):
        return np.stack([self.x0 + (np.asarray(ix) + 0.5) * self.res, self.y0 + (np.asarray(iy) + 0.5) * self.res], -1)

    def inside(self, ix, iy):
        return (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny)

    def extent(self):                       # for matplotlib imshow(origin='lower')
        return [self.x0, self.x0 + self.nx * self.res, self.y0, self.y0 + self.ny * self.res]


def grid_spec(P_list, res: float = 0.05, margin: float = 0.3) -> GridSpec:
    P = np.vstack([p[:, :2] for p in P_list])
    lo = np.percentile(P, 0.5, axis=0) - margin; hi = np.percentile(P, 99.5, axis=0) + margin
    lo = np.floor(lo / res) * res
    return GridSpec(float(lo[0]), float(lo[1]), int(np.ceil((hi[0] - lo[0]) / res)), int(np.ceil((hi[1] - lo[1]) / res)), res)


# ---------------------------------------------------------------- maps
def local_floor(P, Hgt, spec: GridSpec, max_h: float = 0.15, win: int = 9):
    """Per-cell floor offset: the median of the lowest surface heights of nearby cells where the floor shows.
    Monocular depth bends the floor by a few centimetres; measuring obstacles from the local floor keeps that
    from turning carpet into obstacles."""
    from scipy.ndimage import generic_filter
    ix, iy = spec.cell(P[:, :2]); ok = spec.inside(ix, iy)
    low = np.full(spec.nx * spec.ny, np.inf)
    np.minimum.at(low, iy[ok] * spec.nx + ix[ok], Hgt[ok])
    low = low.reshape(spec.ny, spec.nx)
    cand = np.where(low < max_h, low, np.nan)
    med = generic_filter(cand, lambda v: np.nanmedian(v) if np.isfinite(v).any() else np.nan, size=win, mode='nearest')
    return np.nan_to_num(med, nan=0.0)


def occupancy(P, Hgt, spec: GridSpec, robot: Robot, walked=None, floor_tol: float = 0.06, min_pts: int = 3,
              stable=None, floor=None):
    """2D occupancy (rows = y, cols = x) for one robot from fused surface points with heights above the floor.
    occupied: at least `min_pts` self-consistent surface points between the robot's step height and its height
    (measured from the local floor); free: floor seen there (or walked on by the camera operator) and nothing in
    that band; unknown otherwise."""
    ix, iy = spec.cell(P[:, :2])
    ok = spec.inside(ix, iy)
    flat = iy[ok] * spec.nx + ix[ok]
    h = Hgt[ok] - (floor.ravel()[flat] if floor is not None else 0.0)
    st = stable[ok] if stable is not None else np.ones(len(h), bool)
    n = spec.nx * spec.ny
    occ = np.bincount(flat[(h > robot.step) & (h < robot.height) & st], minlength=n)
    flo = np.bincount(flat[np.abs(h) <= floor_tol], minlength=n)
    g = np.full(n, UNK, np.int8)
    g[flo > 0] = FREE
    if walked is not None:
        g[walked.ravel() & (g == UNK)] = FREE
    g[occ >= min_pts] = OCC
    g = g.reshape(spec.ny, spec.nx)
    return _fill_holes(_drop_specks(g))


def _drop_specks(g, max_cells: int = 3):
    """Isolated obstacles of at most `max_cells` cells (under ~10 cm across) are depth noise far more often than
    real objects; they become unknown instead of being inflated into a 0.6 m wall."""
    from scipy.ndimage import label
    lab, n = label(g == OCC, structure=np.ones((3, 3)))
    if n == 0:
        return g
    size = np.bincount(lab.ravel())
    small = (size <= max_cells); small[0] = False
    g = g.copy(); g[small[lab]] = UNK
    return g


def _fill_holes(g):
    """Unknown cells with free cells on all four sides (gaps between depth samples) become free."""
    u = g == UNK
    f = g == FREE
    p = np.pad(f, 1)
    four = p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:]
    g = g.copy(); g[u & four] = FREE
    return g


def walked_cells(views, spec: GridSpec, epoch: str, back: float = 0.25, radius: float = 0.15):
    """Where the camera operator stood: a little behind the phone, on the floor. Floor someone walked on is
    floor a robot can use (unless something is on it now)."""
    m = np.zeros((spec.ny, spec.nx), bool)
    r = int(np.ceil(radius / spec.res))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    disc = (xx ** 2 + yy ** 2) * spec.res ** 2 <= radius ** 2
    for v in views.values():
        if v.epoch != epoch:
            continue
        f = v.forward.copy(); f[2] = 0
        nf = np.linalg.norm(f)
        p = v.C[:2] - (f[:2] / nf * back if nf > 1e-6 else 0)
        ix, iy = spec.cell(p)
        for dy, dx in zip(yy[disc], xx[disc]):
            jx, jy = ix[0] + dx, iy[0] + dy
            if 0 <= jx < spec.nx and 0 <= jy < spec.ny:
                m[jy, jx] = True
    return m


def change_regions(changes: list, spec: GridSpec, pad: float = 0.12):
    """Per change id: boolean cell mask of the footprints it touches (both places for a move)."""
    out = {}
    yy, xx = np.mgrid[0:spec.ny, 0:spec.nx]
    C = spec.centre(xx, yy)
    for c in changes:
        objs = [c['from_obj'], c['to_obj']] if c['type'] == 'moved' else [c['obj']]
        m = np.zeros((spec.ny, spec.nx), bool)
        for o in objs:
            fp = o['footprint']; a = np.radians(fp['angle'])
            ux, uy = np.array([np.cos(a), np.sin(a)]), np.array([-np.sin(a), np.cos(a)])
            d = C - np.array(fp['center'])
            half = np.array(fp['size']) / 2 + pad
            m |= (np.abs(d @ ux) <= half[0]) & (np.abs(d @ uy) <= half[1])
        out[c['id']] = m
    return out


def apply_edits(before, after, regions: dict, ids=None):
    """Yesterday's map with each verified change patched in from the after video, only inside its footprint."""
    g = before.copy()
    for cid, m in regions.items():
        if ids is None or cid in ids:
            g[m] = after[m]
    return g


# ---------------------------------------------------------------- planning
def _disk(r_cells):
    yy, xx = np.mgrid[-r_cells:r_cells + 1, -r_cells:r_cells + 1]
    return (xx ** 2 + yy ** 2 <= r_cells ** 2).astype(np.uint8)


def clearance(g, res):
    """Distance (m) from every cell to the nearest occupied cell."""
    from scipy.ndimage import distance_transform_edt
    occ = g == OCC
    if not occ.any():
        return np.full(g.shape, 99.0)
    return distance_transform_edt(~occ) * res


def traversable(g, robot: Robot, res: float):
    """Cells where the robot's centre can be: free, and no obstacle within its radius."""
    return (g == FREE) & (clearance(g, res) > robot.radius)


def astar(trav, clear, start, goal, res: float, radius: float, w: float = 2.0):
    """8-connected A*; steps near obstacles cost more (like a Nav2 inflation layer), so routes keep a margin."""
    ny, nx = trav.shape
    if not (trav[start[1], start[0]] and trav[goal[1], goal[0]]):
        return None
    pen = 1.0 + w * np.exp(-6.0 * np.maximum(clear - radius, 0.0))
    INF = np.inf
    gs = np.full(trav.shape, INF); gs[start[1], start[0]] = 0
    came = {}
    h = lambda x, y: math.hypot(x - goal[0], y - goal[1])
    pq = [(h(*start), 0.0, start)]
    nb = [(1, 0, 1), (-1, 0, 1), (0, 1, 1), (0, -1, 1), (1, 1, math.sqrt(2)), (1, -1, math.sqrt(2)),
          (-1, 1, math.sqrt(2)), (-1, -1, math.sqrt(2))]
    while pq:
        f, gc, (x, y) = heapq.heappop(pq)
        if (x, y) == tuple(goal):
            path = [(x, y)]
            while path[-1] in came:
                path.append(came[path[-1]])
            return path[::-1]
        if gc > gs[y, x]:
            continue
        for dx, dy, c in nb:
            u, v = x + dx, y + dy
            if 0 <= u < nx and 0 <= v < ny and trav[v, u]:
                if dx and dy and not (trav[y, u] and trav[v, x]):
                    continue
                ng = gc + c * pen[v, u]
                if ng < gs[v, u]:
                    gs[v, u] = ng; came[(u, v)] = (x, y)
                    heapq.heappush(pq, (ng + h(u, v), ng, (u, v)))
    return None


def reach(trav, start):
    """Cells reachable from start (4-connected flood fill)."""
    from scipy.ndimage import label
    lab, _ = label(trav)
    l = lab[start[1], start[0]]
    return lab == l if l > 0 else np.zeros_like(trav)


def smooth(path, trav):
    """String pulling: keep a waypoint only where the straight line to the next one would leave free cells."""
    if not path or len(path) < 3:
        return path
    def los(a, b):
        n = int(max(abs(b[0] - a[0]), abs(b[1] - a[1])) * 2) + 1
        xs = np.round(np.linspace(a[0], b[0], n)).astype(int); ys = np.round(np.linspace(a[1], b[1], n)).astype(int)
        return trav[ys, xs].all()
    out = [path[0]]; i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1 and not los(path[i], path[j]):
            j -= 1
        out.append(path[j]); i = j
    return out


def path_length(path, res):
    if not path or len(path) < 2:
        return 0.0
    p = np.asarray(path, float)
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum() * res)


def densify(path, spec: GridSpec, step: float = 0.05):
    if not path:
        return np.zeros((0, 2))
    P = spec.centre(*np.asarray(path).T)
    out = [P[0]]
    for a, b in zip(P[:-1], P[1:]):
        n = max(1, int(np.linalg.norm(b - a) / step))
        out += [a + (b - a) * k / n for k in range(1, n + 1)]
    return np.asarray(out)


def hausdorff(A, B):
    if len(A) == 0 or len(B) == 0:
        return np.inf
    from scipy.spatial import cKDTree
    return float(max(cKDTree(B).query(A)[0].max(), cKDTree(A).query(B)[0].max()))


# ---------------------------------------------------------------- starts and goals
def start_cell(trav, spec: GridSpec, landmarks, views, max_dist: float = 1.5):
    """Where the robot comes in: the door if one was found, else where the before video started. The start is
    the cell nearest to it in the largest free region within reach (so a small pocket of free floor next to the
    door does not trap the robot)."""
    from scipy.ndimage import label
    door = [l for l in landmarks if l['label'] == 'door']
    if door:
        p, why = np.array(door[0]['xy']), 'the door'
    else:
        first = min((v for v in views.values() if v.epoch == 'before'), key=lambda v: v.frame)
        p, why = first.C[:2], 'where the before video started'
    lab, n = label(trav, structure=np.ones((3, 3)))
    ys, xs = np.nonzero(trav)
    if not len(xs):
        return None, why
    d = np.linalg.norm(spec.centre(xs, ys) - p, axis=1)
    near = d <= max_dist
    if not near.any():
        return None, why
    sizes = np.bincount(lab.ravel())
    comp = max(set(lab[ys[near], xs[near]].tolist()), key=lambda c: sizes[c])
    m = near & (lab[ys, xs] == comp)
    k = int(np.argmin(np.where(m, d, np.inf)))
    return (int(xs[k]), int(ys[k])), why


def footprint_dist(C, lm):
    fp = lm['footprint']; a = np.radians(fp['angle'])
    ux, uy = np.array([np.cos(a), np.sin(a)]), np.array([-np.sin(a), np.cos(a)])
    d = C - np.array(fp['center'])
    q = np.stack([d @ ux, d @ uy], -1)
    half = np.array(fp['size']) / 2
    out = np.maximum(np.abs(q) - half, 0)
    closest = np.array(fp['center']) + np.clip(q[:, :1], -half[0], half[0]) * ux + np.clip(q[:, 1:], -half[1], half[1]) * uy
    return np.hypot(out[:, 0], out[:, 1]), closest


def goal_for(lm, trav, reachable, spec: GridSpec, robot: Robot, dist_from_start=None):
    """A pose from which the robot faces the piece of furniture: a reachable cell whose edge is 5-45 cm from
    the furniture's footprint; the nearest such cell to the start wins. -> (cell, yaw) or (None, None)."""
    ys, xs = np.nonzero(trav & reachable)
    if not len(xs):
        return None, None
    C = spec.centre(xs, ys)
    d, closest = footprint_dist(C, lm)
    gap = d - robot.radius
    ok = (gap >= 0.0) & (gap <= 0.45)
    if not ok.any():
        return None, None
    idx = np.nonzero(ok)[0]
    if dist_from_start is not None:
        k = idx[np.argmin(dist_from_start[ys[idx], xs[idx]] + 2.0 * gap[idx])]
    else:
        k = idx[np.argmin(gap[idx])]
    v = closest[k] - C[k]
    return (int(xs[k]), int(ys[k])), float(np.degrees(np.arctan2(v[1], v[0])))


def geodesic(trav, start):
    """Path distance (cells) from start over traversable cells (Dijkstra, 8-connected)."""
    ny, nx = trav.shape
    D = np.full(trav.shape, np.inf); D[start[1], start[0]] = 0
    pq = [(0.0, start)]
    nb = [(1, 0, 1), (-1, 0, 1), (0, 1, 1), (0, -1, 1), (1, 1, 1.4142), (1, -1, 1.4142), (-1, 1, 1.4142), (-1, -1, 1.4142)]
    while pq:
        d, (x, y) = heapq.heappop(pq)
        if d > D[y, x]:
            continue
        for dx, dy, c in nb:
            u, v = x + dx, y + dy
            if 0 <= u < nx and 0 <= v < ny and trav[v, u] and d + c < D[v, u]:
                if dx and dy and not (trav[y, u] and trav[v, x]):
                    continue
                D[v, u] = d + c; heapq.heappush(pq, (d + c, (u, v)))
    return D


# ---------------------------------------------------------------- one robot, one map
def _nearest_trav(trav, cell, max_cells: int = 6):
    """The traversable cell nearest to `cell` (within max_cells), or None."""
    if cell is None:
        return None
    if trav[cell[1], cell[0]]:
        return tuple(cell)
    ys, xs = np.nonzero(trav)
    if not len(xs):
        return None
    d = np.hypot(xs - cell[0], ys - cell[1])
    k = int(np.argmin(d))
    return (int(xs[k]), int(ys[k])) if d[k] <= max_cells else None


def plan_all(g, robot: Robot, spec: GridSpec, landmarks, views, ref=None):
    """Routes from the start to a goal pose in front of every piece of furniture. With `ref` (yesterday's plan),
    the same start and the same goal poses are kept wherever they are still free, as a robot with stored goals
    would; a goal moves only when something now stands on it."""
    clear = clearance(g, spec.res)
    trav = (g == FREE) & (clear > robot.radius)
    why = ref.get('start_from') if ref else None
    start = _nearest_trav(trav, ref.get('start')) if ref and ref.get('start') else None
    if start is None:
        start, why = start_cell(trav, spec, landmarks, views)
    if start is None or not trav[start[1], start[0]]:
        return dict(start=None, start_from=why, routes={}, reach=np.zeros_like(trav), trav=trav, clear=clear)
    R = reach(trav, start)
    D = geodesic(trav, start)
    routes = {}
    seen = {}
    for lm in landmarks:
        if lm['label'] not in GOAL_CLASSES:
            continue
        key = lm['label'] if lm['label'] not in seen else f"{lm['label']} {seen[lm['label']] + 1}"
        seen[lm['label']] = seen.get(lm['label'], 0) + 1
        cell, yaw = None, None
        old = (ref or {}).get('routes', {}).get(key, {})
        if old.get('reachable'):
            c0 = tuple(old['cells'][-1])
            if trav[c0[1], c0[0]] and R[c0[1], c0[0]]:
                cell, yaw = c0, old['yaw_deg']
        if cell is None:
            cell, yaw = goal_for(lm, trav, R, spec, robot, D)
        if cell is None:
            routes[key] = dict(reachable=False, landmark=lm['label'])
            continue
        path = astar(trav, clear, start, cell, spec.res, robot.radius)
        if path is None:
            routes[key] = dict(reachable=False, landmark=lm['label'])
            continue
        sp = smooth(path, trav)
        routes[key] = dict(reachable=True, landmark=lm['label'], goal=spec.centre(*cell).tolist(), yaw_deg=round(yaw, 1),
                           cells=path, waypoints=spec.centre(*np.asarray(sp).T).round(3).tolist(),
                           length_m=round(path_length(sp, spec.res), 2),
                           min_clearance_m=round(float(min(clear[y, x] for x, y in path)), 2))
    return dict(start=start, start_xy=spec.centre(*start).round(3).tolist(), start_from=why, routes=routes, reach=R,
                trav=trav, clear=clear)


def compare_route(a, b, spec: GridSpec):
    """-> status of a route from yesterday's map (a) to the updated map (b)."""
    if not a.get('reachable') and not b.get('reachable'):
        return 'unreachable'
    if a.get('reachable') and not b.get('reachable'):
        return 'blocked'
    if not a.get('reachable'):
        return 'opened'
    dl = b['length_m'] - a['length_m']
    if max(a['length_m'], b['length_m']) < 0.3:
        return 'unchanged'                               # already there from the door
    hd = hausdorff(densify(a['cells'], spec), densify(b['cells'], spec))
    if hd < 0.3 and abs(dl) < 0.2:
        return 'unchanged'                               # small adjustments are not worth reporting
    if dl >= 0.2:
        return 'longer'
    if dl <= -0.2:
        return 'shorter'
    return 'rerouted'


# ---------------------------------------------------------------- everything, for both robots
def navigation(P, Hgt, views, changes, landmarks, out_dir, robots=ROBOTS, res=0.05, log=print, stable=None):
    """P, Hgt: dicts epoch -> fused points / heights (stable: epoch -> self-consistent points). Writes maps/ and
    returns a summary dict (also maps/routes.json)."""
    os.makedirs(out_dir, exist_ok=True)
    spec = grid_spec([P['before'], P['after']], res)
    floors = {ep: local_floor(P[ep], Hgt[ep], spec) for ep in ('before', 'after')}
    regions = change_regions(changes, spec)
    arrived = change_regions([dict(c, type='added', obj=c['to_obj']) if c['type'] == 'moved' else c
                              for c in changes], spec, pad=0.05)       # where something new now stands
    summary = dict(grid=asdict(spec), robots={}, frame='room', files=[])
    for robot in robots:
        maps = {}
        for ep in ('before', 'after'):
            maps[ep] = occupancy(P[ep], Hgt[ep], spec, robot, walked=walked_cells(views, spec, ep),
                                 stable=None if stable is None else stable[ep], floor=floors[ep])
        maps['updated'] = apply_edits(maps['before'], maps['after'], regions)
        plans = {'before': plan_all(maps['before'], robot, spec, landmarks, views)}
        plans['updated'] = plan_all(maps['updated'], robot, spec, landmarks, views, ref=plans['before'])
        # cause of every route change: re-plan with each edit left out
        rows, alt_plans = [], {}

        def without(cid):
            if cid not in alt_plans:
                others = [x['id'] for x in changes if x['id'] != cid]
                alt_plans[cid] = plan_all(apply_edits(maps['before'], maps['after'], regions, others), robot, spec,
                                          landmarks, views, ref=plans['before'])
            return alt_plans[cid]
        for key, a in plans['before']['routes'].items():
            b = plans['updated']['routes'].get(key, dict(reachable=False))
            st = compare_route(a, b, spec)
            row = dict(goal=key, status=st, before_m=a.get('length_m'), updated_m=b.get('length_m'), causes=[])
            if a.get('reachable') and b.get('reachable'):
                row['goal_moved_m'] = round(float(np.linalg.norm(np.subtract(a['goal'], b['goal']))), 2)
            if st not in ('unchanged', 'unreachable'):
                for c in changes:
                    alt = without(c['id'])['routes'].get(key, dict(reachable=False))
                    if compare_route(a, alt, spec) != st or (st in ('longer', 'shorter', 'rerouted') and alt.get('reachable')
                                                             and abs(alt['length_m'] - b['length_m']) > 0.05):
                        row['causes'].append(c['id'])
            # does yesterday's route now run into something that arrived?
            if a.get('reachable'):
                xy = spec.centre(*np.asarray(a['cells']).T)
                new_occ = (maps['updated'] == OCC) & (maps['before'] != OCC)
                hit, cl = [], np.inf
                for c in changes:
                    if c['type'] == 'removed':
                        continue
                    m = arrived[c['id']] & new_occ
                    if m.any():
                        ys, xs = np.nonzero(m)
                        from scipy.spatial import cKDTree
                        dmin = float(cKDTree(spec.centre(xs, ys)).query(xy)[0].min()) - spec.res / 2
                        if dmin < robot.radius:
                            hit.append(c['id']); cl = min(cl, max(0.0, dmin))
                if hit:
                    row['old_route_clearance_m'] = round(float(cl), 2)
                    row['old_route_hits'] = hit
            rows.append(row)
        area = {k: round(float(plans[k]['reach'].sum() * res * res), 2) for k in plans}
        diff_upd = (maps['updated'] != maps['before']); diff_after = (maps['after'] != maps['before'])
        known = (maps['before'] != UNK) & (maps['after'] != UNK)
        edits = dict(
            vacated_m2=round(float(((maps['before'] == OCC) & (maps['updated'] == FREE)).sum() * res * res), 2),
            new_obstacle_m2=round(float(((maps['before'] != OCC) & (maps['updated'] == OCC)).sum() * res * res), 2),
            now_unknown_m2=round(float(((maps['before'] != UNK) & (maps['updated'] == UNK)).sum() * res * res), 2),
            cells_changed_by_edits_m2=round(float(diff_upd.sum() * res * res), 2),
            cells_changed_by_remapping_m2=round(float(diff_after.sum() * res * res), 2),
            remapping_changes_where_both_known_m2=round(float((diff_after & known).sum() * res * res), 2))
        files = []
        for k, g in maps.items():
            files += export_ros(g, spec, os.path.join(out_dir, f'{robot.name}_map_{k}'))
        # poses from which to inspect each change (go and look at the new backpack, at where the suitcase was), with
        # the route there on the updated map: every room with a change gets a route, furniture or not
        pu = plans['updated']
        insp = {}
        if pu.get('start') is not None:
            D = geodesic(pu['trav'], pu['start'])
            for c in changes:
                o = c.get('to_obj') or c['obj']
                lm = dict(label=c['label'], footprint=o['footprint'])
                cell, yaw = goal_for(lm, pu['trav'], pu['reach'], spec, robot, D)
                if cell is not None:
                    e = dict(reachable=True, goal=spec.centre(*cell).tolist(), yaw_deg=round(yaw, 1))
                    path = astar(pu['trav'], pu['clear'], pu['start'], cell, spec.res, robot.radius)
                    if path is not None:
                        sp = smooth(path, pu['trav'])
                        e.update(waypoints=spec.centre(*np.asarray(sp).T).round(3).tolist(),
                                 length_m=round(path_length(sp, spec.res), 2))
                    insp[f"check_{c['id']}_{c['label'].replace(' ', '_')}"] = e
        goals_path = os.path.join(out_dir, f'{robot.name}_goals.yaml')
        export_goals(dict(pu, routes={**pu['routes'], **insp}), robot, goals_path)
        files.append(goals_path)
        summary['robots'][robot.name] = dict(
            profile=asdict(robot), start=plans['before'].get('start_xy'), start_from=plans['before']['start_from'],
            inspect_changes=insp,
            reachable_area_m2=area, routes=rows, edits=edits,
            plans={k: {g: {kk: vv for kk, vv in r.items() if kk != 'cells'} for g, r in plans[k]['routes'].items()}
                   for k in plans})
        summary['files'] += [os.path.relpath(f, os.path.dirname(out_dir)) for f in files]
        np.savez_compressed(os.path.join(out_dir, f'{robot.name}_grids.npz'), **{f'map_{k}': g for k, g in maps.items()},
                            **{f'trav_{k}': plans[k]['trav'] for k in plans}, **{f'reach_{k}': plans[k]['reach'] for k in plans},
                            spec=np.array([spec.x0, spec.y0, spec.nx, spec.ny, spec.res]))
        moved = ', '.join(f"{r['goal']} {r['status']}" for r in rows if r['status'] not in ('unchanged', 'unreachable'))
        log(f"  {robot.name}: reachable floor {area['before']} -> {area['updated']} m2; "
            f"routes: {moved or 'no route changes'}")
    summary['sentences'] = route_sentences(summary, changes)
    json.dump(summary, open(os.path.join(out_dir, 'routes.json'), 'w'), indent=1,
              default=lambda o: o.item() if hasattr(o, 'item') and getattr(o, 'ndim', 0) == 0 else o.tolist())
    return summary


def route_sentences(summary, changes):
    label = {c['id']: c['label'] for c in changes}
    out = {}
    for name, r in summary['robots'].items():
        prof = r['profile']
        s = []
        for row in r['routes']:
            who = ' and the '.join(label[c] for c in row['causes']) if row['causes'] else None
            g = row['goal']
            if row['status'] == 'blocked':
                s.append(f"The {g} can no longer be reached" + (f": the {who} {'are' if ' and ' in who else 'is'} in the way."
                                                                       if who else '.'))
            elif row['status'] == 'opened':
                s.append(f"The {g} can now be reached" + (f": moving the {who} opened a gap." if who else '.'))
            elif row['status'] in ('longer', 'shorter', 'rerouted'):
                d = (row['updated_m'] or 0) - (row['before_m'] or 0)
                how = {'longer': f'{abs(d):.1f} m longer', 'shorter': f'{abs(d):.1f} m shorter',
                       'rerouted': 'about as long'}[row['status']]
                why = f"because of the {who}" if who else 'because the map changed'
                if row.get('goal_moved_m', 0) >= 0.2:
                    s.append(f"To reach the {g} it now stops at a different spot ({row['goal_moved_m']:.1f} m from "
                             f"yesterday's), {why}; the route is {how}.")
                else:
                    s.append(f"The route to the {g} changed {why}; it is now {how}.")
        groups = {}
        for row in r['routes']:
            if row.get('old_route_hits'):
                groups.setdefault(tuple(row['old_route_hits']), []).append(row)
        for hits, rows in groups.items():
            hit = ' and the '.join(label[c] for c in hits)
            goals = [x['goal'] for x in rows]
            gl = goals[0] if len(goals) == 1 else ', the '.join(goals[:-1]) + ' and the ' + goals[-1]
            cl = min(x['old_route_clearance_m'] for x in rows)
            plural = 's' if len(goals) > 1 else ''
            if cl < 0.05:
                s.append(f"Yesterday's route{plural} to the {gl} would now run straight into the {hit}.")
            else:
                s.append(f"Yesterday's route{plural} to the {gl} would pass within {cl:.2f} m of the {hit}, inside its "
                         f"{prof['radius']:.2f} m radius.")
        a = r['reachable_area_m2']
        s.append(f"Floor it can reach from {r['start_from']}: {a['before']:.2f} m² before, {a['updated']:.2f} m² now.")

        out[name] = s
    return out


# ---------------------------------------------------------------- export
def export_ros(g, spec: GridSpec, stem: str):
    """ROS map_server pair: <stem>.pgm (254 free, 0 occupied, 205 unknown) and <stem>.yaml (origin = lower-left)."""
    img = np.full(g.shape, 205, np.uint8)
    img[g == FREE] = 254; img[g == OCC] = 0
    img = img[::-1]                                    # PGM row 0 is the top (max y)
    with open(stem + '.pgm', 'wb') as f:
        f.write(f'P5\n# deltanav occupancy, room frame\n{img.shape[1]} {img.shape[0]}\n255\n'.encode())
        f.write(img.tobytes())
    with open(stem + '.yaml', 'w') as f:
        f.write(f"image: {os.path.basename(stem)}.pgm\nmode: trinary\nresolution: {spec.res}\n"
                f"origin: [{spec.x0:.3f}, {spec.y0:.3f}, 0.0]\nnegate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.25\n")
    return [stem + '.pgm', stem + '.yaml']


def export_goals(plan, robot: Robot, path: str):
    """Named goal poses for a Nav2 waypoint follower (frame 'map' = the room frame)."""
    lines = [f"# goal poses for the {robot.label or robot.name}, in the room frame (= map frame of the exported maps)",
             "# yaw: radians, facing the furniture; quaternion about z for geometry_msgs/Pose",
             "frame_id: map", "poses:"]
    for name, r in plan['routes'].items():
        if not r.get('reachable'):
            continue
        yaw = math.radians(r['yaw_deg'])
        lines.append(f"  - name: {name.replace(' ', '_')}\n    x: {r['goal'][0]:.3f}\n    y: {r['goal'][1]:.3f}\n"
                     f"    yaw: {yaw:.4f}\n    orientation: {{x: 0.0, y: 0.0, z: {math.sin(yaw / 2):.4f}, w: {math.cos(yaw / 2):.4f}}}")
    if plan.get('start_xy'):
        lines.append(f"start:\n  x: {plan['start_xy'][0]:.3f}\n  y: {plan['start_xy'][1]:.3f}")
    open(path, 'w').write('\n'.join(lines) + '\n')


# ---------------------------------------------------------------- 3D occupancy (for robots that are not flat)
def occupancy3d_updated(occ_b, occ_a, origin, voxel, changes, floor_c, pad: float = 0.12):
    """Yesterday's 3D occupancy with each verified change patched in from the after video inside its box."""
    from .geometry import floor_z
    g = occ_b.copy()
    nx, ny, nz = occ_b.shape
    ix, iy, iz = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing='ij')
    X = np.asarray(origin) + (np.stack([ix, iy, iz], -1) + 0.5) * voxel
    for c in changes:
        objs = [c['from_obj'], c['to_obj']] if c['type'] == 'moved' else [c['obj']]
        for o in objs:
            fp = o['footprint']; a = np.radians(fp['angle'])
            ux, uy = np.array([np.cos(a), np.sin(a)]), np.array([-np.sin(a), np.cos(a)])
            d = X[..., :2] - np.array(fp['center'])
            half = np.array(fp['size']) / 2 + pad
            hz = X[..., 2] - floor_z(X[..., :2].reshape(-1, 2), floor_c).reshape(X.shape[:3])
            m = (np.abs(d @ ux) <= half[0]) & (np.abs(d @ uy) <= half[1]) & (hz <= o['z_range'][1] + pad)
            g[m] = occ_a[m]
    return g


def export_occupancy3d(path_stem, grids: dict, origin, voxel, floor_c=None):
    """<stem>.npz (int8 grids: 1 occupied, 0 free, -1 unknown; origin = corner of voxel [0,0,0]; room frame) and
    <stem>_occupied_updated.ply (occupied voxel centres, coloured by height) for any point-cloud viewer."""
    np.savez_compressed(path_stem + '.npz', origin=np.asarray(origin, float), voxel=float(voxel),
                        **{k: v for k, v in grids.items()},
                        legend=np.array(['1 = occupied', '0 = free (seen through)', '-1 = unknown']))
    g = grids.get('updated', next(iter(grids.values())))
    idx = np.argwhere(g == 1)
    P = np.asarray(origin) + (idx + 0.5) * voxel
    z = P[:, 2]; t = np.clip((z - z.min()) / max(1e-6, np.ptp(z)), 0, 1)
    C = np.stack([255 * t, 80 + 120 * (1 - np.abs(2 * t - 1)), 255 * (1 - t)], 1).astype(np.uint8)
    from .mesh import write_points_ply
    write_points_ply(path_stem + '_occupied_updated.ply', P.astype(np.float32), C)
    free = {k: round(float((v == 0).sum() * voxel ** 3), 2) for k, v in grids.items()}
    return dict(free_m3=free, voxel=voxel, files=[path_stem + '.npz', path_stem + '_occupied_updated.ply'])
