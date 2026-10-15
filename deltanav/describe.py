"""Plain-language descriptions generated from the structured change list, so the words and the geometry can
never disagree: every number and place in a sentence is read from the change record."""
from __future__ import annotations
import math
from .places import where, where_from_to, nice, locate, nearest_wall


def _colour(o) -> str:
    c = o.get('color_name') or ''
    return c + ' ' if c else ''


def _turn(c) -> str:
    yaw = c.get('yaw_deg')
    if yaw is not None and c.get('yaw_confident') and abs(yaw) >= 20:
        return f" and turned about {int(round(abs(yaw) / 5) * 5)}°"
    if c.get('turned'):
        return ' and was turned (the angle is uncertain)'
    return ''


def _same_place(target, p, q, bounds) -> str:
    """A move that starts and ends at the same kind of place: 'across the open floor area', 'along the wall'.
    '' when both ends are at the same furniture (the sentence then says it is still there)."""
    if target == 'the open floor area':
        return 'across the open floor area'
    if target == 'the middle of the room':
        return 'within the middle of the room'
    if target == 'the wall':
        return 'along the wall' if nearest_wall(p, bounds) == nearest_wall(q, bounds) else 'from one wall to another'
    if target == 'a corner of the room':
        return 'within its corner' if math.dist(p[:2], q[:2]) < 1.0 else 'from one corner of the room to another'
    return ''


def describe(changes: list, landmarks: list, bounds) -> list[str]:
    out = []
    for c in changes:
        lab = nice(c['label'])
        if c['type'] == 'moved':
            f, t = c['from_obj'], c['to_obj']
            turn = _turn(c)
            a, b = locate(f['centroid'], landmarks, bounds, lab), locate(t['centroid'], landmarks, bounds, lab)
            head = f"The {_colour(t)}{lab} moved {c['displacement_m']:.1f} m"
            if a != b:
                out.append(f"{head}, from {where_from_to(f['centroid'], landmarks, bounds, lab)} to "
                           f"{where_from_to(t['centroid'], landmarks, bounds, lab)}{',' if turn else ''}{turn}.")
            else:                                     # the same kind of place at both ends
                way = _same_place(b[1], f['centroid'], t['centroid'], bounds)
                out.append(f"{head} {way}{turn}." if way else
                           f"{head}{turn}, and is still {where(t['centroid'], landmarks, bounds, lab)}.")
        elif c['type'] == 'removed':
            o = c['obj']; e = o.get('evidence', {})
            seen = (f" {e['views_saw_through']} frames of the after video look straight through the space it filled."
                    if e.get('views_saw_through') else ' the space it filled is now seen to be empty.')
            loc = where(o['centroid'], landmarks, bounds, lab)
            out.append(f"The {_colour(o)}{lab} {loc}{',' if ',' in loc else ''} is gone:{seen}")
        else:
            o = c['obj']
            h = o['z_range'][1]
            obstacle = ' It stands on the floor, so it is a new obstacle.' if o['z_range'][0] < 0.2 else ''
            out.append(f"A {_colour(o)}{lab} ({h:.2f} m tall) appeared {where(o['centroid'], landmarks, bounds, lab)}.{obstacle}")
    return out


def evidence_line(c) -> str:
    """How the verdict was reached, in one line."""
    def ev(o, other):
        e = o.get('evidence', {})
        if not e.get('views_looked'):
            return ''
        t, s_, oc = e['views_saw_through'], e.get('views_saw_surface', 0), e.get('views_occluded', 0)
        txt = f"{t} {other}-video frames saw straight through, {s_ if s_ else 'none'} saw a surface there"
        return txt + (f" ({oc} had it hidden behind something)" if oc else '')
    if c['type'] == 'moved':
        a = ev(c['from_obj'], 'after'); b = ev(c['to_obj'], 'before')
        return '; '.join(x for x in (f'old place: {a}' if a else '', f'new place: {b}' if b else '') if x)
    return ev(c['obj'], 'after' if c['type'] == 'removed' else 'before')
