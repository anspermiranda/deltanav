"""The words and numbers come from the same records: sentences, relations and the before -> after transform."""
import os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from deltanav.ledger import transform
from deltanav.describe import describe


def obj(cx, cy, w, d, h, ev=None):
    return dict(centroid=[cx, cy, h / 2], footprint=dict(center=[cx, cy], size=[w, d], angle=0.0), z_range=[0.0, h],
                color_name='black', evidence=ev or {}, sigma_m=0.03)


LMS = [dict(label='desk', xy=[0.6, 0.35], footprint=dict(center=[0.6, 0.35], size=[1.2, 0.6], angle=0.0), z=[0, 0.75], support=50),
       dict(label='bed', xy=[2.5, 2.2], footprint=dict(center=[2.5, 2.2], size=[1.4, 1.95], angle=0.0), z=[0, 0.55], support=80),
       dict(label='wardrobe', xy=[3.75, 0.6], footprint=dict(center=[3.75, 0.6], size=[0.5, 1.0], angle=0.0), z=[0, 1.9], support=60)]
BOUNDS = ((0.0, 0.0), (4.0, 3.2))


def test_transform_maps_before_pose_to_after_pose():
    c = dict(yaw_deg=57.0, from_obj=obj(1.2, 0.9, 0.5, 0.5, 0.9), to_obj=obj(2.6, 1.0, 0.5, 0.5, 0.9))
    T = np.array(transform(c))
    assert np.allclose(T @ np.r_[c['from_obj']['centroid'], 1], np.r_[c['to_obj']['centroid'], 1], atol=1e-3)
    assert np.isclose(np.degrees(np.arctan2(T[1, 0], T[0, 0])), 57.0, atol=1e-2)


def test_sentences_read_the_numbers_and_places():
    ch = [dict(type='moved', label='office chair', from_obj=obj(0.6, 0.85, 0.5, 0.5, 0.9), to_obj=obj(3.45, 1.15, 0.3, 0.3, 0.9),
               displacement_m=2.87, yaw_deg=57.0, yaw_confident=True, turned=True),
          dict(type='removed', label='suitcase', obj=obj(2.0, 0.6, 0.6, 0.3, 0.62, dict(views_saw_through=48, views_looked=50)))]
    s = describe(ch, LMS, BOUNDS)
    assert s[0].startswith('The black office chair moved 2.9 m, from the desk') and 'turned about 55°' in s[0], s[0]
    assert 'between the bed and the wardrobe' in s[0], s[0]
    assert s[1].startswith('The black suitcase') and '48 frames of the after video' in s[1], s[1]


def test_sentences_without_furniture_say_where_in_the_room():
    ch = [dict(type='moved', label='chair', from_obj=obj(1.0, 0.2, 0.5, 0.5, 0.9), to_obj=obj(2.6, 0.2, 0.5, 0.5, 0.9),
               displacement_m=1.6, yaw_deg=None, turned=False),
          dict(type='moved', label='box', from_obj=obj(2.0, 1.6, 0.4, 0.4, 0.4), to_obj=obj(0.3, 0.3, 0.4, 0.4, 0.4),
               displacement_m=2.1, yaw_deg=None, turned=False),
          dict(type='added', label='guitar', obj=obj(3.7, 2.9, 0.4, 0.3, 1.0)),
          dict(type='removed', label='lamp', obj=obj(0.8, 1.6, 0.3, 0.3, 1.2, dict(views_saw_through=9)))]
    s = describe(ch, [], BOUNDS)
    assert s[0] == 'The black chair moved 1.6 m along the wall.', s[0]
    assert s[1] == 'The black box moved 2.1 m, from the middle of the room to a corner of the room.', s[1]
    assert s[2].startswith('A black guitar (1.00 m tall) appeared in a corner of the room.'), s[2]
    assert s[3].startswith('The black lamp in the open floor area is gone: 9 frames'), s[3]


def test_a_move_beside_the_same_furniture_says_it_is_still_there():
    ch = [dict(type='moved', label='chair', from_obj=obj(1.7, 1.6, 0.5, 0.5, 0.9), to_obj=obj(1.7, 2.3, 0.5, 0.5, 0.9),
               displacement_m=0.7, yaw_deg=40.0, yaw_confident=True, turned=True)]
    s = describe(ch, LMS, BOUNDS)
    assert s[0] == 'The black chair moved 0.7 m and turned about 40°, and is still next to the bed.', s[0]


def _lm(label, cx, cy, w, d, support=50):
    return dict(label=label, xy=[cx, cy], footprint=dict(center=[cx, cy], size=[w, d], angle=0.0), z=[0, 0.9],
                support=support)


def test_kitchen_and_bathroom_fixtures_need_company():
    from deltanav.places import _in_context
    bedroom = [_lm('bed', 2.5, 2.2, 1.4, 2.0), _lm('fridge', 3.7, 0.5, 0.6, 0.6)]      # a wardrobe misread as a fridge
    assert [l['label'] for l in _in_context(bedroom)] == ['bed']
    kitchen = [_lm('fridge', 0.4, 0.4, 0.6, 0.6), _lm('sink', 2.0, 0.3, 0.6, 0.45), _lm('counter', 2.0, 0.3, 3.0, 0.6)]
    assert {l['label'] for l in _in_context(kitchen)} == {'fridge', 'sink', 'counter'}


def test_a_sink_in_a_counter_is_kept_but_duplicates_merge():
    from deltanav.places import _dedupe
    out = _dedupe([_lm('counter', 2.0, 0.3, 3.0, 0.6, 400), _lm('sink', 2.0, 0.3, 0.6, 0.45, 120)])
    assert {l['label'] for l in out} == {'counter', 'sink'}
    out = _dedupe([_lm('cabinet', 3.7, 0.6, 0.5, 1.0, 300), _lm('wardrobe', 3.7, 0.6, 0.55, 1.0, 200)])
    assert [l['label'] for l in out] == ['wardrobe']


def test_viewer_page_carries_three_js_inline():
    from deltanav.viewer import inline_scripts, CDN
    tpl = open(os.path.join(os.path.dirname(__file__), '..', 'deltanav', 'viewer_template.html')).read()
    html = inline_scripts(tpl)
    for url in CDN:
        assert url in tpl and f'<script src="{url}">' not in html
    assert 'THREE.OrbitControls' in html and html.count('<script') == html.count('</script>')
