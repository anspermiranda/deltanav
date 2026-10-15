"""Instance segmentation of keyframes (YOLO11-seg for COCO classes, YOLOE prompt-free for an open vocabulary)
and the rules that turn raw detector labels into names that fit the measured geometry."""
from __future__ import annotations
import os
import numpy as np


def run_yolo(frames_dir: str, names: list[str], weights: str = 'yolo11m-seg.pt', imgsz: int = 1024, conf: float = 0.2,
             max_det: int = 150, log=print, partial: str | None = None) -> dict:
    """-> name -> list of dict(label, conf, box (4,), poly (K,2) in keyframe pixels).
    One image at a time (streamed), keeping only polygons: an open-vocabulary model returns many full-size masks
    per image, and holding them for a batch takes gigabytes. With `partial`, progress is saved every 40 images
    and a rerun resumes from there."""
    import gc, pickle
    from ultralytics import YOLO
    m = YOLO(weights_path(weights))
    out = pickle.load(open(partial, 'rb')) if partial and os.path.exists(partial) else {}
    todo = [n for n in names if n not in out]
    for k, n in enumerate(todo):
        for r in m.predict(os.path.join(frames_dir, n), imgsz=imgsz, conf=conf, max_det=max_det, verbose=False, stream=True):
            dets = []
            polys = r.masks.xy if r.masks is not None else [None] * len(r.boxes)
            for b, c, cf, pg in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy(), polys):
                dets.append(dict(label=m.names[int(c)], conf=float(cf), box=b.astype(np.float32),
                                 poly=None if pg is None or len(pg) < 3 else np.asarray(pg, np.float32)))
            out[n] = dets
            del r
        if (k + 1) % 40 == 0 or k + 1 == len(todo):
            gc.collect()
            if partial:
                pickle.dump(out, open(partial + '.tmp', 'wb')); os.replace(partial + '.tmp', partial)
            log(f'    {os.path.basename(weights)}: {len(out)}/{len(names)}')
    return out


def weights_path(weights: str) -> str:
    """A bare model name ('yolo11m-seg.pt') is kept in ~/.cache/deltanav/weights instead of the current folder
    (ultralytics downloads it there on first use)."""
    if os.path.exists(weights) or os.path.dirname(weights):
        return weights
    d = os.path.join(os.path.expanduser('~'), '.cache', 'deltanav', 'weights')
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, weights)


# Open-vocabulary labels include whole-scene and material tags; these never name an object.
STUFF = {'home interior', 'room', 'carpet', 'floor', 'wall', 'ceiling', 'linen', 'home', 'interior', 'office', 'house',
         'apartment', 'flooring', 'wood', 'tile', 'hardwood', 'interior design', 'ward', 'corridor', 'hallway', 'shadow',
         'light', 'stage', 'studio', 'property', 'rug', 'mat', 'blanket', 'quilt', 'bedding', 'bed sheet', 'pillowcase',
         'bedcover', 'doormat', 'playroom', 'restroom', 'curtain', 'shutter', 'window blind', 'blind'}
SYNONYMS = {'canopy bed': 'bed', 'bed frame': 'bed', 'headboard': 'bed', 'infant bed': 'bed', 'day bed': 'bed',
            'luggage': 'suitcase', 'mountaineering bag': 'backpack', 'school bag': 'backpack', 'rucksack': 'backpack',
            'swivel chair': 'office chair', 'computer chair': 'office chair', 'office desk': 'desk', 'computer desk': 'desk',
            'writing desk': 'desk', 'information desk': 'desk', 'bedroom window': 'window', 'window frame': 'window',
            'dining table': 'table', 'sports ball': 'ball', 'football': 'ball', 'soccer ball': 'ball', 'closet': 'wardrobe',
            'cupboard': 'wardrobe', 'armoire': 'wardrobe', 'bedside table': 'nightstand', 'chest of drawers': 'drawers',
            'heater': 'radiator', 'side cabinet': 'cabinet', 'file cabinet': 'cabinet', 'shelve': 'shelf',
            'bookcase': 'bookshelf', 'tv': 'television', 'sofa': 'couch',
            # kitchens, bathrooms, living rooms (COCO and open-vocabulary spellings -> one name)
            'refrigerator': 'fridge', 'gas stove': 'stove', 'cooker': 'stove', 'kitchen stove': 'stove',
            'kitchen sink': 'sink', 'bathroom sink': 'sink', 'washbasin': 'sink', 'toilet bowl': 'toilet', 'tub': 'bathtub',
            'kitchen counter': 'counter', 'countertop': 'counter', 'kitchen cabinet': 'cabinet', 'washer': 'washing machine',
            'dresser': 'drawers', 'tv stand': 'cabinet', 'entertainment center': 'cabinet', 'grand piano': 'piano',
            'upright piano': 'piano', 'staircase': 'stairs', 'bunk bed': 'bed', 'microwave oven': 'microwave'}
# furniture and fixtures that stay put: used as landmarks in the descriptions and as robot goals
CORE_LANDMARKS = {'bed', 'nightstand', 'desk', 'table', 'wardrobe', 'radiator', 'window', 'door', 'shelf', 'bookshelf',
                  'drawers', 'couch', 'television', 'mirror', 'cabinet'}
KITCHEN = {'fridge', 'oven', 'stove', 'sink', 'counter', 'kitchen island', 'washing machine'}
BATHROOM = {'toilet', 'bathtub', 'shower', 'sink', 'washing machine'}
LANDMARKS = CORE_LANDMARKS | KITCHEN | BATHROOM | {'fireplace', 'piano', 'stairs'}
# fixtures built into a counter or a cabinet run: their footprints overlap it without being the same thing
NESTED = {'sink', 'stove', 'oven', 'washing machine', 'fridge'}
# names that refer to the same kind of thing: votes are pooled per family and moves matched per family
# ('office chair' in one video and 'chair' in the other are the same object; a suitcase is never a backpack)
FAMILIES = {'chair': {'chair', 'office chair', 'armchair', 'stool', 'gaming chair', 'bar stool', 'folding chair',
                      'rocking chair', 'bean bag chair'},
            'bag': {'backpack', 'handbag', 'bag', 'gift bag', 'saddlebag', 'messenger bag', 'shoulder bag', 'paper bag',
                    'clutch bag', 'duffel bag', 'tote bag', 'diaper bag', 'shopping bag', 'briefcase', 'gym bag',
                    'sports bag', 'laundry bag', 'plastic bag', 'travel bag', 'laptop bag'},
            'suitcase': {'suitcase', 'trolley case'},
            'box': {'box', 'storage box', 'cardboard box', 'carton', 'hatbox', 'crate'},
            'plant': {'potted plant', 'houseplant', 'plant', 'flower pot'},
            'lamp': {'lamp', 'table lamp', 'floor lamp', 'desk lamp', 'lampshade'},
            'bin': {'bin', 'waste container', 'recycling bin', 'trash can', 'wastebasket', 'garbage can', 'dustbin'},
            'basket': {'basket', 'laundry basket', 'hamper'},
            'toy': {'toy', 'stuffed animal', 'stuffed toy', 'teddy bear', 'doll', 'toy car'},
            'guitar': {'guitar', 'acoustic guitar', 'electric guitar', 'bass guitar'},
            'fan': {'fan', 'electric fan'},
            'speaker': {'speaker', 'loudspeaker', 'subwoofer'},
            'pillow': {'pillow', 'throw pillow', 'cushion'},
            'monitor': {'monitor', 'computer monitor'},
            'vacuum': {'vacuum', 'vacuum cleaner'}}
_FAM = {lab: fam for fam, labs in FAMILIES.items() for lab in labs}
# rough physical size ranges (min height, max height, max footprint side) in metres
SIZE_PRIORS = {'chair': (0.35, 1.4, 1.0), 'office chair': (0.35, 1.4, 1.0), 'suitcase': (0.3, 1.0, 0.9),
               'backpack': (0.2, 0.75, 0.7), 'handbag': (0.1, 0.6, 0.6), 'bag': (0.1, 0.8, 0.9), 'box': (0.08, 1.0, 1.0),
               'ball': (0.1, 0.4, 0.4), 'plant': (0.15, 2.0, 1.0), 'potted plant': (0.15, 2.0, 1.0),
               'bottle': (0.05, 0.45, 0.2), 'cup': (0.03, 0.25, 0.2), 'laptop': (0.01, 0.45, 0.5), 'book': (0.01, 0.4, 0.5),
               'lamp': (0.15, 2.0, 0.8), 'bin': (0.15, 1.1, 0.8), 'basket': (0.1, 1.0, 0.9), 'fan': (0.2, 1.6, 0.8)}


def is_stuff(label: str) -> bool:
    return label in STUFF or label.endswith(' room')


def family(label: str) -> str:
    return _FAM.get(label, label)


def normalise(label: str) -> str:
    return SYNONYMS.get(label, label)


def plausible(label: str, height: float, longest: float) -> bool:
    pr = SIZE_PRIORS.get(label) or SIZE_PRIORS.get(family(label))
    if pr is None:
        return True
    lo, hi, side = pr
    return lo * 0.7 <= height <= hi * 1.3 and longest <= side * 1.3


def landmark_check(label: str, height: float, longest: float, shortest: float):
    """Furniture names checked against the measured footprint: a 'wardrobe' 15 cm deep and door-sized is the
    door (open-vocabulary models confuse the two); implausible names are dropped (None)."""
    door_like = shortest < 0.25 and 0.55 <= longest <= 1.15 and height >= 1.5
    if label in ('wardrobe', 'cabinet', 'shelf', 'bookshelf') and door_like:
        return 'door'
    if label == 'bed' and not (1.4 <= longest <= 2.6):
        return None
    if label == 'wardrobe' and (height < 1.2 or shortest < 0.25):
        return 'cabinet' if height >= 0.4 else None
    if label in ('desk', 'table') and not (0.45 <= height <= 1.2):
        return None
    if label == 'nightstand' and (height > 0.9 or longest > 0.9):
        return None
    if label == 'door' and not door_like and height < 1.5:
        return None
    if label in ('counter', 'kitchen island') and not (0.6 <= height <= 1.3 and longest >= 0.6):
        return None
    if label == 'fridge' and (height < 0.45 or longest > 1.3):
        return None
    if label in ('washing machine', 'toilet') and (height > 1.3 or longest > 1.0):
        return None
    if label == 'bathtub' and not (1.1 <= longest <= 2.1):
        return None
    if label in ('oven', 'stove', 'sink') and longest > 1.4:
        return None
    return label


def merge_detections(*sources: dict) -> dict:
    """Combine detections from several models, normalising names and dropping scene/material tags."""
    out = {}
    for src in sources:
        for name, dets in src.items():
            for d in dets:
                lab = normalise(d['label'])
                if is_stuff(lab):
                    continue
                e = dict(d); e['label'] = lab
                out.setdefault(name, []).append(e)
    return out
