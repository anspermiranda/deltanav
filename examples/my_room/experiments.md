| Test | Frames (before / after) | This pipeline | Naive point-cloud diff |
|---|---|---|---|
| control_before_chest_vs_waist | 101 / 79 | nothing | 3 objects |
| control_after_chest_vs_waist | 100 / 80 | nothing | 2 objects |
| cross_before_chest_vs_after_waist | 101 / 80 | moved office chair, removed saddlebag | 5 objects |
| cross_before_waist_vs_after_chest | 79 / 100 | moved office chair, removed suitcase, added backpack | 9 objects |
| full | 180 / 180 | moved office chair, removed suitcase, added backpack | 12 objects |
| never_filmed | 180 / 115 | moved office chair (place: 100.0% could not be checked) | 5 objects |
| only_occluded | 180 / 131 | moved office chair (place: 100.0% could not be checked) | 10 objects |

Verification loop: start from 25 after frames that never saw the place; 155 candidate frames (65 of them see it). Planner order: removal confirmed after 8 frame(s), 84% of the surface checked after 10. Random order: confirmed after a median of 13.0 (never within 30 frames in 3 of 10 orders), 61% checked after 10.
