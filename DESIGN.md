# DeltaNav: design note

## The problem, restated

Two handheld videos of one room, filmed at different times along different paths. Some objects were moved,
added or removed. The difficulty is not finding *differences*, which are everywhere. It is that most
differences are not changes:

- **Viewpoint.** The same surface looks different, or is reconstructed slightly differently, from another
  angle or height.
- **Occlusion.** Something is hidden in one video, for example behind the chair that was moved in front of
  it.
- **Incomplete observation.** A corner simply was not filmed the second time.

A geometric diff of two reconstructions treats all three as change. The design follows from one decision:
**a change has to be observed, not inferred from missing data.** Every surface point gets one of three
answers (gone / same / could not be checked), and only *gone* with strong, multi-frame evidence becomes a
change. For a robot this is the safe failure mode: an *unknown* region is something to look at again, a
false change is a wrong map.

## Pipeline and the main choices

**1. One reconstruction for both videos, not two aligned ones.** Aligning two separate reconstructions (ICP,
or matching their features afterwards) is pulled off course by exactly the things that changed, and it
doubles the scale ambiguity. Instead, sharp keyframes from both videos go into a single structure-from-motion
problem (pycolmap: SIFT, sequential matching with loop detection inside each video, vocabulary-tree
retrieval across the videos, global mapper, one shared camera). The unchanged room anchors both; changed
objects simply produce no consistent matches. On my room all 360 keyframes registered in one model.
After the depth stage, the walls are aligned with x and y again from the dense points: SfM points gather on
textured clutter and plain painted walls give almost none, so the first estimate was 14° off. The lens
prior comes from the video's own metadata (the iPhone writes the 35 mm-equivalent focal length), and the
reconstruction refines it.

**2. Up, floor and walls.** The phone is held roughly upright, so the cameras' x-axes are nearly
horizontal. Gravity is the direction most orthogonal to all of them (the smallest eigenvector), refined
with the floor. This beat averaging camera "up" vectors, which is biased by how much you tilt the phone
down. Yaw snaps to the dominant wall directions, and the origin goes on the floor in a corner, so every
number in the output reads as "metres from the corner, along the walls". Heights are measured from a smooth
floor surface fitted to the dense points, because monocular depth bends large planes by a few centimetres.

**3. Dense metric depth from a network, aligned to the reconstruction.** I considered MVS (COLMAP dense),
which needs CUDA, is slow, and leaves holes on white walls and carpet. I also considered NeRF / Gaussian
splatting, which takes hours per video and still needs a visibility test afterwards. Monocular metric
depth (Depth Anything V2, indoor) is complete and fast, but needed two fixes:

- **Lens.** Metric depth networks learn the focal length of their training camera. The 0.5× iPhone lens
  sees about 95° horizontally, the network was trained at about 60°, so raw depths came out about 1.9× too
  large (a person's chest height read as 2.5 m). Scaling by the ratio of focal lengths at the network input
  fixed it.
- **Low-frequency bias.** Per frame, a smooth multiplicative correction field (7 terms in log-depth, robust
  IRLS) is fitted to the reconstruction's own 3D points (median error 4.1% → 1.6%).
  A pixel survives only if two neighbouring frames see a surface at the same depth (within 5%).

**4. The see-through test** (`change.py`). Each surface point of video A is projected into every frame of
video B and compared with B's depth there:

- depth beyond the point by more than max(10 cm, 10%): the camera saw *through* it, a **free** vote;
- within max(6 cm, 6%): **same**;
- in front: **occluded**, no vote;
- out of frame: no vote.

The thresholds are deliberately asymmetric, because "it's gone" is the claim that needs the stronger
evidence. A point is *gone* only with at least 3 free votes, at least 70% of the decisive votes, and at
least 25% of all frames that pointed there. The last condition keeps a place that is seen through once and
hidden thirty times as *could not be checked*.

Two gates sit around the test:

- **Self-consistency.** A point is tested only if its own video saw it consistently (3 or more frames agree,
  at most 25% see through it). This stops depth glitches in one video from turning into "changes".
- **The object-level frame check.** A candidate object is kept only if at least 5 frames of the other video
  saw through its place *as a whole*, and fewer than 10% of the frames that saw the place clearly saw a
  surface there. A mesh chair back really is see-through from close up, and a wardrobe edge seen at a
  grazing angle fools individual points. Whole frames rarely agree with them.

Running the test in both directions gives both kinds of change. Before-points seen through by the after
video are things that **left**. After-points whose place the before video saw through are things that
**arrived**. Every unknown point also records *why* it is unknown (never in view, hidden behind something,
too few views, or inconsistent in its own video), and the report shows that breakdown.

**5. From points to objects.** DBSCAN on the gone points, with outlier trimming, a robust footprint
rectangle, and a minimum size and height. Parts stacked on top of each other (a chair's seat and its base)
are joined. Names come from voting over instance masks in the frames that actually see the object, after a
depth test so that a mask belonging to something in front doesn't vote. Two models vote: YOLO11-seg (COCO,
reliable for its 80 classes) and YOLOE prompt-free (open vocabulary, so "office chair", "wardrobe",
"nightstand"). Votes are pooled by family ("office chair" and "chair" are the same thing; a suitcase is never
a backpack), and a name has to agree with the measured size. The same check fixes furniture: YOLOE called
the 15 cm-deep, 0.85 m-wide, 1.8 m-tall slab by the door a "closet"; it is door-shaped, so it is named the
door. Colour comes from grey-world white-balanced pixels in CIELAB, because the room's warm light made the
navy suitcase read as brown.

**6. Moves and rotation.** Removed and added objects are paired with the Hungarian algorithm on name, size
and colour. For rotation I first used ICP (yaw only); on this data its fitness was almost flat over all
angles, because partial, blob-like point clouds of a chair fit about as well at any angle. The rotation now
comes from appearance: SIFT matches between the frames that see the object in each video are lifted to 3D
with the depth maps, then a 2-point RANSAC finds yaw and translation. When appearance is not conclusive, the
geometric estimate is reported with its confidence ("turned, angle uncertain"). Every move is also written as
a 4×4 rigid transform in the room frame.

**7. Uncertainty.** Each position carries a 1-sigma error: the depth error (1.6%) times the
typical viewing distance, combined with the voxel quantisation. A move's distance carries both ends'
errors.

**8. Language.** Sentences are generated from the change records, never written separately, so every number
and place in the text is the geometry's. Places come from furniture landmarks (instance masks lifted to 3D
footprints by multi-detection consensus) and a few spatial relations: *at the desk*, *between the bed and
the wardrobe*, *at the foot of the bed* (the end of the bed away from the wall), *against the wall*. The
landmark vocabulary also covers living rooms, offices, kitchens and bathrooms (fridge, oven, sink, counter,
toilet, bathtub, fireplace, ...). A kitchen or bathroom fixture is kept only if a second one is found: COCO
YOLO called my wardrobe a "refrigerator" 23 times, and a lone fridge in a bedroom is a misreading. Fixtures
built into a counter (a sink, a hob) are kept next to it rather than merged with it. With no furniture in
reach, places are described relative to the room: *in a corner of the room*, *along the wall*, *in the
middle of the room*.

**9. Surfaces.** A TSDF (truncated signed distance) volume at 1.5 cm is fused from the metric depth maps on
the GPU when there is one, then marching cubes, small-piece removal and Taubin smoothing. Three surfaces:
before only, after only, and a **shared** one from both videos with the changed objects cut out (the part of
the room both videos agree on), plus a mesh of each changed object in each of its versions.

## The robot's side

**Map edits, not re-mapping.** The change set is applied to yesterday's occupancy map as local patches:
inside each change's footprint the map takes what the after video saw; everywhere else it stays as it was.
Re-mapping from the after video alone would rewrite every cell the second walk-through saw differently
(depth noise, coverage); the report gives both areas so the difference is visible.

**Height-aware occupancy, per robot.** A cell is occupied for a robot if at least three self-consistent surface
points lie between 12 cm and the robot's own height, measured from the local floor (the median of the lowest
surfaces nearby, because monocular depth bends the floor by a few centimetres); free if the floor was seen there
(or the camera operator walked there) and nothing is in that band. So the desk top blocks a 1.8 m humanoid but
not a 0.5 m robot, which can drive under it. Below 12 cm, monocular depth cannot tell a low object from a bump in
the floor (at 5-8 cm the carpet itself showed up as obstacles), so lower things are left out of the maps and the
limitation is stated. Isolated one- to three-cell obstacles are treated as noise rather than inflated into a
wall. Unknown cells are not traversable.

**Routes with causes.** A* (8-connected, with a cost that rises near obstacles, like a Nav2 inflation layer)
from the door to a goal pose in front of each piece of furniture, on yesterday's map and on the updated map. The
start and the goal poses are kept from yesterday's plan wherever they are still free, as a robot with stored
goals would behave; a goal moves only when something now stands on it. Each route that changed is traced back
to its cause by re-planning with that one edit left out (counterfactual attribution). Yesterday's route is also
checked against what arrived: "it would now run straight into the office chair". Every change also gets an
inspection pose ("go and look at the new backpack").

**Exports.** ROS `map_server` maps (PGM + YAML, room frame = map frame) for before, updated and after, per
robot, and named goal poses (x, y, yaw + quaternion) for a Nav2 waypoint follower.

**3D occupancy.** The TSDF volumes give a 3D grid at 4.5 cm: occupied (a surface), free (seen through by at
least two frames), unknown. The same edit rule produces the updated grid. A humanoid needs this: its arms and
head move through the space above the floor that a 2D map flattens away.

**Where to look next.** The surfaces that could not be checked are where a robot should look. Candidate
positions on reachable floor get a 360° depth panorama of the room as it is now (the after video's
surfaces), so one panorama serves every viewing direction; each (position, yaw, pitch) is scored by the
unknown surface it would see with nothing nearer in the way, floor-level surfaces weighted most (that is where
a change matters to a robot), and the best views are chosen greedily so that each covers what the previous
ones did not. In the verification-loop experiment the same scoring runs in a closed loop (pick a frame, re-check,
re-plan) and confirms the hidden removal in 8 frames against a median of 13 in random order (README).

## What the evaluation covers, and what it doesn't

I didn't measure ground truth with a tape, so the tests are built so that the correct answer follows from
how the videos were recorded. The two laps of one video (chest and waist height) give no-change pairs
with large viewpoint differences. Dropping frames simulates "never filmed" and "only seen hidden" exactly
where a real removal happened. The reported distances and rotation are measurements, not checked against a
tape; the metric checks are indirect (camera heights, the bed against a standard mattress size, two
independent rotation estimates).

## Trade-offs

- **Misses over false alarms.** For a robot's map, a false change makes it avoid a free corridor or forget
  a real obstacle; an *unknown* region is safe. So every threshold leans towards *unknown*.
- **Object size floor.** About 150 surface points at 2.5 cm, at least 12 cm tall, and seen through by at
  least 5 frames. Small things (a phone, a cup) are below it. A finer voxel grid would lower it at the cost
  of more depth noise per voxel.
- **Speed vs completeness.** 180 keyframes per video keep the reconstruction and the networks to minutes on
  a laptop GPU. Denser frames help mainly at the corners of the room.
- **Monocular depth.** Fast and complete, but it smooths thin parts (chair legs, mesh), which shows up as
  "could not be checked", not as false change.

## Related work, and what is different here

| Work | Input | How absence is decided |
|---|---|---|
| [Fehr et al., ICRA 2017](https://tisl.utm.utoronto.ca/publication/201705-icra-tsdf_change_detection/icra17-tsdf_change_detection.pdf), TSDF-based change detection | RGB-D, posed | differences between TSDF reconstructions of each session |
| [Panoptic Multi-TSDFs](https://arxiv.org/pdf/2109.10165) (Schmid et al., ICRA 2022, [code](https://github.com/ethz-asl/panoptic_mapping)) | RGB-D + panoptic segmentation | per-submap TSDF consistency: present / absent / persistent |
| [Khronos](https://arxiv.org/pdf/2402.13817) (Schmid et al., RSS 2024, [code](https://github.com/MIT-SPARK/Khronos)) | RGB-D SLAM | free-space evidence against object reconstructions, spatio-temporal |
| [3DGS-CD](https://arxiv.org/pdf/2411.03706) (2024) | multi-view images of the old scene + a few new images | 3D Gaussian splatting render vs new images, with segmentation masks |
| [Gaussian Difference](https://arxiv.org/html/2502.16941v1) (2025) | two image sets | instance-aware 3D Gaussian fields of both scenes |
| [RIO / 3RScan](https://arxiv.org/pdf/1908.06109) (Wald et al., ICCV 2019, [data](https://github.com/WaldJohannaU/3RScan)) | RGB-D rescans of changing rooms | benchmark for re-localising moved objects |

The closest idea is the free-space reasoning in Fehr et al. and Khronos: a thing is gone because space where
it was is now observed empty, not because a match is missing. Those systems run on RGB-D streams with
known poses. This repository applies the same principle to two ordinary handheld phone videos with no depth
sensor and no poses (joint SfM + corrected monocular metric depth), makes the *could not be checked* state
explicit and reported, and closes the loop for a robot: verified changes become map edits with attributed
route changes, and the unknowns become next views.

## Next

- **Online, on the robot.** Keep the visibility evidence per voxel over time (free / occupied log-odds, as
  in OctoMap, plus *last seen* and *by whom*) and update it from the robot's own depth camera. Changes then
  become map edits with provenance, and stale regions decay towards unknown instead of staying wrong.
- **Better depth.** Use the iPhone's LiDAR depth when available, or a video depth model for temporally
  consistent depth, and model mirrors and glass instead of masking them.
- **Re-identification across rooms.** An object that left this room and appeared in another is a move,
  not a removal plus an addition.
- **Gaussian splatting for the picture, not the decision.** A splat model would make the 3D view
  photo-real; the change decision should stay with the visibility test.
