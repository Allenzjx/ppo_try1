# Read-only RR TOP/contact evidence: C28032

## Conclusion

**The measured RR event at tick 5999 is supported by near-top, upward load-bearing contact.** It is not evidence of a vertical-wall-only contact being mislabeled merely because `top_gap_min_m = -0.015`. The cached bottom is only **1.849 mm below top**, and the native obstacle-pair normal force is almost entirely **+14.205 N vertical**, at a contact point just inside the front edge and near the top plane. Subsequent backward motion does not invalidate the previously measured active lift or crossing; it shows that the achieved location/contact was not maintained.

There are nevertheless two real observation limitations: the wheel-bottom cache does not rotate its stored world extents with later wheel orientation, and evaluator load/support use force magnitude rather than the vertical component. Neither limitation establishes that this particular placement was false.

No Python, Isaac, production modification, threshold change or new gate was used. Only the report was written while the P10 training continued.

## Exact 120 Hz evidence

Completed C28032 evaluation source:

`runs/ppo_semantic_v3/validation/20260906T0735108936738Z_g64abc5357d00_b94fa750838144058479e4640e9a831d/physical_observations.jsonl`

The corresponding `stage_transition_evidence.jsonl` confirms RR qualified at **5820**, crossed at **5998**, and placed at **5999**. These are separate history events, not an initial-clearance label being counted as success.

All positions are world coordinates. The obstacle front is **0.521312173774 m**, top **0.050000 m**.

| Sample | Center ahead of front, mm | Cached bottom minus top, mm | Obstacle normal-force vector, N | Mean contact point relative to front/top, mm | Ground |
| --- | ---: | ---: | --- | --- | --- |
| C tick 5820, qualified AIR | −21.374 | +0.442 | (0, 0, 0), inactive | Finite point exists but is **not active-contact evidence** | false |
| C tick 5996, before center crossing | −0.946 | −1.947 | (−0.000003, −0.000004, +14.351) | (+2.837, +0.887) | false |
| C tick 5998, crossing | +0.137 | −1.871 | (−0.181676, −0.017623, +14.884320) | (+3.717, +0.918) | false |
| C tick 5999, placement | +0.564 | −1.849 | (−0.000001253, −0.000001579, +14.204618) | (+3.570, +0.869) | false |
| C tick 6020, beginning retreat | +1.337 | −1.753 | (−0.134859, −0.091965, +14.202424) | (+4.297, +0.838) | false |
| C tick 6040, center back outside | −3.204 | −1.871 | (−0.962664, −0.150829, +13.290790) | (+3.306, +0.937) | false |

At **5999 / 49.991667 s**, full RR geometry is:

```text
center = (0.521875917912, -0.319462507963, 0.098140925169) m
bottom = (0.521875917912, -0.319462507963, 0.048150722350) m
obstacle contact point = (0.524882137775, -0.333171606064, 0.050869360566) m
normal-force magnitude = 14.204618453980 N
wheel-body linear velocity = (0.056212585, 0.050320704, -0.003039671) m/s
```

Obstacle contact remains active in every inspected tick 5996–6004, and its vertical force dominates throughout. Before the center crossed, the wheel was already contacting the top near the edge. That is physically consistent with a finite-size wheel; it does not require the immediately preceding tick to be AIR.

## Interior TOP and ground/wall controls

The current P10 teacher-prefix receipt provides an independent RR interior-TOP observation at **tick 7584 / 63.2 s**:

`runs/ppo_semantic_v3/train/20260906T0748083849450Z_g64abc5357d00_525e380c18ae4039ad93d764e7a06f50/prefix_evidence.jsonl`, `kind=policy_credit_start`, `start.credit_start_observation`.

- RR center is **91.978 mm** beyond the front; cached clearance is **−1.638 mm**, similar to C5999's small negative proxy gap.
- Exact obstacle-pair force is `(0.000000482, 0.000000173, +13.884838) N`; contact point is `(0.609606862, -0.337471008, 0.051017601) m`, i.e. **88.295 mm inside front / 1.018 mm above top**. Ground pair is inactive; measured RR placement history is true.
- Three retained force-history samples are active, with Fz **13.884838–14.354591 N**. This is a short native contact-history positive control, not a claim that three ticks alone prove a long stable dwell.
- The same receipt's RL is a ground control: cached gap **−47.864 mm**, ground Fz **2.587906 N**, obstacle inactive, ground contact point z **2.043 mm**.

A direct wall/ground negative control exists in the same C evaluation at **tick 5514 / 45.95 s**, before the later successful RR lift:

- RR center-front **−48.288 mm**, cached clearance **−50.221 mm**.
- Obstacle force **(−18.424582, −0.0000011, −0.0000015) N**; point is approximately **0.183 mm outside front / 4.364 mm below top**. Ground is simultaneously active with Fz **2.041747 N**.
- This is side-dominated normal contact, clearly unlike C5999. Its large force norm alone must not be interpreted as vertical support. The current `loaded` geometry test is false here, so this observed negative is **not** an actual erroneous TOP placement.

## What the production quantities mean

1. **Geometry is a translated cached extent, not a continuously rotated bottom.** `sensing/geometry.py:113–171` queries collision bounds on first encounter, saves initial world-axis min/max offsets, then only adds current body translation. `bottom_w_m` takes current center x/y and cached minimum z. The lower-level USD provider (`geometry.py:235–256`) can transform mesh points using a supplied quaternion, but `ColliderGeometryCache` does not call it again once the shape is cached. C samples at ticks 0, 5820 and 5999 all have exactly **49.990202819 mm** center-to-bottom offset despite different wheel quaternions. Consequently, later tilt/camber changes are not incorporated in that extent; this read-only analysis did not recompute collider vertices and cannot quantify the actual bottom error or attribute the 1.8 mm gap solely to camber. It uses no guessed wheel radius.

2. **Force magnitude is not vertical load.** `contact_classifier.py:146–180` uses `||force_w_n||` for hysteretic active detection and labels that scalar `normal_force_n`. `semantic_supervisor.py:294–299,418–425` sums active ground/obstacle magnitudes for per-leg force, support and normalized load fraction. It does not use Fz. `loaded` at lines 373–374 requires active obstacle pair, top-gap/XY geometry, and center past front; placement requires crossing history and two loaded samples (406–408). Neither `loaded` nor `placed` inspects the force direction or contact point.

3. **Body-pair identification is not surface identification.** The source is exact wheel/obstacle-body `ContactSensor.force_matrix_w`, not a separate top-face sensor. IsaacLab's local `contact_sensor_data.py:28–34,97–113` documents normal-force vectors by body pair and the **average** contact-point position. Individual manifold points/normals and obstacle face IDs are not preserved in these fields. `sensor_reader.py:145–160` forwards that vector and optional point; nonfinite points become `None` (506–515). An inactive sample can contain a finite point—as observed at 5820—so point presence alone is never contact proof.

These vector/point records are sufficient to distinguish the clearly horizontal negative control from the vertical near-top C5999 and interior positive control. They cannot universally separate every mixed corner/top/front manifold, recover a camber-correct exact mesh bottom, or certify long-term support from a single placement latch. The observed C5999 event remains credible near-top load-bearing placement following a real qualified crossing; maintaining it is a separate physical control outcome.
