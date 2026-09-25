# CP229120 deterministic P05 failure: bounded review

**Result:** 57.8667 s / 868 decisions / tick 6944, P05 age 40 s, `INCOMPLETE_CONTROLLER_BLOCKED` from `LOCAL_BOUNDED_RECOVERY_EXHAUSTED`. FR placed; FL never crossed or placed. This is **task incompletion**, not an actual hard-safety termination.

Formal source: `runs/ppo_rr_rl_timing_policy_learning_v1/video_eval/validation/20260924T1008401445075Z_g72e63592bdf4_3f1b5877a524484aab1eecca97694515/source`. CP229120, runtime `72e63592bdf4`, deterministic conditional mean, seed 4001; natural P01, existing FL helper declared, rear helpers disabled.

## What failed first

FL acquired measured lift at **17.9500 s / tick 2154**. It briefly had positive wheel-bottom clearance above the top height, but remained behind the front edge. At **22.5417 s / tick 2705**, that clearance first fell below zero again while still AIR and **138.676 mm behind the edge**. The executed FL hip/knee were **47.686/−58.000°**, actual **47.655/−57.989°**: the leg was following its requested posture, yet the floating-body geometry was losing clearance.

The source four-wheel stop was first recorded at **26.9333 s**, **4.3917 s after** this loss of clearance. At **27.1167 s / tick 3254**, an actual ground contact of **11.756 N** revoked FL lift qualification before crossing. Tick 3256 briefly reads AIR again, but qualification correctly remains revoked. There was no subsequent new qualified FL lift.

| Time s | FL hip FINAL / actual ° | FL knee FINAL / actual ° | Front distance / top gap mm | FL contact / force N | Base-origin world z mm |
| --- | ---: | ---: | ---: | --- | ---: |
| 19.0000 | +47.75 / +47.40 | -57.76 / -57.62 | -169.61 / +23.47 | AIR / 0.000 | 65.329 |
| 21.8667 | +47.66 / +47.64 | -58.00 / -57.99 | -138.44 / +11.47 | AIR / 0.000 | 57.072 |
| 23.8667 | +47.71 / +47.69 | -58.00 / -57.99 | -131.48 / -15.45 | AIR / 0.000 | 44.397 |
| 27.1167 | +47.74 / +47.62 | -43.70 / -52.25 | -83.34 / -50.20 | GROUND / 11.756 | 27.610 |
| 57.8667 | +20.58 / +20.92 | -32.54 / -32.90 | -177.93 / -50.49 | GROUND / 2.478 | 59.647 |

Base-origin height is a measured coordinate, **not body collision clearance**. Between 19.0000 s and first ground contact, it fell 65.329→27.610 mm and CoM z fell 150.745→117.892 mm while FL was still short of the edge. These observations identify a whole-body clearance/approach failure; they do not establish one joint or wheel as its sole cause.

## Why this is not a failed descent-helper budget

All **600 P05 decision endpoints** show FL helper **WAIT, initialized=0**; no DESCEND or BLOCKED helper state. There are **0 within-top-XY and 0 TOP endpoints**, and no crossing/placement event. The helper never acquired a legal capture condition and consumed no initialized hip-descent budget.

The source motion reaches its finite endpoint at the first sampled **27.6667 s / P05 age 9.8 s**. At that point FL is already GROUND, unqualified, 65.529 mm behind the edge and about 50 mm below top height. The existing approach recovery correctly reports no current qualified AIR capture path. Thus pending-capture soft continuation to P06 is unavailable; the **general 40 s P05 task-recovery window**, not the helper's travel budget, ultimately expires.

At the terminal tick, the physical evaluator is **VALID** with **no physical safety termination**; observations are finite and exact base-link/obstacle contact is false. The recorded common-task rejection is therefore genuine incompletion, not a recording acceptance artifact.

## Minimal comparison with the usable original probe-v2 prefix

Only diagnostic rows **0–645** were read, ending at FL placement. Every row has original student raw == applied raw, no overridden channel and no active intervention. Rows 998+ were excluded. This is CP225280 on `b0ff99729b1d`, seed 4001, not CP229120 and not a full-success probe claim.

The P05 clock makes the relevant difference clear:

| P05 age s | FL top gap mm: CP229120 / prefix | FL front distance mm: CP229120 / prefix | FL knee FINAL °: CP229120 / prefix | CoM z mm: CP229120 / prefix |
| --- | ---: | ---: | ---: | ---: |
| 2 | +22.32 / +41.21 | -161.58 / -152.71 | -58.00 / -58.00 | 149.64 / 157.93 |
| 4 | +11.47 / +50.56 | -138.44 / -124.01 | -58.00 / -58.00 | 144.54 / 161.71 |
| 6 | -15.45 / +58.57 | -131.48 / -111.60 | -58.00 / -58.00 | 131.13 / 164.22 |

Both have the same source FL posture **48.2/−36.7°** in this interval, and the same mapped baseline **49.45/−37.95°**; both FINAL knees reach −58°. The retained prefix nevertheless preserves clearance and gets ahead in front distance: the relevant divergence is already present during the **2–6 s P05 window**, not only at later helper entry. Different whole-body/policy targets and resulting poses remain possible causes; similar FL knee targets are not proof of equivalent geometry.

In the prefix, source-endpoint recovery remains eligible at P05 age 9.8 s (FL qualified AIR, front −33.464 mm, gap +28.525 mm), with nominal forward wheel commands retained at +0.3 rad/s. It reaches XY at 35.2667 s, begins DESCEND at the 35.6000 s endpoint, and actually places FL at **43.0417 s / tick 5165**. Those original-student predecessor actions precede the later independent probe.

CP229120 instead reaches that same source endpoint unqualified on ground; N remains stopped and the raw residual still gives final FL wheel about −0.707 rad/s. Re-enabling all wheels or the helper unconditionally would not be an evidence-based diagnosis. Clearance was lost before the stop, and both trajectories include negative FL wheel targets while AIR.

Scope: no source/config/reward/model changes, no Torch/Isaac, no new simulation or test suite, no changes to video/DELIVERY/RECOVERY. The source versions and weights differ: this comparison establishes physical facts and a short candidate investigation window, not a clean single-factor causal attribution.

