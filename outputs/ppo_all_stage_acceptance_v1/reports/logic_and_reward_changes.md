# All-stage logic / reward revision

Scope is the existing ppo_try1 architecture. Frozen A files, assets, scene, physical
actuators, hashes and all historical runs remain unchanged. The new physical
sensing adapter lives only in `ppo/semantic_physical_sensing.py` and wraps the
unchanged atomic reader. A/B/C use the same versioned common task evaluator;
old A initialization is not silently declared paired with B/C.

Confirmed fixes: pose-dependent collider bounds; exact contact/vertical bearing
versus reaction norm; missing base contact evidence; missing-Q no longer implies
powered wall climb; active whole-body early-TOP route; real traversal/body bounds
versus strict settling quality; independent preparation/transfer maturity;
captured old nominal owner retirement; P06 current-distance rolling recovery;
P03 FL/RL wheel cancellation ranges and minimum nonshrinking continuation;
advanced handoff acceptance and symmetric declared measurement tolerance.

All 12 channels remain active in all phases. Mapper/filter/residual/history are
not reset on a phase transition. Newer legitimate receiver/recovery owners remain
allowed after a previous swing was captured. A current ground regression cannot
borrow a historical placement to claim present support.

All local limits remain finite task terminals, distinct from the 200-second
global deadline and from external evaluator windows. Current physical progress
provides a bounded recovery allowance: min(10s, half the original stage limit)
times progress squared. This is a proximity/recovery allowance, not a claim of
positive velocity; it never extends the hard bound through jitter. Its operands
are already observed in the372-vector. Original six-second stall is diagnostic.

Reward configuration bytes and five reward families are retained: one task
family, three active dense quality families and one disabled weak regularizer. The single
physical PBRS potential uses current body-AABB rectangle proximity, actual four-wheel /
linear / angular speed ratios and current verified TOP support for its existing
finish share (.65/.25/.10 geometry/control/fixed-post allocation). The .25m
distance scale is reused. Commanded wheel near-zero/home similarity no longer
gates this share. A pre-event body-forward flat region identified during review
was corrected before the first new PPO sample; it was not proven to cause a live failure.
There is no added phase-transition reward or duplicate event bonus. Same gamma
.9985, lambda .99, 128 rollout; actual applied-action smoothness and measured
role-transfer attitude allowance remain. Task terminals still use Phi_next=0;
ordinary phase switches are not done and keep cross-stage GAE credit.

Migration is explicit: preserve all372 actor/critic parameters, learned std,
identity normalizers, verified RNG, lifetime counters and spent budgets. Verify
source Adam state, then deliberately reset moments at lr=3e-5. New empty rollout;
no physical snapshot inheritance. The continued_response role feature now means
actual transfer maturity, not earlier workspace response age. P13 phase_progress
now reversibly encodes no event (0), event/no timer (.25), healthy fixed-post
(.50+.25*t), latched region loss (.80+.19*t), or true success (1). t covers the
full fixed1s window including its last0.1s. This is separate from physical Phi;
schema and encoder bytes remain unchanged. Started post observation retains the
CAPTURE label. Its fixed end is protected from local interruption, not renewed;
the global200s budget remains unchanged. This explicitly versions local deadline
progress semantics, and does not claim the entire environment is fully Markov.
Same-input tensor
identity is not a claim of unchanged physical trajectories.

Evidence classes remain separate: code/synthetic tests; current-version real
diagnostics; credited policy trajectories; teacher prefixes; full natural-P01
reloaded evaluation. B2 at2f942 recorded900dec/60s, P01–P09, FR/FL Q/C/P,
RR initial lift only, ending by external diagnostic window with ambiguous RR
contact. This is not PPO training or full task success. Final pretrain runtime
a8b1484 changes P13 only relative to B2; B2 is not a fully paired final evaluation.
True C sampling and checkpoint accounting are in training_manifest.json.
