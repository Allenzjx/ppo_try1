# Task-conditioned hip/wheel continuation (adopted at ee5a965)

Source is the verified actual latest CP185856, update1417, optimizer steps28340.
Accepted 73.808333s N_ref and same-control prior B remain intact. No new learned
decision has occurred before this version's actual training logs say otherwise.

## Distribution and learning state

Keep the 372-input actor/critic parameters, complete Adam moments and parameter
groups, effective LR1e-5, Identity normalizers, RNG, mean, rho=.9 and action caps.
The joint migration changes the observable state-dependent Gaussian sigma and
soft reward/potential semantics; it does not claim identical MDP semantics.
The mean kernel is identical only for identical numerical observations and
weights. Current rolling retention changes the existing global-Phi observation
feature's semantics; a fixed physical state is not necessarily the same encoded
input after migration. This is explicitly bound in the migration record, not
misrepresented as a reward-only change or proof of physical trajectory identity.
Fresh rollout only. No manual diagnostic action enters PPO storage. No auxiliary
supervision, directional bias, sampling rejection, new action representation or
hidden zero fallback is selected.

Canonical order: FL hip/knee, FR hip/knee, RL hip/knee, RR hip/knee,
FL/FR/RL/RR wheels. `sigma_raw = exp(learned_log_sigma)*.25*B/current_cap`.
B's first8 units are servo degrees; last4 are wheel rad/s. B is not the actual
actuator sigma. Tanh, saturation, HISTORY, mapper and physical limits still act;
actual same-draw raw sigma and applied response must be inspected in each run.
Full positive B tables and exact observable gates are pinned in the policy
contract, used by sampling, old/current likelihood, loading and stochastic eval.

Actual adopted B table (pairs are hip/knee degrees; wheels FL/FR/RL/RR rad/s):

| Observable task | FL | FR | RL | RR | Four wheels |
| --- | --- | --- | --- | --- | --- |
| FR usable AIR approach | 9/12 | 12/18 | 12/12 | 6/9 | .6/.6/.6/.6 |
| FL crossed, AIR, capture pending | 24/24 | 9/12 | 6/9 | 6/9 | 1/.6/1/.6 |
| P06 current front-support rolling proxy | 12/12 | 12/12 | 12/12 | 12/12 | 1.2/1.2/1/.6 |
| P06 current support recovery proxy | 24/24 | 12/18 | 12/12 | 12/18 | 1.2/1.2/1/.6 |
| RR preparation | 24/24 | 12/18 | 24/24 | 12/18 | 1.2/1.2/1/.6 |
| RR currently qualified lift | 24/24 | 12/18 | 24/24 | 24/36 | 1.2/1.2/1/.6 |

These are symmetric innovation opportunities, not commanded offsets, actual
standard deviations, masks or hard acceptance. FR/FL gap gates blend over
0–15mm/0–3mm respectively; P06 exit blends at nearest-rear -.27 to -.22m.
Rules apply in listed order, so current RR qualification overrides preparation
rather than adding its scale. Default B equals current cap except retained
P06+ FR-knee B24. The observed front-contact proxy used for exploration is not
an exact TOP classification and never substitutes for the physical evaluator.
Source: semantic_policy_distribution.TASK_CONDITIONED_PHYSICAL_B_TABLE and
semantic_history_actor.task_conditioned_physical_scales at adopted ee5a965.

- FR usable AIR approach: retain RL hip opportunity, reduce large RR support
  disturbances and unrelated channels. Symmetric exploration, no fixed RL angle.
- FL crossed/pending AIR: FL hip B18->24 increases its innovation opportunity;
  FL knee B24 remains unchanged, while unrelated support-leg noise decreases.
  Current support, not placed history, selects recovery versus rolling later.
- P06 rolling: all8 linkage B12, full wheel opportunity. Recovery restores FL
  hip/knee B24 and modest support corrections. The nearer rear leg controls the
  continuous -.27 to -.22m exit; RR current qualification overrides preparation.
- RR preparation/carry: FL/RL reconfiguration and RR carry opportunities remain
  nonzero. Current qualification is not an old AIR-history proxy or fixed timer.

## Reward and limits

Retain old effective front quality coefficient .015–.03/s in P01/P02. Add at
most .03/s for a bounded 20mm collider/obstacle AABB separation deficit in
P01/P02/P05/P06/P07/P08/P09. This is a conservative geometric lower bound, not
base height or exact mesh clearance, and gives no extra benefit above enough
space. Per-physics-tick dt/coefficient/cost/geometry audit is emitted.

Reuse existing front placement retention share for current near-top proximity
and verified TOP bearing during rolling; no duplicate touchdown bonus. Fade out
  before rear preparation via real nearest-rear edge distance, with no permanent
two-front-contact constraint. Existing FL coarse/fine gap shaping remains; no
new fixed hip target or larger contact-force reward. No linkage-motion penalty
yet: existing evidence shows real forward progress, not a proven linkage-driven
stall. All hard collision/wheel-only/FALL/NaN/limit and nominal logic unchanged.

## Initial real-data checkpoint, not a stopping budget

First intended mixed2048 learner decisions, adjusted by actual coverage:

1. 512 naturalP01 full-task decisions for FR quality and own-policy entrances.
2. 512 from realP04 predecessor through FL capture/hold/P06 (legal N prefix).
3. 512 from realP06 through rolling and RR preparation (legal N prefix).
4. 128 from realP08 before current RR lift through carry/placement (legal N prefix).
5. 384 naturalP01 after the rear updates, to retain actual front quality exposure
   immediately before the first fixed-checkpoint evaluation.

This allocation was adjusted after two observed P09 collision episodes from
P04/P06 courses: both already entered preparation with negative RL hip and
positive RR hip residual history. A clean physical P08 entry tests that confound,
while the final P01 block re-exposes the front task after rear updates. The
initial total remains2048 actual learner decisions; the adjustment changes no
reward, source, sigma table, phase acceptance or active rollout.

At the first actual P08 takeover, zero residual HISTORY still grew RR hip/knee
REQUEST to +10.385/-8.514deg within15decisions; episode ended with body collision
after61 learner decisions and no qualified RR lift. A verified stop-after-update
request now seals that originally256 block at its first complete128 update,
without truncating collection/GAE/update or dropping failures. The extra128
decisions move to the final naturalP01 block (384 total), keeping the2048 total
while bringing fixed-checkpoint videos forward. This is not a success gate.

Prefix actions, diagnostic steps and video frames have zero learner credit.
The inherited policy's own failure entrances remain in data; no deletion of
failures or phase-done resets. Each128-decision official update has a saved
round-trip checkpoint. Inspect actual phase counts/reward/GAE/normalized
advantage/likelihood and selected hip gradients, not curriculum labels.

Then reload one saved CP for deterministic naturalP01, sameCP stochastic seed,
and compatible N video. Preserve incomplete/failure tails. Continue updates
according to the first actual unfinished physical task, not a synthetic-test
pass or a lucky diagnostic capture. If pure PPO does not absorb a physically
verified useful direction, assess the separately authorized finite auxiliary
option with its own budget and fresh PPO collection; do not activate it silently.

## Next block after the first fixed-checkpoint videos (not yet collected)

CP187904 deterministic natural P01 evaluation reached FR placement but ended
at P05 after 50.825s, FL AIR/gap8.078mm, LOCAL_BOUNDED_RECOVERY_EXHAUSTED.
The first mixed2048 had only896 naturalP01 decisions split into short512/384
blocks, both stopping before this late incomplete-task event. Prefix courses
did include rear BODY_COLLISION, but that is not evidence that a full natural
P01 timeout was optimized. Do not pretend the evaluation itself adds credit.

After sameCP stochastic and current-version B videos, continue from the actual
latest compatible saved CP with2048 naturalP01 full_episode decisions, seed1001,
N1/cuda:0, rollout128, checkpoint everyupdate. This longer collection keeps
ordinary phase continuity, permits complete front/rear task endpoints within
the block, and preserves failures. It is data allocation, not a reward/rho/LR/
mapper change; all12 capacity/sigma and existing nonzero quality stay active.
There is no new success gate or auxiliary loss. Assess actual phase coverage
and first unfinished tasks after the completed updates. Longer collection is
not a promise of convergence or a substitute for additional rear coverage.

Mapper gain8 with4tick feedback has a measured P05 two-point limit cycle;
gain2 is only an isolated, untested suggestion. It is NOT adopted or applied
to the live videos/next rollout, and preserved N_ref is not overwritten.

## After CP189952: finite direction check and missing carry credit

The preceding natural-P01 block actually completed2048 decisions/16updates;
the branch now has4096/32. Three completed training episodes reached P09,
temporarily qualified RR, lost current lift and ended in BODY_COLLISION.
P10-P13 still have zero learner samples. This is not evidence that the latest
deterministic policy completes FL placement or the whole task.

First finish and deliver the unchanged CP189952 natural-P01 deterministic and
stochastic recordings. One bounded FL_minus3 diagnostic may then reuse the
reviewed pre-action logger on that checkpoint: physical verification only,
no auxiliary update, no PPO credit, and not an optimizer success prerequisite.
Do not simultaneously change knee, reward, HISTORY or mapper gain.

Then collect a bounded512 learner-decision carry supplement using the existing
successful_nominal prefix, phase_suffix/P09/teacher_offset_decisions35, followed
by renewed natural-P01 collection. Keep the current distribution/reward/Adam.
The offset is a curriculum selector, not an RR timing guard or injected pose.
Current B first observes P09 at decision647/tick5176;35 further real N decisions
end at682/tick5456. There RR is currently qualified/AIR, not crossed or placed;
FR/RL support and FL AIR/zero bearing are retained. A new rollout must report
its own actual entry, not assume it reproduces this reference exactly.

All real N prefix decisions are excluded from learner counts/storage/GAE;
the core observation, mapper, HISTORY, clock and physical events pass through
unchanged at takeover. Phase/offset misses retain the existing explicit single
fresh-P01 fallback and must be reported as such. This supplement does not train
the preceding N preparation and cannot replace the already collected natural
and P04/P06/P08 predecessor courses. A suffix completion is never full-P01 PPO
success. Actual coverage and endpoints, not the requested course name, decide
whether this supplied missing later-task learning opportunities.
