# CP199168 checkpoint: real training and fixed-model evaluation

This is an intermediate delivery, not completion of the requested full learned
traversal. The successful nominal remains preserved. Latest learned checkpoint
is not a verified full-task winner.

## Delivered media

- Deterministic: `CP199168_deterministic_review/PPO_PLUS_LIMITED_AUX_CP199168_deterministic_P01_full.mp4`.
  Natural P01, 54.625 s physical / 54.666667 s encoded, 820 frames at 15 fps,
  1280×870, normal speed, full failure tail retained. Full decode valid,
  820 unique frames, zero black frames, continuous monotonic PTS.
  SHA256 `142c4769c496b924df4a7406be47ef567c4f84859aedc6f45a901a5632392631`.
  Root inspected decoded first/final frames and delivered the video inline.
- Nominal comparison: `CP199168_deterministic_B_current_pair/N_vs_CP199168_PPO_PLUS_LIMITED_AUX_receiving_wheel_sigma_x3.mp4`.
  Successful B=N+0 versus the above deterministic C; not original FSM A.
  1108 frames, 15 fps, 1920×688; full decode valid, zero black frames.
  C ends at 54.625 s and is explicitly labelled FROZEN afterward; this is not
  continued stable physics. Root inspected the final comparison frame and
  delivered it inline. B retains its real 73.808333 s task success.
- Same-checkpoint stochastic seed4101: `CP199168_stochastic_seed4101_review/PPO_PLUS_LIMITED_AUX_CP199168_stochastic_P01_full_seed4101.mp4`.
  Natural P01, 57.716667 s physical / 57.733334 s encoded, 866 frames at 15 fps,
  1280×870, normal speed, full failure tail retained. Full decode valid,
  866 unique frames, zero black frames, continuous monotonic PTS.
  SHA256 `149d760004d3cd0008f10b1e3dfcd6131ad28ca8f991527894c2fddff9a3ac32`.
  P09 `TASK_FAILURE_BODY_COLLISION` at tick6926; no full success.
  `CP199168_same_checkpoint_modes.json` binds it to the deterministic capture at
  exact CP199168, scene/reset seed4001 and stochastic policy seed4101; it makes
  no same-measured-initial-state or statistical claim.

Method label is **PPO + LIMITED AUX**. The existing finite auxiliary event has
7 accepted / 8 attempted SGD steps, separate from all PPO counts. No new
auxiliary event, deployment teacher, permanent residual mask or hidden zero
fallback was added. Receiving sigma profile is explicit in metadata; its gate
was not reached in the newly completed training block or deterministic attempt.

## Actual training and production change

Commit `649ccd906421d06c8b5c699f28101730910e885c` changes seven runtime files:
`semantic_receiving_wheel_profile.py`, `semantic_receiving_wheel_sigma.py`,
`semantic_policy_distribution.py`, `semantic_training.py`,
`semantic_migration.py`, `semantic_cli.py`, and
`semantic_checkpoint_prefix_policy.py`. One portable unit test was added.
Production targeted suite: 104 tests PASS. The earlier collection error was
missing PYTHONPATH, not a failed test result.

Only conditional FR/RR wheel Gaussian sigma is multiplied by 3 in P10–P12
when RR placed-history observation157 is one. This is historical receiving
continuation/recovery, not proof of current support. Other ten sigmas, mean,
rho=.9/HISTORY, capacities, nominal, mapper, reward, evaluator and physics are
unchanged. Sampling, stored old likelihood, current likelihood, entropy,
checkpoint loading and stochastic evaluation use one kernel. No action-screening
or sign bias. Negative exploration also increases; capacities are not a safety proof.

Explicit same-physical-MDP stochastic-kernel migration from CP197120 retained
actor/critic, full Adam state and effective LR, Identity normalizer, RNG, branch
origin and AUX ledger. Fresh empty rollout; zero migration learning credit.
Actual first128/20-minibatch audit passed. There was no independent GPU RNG
replay, and the receiving gate was inactive in this front-only first rollout.

Block14 run `train/20260921T1355291518150Z_g649ccd906421_aac78369d59846409cb2d4039083e87a`
adds **2048 policy decisions / 16 PPO updates / 320 optimizer steps**, all natural
P01, no teacher prefix. Actual LR retained/adapted normally and ended at1e-5.

|Phase|P01|P02|P03|P04|P05|P06|P07|P08|P09|P10|P11|P12|P13|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|New actual samples|8|961|12|3|382|418|2|2|260|0|0|0|0|
|Branch cumulative|32|3343|55|18|2791|3110|12|15|2341|14|678|903|0|

Branch total13312 decisions /104 PPO /2080 optimizer; lifetime199168 /1521 /
30420. Prior nominal-prefix7632 decisions remain zero learner credit. Actual
quality front cost .06094348182919586 and geometry cost .17067929525151676
entered reward, signed sum−.23162277708071283; quality was not merely rescored.
Receiving×3 active samples=0: no rear benefit is established by this block.

Three task terminals: P09 BODY_COLLISION47.491667s; P02
INCOMPLETE_CONTROLLER_BLOCKED15.683333s; P09 BODY_COLLISION49.533333s. Final
P05 tail23.733333s is a nonterminal budget boundary, not another task failure.
Training lifecycle SUCCEEDED means sealed training, not successful traversal.

## Checkpoint and deterministic physical result

`checkpoints/history/checkpoint_step_000199168.pt`

SHA256 `e98957a4072b01eff9d9f20838b07dd0890bc997ef7a268061b929ae8aa5b673`.
1521 PPO updates /30420 optimizer steps, verified save/load roundtrip.

Deterministic run `video_eval/validation/20260921T1429210894242Z_g649ccd906421_0c5fc0498bbd4acca26ca3dde5d10193`:
FR qualified/cross/placed ticks23/2064/2074. FL crossed3190, never placed.
Terminal6555/54.625s, **P05 INCOMPLETE_CONTROLLER_BLOCKED**. Final FL AIR,
gap7.781872mm, bearing0N. No full success. Wrapper exit1 reflects failed-task
lifecycle; source manifest and video writer were sealed normally.

FR physical event window comparison:

|Metric|Successful B|CP194560 det|CP199168 det|
|---|---:|---:|---:|
|FR capture time s|12.5167|14.9583|17.2833|
|Roll RMS rad|.231919|.189668|.180907|
|Pitch RMS rad|.183521|.180312|.178496|
|True Euler-rate RMS rad/s|.117889|.092517|.088806|
|Peak tilt rad|.309464|.280743|.270717|
|Minimum body collider world z mm|91.272|94.868|96.486|
|Usable-air pre-cross FR mean gap mm|98.762|89.569|91.641|
|Body net forward mm|179.136|172.560|173.213|

CP199168 is38.08% slower than B and15.54% slower than CP194560. Its lower
tilt/rate does not establish equal-speed superiority. The measured body did not
collapse lower in this FR window; average FR gap remains7.121mm below B.
RL hip mean33.198° versus B37.901° is a simultaneous whole-body observation,
not isolated proof that RL alone caused the improvement. No statistical claim.

## Remaining physical problem and bounded evidence

Actual training FL placement did occur twice, but P06 retention degraded.
In the first episode, FL placed2957, P06entry2960, first AIR3097. Source four
wheel targets+.3rad/s were preserved; at the first AIR endpoint all four final
targets were nonzero. Final=mapped nominal+effective residual exactly; no mask
or same-tick cap-transition loss was found. By3496 FL had continuous AIR127ticks,
gap29.864mm. Successful B retained FL contact for2481/2481 P06 physics samples.
This comparison is not a same-state single-channel causal experiment.

The FL retention loss did enter potential shaping (−.120090 on first loss),
but its standardized advantage was still+1.004012;4/5 repetitions were clipped.
Later AIR samples included both signs. Ordinary phase changes are not done;
the128-step boundary uses real critic bootstrap, not zero future return.
No probability or reward-sign implementation error was found in these windows.
The compact supporting reports preserve exact evidence limits and missing fields.

The reviewed finite FLhip−1°/knee+1° ACK-anchored diagnostic has12 CPU tests
PASS and naturally sealed at54.625s/P05 INCOMPLETE_CONTROLLER_BLOCKED
(`diagnostics/FL_hip_minus1_knee_plus1_CP199168_20260921_01`). Actual prefix
0–3256 is exactly matched; hold minimum FLgap4.653mm vs6.865mm control,
but no actual FL contact or placement anywhere. After release all326 sampled
endpoints remain AIR. Full model state unchanged; zero PPO/AUX credit, not a
learned success or capture label. See CP199168_FL_hip_knee_evidence.md/json.
The successful N and all historical checkpoints/results remain untouched.
The stochastic MP4 was also inspected and delivered inline by root.
