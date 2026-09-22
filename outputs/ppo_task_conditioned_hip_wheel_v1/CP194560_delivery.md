# CP194560 — PPO + LIMITED AUX, full attempts

Saved CP194560 SHA20d327d6c9b62a1b459dad61abbae7181a7b1dcae23ca26a5f51ba63e7399aca;
official save/load roundtrip passed. Lifetime194560 decisions/1485 PPO updates/
29700 optimizer steps. Task branch8704/68/1360. Independent finite auxiliary
learning remains exactly7 accepted/8 attempted SGD steps, zero PPO credit.
No teacher, manual joint action, diagnostic mask or hidden zero fallback is used
in deployment. Successful N and original pure-PPO checkpoints remain preserved.

## Deterministic complete attempt — delivered

`CP194560_deterministic_review/PPO_PLUS_LIMITED_AUX_CP194560_deterministic_P01_full.mp4`

- NaturalP01, physicalseed4001, fixed saved weights; source run
  `20260921T1117298582122Z_g97ecd305afb5_26ae0caab77f4c1bab164b45635744d4`.
- 6276ticks/52.3s;785frames15fps, encoded52.333334s due final display interval.
  Full decode passed; root viewed first/last decoded PNGs and delivered inline.
  SHA8559fb5d5c38db72d87a937d8bcadcf76c11b85f11a7a21d97a1c68d8ab1e086.
- TASK_INCOMPLETE/P05, INCOMPLETE_CONTROLLER_BLOCKED. FRplaced1795,
  FLcross2914 but neverplaced; finalFL AIR/16.25259249mm/0N. No bodycollision.
  Full waiting tail retained. Wrapper exited1 for physical task incompletion,
  while source writer and export sealed successfully; do not erase exit status.
- FL hip negative residual really reached the execution chain (about−2.07deg
  at3568), but simultaneous FL knee/body/other-leg state did not produce contact.
  It is not a missing-action proof or a successful landing policy.

## Same-checkpoint stochastic complete attempt — delivered

`CP194560_stochastic_seed4101_review/PPO_PLUS_LIMITED_AUX_CP194560_stochastic_P01_full_seed4101.mp4`

- NaturalP01, physicalseed4001/policyseed4101, same savedCP194560; source
  `20260921T1136245327456Z_g97ecd305afb5_1e424676283e49bdbee680a08111d59e`.
- 8219ticks/68.4916667s,1028frames15fps1280x870; encoded68.533334s due final
  frame interval. Full decode passed,1028unique frames/0black-like. Root viewed
  first/last decoded PNGs and delivered inline/openqueued. SHA
  b7e3a0156744d0479be6f9142efd1fc9a0f0575f6bbe565715f6deac8dca9d36.
- TASK_INCOMPLETE/P09, INCOMPLETE_CONTROLLER_BLOCKED. FRplaced1660/FLplaced2986;
  RRqualified4887, regrounded/revoked4928, never crossed/placed. FinalFL
  AIR186.416mm/0N, RRground; no physical collision termination. Full wait tail
  retained. Wrapper exit1 reflects incomplete task, not video writer failure.
- Dedicated auxiliary-aware same-model modes validation completed successfully:
  `CP194560_same_model_modes.json`. No hidden teacher, diagnostic mask or new
  auxiliary steps during either evaluation. This is not a learned full success.

## N comparison — delivered

`CP194560_N_pair/N_vs_CP194560_PPO_PLUS_LIMITED_AUX.mp4`, SHA
1e39d1acf5c40230dc200b918b3594f8fc6236ee59e196b20a323cae3150d8ee.
1108frames/15fps/1920x688; full decode passed, root previewed first/last decoded
frames and delivered inline. Left preservedB SUCCESS73.808333s; rightC
TASK_INCOMPLETE52.3s. Right323 added FROZEN frames are display only, not physics.
The complete actual pair gate passed using the officially revalidated quantity-
only control boundary; full runtime/evaluation configuration hash equality is
not claimed. Filename, picture and receipt all identify PPO+LIMITEDAUX.

## Evidence limits and continuing work

`CP194560_FR_window_readonly.md`: versusCP192512,rateRMS−6.10%,peaktilt−1.59%,
body minimum+1.707mm, but0.475s slower andpitchRMS higher. RelativeN still
2.442s slower andFR mean gap9.193mm lower. No unqualified stability superiority.

`CP194560_fixed_state_comparison.md`: finite aux FL network direction was
retained/strengthened by2048 later truePPO decisions on identical originalH,
but fixed-state action changes do not establish natural capture or safety.
`block10_FL_capture_credit.md`: two real training captures had positive immediate
rewards but negative standardized advantages; no automatic reward bug inferred.

Block10 actual coverageP01..P13:
[4,433,8,2,375,392,2,2,486,5,339,0,0]. First natural training episode achieved
RRcross/place then lost retention and stoppedP11; secondP09 budgettail was
nonterminal. Neither is a full learned success or a fixed-model success video.
Further genuine preparation/capture/hold and rear continuation learning remains.

Actual continuation started12:00UTC: block11 P04 successful-nominal real-prefix
course, requested1024 fresh learner decisions fromCP194560, same production
version. Prefix gets0 learner credit. Its future outcomes/counts are not included
in this fixed-model video report.
