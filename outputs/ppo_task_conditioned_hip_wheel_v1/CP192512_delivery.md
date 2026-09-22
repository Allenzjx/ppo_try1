# CP192512 — pure-PPO full attempts (three videos delivered)

At the capture boundary, saved formal CP192512 had 192512 lifetime policy
decisions,1469 PPO updates,29380 optimizer steps; task branch6656/52/1040.
The weights in all three videos have no auxiliary update. Later continuation
is separately identified below. Successful N and earlier artifacts are unchanged.

## Delivered deterministic attempt

- `CP192512_deterministic_review/CP192512_deterministic_P01_full.mp4`
- Physical51.833333s,6220ticks,778frames/15fps; encoded51.866667s reflects the
  terminal display interval, not extra physics. Full decode passed; first/last
  decoded frames viewed by root. SHA3c427ff2f01143fb7a528fb90ec404d301204d20f47d7ed3c7c346bbec73cea3.
- NaturalP01, same full12 learned policy, no teacher/manual intervention.
  First unfinished task P05: FL crossed2785 but never contacted/placed; final
  AIR gap16.259794mm,bearing0. `INCOMPLETE_CONTROLLER_BLOCKED`, local bounded
  recovery exhausted. No body collision and no complete success.
- Source run `20260921T1006004011698Z_g97ecd305afb5_78f9ffd75d504f0e8447f44e61d134ce`.
  Source media sealed normally; wrapper exited1 because lifecycle recorded task
  incompletion as DIAGNOSTIC_FAILURE, not because the simulator or writer failed.

## Delivered N comparison

- `CP192512_N_pair/N_vs_CP192512_deterministic_quantity_boundary.mp4`
- Left preserved B=N+0 SUCCESS73.808333s; right same saved CP192512 deterministic
  TASK_INCOMPLETE51.833333s. 1108frames/15fps, full decode passed and root previewed.
  Right330post-windowframes explicitly FROZEN FRAME, zero further physics evidence.
  SHAe3b2697f3e70709e580b089b29576e85f33c617f71cacab69467982214089ae6.
- Runtime hashes and execution-profile hashes differ only across the officially
  revalidated quantity-only boundary. All physical/control semantics unchanged;
  metadata does NOT claim equal complete runtime/evaluation configuration hashes.
  `pair_quantity_only_v2.py` checks all six exact versioned configurations and
  permits only the bound profile hash difference. Old helper/failed attempt retained.

## Same-model stochastic attempt

- `CP192512_stochastic_seed4101_review/CP192512_stochastic_P01_full_seed4101.mp4`
- SameCP192512, physicalseed4001, policyseed4101, naturalP01; same-weight/runtime
  check passed in `CP192512_same_model_modes.json`. No auxiliary or teacher.
- Physical48.125s/5775ticks/722frames; encoded48.133334s, full decode passed,
  root viewed first/last decoded frames and delivered inline. SHA
  a01bcee418f970fcc070121ea309d38440a95fdcecb0d8091e300fedb2ef6349.
- Run `20260921T1024030822863Z_g97ecd305afb5_57f370bd94e9499e8ab7a10b18115e10`.
  Actual BODY_COLLISION/P09. FR placed1661, FL placed2986, but subsequent FL
  bearing was not retained. RR qualified4861, revoked4936 after ground contact,
  never crossed/placed. First unfinished task: maintained RR lift and carry.
  Full final7tick action and failure tail retained. This stochastic partial
  progress is not deterministic FL capability or full success.

## Later training lineage (not the weights in these videos)

After all three videos were sealed and delivered, a separate finite auxiliary
update was applied to CP192512:7 accepted/8 attempted independent SGD steps,
step8 rolled back on KL>.1. Receipt `finite_aux_CP192512_execution_01.json`.
This does not add PPO credit or relabel any video above as auxiliary-trained.

## Event-window evidence already available

`CP192512_FR_window_readonly.md` compares the complete FR physical window.
VersusCP189952:0.4833s faster and body minimum+1.584mm, FR mean gap nearly
unchanged, but angular-rateRMS+6.05%. VersusN, still slower and lowerFRgap;
not a claim of globally superior stability. `CP192512_P05_readonly.*` records
the residual/execution/contact chain; a smaller AIR gap is not FL capture.
