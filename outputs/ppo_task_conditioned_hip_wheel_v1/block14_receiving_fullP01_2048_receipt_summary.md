# Block 14 receiving-profile training receipt summary

Status: **PASS** — the formal receiving-profile bridge accepted the naturally sealed run using the detached historical quantity runtime and the original LIMITED AUX execution receipt. This was post-hoc CPU validation only; no Isaac, training, optimizer, or video process was touched.

## Sealed credit and checkpoint

- Run: `20260921T1355291518150Z_g649ccd906421_aac78369d59846409cb2d4039083e87a`
- New credit: 2,048 policy decisions (`197121..199168`), 16 PPO updates, 320 optimizer steps.
- Final checkpoint: `checkpoint_step_000199168.pt`, SHA-256 `e98957a4072b01eff9d9f20838b07dd0890bc997ef7a268061b929ae8aa5b673`.
- Lifetime counters: `199168 / 1521 / 30420`; save/load round-trip true; effective Adam LR `1e-5`.
- Full-episode budget: `102528 -> 104576` of `131072`; 2,048 spent here, 26,496 remain, no counter reset.

## Actual phase coverage

| Phase | P01 | P02 | P03 | P04 | P05 | P06 | P07 | P08 | P09 | P10 | P11 | P12 | P13 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Decisions | 8 | 961 | 12 | 3 | 382 | 418 | 2 | 2 | 260 | 0 | 0 | 0 | 0 |

The three completed terminals match the sealed audit rows exactly:

- decision 197833: P09 `BODY_COLLISION`;
- decision 198069: P02 `INCOMPLETE_CONTROLLER_BLOCKED`;
- decision 198812: P09 `BODY_COLLISION`.

The last sample (decision 199168, P05) is a budget boundary, not a success or failure. Natural-P01 full-task successes: 0; suffix successes: 0.

## Receiving gate and quality

All 2,048 request rows satisfy the profile's row-level gate/multiplier identity. Actual `RR_placed_history=true` rows: 0; actual P10-P12 rows: 0; therefore receiving x3 active rows: **0**. This run did not exercise the active receiving branch, so the zero count must not be presented as active-branch physical evidence.

Verified quality contributions are front cost `0.06094348182919586`, known geometry cost `0.17067929525151676`, and signed body-stability contribution `-0.23162277708071283`; receipt arithmetic is exact. Unknown terminal geometry remains null rather than being relabeled measured zero.

Lineage remains `PPO + LIMITED AUX`: 7 accepted / 8 attempted auxiliary steps, zero PPO credit, teacher not deployed. Historical quantity validation used the clean detached `97ecd305afb5c43b095e1206ce8917e6e742b1bb` runtime and is explicitly post-hoc only.

Formal receipt: `block14_receiving_fullP01_2048_receipt.json` (1,645,228 bytes), SHA-256 `ee8afe0bb171eab8cf14d0afbd7fbedc1951e27626b6ebb912cfeee023196c1f`.

No physical success or video-validation claim is made here.
