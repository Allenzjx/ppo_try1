# CP201728 deterministic front-data feasibility — three P02 samples only

**Feasible for the three inspected P02 states, with explicit numerical-verification limits.** No missing field required guessing or zero-filling for these samples. This is not a published training dataset or a claim that all P01/P02 states are ready.

Source: `video_eval/validation/20260922T0450447593129Z_g0001c3138b0b_f412423658e844e386aef04f0f53bcdc/source`; original checkpoint201728 SHA256 `a4eb243ce07ad4a9ed1cf2f7f9a22e951181bdb6e0716361b8eaeb173acfd5b6`.

| Policy decision | Input episode tick / time | Maximum full12 conditional-mean error, CPU reconstruction vs recorded CUDA |
|---:|---:|---:|
| 3 | 16 /0.133333s | 3.7252903e−9 |
| 4 | 24 /0.200000s | 7.4505806e−9 |
| 5 | 32 /0.266667s | 9.3132257e−10 |

All three source raw actions exactly equal their recorded deterministic conditional means; these are actual executed mean actions, unlike block03 stochastic raw samples. CPU base-mean maximum error2.9802322e−8; effective-sigma maximum error1.1175871e−8. The recorded previous-raw HISTORY,12 assist and5 pending features match the reconstructed float32 values exactly. No optimizer or Gaussian sampling was run.

Reconstruction uses only explicit source relationships:

- Current physical sample plus the immediately preceding **physics tick** for quaternion/rate derivatives; not the previous15Hz decision.
- Current semantic-task endpoint from the preceding decision, which ends exactly at the new action's start tick.
- Current capture row's next nominal and native ACK; current/prior physical ACKs for the six residual/drive/nominal HISTORY groups.
- The next dispatch's explicitly recorded `tracking_reference_evidence.mapper_pre_state`, which is the unchanged state before that dispatch; its matching previous-ACK timestamp, feedback tick and `previous_final_drive_servo_deg` establish the current mapper summary. This does not substitute next-dispatch targets for current state.
- The original encoder/schema/env-history/actor/HISTORY/sigma source files were checked byte-identical to checkpoint201728's bindings before use.

**Limits:** output agreement is not a mathematical proof of389-input equality: the original full389 vector was never directly saved, and a12-output network is not invertible. The CPU/CUDA outputs are not bitwise identical; the measured differences are tiny, consistent with float32 backend roundoff, but are reported rather than hidden. The construction provides field-level provenance plus per-channel numerical checks, not an invented direct-observation hash. Source P01 reset tick0 and a full front window were not reconstructed or certified in this bounded check. No old failed suffix was reclassified.

Independent reader: `inspect_CP201728_P02_observation_reconstruction.py`; CPU process exited0. It reads only the first5 decisions and first34 physical/33 capture/native rows, plus the original checkpoint. No v1 hash-bound helper, runtime, configuration or real checkpoint was modified. No data was admitted to AUX/PPO; the current formal block06/deterministic work remains unaffected. Whether a future finite deterministic rehearsal improves the natural policy remains unknown.
