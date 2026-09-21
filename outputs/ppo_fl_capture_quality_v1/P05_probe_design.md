# P05 FL controllability diagnostic — design, not a repair claim

The immutable CP178432 natural-P01 failure remains failure. Its sealed 3408-tick chain was reused; the frequency window is ticks3000–5800, 2801 samples at120Hz. No new physics or actor forward was used to infer that trajectory.

## Measured frequency and reference

Mapper FL-hip targets update every4 physics ticks (30Hz), but repeat a four-update cycle: the dominant target and actual-joint frequency is **7.5Hz**, not30Hz. Mapper target peak-to-peak2.5deg; final target2.6342deg; measured hip .5947deg. At7.5Hz, measured hip lags final target67.685deg, coherence≈1; wheel-bottom gap lags72.885deg. These are closed-loop relationships, not open-loop causal transfer estimates. Raw FL policy varies mostly below.5Hz in this window.

All700 eligible FL-hip feedback samples used the previous filtered REQUEST reference; none clipped. `c1=clip(c0+8*r_previous)` held exactly. If a constant residual±1.5deg is physically achieved, the current reference law asks for zero additional nominal correction rather than cancelling that residual. Pure arithmetic examples confirm this, but do not prove real closed-loop tracking. Knee was not actively tracking in the tail; its prior+1.25deg mapper bias was held. No mapper change is presently justified by a supposed omitted reference.

## Geometric amplitude and headroom

Read-only frozen USD joint frames were hash-matched to the actual asset; measured base pose, canonical q and captured standing offsets reconstruct FL wheel center within.0001mm at three saved ticks. This is a center derivative with fixed measured base, not a lowest-collider or contact-dynamics model.

| Local estimate | Hip | Knee |
|---|---:|---:|
| World X mm / canonical deg |1.784–1.794|2.731–2.732|
| World Z mm / canonical deg |2.639–2.646|.237–.248|

Use two separate legal natural resets, **hip−1.5deg** and **hip+1.5deg**. Estimated vertical displacement is∓3.96mm, comparable to the observed millimetre gap, with about∓2.68mm fore-aft movement. Optional third case hip−1.5/knee+1deg estimates−3.72mm Z and approximately+.05mm X. These are small against P05 caps18/24deg and measured native targets far from reserved hard bounds. Existing projection, headroom,60deg/s residual slew and150deg/s final slew remain authoritative. Live read-only Jacobian checks the actual sign/scale before any probe; physical body/contact response can differ.

## Actual experiment protocol

`run_fl_controllability_probe.py` runs one case per process. CP178432 is officially integrity-loaded against its saved contract, while a separate receipt reports the actual runtime and explicit reviewed differences. Old RR six configs are byte-identical; actor/backend/env/mapper/observation/physics differences are rejected. Supervisor AST differences must be only the five new soft-reward symbols, not TaskEvaluator or N. No checkpoint/normalizer/optimizer is updated or saved.

Start at naturalP01 and follow the actual deterministic policy through P05. Trigger only after its source endpoint, with current valid FL AIR, already crossed, in top XY and not captured. No teleport, state/observation injection, snapshot restoration or historical entry equality. Triggered FL-channel residual uses the last real filtered request as a fixed anchor; add the finite offset to that anchor, not recursively to each new HISTORY mean. All unselected channels retain the actual current policy output, and actual issued raw remains in the ordinary history.

Ramp8 decisions(.533s), hold32(2.133s), release8(.533s), then12(.8s) ordinary-policy follow decisions. Real capture or leavingP05 begins release immediately: no continuing forced descent into established contact. This observation duration is a diagnostic, not a new task dwell condition. Finite episode cap30s; task terminals always take priority. All standard120Hz physical/native logs plus per-decision baseline/injected raw are saved. A successful contact is external controllability evidence, never formal all-channel PPO success.

Six helper tests and official CP178432 CPU load passed; no Isaac run was performed by the helper author. `sealed_FL_frequency_and_probe_design.json` stores numerical evidence. Parent owns sequential physical execution.

After each natural seal, use:

```powershell
& C:/Users/kskzz/miniconda3/envs/env_isaaclab/python.exe outputs/ppo_fl_capture_quality_v1/summarize_fl_probe.py RUN_DIRECTORY --output-stem outputs/ppo_fl_capture_quality_v1/CASE_result
```

The summarizer separates not-triggered, triggered, complete, true contact/placed, P06 entry and current support retention, with original failure labels preserved. It will not inspect an active run.
