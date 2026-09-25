# Formal deterministic export

After a formal evaluation is naturally sealed, run the standard-library
wrapper below. It derives the runtime HEAD, actual composite checkpoint and
sidecar hashes, and all local training counts from the sealed source; none are
hand-entered.

```powershell
$repo = 'C:\robotics_sim\wlr_robot\fsm_base_on_recording_ppo_phase_v1'
$env:CUDA_VISIBLE_DEVICES = '-1'
& 'C:\Users\kskzz\miniconda3\python.exe' `
  "$repo\outputs\ppo_rr_capture_first_cp225280_v1\export_formal_from_sealed.py" `
  --source '<SEALED_FORMAL_SOURCE_ABSOLUTE_PATH>' `
  --destination "$repo\outputs\ppo_rr_capture_first_cp225280_v1\video_review\<UNIQUE_FORMAL_REVIEW>"
```

This prints the exact immutable plan without encoding. Review it, then append
`--execute`. Diagnostic manifests are rejected rather than silently exported
as formal results.

For the planned first 2048-local checkpoint, the display identity is
`CP227328_local002048`. The suffix is mandatory: this is the isolated local-RR
branch and is not the historical soft-KL checkpoint that shares numeric step
227328.
