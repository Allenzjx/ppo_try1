# Superseded direction-probe analysis artifacts

The following intermediate JSON files are preserved but must not be used:

- `student_entry_direction_probe_v2_analysis.json`
- `student_entry_direction_probe_v2_analysis_clock_aligned.json`
- `student_entry_direction_probe_v2_analysis_final.json`

They used `native_readback_audit.native_drive_target_full12`, which is mapper
nominal N, under an incorrect `FINAL` label. No underlying source log was
changed. The corrected authority is
`student_entry_direction_probe_v2_analysis_ACK_FINAL.json`, where executed
FINAL is sourced only from `atomic_ACK.drive_target_full12` and independently
requires verified, tick-matched actuator readback. The nominal mapper command
is retained under the explicit `mapper_nominal_native_target` label.
