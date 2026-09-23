# CP220544 ancestor / progress-handoff v5 media draft

Status: **prepared only; not executed**. The referenced video source is still
writer-owned. Do not run the command below until root explicitly confirms the
run is sealed and authorizes CPU media work.

Exporter:

`outputs/ppo_rr_capture_then_rl_transfer_v1/export_rr_capture_video_v5_CP220544.py`

Intended sealed source:

`runs/ppo_rr_capture_then_rl_transfer_v1/video_eval/validation/20260923T0718224682359Z_gc53119ab332f_886689c9a357416293bc8242856d0be6/source`

Reserved unique destination:

`outputs/ppo_rr_capture_then_rl_transfer_v1/video_review/CP220544_ancestor_progress_handoff_v5_review`

Future command, only after seal permission:

```powershell
$env:CUDA_VISIBLE_DEVICES = '-1'
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
python outputs/ppo_rr_capture_then_rl_transfer_v1/export_rr_capture_video_v5_CP220544.py `
  --source runs/ppo_rr_capture_then_rl_transfer_v1/video_eval/validation/20260923T0718224682359Z_gc53119ab332f_886689c9a357416293bc8242856d0be6/source `
  --destination outputs/ppo_rr_capture_then_rl_transfer_v1/video_review/CP220544_ancestor_progress_handoff_v5_review
```

The adapter binds checkpoint SHA
`308122a3c8e760733fbe8370bfdf399a25718148c82333deb41a1265e4e13895`
and counters `220544 / 1688 / 33760`. These are the explicitly selected
front-validated ancestor weights. The later CP221184 checkpoint and its
640/5/100 learning credit are not used or borrowed. The RR branch contribution
for this checkpoint is `0 / 0 / 0`; migration and evaluation add no learning.

The visible controller label is progress-handoff v5 with public
`DESCEND_PROGRESS` / `CAPTURED_FOLLOW`, bounded 52-degree / 44-second totals,
common FL/RR capture assists, support-wheel v4, and inherited limited AUX.
This is not pure policy. The policy contract/kernel remains the same410
`rr_capture_then_rl_transfer_history_v1`; the controlled MDP changed.

Outputs remain data-dependent:

- the full natural-speed episode includes every decoded source frame and its
  failure tail, up to the 200-second/3000-frame bound;
- RR-to-RL detail is emitted only for the RR/RL interval actually reached;
  otherwise a truthful predecessor tail is used and the missing RR window is
  recorded;
- no RL placement or task-success label is emitted without this episode's
  evidence;
- the historical accepted N reference is clearly marked non-fresh and freezes
  visibly after its own end;
- derived videos rebuild a continuous 15 fps PTS grid from decoded frame order;
  source container duration is never treated as physics duration.

No source, checkpoint, production file, existing exporter, or existing media
artifact is modified by this draft.
