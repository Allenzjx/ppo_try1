# Parent review: completed natural-P01 N+0 timeline

Reviewed the generated receipt, independently imported CSV scalars, and displayed `contact_columns.png` after generation on 2026-09-10. The eight contact headers and seven preview data rows are readable, unselected, and contained within cells; the longest header wraps without overlap. This preview is not a visual review of every data row or a video.

The CSV contains 1,108 rows and 263 columns. All rows are `semantic_prior_eval`, `zero_residual_eval=true`, and credited/optimized/saved policy decision=false. Phase counts are P01=2, P02=182, P03=4, P04=1, P05=146, P06=310, P07=1, P08=1, P09=461; later phases=0. The completed run has one actual P09 task-incomplete terminal, not a collection cutoff or PPO success.

The authoring engine recalculated and verified the typed scalar grid. Its exact episode/tick/time physical join matched all 1,108 decisions, with no missing/invalid clocks or nearest-sample substitution. AIR, edge-related contact, ground contact, and true top-placement evidence remain separate. No teacher or zero-residual samples contribute PPO training credit, and no fixed RR hover-duration requirement was introduced.
