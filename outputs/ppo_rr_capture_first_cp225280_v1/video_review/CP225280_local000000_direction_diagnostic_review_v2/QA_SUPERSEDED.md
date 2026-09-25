# QA superseded

Do not deliver this v2 directory. Its HUD displayed the raw RR
`bearing_verified` sensor flag as `bearing`, which could misleadingly show
`AIR / 0 N / bearing=True`. The physical result and sealed source are
unchanged. Use the adjacent immutable `_v3` directory, where current bearing is
read from `rr_capture_local.metrics.current_top_bearing` and the raw sensor flag
is explicitly labelled as insufficient on its own.
