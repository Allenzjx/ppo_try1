# First fixed-version check timeline review

The authored CSV contains 1,659 actual decision rows and 263 columns: 1,024
credited training decisions from three completed blocks and 635 deterministic
evaluation decisions. Teacher prefixes are excluded. This is a fixed snapshot,
not a live or cumulative claim about later training.

The parent inspected `contact_columns.png` after generation. Headers and all
shown rows are legible; AIR, ground, obstacle-pair, TOP, non-TOP and availability
remain separate columns. Boolean checkbox rendering is only the QA view; the
CSV preserves typed true/false values. No merged-cell or clipped-data issue was
observed. The long non-TOP header wraps without loss of content.

The artifact engine recalculated and the exporter compared every authored scalar
against its source-derived value before serialization. Completed evaluation
observations were joined by exact episode, physical tick, time and dt; no nearest
sample substitution was used. Body measurements missing from training logs stay
blank with explicit validity flags. Current RR qualification is distinct from
historical lift events. Durations are diagnostic, not fixed-hover acceptance.

Neither the timeline nor this visual review changes task verdicts. The evaluated
policy remains incomplete at P05; this is not a successful P09/full-task result.
