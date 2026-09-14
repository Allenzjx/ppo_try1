# Timeline snapshot verification

The parent viewed `contact_columns.png` after the bundled artifact engine authored,
recalculated and checked every scalar in this CSV snapshot. All eight contact
column headings and displayed rows are readable; booleans render as checkboxes
only in this QA image and remain true/false in the CSV.

The new snapshot contains 1,787 actual decision rows and 263 columns: 1,152 saved
on-policy decisions from four completed training runs and 635 decisions from the
completed natural-P01 deterministic evaluation. The active P10 run is excluded.
The earlier CSV and its QA evidence remain unchanged.

The completed P07 block includes real FRONT_WALL/OBSTACLE_AMBIGUOUS interaction,
ground-return qualification revocation and a later RR crossing event, but no RR
placement or full task success. Historical crossing is not current top-region
occupancy. Unavailable measurements stay blank with validity fields; no teacher
prefix receives credit and no fixed RR holding timer is an acceptance condition.

This is tabular authoring/visual QA, not a new physical replay, contact-force
causality test, checkpoint reload or successful-video validation. See the receipt
and `p07_edge_crossing_block128.md` for provenance and physical limitations.
