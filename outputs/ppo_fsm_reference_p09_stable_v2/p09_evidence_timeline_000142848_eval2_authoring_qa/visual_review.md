# Visual and typed CSV verification

Reviewed the generated contact_columns.png at native displayed resolution on 2026-09-10.
All eight contact headers and the first seven data rows are legible. Long headers wrap;
no cell text overlaps adjacent columns. QA checkboxes represent Boolean values only;
the CSV retains literal true/false values, with missing measurements left blank.

This immutable snapshot contains 3,556 rows and 263 columns: 2,304 saved training
decisions from six completed runs and 635 + 617 deterministic evaluation decisions.
It excludes the active 4,096-decision run and the zero-action video diagnostic.
The builder verified every authored scalar, recalculated the typed grid, and
joined evaluation measurements by exact episode, tick, time and duration.
Training rows retain missing root attitude/linear measurements as blanks; actual
CoM or command values are not substituted. Unknown load is not zero load.

The original 1,659-row and 1,787-row CSVs and their QA remain untouched.
This CSV is diagnostic evidence, not a full-P01 success or stability-superiority claim.
