# Block7 timeline review

Created from the completed natural-P01 block7 only, using the existing bundled Artifact Tool builder. The operation marker was run once before this new CSV. Older CSVs, production and the active P10 run were not changed.

The parent inspected `contact_columns.png`: all eight contact headers and seven data rows are visible, readable and aligned, with no clipped text or overlapping cells. Checkbox display is authoring-preview representation only; the CSV contains literal true/false values.

Independent PowerShell CSV readback confirms 2,560 rows, 263 columns, one source run, 2,560 optimized and saved decisions, zero teacher-credit rows and 10 terminal rows. Phase counts are 24/1265/42/19/989/178/3/3/37 for P01-P09, with no P10-P13 rows. They match the completed training manifest and fixed progress snapshot. Every scalar was checked against the authored grid by the reused builder after recalculation.

This is an incremental block7 table, not a replacement for the prior 3,556-row timeline: those files cover disjoint training blocks. Together they cover 4,864 saved training decisions plus 1,252 evaluation decisions, without counting the active block8.

Missing full raw physical streams remain explicit. All 2,560 CSV body roll/pitch and linear-vector fields remain blank; actual joint readbacks use the documented measured-margin fallback, not commanded angles. The separate later-episode diagnostic decodes available terminal-only float32 vectors, but this unchanged CSV builder does not substitute those for a complete physical stream or fabricate missing nonterminal measurements. Unknown load remains blank rather than zero. These files do not establish full-task success or FSM-relative improvement.
