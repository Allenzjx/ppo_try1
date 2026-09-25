# Successful N: fixed FR-receiver COM projection

This addendum uses only the existing 21 exact-tick successful-N table; it does not rescan the large sealed logs, interpolate ticks, or run physics. The fixed receiver axis is set once at the RR-loaded pre-late row and is never updated as FR moves.

## Fixed direction and evidence boundary

- Start: tick 6152; COM xy=[0.7024185999049787, -0.11749699210267131]; logged FR-wheel support-point xy=[1.0443284511566162, -0.35766610503196716]; unit direction=[0.8182951081832207, -0.5747983263053321].
- RR load at start: 6.484 N. The direction therefore begins in an actually RR-loaded context, not an AIR-RR assumption.
- The compact table retained the ContactSensor support point owned by `front_right_wheel`, not the rigid-body wheel-center xy. That distinction is explicit in JSON; the support point is used as the only already-extracted actual FR receiver location.
- Base orientation was not retained. Yaw displacement is therefore N/A; logged yaw rate is shown per row and is not sparsely integrated.

## Actual COM response and loads

|event|tick / t|dx / dy / fixed-FR projection (mm)|COM v·d (m/s)|yaw rate (rad/s)|FL / RR load (N)|RL context|
|---|---:|---:|---:|---:|---:|---|
|PRE_LATE_RR_LOADED_CONTEXT|6152 / 51.267s|0.0 / 0.0 / 0.0|0.015|0.013|5.928 / 6.484|GROUND front/gap=-159.1/-50.4mm|
|P09_LATE_ATOMIC_FL_RL_LAUNCH|6156 / 51.300s|0.4 / -0.1 / 0.4|0.009|-0.013|8.192 / 8.006|GROUND front/gap=-158.7/-50.5mm|
|P09_LATE_EARLY_RESPONSE|6160 / 51.333s|1.0 / -0.8 / 1.3|0.044|-0.020|10.138 / 9.688|GROUND front/gap=-158.8/-50.5mm|
|P10_RR_KNEE_POSITIVE_ENDPOINT|6168 / 51.400s|6.5 / -3.0 / 7.1|0.119|-0.149|11.022 / 14.347|GROUND front/gap=-157.1/-50.7mm|
|P11_RESPONSE_FR_AIR_RR_LOADED|6176 / 51.467s|19.0 / -2.9 / 17.2|0.158|-0.072|11.846 / 11.136|GROUND front/gap=-150.7/-51.0mm|
|P12_RL_KNEE_POSITIVE_AND_FR_HIP_RESPONSE|6201 / 51.675s|48.5 / 2.5 / 38.2|0.013|-0.141|14.454 / 13.999|GROUND front/gap=-139.8/-50.0mm|
|P12_FOUR_WHEEL_REVERSE_BEGIN|6233 / 51.942s|73.4 / -7.6 / 64.4|0.190|-0.405|15.695 / 14.724|AIR front/gap=-94.7/-50.0mm|
|RL_QUALIFIED_EVENT|6251 / 52.092s|89.6 / -24.5 / 87.4|0.085|-0.084|14.678 / 14.595|QUALIFIED; AIR front/gap=-56.5/-42.0mm|
|REVERSE_AND_LOAD_RESPONSE|6256 / 52.133s|92.0 / -27.0 / 90.8|0.078|-0.036|14.032 / 14.399|AIR front/gap=-54.4/-33.4mm|
|RR_SHORT_AIR_SNAPSHOT|6272 / 52.267s|96.0 / -36.2 / 99.4|-0.003|0.032|4.964 / 0.000|AIR front/gap=-55.2/4.1mm|
|FIRST_REVERSE_STOP|6497 / 54.142s|62.3 / -31.1 / 68.9|0.001|0.028|14.545 / 14.146|AIR front/gap=-90.4/-0.0mm|
|SECOND_STOP_BEFORE_RL_CROSS|6652 / 55.433s|83.3 / -34.9 / 88.2|0.013|0.029|14.840 / 12.371|AIR front/gap=-3.9/65.2mm|
|RL_FRONT_EDGE_CROSSED_EVENT|6658 / 55.483s|83.6 / -34.9 / 88.5|0.007|0.008|13.907 / 12.191|FRONT_EDGE_CROSSED; AIR front/gap=0.7/63.8mm|
|RL_PLACED_EVENT|6727 / 56.058s|86.3 / -35.6 / 91.0|-0.007|0.026|4.126 / 4.853|PLACED; OBSTACLE front/gap=69.5/-0.2mm|
|RL_CAPTURE_RESPONSE|6728 / 56.067s|86.0 / -35.7 / 90.9|-0.007|0.035|3.862 / 5.396|OBSTACLE front/gap=69.7/-0.2mm|

RL qualification, crossing, and placement are separately logged at ticks 6251, 6658, and 6727; none is inferred from joint angles or COM direction.

## CP225280 entrance contrast

|event|tick / t|RR TOP / load|RR front / gap (mm)|local 0.5s COM→FR (mm)|
|---|---:|---:|---:|---:|
|P09_ENTRY|6752 / 56.267s|False / 1.410|-219.6 / -50.2|5.4|
|RR_QUALIFIED|6995 / 58.292s|False / 0.000|-177.0 / -41.3|27.3|
|RR_CROSSED_AIR|7979 / 66.492s|False / 0.000|0.0 / 59.0|4.5|
|TERMINAL|10940 / 91.167s|False / 0.000|99.9 / 56.1|0.1|

CP225280 never supplies an equivalent loaded RR transfer start in these selected rows; its 0.5 s projection axes were recomputed at each local window. Those numbers are descriptive, not time-aligned with—or projections onto—the successful-N fixed axis.

The positive/negative fixed-axis projection is measured whole-body COM motion toward/away from the frozen FR receiver direction. It does not attribute that motion to FL, RR, a joint angle, or any single command channel.
