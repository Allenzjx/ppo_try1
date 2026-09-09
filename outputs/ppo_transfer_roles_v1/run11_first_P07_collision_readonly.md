# Run11 first P07 handoff → collision — bounded read-only

Run `20260908T0418530171278Z_gdb991aa68103_aacec4db672a4cc6be3043314a075baf`. Only first accepted handoff, first14 policy rows (g134273–134286) and first completed terminal; no later episode/optimizer analysis.

**Actual outcome:**14 decisions /109 policy physics ticks after teacher tick5952; terminal6061/50.508333s, P09 `BODY_COLLISION`, physical `TASK_FAILURE_BODY_COLLISION`, valid=true, success=false. This is a recorded hard physical failure, not incomplete/no-data. Compact terminal lacks the exact base-pair force, point, persistence count, penetration and final body pose: the detailed contact trigger cannot be independently reconstructed here.

## Handoff and stage predicates

First accepted P07 credit begins **5952/49.6s**, offset0,744 teacher decisions/5952 teacher calls excluded from PPO. FR/FL teacher Q/C/P remain71/1665/1695 and2461/3115/3583; RR/RL Q/C/P false. Actual FL is TOP/support/load.199119, RR GROUND/support/.247820/front−205.029mm/gap−50.920mm; all four wheels support. Base position[.619228,−.005971,.093921]m, quaternion wxyz[.999654,−.020910,−.015934,−.000541]; speed.019141m/s/angular.013666rad/s. CoM velocity[.016460,−.000910,.004499]m/s. Collision/penetration false/0 at handoff. Base height is **not collider clearance**; subsequent compact rows do not contain body clearance or directional CoM/pose.

P07 lasts8ticks; P08 lasts16; P09 receives11 decisions/85ticks. At5960, P07 completed current edge/workspace and role-preparation predicates. P08 **waited** at5968 with RR GROUND/load.254446 and transfer_ready.232985, then passed at5976 with current load.068885, GROUND, AIR=false and Q/C/P=false. This is the allowed measured low-load branch, not a false declaration of qualified lift. First full P09 boundary5984 still has RR GROUND/load.025820. No longer static-support or initial-AIR gate is implied.

| Tick | RR current load / AIR | FL gap mm / load | FL hip nominal → final drive ° |
|---:|---|---|---|
|5952|.247820 / false|+.256 / .199119|22.8 →21.55|
|5960|0 / true|+7.170 /0|31.55 →30.8|
|5976|.068885 / false|+27.985 /0|46.1 →40.868|
|6000|0 / true|+64.449 /0|49.2 →52.299|
|6032|.127420 / false|+204.668 /0|49.2 →63.577|
|6061|.147359 / false|+280.625 /0|49.2 →65.312|

## Execution versus attribution

All14 native summaries total109 verified/effect ticks,107 own-phase ticks (one transition hold after each handoff), no state writes and no headroom-clipped channels. All14 boundaries retain RR Q/C/P=false; its best sampled AIR gap is still−42.176mm. FL is AIR/load0 at every policy boundary despite historical placement. Body angular speed reaches1.069036rad/s at6024.

Both action components matter: FL nominal hip rises22.8→49.2°, residual later reaches+16.786901°, final target peaks68.881358°. At terminal FL knee nominal−13.4/residual+33.259208/final+21.109208°. FR nominal wheel+.3 becomes−.63, with final−.890887 at6032 from residual−.260887; this is actual inherited advisory plus PPO, not PPO-only motion.

Source `from_live_prefix/from_handoff` retains the supervisor/history and seeds executed nominal; `MotionExecutor.start_phase` does not replace an existing seed with a fixed entry pose. It starts a new finite source queue, not a copy of the entire teacher queue; `_continuous_advisory` intentionally continues touched predecessor channels into P08/P09. Thus the inherited P07 changes are observable and relevant, **not a demonstrated stale-stage/replay or dispatch defect**. They also mean the suffix is not guaranteed equivalent to continuing the original teacher.

**Conclusion:** no actionable handoff/entry-validation defect is established in this window. A nominal-versus-role advisory review may use the measured loss of current FL support, but these combined-command observations cannot isolate nominal, PPO, or contact as the unique cause. Preserve this failure and teacher-credit boundary; no new gate, production change or simulation was introduced.

