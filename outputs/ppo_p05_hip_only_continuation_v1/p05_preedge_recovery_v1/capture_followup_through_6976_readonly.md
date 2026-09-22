# Actual FL capture and loss, fixed evidence through tick 6976

Same new `164147...13764630c7a0468285bacb6aa3607ee8` headless run. Existing stage events plus native contexts and actual physical contact pairs were available; no future rows or physics were requested.

| Event | Exact observation tick | Simulation time |
|---|---:|---:|
| FL crosses front edge | 6319 | 52.658333 s |
| P06 entry, FL not yet placed | 6320 | 52.666667 s |
| First FL TOP contact | 6350 | 52.916667 s |
| FL placement recorded | 6351 | 52.925000 s |
| First current TOP loss / AIR after placement | 6624 | 55.200000 s |
| Brief TOP reacquisition | 6626 | 55.216667 s |
| AIR again | 6627 | 55.225000 s |

The assist changes WAIT→DESCEND on dispatch 6320 using source observation 6319; DESCEND→HOLD on dispatch 6351 using the first TOP observation 6350. After the loss it retries descent. The first BLOCKED transition is **dispatch episode tick 6869**, using observation **6868** (57.233333 s), reason `gap_not_improving`, gap 5.648 mm; hip target 15.784889679°, held knee −32.082981907°.

At observation 6976 / 58.133333 s, placement **history remains true**, but current TOP and obstacle pair are false, both actual ground/obstacle contact pairs are inactive (AIR), and gap is 11.638261 mm. Historical placement is not current support. This is capture followed by loss, not maintained support or completed-task success.

Timing note: native dispatch's internal physics tick is source-observation tick +180, while episode dispatch tick is source-observation tick +1. JSON preserves both clocks and selected original contexts. This additional read did not alter controls, reward, runtime or optimizer.
