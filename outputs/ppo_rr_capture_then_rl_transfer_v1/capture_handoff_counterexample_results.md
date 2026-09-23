# Synthetic P09 capture/handoff deadline counterexamples

Nine tests passed in 2.39 seconds; CPU process exited with code 0. Runtime HEAD: `e24a3c2630b0b95439a7a71fe6a8a3f610a9a385`. No production change, simulator, checkpoint change, training credit, or inspection of live training data. The ongoing training block is not stopped or gated by this result.

The test uses the existing `airborne_rr` fixture, production RR assist, physical evaluator and supervisor, with adjacent sensor ticks. FR/FL placement and RR lift/cross history are earned through synthetic measurements, not manually assigned. Deadline clock origins are positioned using the established unit-test pattern (P09 age 45 seconds, normal episode age 100 seconds); this is not a real 45-second Isaac trajectory. The feedback envelope is a synthetic API fixture, not an independent native dispatch ACK.

| Case | Observed production behavior |
| --- | --- |
| Expired local deadline; placement on tick modulo8=1,3,7 | First TOP sample carries pre-contact DESCEND and survives. Second TOP sample establishes placed, but fresh assist HOLD disables recovery permission; P09 terminates `LOCAL_BOUNDED_RECOVERY_EXHAUSTED` before the next decision boundary. |
| Same contact sequence before the deadline | TOP count1→2; placed remains true; HOLD continues without termination and P10 begins on the next modulo8=0 tick. |
| Expired P09 deadline; second TOP exactly modulo8=0 | Supervisor hands off to P10 before its local-timeout calculation; no terminal. |
| Same decision-boundary capture at episode age200 | `GLOBAL_FINITE_TASK_DEADLINE` still terminates; no success claim. |
| TOP followed by actual ground contact | No capture-hold waiver; no placed history; local incomplete termination remains. |
| Real placement history followed by AIR, non-decision tick | No current TOP or HOLD recovery permission; local incomplete termination remains. The existing `placed_RR` goal can nevertheless equal1 because the legacy current-usable-placement predicate admits qualified AIR. |
| Weak active obstacle pair, then first force-qualified TOP | The weak pair is correctly not classified TOP. The assist sees contact and enters HOLD; the next current TOP count is1, placed is still false, and the expired local deadline terminates before sample2. |

The simplest concrete failing chain was input RR airborne at tick9, TOP count1 at tick10 with DESCEND/fresh feedback/local warning, then TOP count2 and real placed at tick11 with HOLD/fresh feedback/no local warning. `hold_elapsed_s=1/120` at the terminal tick. The phase-boundary control places at tick16 and reaches P10, while the unexpired modulo1 control places at tick17 and remains alive until its tick24 P10 handoff. These are synthetic counterexamples, not the termination cause of any particular Isaac episode.

## Narrow possible repair, not implemented

At a later legal boundary, a finite contact-confirmation/handoff allowance could reuse the already cumulative public HOLD clock, bounded to10/120 seconds, plus completed-handoff-pending until the next decision boundary. It must require fresh committed feedback, currently valid measured TOP/bearing, legal capture geometry and the existing other-support conditions. Preserve global200 seconds, contact/safety failures, ground revocation and actual two-sample placement requirements. Do not reset cumulative HOLD on contact toggles or award placement/phase completion from this allowance.

Importantly, `all completion goals ==1` alone is insufficient for the new contact-hold exemption: the old-placed-AIR negative case already satisfies `placed_RR ==1`. A current-contact/bearing requirement is needed for this narrow exemption; it should not silently redefine existing physical task predicates.

Test file: `test_capture_handoff_counterexample.py` in this directory. Executed command (project working directory):

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:PYTHONPATH='src'
& C:/Users/kskzz/miniconda3/envs/env_isaaclab/python.exe -m pytest outputs/ppo_rr_capture_then_rl_transfer_v1/test_capture_handoff_counterexample.py -o addopts='' -q
```
