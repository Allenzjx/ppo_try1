# Block10 episode0 P11 terminal credit (read-only)

Confirmed terminal: global decision **193551**, physical tick **8306**,
69.2166667 s, `P11 / INCOMPLETE_CONTROLLER_BLOCKED`. The final issued action
executed two physical ticks (8304→8306). This is task incompletion, not success.

| Actual terminal quantity | Value |
|---|---:|
| Terminal task event | -40 |
| Potential before → after | .631333739006882 → 0 |
| Potential shaping | -3.15666869503441 |
| Logged double reward | -43.15700202836774 |
| Stored float32 reward / return | -43.15700149536133 / -43.15700149536133 |
| Stored old value | -28.116270065307617 |
| Raw GAE (`return-old_value`, storage dtype) | -15.040731430053711 |
| Stored and actually optimized normalized advantage | -3.012563467025757 |
| `done` / timeout / terminal bootstrap allowed | 1 / false / false |

The terminal return equals its reward: no next-state value or next episode
reward is bootstrapped across this boundary. Phi=0 and the terminal -40 are
actually present, not merely configured. Official whole-rollout standardized
GAE is used; raw-GAE mean=-1.3893221616744995, unbiased std=4.531492710113525.
The actual terminal advantage appears unchanged in likelihood minibatches
3,7,11,15,19 (five epochs, original rollout index14).

`rollout_001478.pt` covers decisions193537–193664. There are **14 samples strictly
before termination + 1 terminal sample**, all P11, then **113 real reset-episode
samples** (P01=2, P02=111). First reset decision193552 ends at tick8/P01.
All15 P11 samples have negative actual raw GAE (-15.04073..-12.46054) and
negative normalized advantage (-3.01256..-2.44317), not only the terminal row.
The only done in this rollout is index14. The rollout's final row belongs to the
new, still-live P02 episode and legitimately uses its own nonterminal tail value;
this does not bootstrap across the earlier P11 terminal.

Contrast: block07's quantity-budget tail at global190464/P12 was **not terminal**:
done0, reason null, bootstrap allowed, terminal event0, Phi=.6187591886612673.
Its tail reward=-.03187956660985947 and return=-28.808279037475586 included the
normal nonterminal tail value. That earlier budget boundary did not supply the
explicit failure credit now present at193551.

Sources: block10 run
`20260921T1043344171428Z_g97ecd305afb5_0168bb1943e74402accb92ea31b10de3`,
terminal `residual_and_projection_audit.jsonl` line1039, first reset line1040,
`rollouts/rollout_001478.pt`, `rollouts/update_001478_likelihood.json` and the
`advantage_audit.jsonl` record whose `ppo_update_intended` is1478.
Old contrast is block07 run `20260921T0849405935866Z_gee5a9651591d_12b59026033c4ad08c274c684926dadc`,
decision line512 and rollout001453. No optimizer, forward, physics, production
edit or extra gate was performed by this small audit. No claim that this one
update alone learns successful placement or propagates a complete Monte Carlo
terminal return through earlier already-updated rollout blocks.
