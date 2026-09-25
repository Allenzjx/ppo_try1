# 128-decision credit horizon: prior sealed P07/P10 only

No strong causal evidence that 128 alone caused RR/RL failure: the first learner RR capture/loss/RL lift and most repeated gain/loss transitions occur within an optimized rollout; ordinary phase changes are nonterminal. There is a real bootstrap dependence and limited outcome coverage: both 384-decision rear blocks end nonterminal, and no numerical same-policy tail V or terminal outcome is logged. Longer continuous rear collection is supported; changing rollout128 to256 is a separate testable variance/credit choice, not an established fix. Previously proven issued-owner persistence, current-support loss and saturated action/proxy coverage are more direct implementation/state issues. Test their already-versioned correction before attributing residual failure to horizon.

Actual gamma=0.9985, lambda=0.99; 128 decisions=8.533333s. GAE TD-residual trace half-life=4.000s; weight at128=0.227960, at256=0.051966. These are formula weights, not a measured alternative-run result.

| Run/update | physical window s | observed rewards sum | discounted rewards only | mean old V | mean return target | raw GAE + / - |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| P07/1710 | 43.000–51.533 | -0.033699 | 0.003222 | -12.77136 | -11.80353 | 116/12 |
| P07/1711 | 51.533–60.067 | -0.521830 | -0.485088 | -12.83410 | -12.23073 | 127/1 |
| P07/1712 | 60.067–68.600 | -0.985634 | -0.901796 | -13.42468 | -12.97403 | 114/14 |
| P10/1713 | 51.133–59.667 | -0.815817 | -0.775103 | -12.76228 | -12.02315 | 124/4 |
| P10/1714 | 59.667–68.200 | -1.091849 | -1.002778 | -12.85813 | -12.66809 | 78/50 |
| P10/1715 | 68.200–76.733 | -0.740865 | -0.673109 | -13.05828 | -12.28697 | 126/2 |

## Actual candidate-event examples

Reward and standardized advantage are original logged values. They are not relabeled from contact success/failure. Nonterminal raw GAE remains unavailable per row.

| Run/tick | RR contact / force N | RL qualified | reward | old V | actual std advantage | rows left in rollout incl. this |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| P07/5688 | TOP / 0.364 | False | 0.181821 | -13.25738 | 0.50497 | 63 |
| P07/5696 | AIR / 0.000 | False | -0.193919 | -12.61950 | -0.66799 | 62 |
| P07/5776 | TOP / 15.097 | True | 0.225874 | -12.76160 | -0.65626 | 52 |
| P07/5808 | TOP / 13.501 | False | -0.223915 | -12.39799 | -1.44177 | 48 |
| P07/5824 | AIR / 0.000 | False | -0.230162 | -12.19539 | -1.45587 | 46 |
| P07/6208 | TOP / 4.164 | False | 0.189769 | -12.39219 | -0.46214 | 126 |
| P07/6232 | AIR / 0.000 | False | -0.214342 | -12.44029 | -0.98159 | 123 |
| P10/6144 | TOP / 15.338 | False | -0.006328 | -14.25587 | 3.17155 | 128 |
| P10/6208 | AIR / 0.000 | False | -0.596615 | -13.91800 | 1.76621 | 120 |
| P10/7192 | TOP / 14.180 | True | -0.001389 | -12.88889 | -0.02631 | 125 |
| P10/7304 | TOP / 11.845 | False | -0.396314 | -12.97152 | 0.16101 | 111 |
| P10/7392 | AIR / 0.000 | False | -0.119344 | -12.89866 | 0.97098 | 100 |
| P10/7496 | GROUND / 49.517 | False | -0.005915 | -12.70401 | 0.67714 | 87 |

## What this does and does not support

P07 learner RR placement5682 and RL qualification5772 both precede the first course update1710; the resulting capture/loss/lift transitions are already in its first128 samples. P10 initial RR placement6133 is in the zero-credit N prefix, while later contact losses and RL attempts are learner samples.
P07 sampled contact/qualification transitions within/across rollout boundaries: 30/0; P10: 12/0. These are endpoint transition brackets, not all120Hz events.
Both runs stop after384 decisions/25.6s of learner collection, with task incomplete and no terminal. Therefore there is no observed terminal return to propagate across the full rear task. All six128 tails bootstrap. Last pre-action V is not the unlogged end-state V; next-block V is measured after a critic update.
Keep the distinction: extending continuous collection to a real rear outcome addresses demonstrated missing outcome coverage; increasing rollout length may reduce reliance on a tail bootstrap but is not proven to repair the already-observed owner/state/action-limit problems.
Source evidence: cooperative_sealed384_summary.md, cooperative_P10_384_summary.md, their exact run journals plus six advantage/optimizer/likelihood records. No current active probe, other history, tensor, or model was read.
