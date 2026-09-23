# V6 same410 migration and isolated ancestor output routing

Prepared and CPU-tested only. No real checkpoint publication, optimizer learning credit, simulator, or main pointer change was performed by this preparation.

The immutable source HEAD is `c53119ab332fe048668e66f0543889a128443ca5`.

| Explicit source | Checkpoint SHA256 | Decisions / PPO / Adam |
| --- | --- | --- |
| ancestor | `308122a3c8e760733fbe8370bfdf399a25718148c82333deb41a1265e4e13895` | 220544 / 1688 / 33760 |
| latest | `ab7fec2a96d780336d696fb29fd1317309e5864e4f554ac9c3e292873bcf29d0` | 221184 / 1693 / 33860 |

Their sidecar hashes are pinned separately in `semantic_rr_contact_onset_migration.py`. Neither source borrows counters or state from the other. Original v5 receipts remain untouched. The v6 migration declares changed control semantics and unchanged same410 numerical codec; migration adds zero decisions, PPO, Adam, and AUX updates and discards old rollout storage. Actor, critic, full Adam, actual LR, Identity, RNG, origins, and AUX ancestry are preserved through official save and independent reload.

The exact eight allowed runtime changes are assist, execution profile, the new v6 migration module, semantic migration routing, semantic training, semantic CLI, and the two train/video PowerShell launchers. No old migration whitelist was weakened.

After the final reviewed commit, the prepared publication entry is:

```powershell
$env:PYTHONPATH='src'
& C:/Users/kskzz/miniconda3/envs/env_isaaclab/python.exe outputs/ppo_rr_capture_then_rl_transfer_v1/publish_rr_contact_onset_v6.py --expected-head <EXACT_NEW_HEAD> --source ancestor --publish
```

Use `--source latest` separately for the distinct latest learned model. The publisher uses the original runner device and full CUDA RNG visibility, unique version/source/head filenames, and never promotes the main pointer. Omitting `--publish` performs validation only.

An ancestor training request must explicitly use `-CheckpointOutputBranch ancestor220544_contact_v6` (Python: `--checkpoint-output-branch ancestor220544_contact_v6`). This isolates history, checkpoint-last copy/pointer, and resume state under `outputs/ppo_rr_capture_then_rl_transfer_v1/branches/ancestor220544_contact_v6`. Natural P01 and checkpoint-policy suffix training can resume only that same branch. An initial published ancestor can be evaluated from the immutable parent without a branch flag; training without the flag is rejected by both CLI and direct training API. First branch use requires a formally published v6 receipt, not an arbitrary v5 source. A populated branch cannot be silently restarted from its ancestor. The main latest path remains unchanged.

CPU results (CUDA hidden; OMP/MKL one thread):

- New v6 migration + output branch tests: **65 passed, 1 skipped** (11.43s). The skip is actual Windows symlink creation permission; a separate simulated resolved-alias rejection test passed.
- Historical v5 strict migration + CLI + RR410 training audit + checkpoint-prefix CLI tests: **53 passed** (10.94s).
- Full synthetic dual-source load/save/reload, unchanged same-input Gaussian/critic, lineage/full-state preservation, normal receipt carry, a real CPU synthetic 128-row/1-update/20-Adam branch run, main five checkpoint files/pointer unchanged, and no-flag rejection were exercised. CPU synthetic updates are not robot PPO learning.
- `git diff --check` passed. All CPU test processes exited.

Prepared file SHA256 values:

- New migration module: `1a40fb6f0c39d1874b08030e25327bec3d1fb81e2dcdc1b79ba75e31aeddbaf3`
- New migration test: `77ca7093de21cf388f5d10e61e795acf381eace1d4c761cc97795e6895d7c52e`
- Branch test: `16081aa5848426bd24da444f06d7d72c7865891c6083e42f86ab09155b3cbc73`
- Publisher: `b928aa3d4db70a308db122b6cf2d1f79f9ac5493d114364dcaec56ae9b47d9e9`
