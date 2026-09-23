"""One explicit CP220544 continuation; no training, physics, or pointer promotion."""
import json
from pathlib import Path
import subprocess
from wlr50_clean.ppo.semantic_cli import runtime_contract
from wlr50_clean.ppo.semantic_migration import checkpoint_metadata, file_sha
from wlr50_clean.ppo.semantic_rr_capture_migration import (
    build_rr_capture_migration, publish_rr_capture_checkpoint, TARGET_EXPERIMENT)

root = Path(__file__).resolve().parents[2]
out = Path(__file__).resolve().parent
source = root/'outputs/ppo_p05_hip_only_continuation_v1/checkpoints/history/checkpoint_step_000220544.pt'
assert file_sha(source) == '56239937cd0aaccdc8b7ea36c6266a41b3bed9df67b00f84a06806d2a6fa09b7'
head = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
contract = runtime_contract(expected_head=head, semantic_version='v3', experiment_id=TARGET_EXPERIMENT)
old = checkpoint_metadata(source)['runtime_contract']
code = {p: sha for p, sha in contract['files'].items()
        if old['files'].get(p) != sha and not p.startswith('configs/')}
plan = build_rr_capture_migration(source, contract,
    reason='Preserve CP220544 learning and front behavior; append observable bounded RR hip-only capture/current-support state. First real direction diagnostic, wheel/reward/source scheduling unchanged.',
    reviewed_code_sha256=code)
plan_path = out/'CP220544_to_RR410_migration.json'
with plan_path.open('x', encoding='utf-8') as f:
    json.dump(plan, f, indent=2, allow_nan=False)
target = out/'checkpoints/history/checkpoint_rr_capture_step_000220544.pt'
target.parent.mkdir(parents=True, exist_ok=True)
assert not target.exists()
receipt = publish_rr_capture_checkpoint(source, contract, plan_path, target)
receipt.update(checkpoint_sha256=file_sha(target), source_checkpoint_sha256=file_sha(source),
               source_git_commit=head, control_version='rr_hip_only_capture_v1',
               physical_evaluation='NOT_YET_EVALUATED', no_pointer_promotion=True)
with (out/'CP220544_RR410_publication.json').open('x', encoding='utf-8') as f:
    json.dump(receipt, f, indent=2, allow_nan=False)
print(json.dumps(receipt))
