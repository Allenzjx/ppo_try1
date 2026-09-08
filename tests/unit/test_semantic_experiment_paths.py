"""Artifact isolation and the narrowly reviewed transfer-role encoder boundary."""
from __future__ import annotations

import copy
import json
import os
import subprocess
from pathlib import Path

import pytest

from wlr50_clean.ppo import semantic_cli as cli
from wlr50_clean.ppo import semantic_migration as migration
from wlr50_clean.ppo.semantic_policy_distribution import HISTORY_POLICY, policy_contract


ROOT = cli.PROJECT_ROOT
EXPERIMENT = "transfer_roles_v1"
GROUPS = ("previous_residual_full12", "previous_previous_residual_full12")


def schema_with_scale(scale):
    schema = json.loads((ROOT / "configs/ppo_semantic_v3/observation_schema.json").read_text())
    # This suite covers the original324 scale boundary, not the separately
    # reviewed role append. Keep its old encoder explicit as production grows.
    schema.pop("transfer_role_features_version", None)
    schema["feature_groups"] = [group for group in schema["feature_groups"]
                                if group["name"] != "transfer_role_context_full48"]
    for group in schema["feature_groups"]:
        if group["name"] in GROUPS:
            group["scale"][3] = scale
    assert sum(group["size"] for group in schema["feature_groups"]) == 324
    return schema


def publish_metadata(path, contract, *, version="v3"):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Metadata-only route tests do not claim a tensor round-trip of these bytes.
    path.write_bytes(b"immutable test checkpoint; no physical or tensor execution")
    data = {
        "schema": "wlr50_clean.semantic_checkpoint.v1", "checkpoint_path": str(path.resolve()),
        "checkpoint_sha256": migration.file_sha(path), "save_load_round_trip": True,
        "semantic_version": version, "seed": 1001, "runtime_contract": contract,
        "runner_config": cli.semantic_runner_config(seed=1001, device="cpu", semantic_version=version,
                                                     policy_version=HISTORY_POLICY),
        "policy_contract": policy_contract(HISTORY_POLICY),
        "global_policy_decisions": 123136, "ppo_updates": 927, "optimizer_steps": 18540,
        "new_mdp_origin_global_policy_decisions": 10112,
        "stage_requested_decisions": {"smoke": 0, "phase_suffix": 55424, "full_episode": 57600},
    }
    path.with_name(path.stem + "_manifest.json").write_text(json.dumps(data), encoding="utf-8")
    return data


def arguments(tmp_path, checkpoint=None, *, warm=False, phase="P01", experiment=EXPERIMENT):
    namespace = "ppo_transfer_roles_v1" if experiment else "ppo_semantic_v3"
    values = ["train", "--semantic-version", "v3", "--expected-head", "a" * 40,
              "--run-dir", str(tmp_path / f"runs/{namespace}/train/test"), "--device", "cpu",
              "--stage", "full_episode" if phase == "P01" else "phase_suffix",
              "--from-phase", phase, "--decisions", "128"]
    if experiment:
        values += ["--experiment-id", experiment]
    if checkpoint:
        values += ["--checkpoint", str(checkpoint)]
    if warm:
        values += ["--new-mdp-warm-start"]
    return cli.parser().parse_args(values)


@pytest.fixture
def routed(tmp_path, monkeypatch):
    # CLI preflight now reads the actual target schema before resolving the
    # actor. A routed old324 checkpoint needs a real matching old324 fixture.
    observation = tmp_path / "configs/ppo_semantic_v3/observation_schema.json"
    observation.parent.mkdir(parents=True)
    observation.write_text(json.dumps(schema_with_scale(6)), encoding="utf-8")
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(cli, "OUTPUT_ROOT", tmp_path / "outputs/ppo_semantic_v2")
    monkeypatch.setattr(cli, "RUNS_ROOT", tmp_path / "runs/ppo_semantic_v2")
    return tmp_path


def test_namespace_keeps_v3_configuration_and_historical_defaults(routed):
    assert cli.version_paths("v3", experiment_id=EXPERIMENT) == (
        routed / "runs/ppo_transfer_roles_v1", routed / "outputs/ppo_transfer_roles_v1",
        routed / "configs/ppo_semantic_v3")
    for version in ("v2", "v3"):
        assert cli.version_paths(version) == tuple(routed / f"{kind}/ppo_semantic_{version}"
                                                  for kind in ("runs", "outputs", "configs"))
    with pytest.raises(ValueError):
        cli.version_paths("v2", experiment_id=EXPERIMENT)
    with pytest.raises(ValueError):
        cli.version_paths("v3", experiment_id="../elsewhere")


@pytest.mark.parametrize("phase", ["P03", "P04", "P05", "P06"])
def test_new_mdp_imports_existing_v3_source_without_resetting_budget(routed, phase):
    source = routed / "outputs/ppo_semantic_v3/checkpoints/history/checkpoint_step_000123136.pt"
    publish_metadata(source, {"semantic_version": "v3"})
    args = arguments(routed, source, warm=True, phase=phase)
    cli.validate_request(args)
    assert args.checkpoint == source and args.decisions == 128 and args.from_phase == phase
    args.decisions = cli.STAGE_BUDGETS[args.stage]
    with pytest.raises(ValueError, match="remaining"):
        cli.validate_request(args)


@pytest.mark.parametrize("warm", [False, True])
def test_prior_v2_root_is_never_an_experiment_source(routed, warm):
    source = routed / "outputs/ppo_semantic_v2/checkpoints/history/source.pt"
    publish_metadata(source, {"semantic_version": "v2"})
    with pytest.raises(ValueError, match="checkpoint"):
        cli.validate_request(arguments(routed, source, warm=warm))


def test_ordinary_resume_requires_new_root_and_matching_contract(routed):
    old = routed / "outputs/ppo_semantic_v3/checkpoints/history/source.pt"
    publish_metadata(old, {"semantic_version": "v3"})
    with pytest.raises(ValueError, match="checkpoint"):
        cli.validate_request(arguments(routed, old))
    target = routed / "outputs/ppo_transfer_roles_v1/checkpoints/history/source.pt"
    contract = {"semantic_version": "v3", "experiment_id": EXPERIMENT}
    publish_metadata(target, contract)
    args = arguments(routed, target)
    cli.validate_request(args)
    cli._preflight_checkpoint(args, contract)
    assert args._policy_version == HISTORY_POLICY and args._warm_start_record is None
    with pytest.raises(ValueError, match="runtime changed"):
        cli._preflight_checkpoint(args, {"semantic_version": "v3"})
    args.run_dir = routed / "runs/ppo_semantic_v3/train/wrong"
    with pytest.raises(ValueError, match="run"):
        cli.validate_request(args)


@pytest.fixture
def boundary(tmp_path, monkeypatch):
    names = ("stage_task_spec.yaml", "execution_profile.yaml", "reward_config.yaml",
             "observation_schema.json", "action_schema.json", "quality_score.yaml")
    files = {}
    for name in names:
        relative = f"configs/ppo_semantic_v3/{name}"
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / relative).read_bytes())
        if name == "observation_schema.json":
            target.write_text(json.dumps(schema_with_scale(4)), encoding="utf-8")
        files[relative] = migration.file_sha(target)
    def git(*args):
        return subprocess.run(["git", "-C", str(tmp_path), *args], check=True, capture_output=True, text=True).stdout.strip()
    git("init", "-q")
    git("config", "user.name", "Metadata test")
    git("config", "user.email", "metadata@example.invalid")
    git("config", "core.autocrlf", "false")
    git("add", "configs")
    git("commit", "-qm", "immutable source configuration")
    old = {"semantic_version": "v3", "source_git_commit": git("rev-parse", "HEAD"),
           "files": files, "runtime_content_sha256": migration.digest(files),
           "physics_hz": 120.0, "decision_hz": 15.0, "task_timeout_s": 200.0,
           "timeout_bootstrap": False, "frozen_A_files": {"frozen": "f" * 64},
           "rsl_rl_version": "5.0.1", "local_runtime_versions": {"test": "CPU"}}
    source = tmp_path / "outputs/ppo_semantic_v3/checkpoints/history/checkpoint_step_000123136.pt"
    publish_metadata(source, old)
    relative = "configs/ppo_semantic_v3/observation_schema.json"
    (tmp_path / relative).write_text(json.dumps(schema_with_scale(6)), encoding="utf-8")
    new = copy.deepcopy(old)
    new.update(experiment_id=EXPERIMENT, source_git_commit="b" * 40)
    new["files"][relative] = migration.file_sha(tmp_path / relative)
    new["runtime_content_sha256"] = migration.digest(new["files"])
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(cli, "OUTPUT_ROOT", tmp_path / "outputs/ppo_semantic_v2")
    return tmp_path, source, old, new


def test_real_metadata_migration_binds_historical_git_scales_and_initial_namespace(boundary):
    root, source, old, new = boundary
    original = source.read_bytes()
    record = migration.build_v3_warm_start_record(source, new, project_root=root)
    assert record["target_stage_requested_decisions"] == {"smoke": 0, "phase_suffix": 55424, "full_episode": 57600}
    assert record["new_mdp_origin_global_policy_decisions"] == 10112
    assert record["experiment_transition"]["source_artifact_namespace"] == "ppo_semantic_v3"
    assert record["experiment_transition"]["target_artifact_namespace"] == "ppo_transfer_roles_v1"
    scale = record["observation_scale_transition"]
    assert scale["columns"] == [210, 222] and scale["factors"] == [1.5, 1.5] and scale["changed"] is True
    assert scale["source_schema_sha256"] == old["files"]["configs/ppo_semantic_v3/observation_schema.json"]
    assert record["optimizer"]["initial_learning_rate"] == 3e-5
    assert record["old_rollout_buffer_inherited"] is False and "policy_kernel_transition" not in record
    args = arguments(root, source, warm=True)
    cli.validate_request(args)
    cli._preflight_checkpoint(args, new)
    assert args._policy_version == HISTORY_POLICY
    initial_name = migration.v3_warm_start_checkpoint_name(record)
    (source.parent / initial_name).write_bytes(b"old namespace must not collide")
    cli._preflight_checkpoint(args, new)
    target = root / "outputs/ppo_transfer_roles_v1/checkpoints/history" / initial_name
    target.parent.mkdir(parents=True)
    target.write_bytes(b"immutable experiment initial")
    with pytest.raises(ValueError, match="already exists"):
        cli._preflight_checkpoint(args, new)
    assert source.read_bytes() == original


def test_scale_exception_not_available_without_experiment_or_corrupt_source(boundary):
    root, source, old, new = boundary
    ordinary = copy.deepcopy(new)
    ordinary.pop("experiment_id")
    with pytest.raises(ValueError, match="preprocessing"):
        migration.build_v3_warm_start_record(source, ordinary, project_root=root)
    broken = copy.deepcopy(new)
    broken["physics_hz"] = 60.0
    with pytest.raises(ValueError, match="physical/runtime"):
        migration.build_v3_warm_start_record(source, broken, project_root=root)
    source.write_bytes(b"corruption")
    with pytest.raises(ValueError, match="integrity"):
        migration.build_v3_warm_start_record(source, new, project_root=root)


def test_actual_runtime_inventory_and_preflight_bind_experiment_without_native_launch(boundary, monkeypatch):
    root, _, _, _ = boundary
    relative = "src/wlr50_clean/ppo/semantic_transfer_roles.py"
    module = root / relative
    module.parent.mkdir(parents=True)
    module.write_text("# synthetic inventory source, not actual role implementation\n")
    frozen = root / "artifacts/ppo_phase_v1_start/frozen_fsm_hashes.json"
    frozen.parent.mkdir(parents=True)
    frozen.write_text(json.dumps({"algorithm": "sha256", "protected_files": {}}))
    for args in (("add", "configs", "src", "artifacts"), ("commit", "-qm", "target inventory")):
        subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)
    head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                          check=True, capture_output=True, text=True).stdout.strip()
    monkeypatch.setattr(cli, "local_versions", lambda: {"test": "no native imports"})
    monkeypatch.setattr(cli, "dispatch_live", lambda *a: pytest.fail("preflight must not launch Isaac"))
    contract = cli.runtime_contract(expected_head=head, semantic_version="v3", experiment_id=EXPERIMENT)
    assert contract["files"][relative] == migration.file_sha(module)
    assert contract["selected_configuration"]["reward_config.yaml"]["path"] == "configs/ppo_semantic_v3/reward_config.yaml"
    old_default = cli.runtime_contract(expected_head=head, semantic_version="v3")
    assert "experiment_id" not in old_default
    run = root / "runs/ppo_transfer_roles_v1/interface_checks/preflight"
    assert cli.main(["preflight", "--run-dir", str(run), "--expected-head", head,
                     "--semantic-version", "v3", "--experiment-id", EXPERIMENT]) == 0
    manifest = json.loads((run / "run_manifest.json").read_text())
    assert manifest["runtime_contract"] == contract
    assert manifest["result"]["isaac_started"] is False
    module.write_text("# changed source\n")
    with pytest.raises(ValueError, match="clean and committed"):
        cli.runtime_contract(expected_head=head, semantic_version="v3", experiment_id=EXPERIMENT)


@pytest.mark.parametrize("mutation", ["other_column", "clip", "order", "mixed", "wrong_target", "dimension"])
def test_two_column_exception_rejects_every_other_encoder_change(mutation):
    old, new = schema_with_scale(4), schema_with_scale(6)
    groups = {group["name"]: group for group in new["feature_groups"]}
    if mutation == "other_column":
        groups[GROUPS[0]]["scale"][2] = 6
    elif mutation == "clip":
        new["clip"] = 21
    elif mutation == "order":
        new["feature_groups"][0], new["feature_groups"][1] = new["feature_groups"][1], new["feature_groups"][0]
    elif mutation == "mixed":
        next(g for g in old["feature_groups"] if g["name"] == GROUPS[1])["scale"][3] = 6
    elif mutation == "wrong_target":
        groups[GROUPS[0]]["scale"][3] = 7
    else:
        groups[GROUPS[0]]["size"] = 13
    with pytest.raises(ValueError):
        migration._transfer_observation_scale_transition(old, new, {"source_sha256": "a", "target_sha256": "b"})


def test_already_migrated_scales_do_not_repeat_compensation():
    schema = schema_with_scale(6)
    record = migration._transfer_observation_scale_transition(schema, schema, {"source_sha256": "a", "target_sha256": "a"})
    assert record["factors"] == [1.0, 1.0] and record["changed"] is False


def test_actual_windows_powershell_param_binding_routing_and_global_lock():
    powershell = Path(os.environ.get("SystemRoot", "C:/Windows")) / "System32/WindowsPowerShell/v1.0/powershell.exe"
    if not powershell.exists():
        pytest.skip("native Windows PowerShell is unavailable")
    path = str(ROOT / "scripts/run_semantic_ppo.ps1").replace("'", "''")
    command = r"""
$tokens = $null; $errors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile('__PATH__',[ref]$tokens,[ref]$errors)
if ($errors.Count) { throw 'production parser errors' }
$assign = @($ast.FindAll({param($n) $n -is [System.Management.Automation.Language.AssignmentStatementAst]},$true))
$selected = @('artifactNamespace','runDir','logDir','arguments','lockPath')
$source = @($assign | Where-Object { $_.Left -is [System.Management.Automation.Language.VariableExpressionAst] -and $_.Left.VariablePath.UserPath -cin $selected -and $_.Operator -eq 'Equals' } | ForEach-Object {$_.Extent.Text})
if ($source.Count -ne 5) { throw 'ambiguous production assignment selection' }
$guards = @($ast.FindAll({param($n) $n -is [System.Management.Automation.Language.IfStatementAst] -and $n.Extent.Text.StartsWith('if (-not [string]::IsNullOrWhiteSpace($ExperimentId)')},$true))
if ($guards.Count -ne 2) { throw 'ambiguous experiment guard/argument selection' }
$body = $ast.ParamBlock.Extent.Text + "`nSet-StrictMode -Version Latest`n" + $guards[0].Extent.Text + "`n" + '$project="C:\offline-routing"; $kind="train"; $runId="synthetic"' + "`n" + ($source -join "`n") + "`n" + $guards[1].Extent.Text + "`n" + '[pscustomobject]@{run=$runDir; log=$logDir; lock=$lockPath; argv=$arguments}'
$block = [scriptblock]::Create($body)
$values = @()
foreach ($phase in @('P03','P04','P05')) { $values += & $block -Command train -ExpectedHead ('a'*40) -SemanticVersion v3 -ExperimentId transfer_roles_v1 -FromPhase $phase }
$old = & $block -Command train -ExpectedHead ('a'*40) -SemanticVersion v3
$invalid = $false
try { & $block -Command train -ExpectedHead ('a'*40) -SemanticVersion v2 -ExperimentId transfer_roles_v1 | Out-Null } catch { $invalid = $true }
@{values=$values; old=$old; invalid=$invalid} | ConvertTo-Json -Depth 5 -Compress
""".replace("__PATH__", path)
    result = subprocess.run([str(powershell), "-NoProfile", "-NonInteractive", "-Command", command],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    assert data["invalid"] is True
    for row, phase in zip(data["values"], ("P03", "P04", "P05")):
        assert row["run"] == r"C:\offline-routing\runs\ppo_transfer_roles_v1\train\synthetic"
        assert row["lock"] == r"C:\offline-routing\runs\ppo_semantic_v2\.single_process.lock"
        assert row["argv"][row["argv"].index("--experiment-id") + 1] == EXPERIMENT
        assert row["argv"][row["argv"].index("--from-phase") + 1] == phase
    assert "ppo_semantic_v3" in data["old"]["run"] and "--experiment-id" not in data["old"]["argv"]
