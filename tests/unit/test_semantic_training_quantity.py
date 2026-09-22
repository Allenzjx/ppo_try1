"""Quantity scheduling does not alter entropy, policy or lifetime accounting."""
from __future__ import annotations
import ast
import inspect
import json
import pytest
from wlr50_clean.ppo import semantic_cli as cli, semantic_training as training

EXPERIMENT = "task_conditioned_hip_wheel_v1"
OLD = {"smoke": 10000, "phase_suffix": 100000, "full_episode": 100000}
NEW = {**OLD, "full_episode": 131072}


def test_explicit_task_quantity_ceiling_keeps_historical_defaults_immutable():
    assert training.STAGE_BUDGETS == OLD
    for experiment in (None, "fl_capture_quality_v1", "transfer_roles_v1", "undeclared"):
        assert training.training_quantity_budgets(experiment) == OLD
    result = training.training_quantity_budgets(EXPERIMENT)
    assert result == NEW
    result["full_episode"] = 0
    assert training.training_quantity_budgets(EXPERIMENT) == NEW
    assert training.STAGE_BUDGETS == OLD
    with pytest.raises(ValueError):
        cli.runtime_contract(expected_head="0"*40, semantic_version="v3", experiment_id="undeclared")


def test_entropy_retains_original_210000_horizon_and_formula():
    tree = ast.parse(inspect.getsource(training.train_semantic))
    actual = next(n.value for n in ast.walk(tree) if isinstance(n, ast.Assign)
        and len(n.targets) == 1 and isinstance(n.targets[0], ast.Attribute)
        and n.targets[0].attr == "entropy_coef")
    expected = ast.parse(".005 + (.001-.005)*min(global_step/sum(STAGE_BUDGETS.values()),1.)", mode="eval").body
    assert ast.dump(actual, include_attributes=False) == ast.dump(expected, include_attributes=False)
    code = compile(ast.Expression(actual), "<actual entropy schedule>", "eval")
    assert sum(training.STAGE_BUDGETS.values()) == 210000
    for step in (0, 100000, 131072, 185856, 190464, 192000, 210000, 241072, 260000):
        value = eval(code, {"global_step": step, "STAGE_BUDGETS": training.STAGE_BUDGETS})
        assert value == .005 + (.001-.005)*min(step/210000, 1.)


@pytest.mark.parametrize("declared", [OLD, {}, {**NEW, "full_episode": 131073},
    {**NEW, "full_episode": 131072.0}, {**NEW, "phase_suffix": 100001}])
@pytest.mark.parametrize("experiment_id", [EXPERIMENT, "p05_hip_only_continuation_v1"])
def test_runtime_profile_budget_guard_requires_exact_explicit_declaration(tmp_path, declared, experiment_id):
    import yaml
    # Only the actual guard is isolated; this is not a substitute for runtime
    # inventory and checkpoint migration validation or a claim of physical execution.
    tree = ast.parse(inspect.getsource(cli.runtime_contract))
    guard = next(n for n in tree.body[0].body if isinstance(n, ast.If)
        and isinstance(n.test, ast.Compare)
        and any(isinstance(c, ast.Constant) and c.value == experiment_id
                for comparator in n.test.comparators for c in ast.walk(comparator)))
    code = compile(ast.Module(body=[guard], type_ignores=[]), "<actual profile guard>", "exec")
    profile = tmp_path / "execution_profile.yaml"
    scope = {"experiment_id": experiment_id, "config_root": tmp_path, "contract": {"training_budgets": NEW}}
    profile.write_text(yaml.safe_dump({"training_budgets": declared}), encoding="utf-8")
    with pytest.raises(ValueError, match="execution profile quantity budget"):
        exec(code, scope)
    profile.write_text(yaml.safe_dump({"training_budgets": NEW}), encoding="utf-8")
    exec(code, scope)


def test_cli_lifetime_ceiling_not_reset_or_inherited_by_other_experiment(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    def arguments(experiment, spent, requested):
        namespace = "ppo_" + experiment
        checkpoint = tmp_path / f"outputs/{namespace}/checkpoints/history/source.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"metadata/path-only test; not a tensor checkpoint")
        checkpoint.with_name(checkpoint.stem+"_manifest.json").write_text(json.dumps({
            "semantic_version": "v3", "stage_requested_decisions": {
                "smoke": 0, "phase_suffix": 100000, "full_episode": spent}}), encoding="utf-8")
        return cli.parser().parse_args(["train", "--semantic-version", "v3", "--experiment-id", experiment,
            "--run-dir", str(tmp_path/f"runs/{namespace}/train/test"), "--expected-head", "a"*40,
            "--checkpoint", str(checkpoint), "--stage", "full_episode", "--decisions", str(requested)])
    args = arguments(EXPERIMENT, 99968, 2048)
    cli.validate_request(args)
    assert not args.new_mdp_warm_start and args.from_phase == "P01" and args.decisions == 2048
    args.decisions = 31104
    cli.validate_request(args)
    args.decisions += 1
    with pytest.raises(ValueError, match="remaining"):
        cli.validate_request(args)
    with pytest.raises(ValueError, match="remaining"):
        cli.validate_request(arguments("fl_capture_quality_v1", 99968, 128))
    with pytest.raises(ValueError, match="stage budget"):
        cli.validate_request(arguments(EXPERIMENT, 0, 131073))
