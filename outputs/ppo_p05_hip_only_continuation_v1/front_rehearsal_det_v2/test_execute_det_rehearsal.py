"""Finite wrapper tests with synthetic CPU actors only; no real fit/physics."""
from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path

import pytest
import torch

import execute_det_rehearsal as m
from test_inspect_det_rehearsal import runner as initial_runner, states


@pytest.fixture(autouse=True)
def cpu_only():
    assert not torch.cuda.is_available(), "tests require CUDA_VISIBLE_DEVICES=-1"
    rng = torch.get_rng_state(); threads = torch.get_num_threads()
    torch.set_num_threads(1); torch.manual_seed(619)
    yield
    torch.set_rng_state(rng); torch.set_num_threads(threads)


def runner():
    run = initial_runner()
    # Synthetic fixture only: establish populated Adam moments before the AUX.
    for p in [*run.alg.actor.parameters(), *run.alg.critic.parameters()]:
        p.grad = torch.linspace(-.03, .05, p.numel()).reshape_as(p)
    run.alg.optimizer.step(); run.alg.optimizer.zero_grad(set_to_none=True)
    with torch.no_grad():
        run.alg.actor.mlp[4].weight[12:] += torch.linspace(-.03, .03, 3072).reshape(12, 256)
    return run


def budget(**changes):
    # Small synthetic test step, NOT the real proposed/root-authorized budget.
    return replace(m.kernel.Budget(2, .05, (3.,) * 8 + (.15,) * 4,
        (3.,) * 8 + (.15,) * 4, .05, .05), **changes)


def protected(run):
    return {"critic": m.training.parameter_hash(run.alg.critic),
        "Adam": m.training.state_hash(run.alg.optimizer.state_dict()),
        "Identity": m.training.state_hash(m.training._normalizers(run)),
        "RNG": m.training.capture_training_rng_state(seed=1001),
        "config": deepcopy(run._semantic_runner_config),
        "LR": (run.alg.learning_rate, m.training.optimizer_learning_rate(run))}


def test_only_P02_column_changes_and_all_training_state_preserved(monkeypatch):
    run = runner(); data = states(run.alg.actor)
    before = deepcopy(run.alg.actor.state_dict()); other = protected(run)
    def forbid(*args, **kwargs): raise AssertionError("PPO Adam step forbidden")
    monkeypatch.setattr(run.alg.optimizer, "step", forbid)
    report = m._fit_p02(run, data, budget=budget(), authorized=True)
    assert report["accepted_auxiliary_updates"] == 2
    assert 0 < report["actually_changed_scalar_count"] <= 256
    assert report["optimized_parameters"] == ["actor.mlp.0.weight[:,1]"]
    for key, value in run.alg.actor.state_dict().items():
        if key == "mlp.0.weight":
            assert torch.equal(value[:, 0], before[key][:, 0])
            assert torch.equal(value[:, 2:], before[key][:, 2:])
        else: assert torch.equal(value, before[key])
    assert protected(run) == other
    assert all(report["protected_entire_Gaussian_bitwise_equal"].values())
    assert report["P01_initial_gradient_exact_zero"]
    assert report["P02_mean_and_log_sigma_may_both_change"]
    assert report["nested_kernel_scope_is_capability_not_this_execution_scope"]
    assert report["actor_parameter_sha256_before"] != report["actor_parameter_sha256_after"]
    assert report["PPO_decisions_added"] == report["PPO_updates_added"] == report["PPO_optimizer_steps_added"] == 0
    assert report["source_observations_directly_saved389"] is False
    assert report["fresh_PPO_rollout_required"] and not report["physical_success_claimed"]


def test_first_trust_rejection_restores_all_actor_and_stops():
    run = runner(); data = states(run.alg.actor)
    actor = m.training.parameter_hash(run.alg.actor); other = protected(run)
    report = m._fit_p02(run, data, budget=budget(max_attempts=32,
        maximum_train_request_shift_full12=(1e-14,) * 12), authorized=True)
    assert report["attempted_auxiliary_optimizer_steps"] == 1
    assert report["accepted_auxiliary_updates"] == report["actually_changed_scalar_count"] == 0
    assert m.training.parameter_hash(run.alg.actor) == actor and protected(run) == other


@pytest.mark.parametrize("bad", ["authorization", "P01_train", "P03_validation", "P02_protection"])
def test_forbidden_scope_never_invokes_fit(monkeypatch, bad):
    run = runner(); data = states(run.alg.actor)
    if bad == "P01_train": data["train_observations"][0, :2] = torch.tensor([1., 0.])
    if bad == "P03_validation": data["validation_observations"][0, :3] = torch.tensor([0., 0., 1.])
    if bad == "P02_protection": data["invariance_observations"][0, :13] = 0; data["invariance_observations"][0, 1] = 1
    def forbid(*args, **kwargs): raise AssertionError("invalid scope reached fit")
    monkeypatch.setattr(m.kernel, "fit", forbid)
    with pytest.raises(ValueError): m._fit_p02(run, data, budget=budget(), authorized=bad != "authorization")


@pytest.mark.parametrize("bad", ["P01_column", "other_tensor"])
def test_unexpected_actor_mutation_restores_entire_actor(monkeypatch, bad):
    run = runner(); data = states(run.alg.actor)
    before = m.training.parameter_hash(run.alg.actor)
    def sabotage(*args, **kwargs):
        with torch.no_grad():
            if bad == "P01_column": run.alg.actor.mlp[0].weight[:, 0].add_(1.)
            else: run.alg.actor.mlp[0].bias.add_(1.)
        return {}
    monkeypatch.setattr(m.kernel, "fit", sabotage)
    with pytest.raises(ValueError): m._fit_p02(run, data, budget=budget(), authorized=True)
    assert m.training.parameter_hash(run.alg.actor) == before


@pytest.mark.parametrize("bad", ["binding", "automatic_options", "attempts", "no_budget_field"])
def test_explicit_budget_rejects_stale_or_automatic_values(monkeypatch, bad):
    envelope = {"schema": "wlr50_clean.explicit_det_P02_aux_budget.v2", "binding": {"synthetic": True},
                "budget": asdict(budget())}
    if bad == "binding": envelope["binding"] = {"stale": True}
    if bad == "automatic_options": envelope["retry_until_pass"] = True
    if bad == "attempts": envelope["budget"]["max_attempts"] = 33
    if bad == "no_budget_field": del envelope["budget"]["learning_rate"]
    monkeypatch.setattr(m, "read", lambda _: envelope)
    with pytest.raises((ValueError, TypeError)): m.budget_from_file("unused", {"synthetic": True})


def metadata():
    origin = dict(global_policy_decisions=0, ppo_updates=0, optimizer_steps=0)
    ledger = {"schema": m.LEDGER_SCHEMA, "events": [{"event_index": 1,
        "fit_report": {"accepted_auxiliary_updates": 32, "attempted_auxiliary_optimizer_steps": 32},
        "original_event1_not_rewritten": True}], "accepted_auxiliary_updates_total": 32,
        "attempted_auxiliary_optimizer_steps_total": 32, "training_lineage_label": "original preserved"}
    return {"seed": 1001, "runtime_contract": {"synthetic_CPU_fixture_not_real_physics": True},
        "global_policy_decisions": 128, "ppo_updates": 1, "optimizer_steps": 20,
        "stage_requested_decisions": {"smoke": 0, "full_episode": 128, "phase_suffix": 0},
        "stage": "full_episode", "sampling": "P01_full_task_only_initial_version",
        "execution_topology": m.inspection.semantic_migration.continuation_topology(
            "P01_full_task_only_initial_version", None, observation_layout=m.inspection.P05_CAPTURE_OBSERVATION_LAYOUT),
        "new_mdp_origin_global_policy_decisions": 0,
        "task_conditioned_hip_wheel_branch": {"counter_origin": deepcopy(origin),
            "auxiliary_mean_learning": {"accepted": 7, "attempted": 8}},
        "p05_capture_assist_branch": {"counter_origin": deepcopy(origin)},
        "capture_feedback_semantics_branch": {"counter_origin": deepcopy(origin)},
        "capture_feedback_semantics_migration": {"original": True},
        "p05_capture_assist_migration": {"original": True}, "rr_postcross_workspace_migration": {"original": True},
        "rr_postcross_workspace_branch": {"schema": "wlr50_clean.rr_postcross_workspace_same389.v1",
            "semantics": "current_qualified_RR_over_top_receiver_retirement_v1", "counter_origin": deepcopy(origin),
            "migration_added_updates": 0, "source_checkpoint_sha256": "a" * 64, m.LEDGER_KEY: ledger}}


def synthetic_binding(infos):
    return {"source_checkpoint": {"path": "SYNTHETIC.pt", "sha256": "b" * 64,
        "PPO_counters": {key: infos[key] for key in m.COUNTERS}}, "helpers": {"synthetic": "c" * 64}}


def synthetic_budget_receipt():
    return {"path": "SYNTHETIC_budget.json", "sha256": "d" * 64, "envelope": {"budget": asdict(budget())}}


def test_event2_preserves_event1_old_7_8_and_no_false_credit():
    run = runner(); data = states(run.alg.actor); infos = metadata(); before = deepcopy(infos)
    report = m._fit_p02(run, data, budget=budget(), authorized=True)
    out = m.append_event(infos, report=report, binding=synthetic_binding(infos),
        data_receipt=data["receipt"], budget_receipt=synthetic_budget_receipt())
    assert infos == before
    events = out["rr_postcross_workspace_branch"][m.LEDGER_KEY]["events"]
    assert events[0] == infos["rr_postcross_workspace_branch"][m.LEDGER_KEY]["events"][0]
    event = events[1]
    assert event["event_index"] == 2 and event["phase_scope"] == ["P02"]
    assert event["fit_report_sha256"] == m.digest(event["fit_report"])
    assert event["kind"] == "finite_supervised_reconstructed_deterministic_P02_raw_actions_not_PPO"
    assert event["PPO_counters_unchanged"] == {key: infos[key] for key in m.COUNTERS}
    assert out["task_conditioned_hip_wheel_branch"] == infos["task_conditioned_hip_wheel_branch"]
    assert out["rr_postcross_workspace_branch"][m.LEDGER_KEY]["accepted_auxiliary_updates_total"] == 34
    report["accepted_auxiliary_updates"] = 0
    with pytest.raises(ValueError): m.append_event(infos, report=report, binding=synthetic_binding(infos),
        data_receipt=data["receipt"], budget_receipt=synthetic_budget_receipt())


def test_synthetic_official_save_fresh_reload_with_all_state_preserved(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "OUT", tmp_path)
    run = runner(); data = states(run.alg.actor)
    source, _ = m.training.save_semantic_checkpoint(run, tmp_path / "synthetic_source.pt", metadata())
    meta = m.inspection.semantic_migration.checkpoint_metadata(source)
    infos = m.training.load_semantic_checkpoint(run, source, contract=meta["runtime_contract"], seed=1001)
    report = m._fit_p02(run, data, budget=budget(), authorized=True)
    saved = m.save_candidate(run, tmp_path / "checkpoint_aux_detP02_SYNTHETIC.pt", infos,
        data=data, report=report, binding=synthetic_binding(infos), budget_receipt=synthetic_budget_receipt())
    assert saved["independent_official_reload_verified"] and not saved["latest_pointer_published"]
    assert saved["source_device_preserved"] == "cpu"
    new = m.inspection.semantic_migration.checkpoint_metadata(Path(saved["path"]))
    assert new["training_rng_state"] == infos["training_rng_state"]
    assert new["optimizer_state_sha256"] == infos["optimizer_state_sha256"]
    assert new["normalizer_state_sha256"] == infos["normalizer_state_sha256"]
    assert new["critic_parameter_sha256"] == infos["critic_parameter_sha256"]
    assert tuple(new[k] for k in m.COUNTERS) == (128, 1, 20)
    assert len(new["rr_postcross_workspace_branch"][m.LEDGER_KEY]["events"]) == 2
    assert not (tmp_path / "checkpoint_last_pointer.json").exists()


def test_real_admission_hash_rejects_substitute_without_fit(monkeypatch, tmp_path):
    path = tmp_path / "false_admission.json"
    path.write_text(json.dumps({"result": m.ADMISSION_RESULT}), encoding="utf-8")
    def forbid(*args, **kwargs): raise AssertionError("admission check reached fit")
    monkeypatch.setattr(m.kernel, "fit", forbid)
    with pytest.raises(ValueError): m.validated_admission(path, {}, {}, path, path)


def test_no_execute_flag_stops_before_any_io_or_fit(monkeypatch):
    def forbid(*args, **kwargs): raise AssertionError("unapproved operation reached I/O/fit")
    monkeypatch.setattr(m, "_new_path", forbid); monkeypatch.setattr(m.kernel, "fit", forbid)
    args = []
    for name in ("checkpoint", "inspection-receipt", "admission", "data", "budget", "aux-checkpoint", "report"):
        args.extend(["--" + name, "unused"])
    args.extend(["--expected-head", "unused", "--expected-checkpoint-sha256", "unused", "--expected-policy-decisions", "0"])
    with pytest.raises(ValueError): m.main(args)
