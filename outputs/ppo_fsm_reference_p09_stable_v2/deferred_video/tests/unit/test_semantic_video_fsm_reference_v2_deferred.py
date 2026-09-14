"""DEFERRED 06716a88 video compatibility tests; NOT RUN.

Run against the reviewed applied package after Isaac stops, not mirrored sources.
Only the explicit CPU test imports Torch, with CUDA disabled. CPU evidence is not
a live video or physical task-success certificate.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import json
import os
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from wlr50_clean.ppo import semantic_cli as runtime_cli
from wlr50_clean.ppo import semantic_video as video, semantic_video_cli as cli
from wlr50_clean.ppo import semantic_migration as migration
from wlr50_clean.ppo import semantic_policy_distribution as policies
from wlr50_clean.ppo import semantic_training as training
from wlr50_clean.ppo.semantic_transfer_roles import ROLE_OBSERVATION_LAYOUT as LAYOUT
from test_semantic_runtime_identity_resume372 import (
    fixture372, refresh, changed, save_metadata, NAMES,
)

EXPERIMENTS = ("all_stage_acceptance_v1", "fsm_reference_p09_stable_v2")
REVIEW = {"reason": "Reviewed video routing/HISTORY372 loading only; control, physics and all six configs unchanged"}
VIDEO = "src/wlr50_clean/ppo/semantic_video.py"


def fixture_video372(root, *, experiment_id="fsm_reference_p09_stable_v2", add_video=False):
    source, old, new = fixture372(root)
    if experiment_id != "fsm_reference_p09_stable_v2":
        selected = {}
        for name in NAMES:
            relative = f"configs/ppo_{experiment_id}/{name}"
            destination = root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes((migration.PROJECT_ROOT / relative).read_bytes())
            value = migration.file_sha(destination)
            old["files"][relative] = new["files"][relative] = value
            selected[name] = {"path": relative, "sha256": value}
        for contract in (old, new):
            contract.update(experiment_id=experiment_id, selected_configuration=copy.deepcopy(selected))
    for relative in migration.VIDEO_FILES:
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("# original video fixture\n", encoding="utf-8")
        if not add_video:
            old["files"][relative] = migration.file_sha(destination)
        if relative == VIDEO:
            destination.write_text("# reviewed video fixture\n", encoding="utf-8")
        new["files"][relative] = migration.file_sha(destination)
    refresh(old)
    refresh(new)
    metadata = migration.checkpoint_metadata(source)
    metadata["runtime_contract"] = old
    save_metadata(source, metadata)
    return source, old, new


def video_plan(root, source, old, new, **kwargs):
    return migration.build_migration_plan(source, new, allowed_changed_files=changed(old, new),
        reason="Explicit same-MDP video compatibility boundary", video_review=REVIEW,
        project_root=root, **kwargs)


@pytest.mark.parametrize("experiment", EXPERIMENTS)
@pytest.mark.parametrize("add_video", [False, True])
def test_video372_requires_its_own_exact_factor_and_all_six_unchanged_configs(tmp_path, experiment, add_video):
    source, old, new = fixture_video372(tmp_path, experiment_id=experiment, add_video=add_video)
    original_bytes = source.read_bytes()
    record = video_plan(tmp_path, source, old, new)
    factor = record["video_instrumentation_factor"]["observation_contract"]
    assert factor["schema"] == "wlr50_clean.video_same_observation_contract.v1"
    assert factor["source_policy_contract"] == factor["target_policy_contract"] == policies.policy_contract(
        policies.HISTORY_POLICY, observation_layout=LAYOUT)
    assert factor["selected_configuration"] == old["selected_configuration"] == new["selected_configuration"]
    assert set(factor["selected_configuration"]) == set(NAMES)
    assert factor["parameter_mapping"] is None and factor["num_envs"] == 1
    assert (record["observation_dimension"], record["action_dimension"]) == (372, 12)
    assert "instrumentation_observation_contract" not in record and "execution_factor" not in record
    assert record["preserve_actor_critic_optimizer_normalizer_rng_and_budget"] is True
    assert record["discard_old_rollout_storage"] is True
    path = tmp_path / "video_plan.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    assert migration.validate_migration_plan(source, new, path, project_root=tmp_path)[
        "video_instrumentation_factor"] == record["video_instrumentation_factor"]
    assert source.read_bytes() == original_bytes


def test_existing_instrumentation_only_factor_is_not_replaced_or_widened(tmp_path):
    source, old, new = fixture372(tmp_path)
    record = migration.build_migration_plan(source, new, allowed_changed_files=changed(old, new),
        reason="Existing instrumentation scope", project_root=tmp_path)
    assert "video_instrumentation_factor" not in record
    assert record["instrumentation_observation_contract"] == migration._instrumentation_observation_contract(
        migration.checkpoint_metadata(source), old, new, changed(old, new), tmp_path, mixed_factors=False)
    assert not (migration.VIDEO_FILES & migration.INSTRUMENTATION_FILES)


def test_video_delta_without_explicit_video_review_is_still_rejected(tmp_path):
    source, old, new = fixture_video372(tmp_path)
    with pytest.raises(ValueError, match="cannot mix"):
        migration.build_migration_plan(source, new, allowed_changed_files=changed(old, new),
            reason="Not video authorization", project_root=tmp_path)


@pytest.mark.parametrize("other", ["prior_evidence", "qualification_evidence", "evaluator_review", "execution_evidence"])
def test_video_review_cannot_mix_task_or_execution_permission(tmp_path, other):
    source, old, new = fixture_video372(tmp_path)
    with pytest.raises(ValueError, match="separate reviewed boundaries"):
        video_plan(tmp_path, source, old, new, **{other: {}})


@pytest.mark.parametrize("protected", [
    "src/wlr50_clean/ppo/semantic_supervisor.py",
    "src/wlr50_clean/ppo/semantic_backend.py",
    "src/wlr50_clean/ppo/semantic_observation.py",
    "src/wlr50_clean/ppo/semantic_reward.py",
    *("configs/ppo_fsm_reference_p09_stable_v2/" + name for name in NAMES),
])
def test_video_review_cannot_change_protected_runtime_or_any_config(tmp_path, protected):
    source, old, new = fixture_video372(tmp_path)
    new["files"][protected] = "b" * 64
    refresh(new)
    with pytest.raises(ValueError, match="cannot change"):
        video_plan(tmp_path, source, old, new)


@pytest.mark.parametrize("fault", [
    "missing_contract", "wrong_layout", "mixed_runner", "topology",
    "missing_action_binding", "extra_binding", "wrong_action_path", "wrong_schema_marker",
])
def test_video372_never_inferred_from_width_or_incomplete_bindings(tmp_path, fault):
    source, old, new = fixture_video372(tmp_path)
    metadata = migration.checkpoint_metadata(source)
    if fault == "missing_contract":
        del metadata["policy_contract"]
    elif fault == "wrong_layout":
        metadata["policy_contract"]["observation_layout"] = "unverified372"
    elif fault == "mixed_runner":
        metadata["runner_config"]["actor"].pop("observation_layout")
    elif fault == "topology":
        metadata["execution_topology"]["num_envs"] = 8
    elif fault in ("missing_action_binding", "extra_binding", "wrong_action_path"):
        for contract in (old, new):
            selected = contract["selected_configuration"]
            if fault == "missing_action_binding":
                del selected["action_schema.json"]
            elif fault == "extra_binding":
                selected["extra"] = {"path": "extra", "sha256": "a" * 64}
            else:
                selected["action_schema.json"]["path"] = "unverified/action_schema.json"
    else:
        relative = "configs/ppo_fsm_reference_p09_stable_v2/observation_schema.json"
        path = tmp_path / relative
        schema = json.loads(path.read_text())
        schema.pop("transfer_role_features_version")
        path.write_text(json.dumps(schema), encoding="utf-8")
        for contract in (old, new):
            contract["files"][relative] = migration.file_sha(path)
            contract["selected_configuration"]["observation_schema.json"]["sha256"] = contract["files"][relative]
    refresh(old)
    refresh(new)
    metadata["runtime_contract"] = old
    save_metadata(source, metadata)
    with pytest.raises(ValueError):
        video_plan(tmp_path, source, old, new)


@pytest.mark.parametrize("experiment", EXPERIMENTS)
def test_explicit_video_routing_and_replay_bind_all_six_current_config_files(tmp_path, monkeypatch, experiment):
    _, _, new = fixture_video372(tmp_path, experiment_id=experiment)
    monkeypatch.setattr(runtime_cli, "PROJECT_ROOT", tmp_path)
    configs = video.video_configuration("v3", experiment_id=experiment)
    assert len(configs) == 6 and {path.name for path in configs.values()} == set(NAMES)
    assert all(path.parent == tmp_path / f"configs/ppo_{experiment}" for path in configs.values())
    source = {"semantic_version": "v3", "experiment_id": experiment, "runtime_contract": new,
              "evaluation_configuration": {key: video.file_record(path) for key, path in configs.items()}}
    assert video.validate_video_configuration(source) == configs
    bad = copy.deepcopy(source)
    del bad["evaluation_configuration"]["action_schema_path"]
    with pytest.raises(video.SemanticVideoError, match="configuration"):
        video.validate_video_configuration(bad)
    bad = copy.deepcopy(source)
    bad["runtime_contract"]["selected_configuration"].pop("action_schema.json")
    with pytest.raises(video.SemanticVideoError, match="six"):
        video.validate_video_configuration(bad)
    bad = copy.deepcopy(source)
    bad["experiment_id"] = None
    with pytest.raises(video.SemanticVideoError, match="experiment"):
        video.validate_video_configuration(bad)
    # The action schema is actually byte-bound, not just named in a manifest.
    action_path = configs["action_schema_path"]
    action_path.write_text(action_path.read_text() + "\n ", encoding="utf-8")
    with pytest.raises(video.SemanticVideoError, match="path/hash"):
        video.validate_video_configuration(source)


@pytest.mark.parametrize("experiment", EXPERIMENTS)
def test_current_A_reader_recipe_and_identical_BC_config_routing(monkeypatch, experiment):
    from wlr50_clean.ppo import isaac_fsm_backend as frozen, residual_direct_env
    from wlr50_clean.ppo import semantic_backend, semantic_env, semantic_physical_sensing
    @dataclass
    class Dependencies:
        reader_from_scene: object
        frozen_controller_marker: object
    marker = object()
    original = Dependencies(lambda *a, **k: "legacy", marker)
    monkeypatch.setattr(frozen, "_load_live_dependencies", lambda: original)
    monkeypatch.setattr(frozen, "IsaacFSMBackend", lambda app, **kw: NS(options=kw))
    monkeypatch.setattr(residual_direct_env, "ResidualEpisodeEnv", lambda backend, **kw: NS(backend=backend, options=kw))
    monkeypatch.setattr(semantic_physical_sensing.SemanticSensorReader, "from_live_scene",
                        lambda scene, adapter, **kw: (scene, adapter, kw))
    baseline = cli.build_video_core(None, role="A", semantic_version="v3", experiment_id=experiment)
    dependencies = baseline.backend.options["dependencies"]
    assert dependencies is not original and dependencies.frozen_controller_marker is marker
    assert dependencies.reader_from_scene(1, 2, 3) == (1, 2, {"backends": 3})
    monkeypatch.setattr(semantic_backend, "SemanticIsaacBackend", lambda app, **kw: NS(options=kw))
    monkeypatch.setattr(semantic_env, "SemanticEpisodeEnv", lambda backend, **kw: NS(backend=backend, options=kw))
    baseline_options = None
    for role in ("B", "C"):
        core = cli.build_video_core(None, role=role, semantic_version="v3", experiment_id=experiment)
        configs = video.video_configuration("v3", experiment_id=experiment)
        assert core.backend.options["execution_profile"] == configs["execution_profile"]
        assert core.backend.options["task_spec_path"] == configs["task_spec_path"]
        assert core.options["action_config"] == configs["execution_profile"]
        assert core.options["observation_schema_path"] == configs["observation_schema_path"]
        assert core.options["reward_config_path"] == configs["reward_config_path"]
        options = (core.backend.options, core.options)
        assert baseline_options is None or baseline_options == options
        baseline_options = options


def test_default_configuration_shape_and_physical_context_limits_remain_unchanged():
    for version in ("v2", "v3"):
        configs = video.video_configuration(version)
        assert len(configs) == 5 and "action_schema_path" not in configs
    assert (video.PRE_TICKS, video.POST_TICKS, video.MAX_FRAMES) == (64, 184, 3000)
    assert video.frames_for_episode(23751) == 3000 and video.frames_for_episode(23752) == 3001


@pytest.mark.parametrize("terminal,expected", [
    ({"success": True, "termination_reason": None}, "SUCCESS"),
    ({"success": False, "termination_reason": "TASK_TIMEOUT"}, "TASK_TIMEOUT"),
])
def test_video_context_cannot_shorten_the_real_200s_task_endpoint(monkeypatch, terminal, expected):
    captured = []
    monkeypatch.setattr(video, "capture_task_interval_frame", lambda recorder, backend, tick: captured.append(tick))
    physical = NS(evaluator=NS(snapshot={"success": False, "termination_reason": None}),
                  observe=lambda *args: None)
    observer = video.EndpointObserver(physical, object(), object(), task_window=True)
    # Old video aborted here solely to reserve 64+184 context ticks. The common
    # task is still live; packaging cannot truncate it or relabel its outcome.
    observer(None, NS(physics_tick=23752), None)
    assert observer.reason is None and captured == [23752]
    physical.evaluator.snapshot = terminal
    with pytest.raises(video._PhysicalEndpoint):
        observer(None, NS(physics_tick=24000), None)
    assert observer.reason == expected and observer.last_frame.physics_tick == 24000


@pytest.mark.parametrize("fault", [
    {"from_phase": "P06"}, {"num_envs": 8}, {"new_mdp_warm_start": True},
    {"teacher_offset_decisions": 1}, {"max_decisions": 128}, {"seed": 1001},
])
def test_video_entry_remains_natural_single_P01_episode_no_prefix_or_warm_start(monkeypatch, fault):
    monkeypatch.setattr(cli, "validate_request", lambda args: None)  # Isolate existing video-only checks.
    args = NS(policy_distribution_migration=False, command="eval", seed=4001, headless=False,
        max_decisions=3000, decisions=None, num_envs=1, from_phase="P01",
        teacher_offset_decisions=0, new_mdp_warm_start=False, mode="semantic_residual_eval")
    assert cli.validate_video_args(args) == "C"
    vars(args).update(fault)
    with pytest.raises(video.SemanticVideoError):
        cli.validate_video_args(args)


def test_real_CPU_saved_HISTORY372_video_reload_retains_Adam_rng_budget_and_deterministic_kernel(tmp_path, monkeypatch):
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    import torch
    from tensordict import TensorDict
    from test_semantic_v3_continuation import Core324

    class Core372(Core324):
        def reset(self, **kwargs):
            return super().reset(**kwargs) + (0.,) * 48
        def step(self, raw):
            result = super().step(raw)
            values = list(result.observation) + [0.] * 48
            values[195:207] = [max(-20., min(20., float(x))) for x in raw]
            result.observation = tuple(values)
            return result

    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        _, old, new = fixture_video372(tmp_path)
        training.seed_training_rngs(1001)
        env = training.SemanticRslAdapter(Core372(), seed=1001, device="cpu")
        env.cfg["semantic_version"] = "v3"
        runner, _ = training.construct_semantic_runner(env, seed=1001, device="cpu",
            policy_version=policies.HISTORY_POLICY, observation_layout=LAYOUT, initialize_actor=False)
        trained = training.train_semantic(runner, env, run_dir=tmp_path / "cpu_source_run",
            output_root=tmp_path / "cpu_source_outputs", stage="full_episode",
            decisions=128, contract=old, seed=1001)
        assert runner.alg.optimizer.state_dict()["state"]
        source = Path(trained["checkpoints"][-1]["checkpoint"])
        metadata = migration.checkpoint_metadata(source)
        path = tmp_path / "video_plan.json"
        path.write_text(json.dumps(video_plan(tmp_path, source, old, new)), encoding="utf-8")
        real_validate = migration.validate_migration_plan
        monkeypatch.setattr(migration, "validate_migration_plan", lambda cp, contract, p:
            real_validate(cp, contract, p, project_root=tmp_path))
        verified = migration.validate_migration_plan(source, new, path)
        args = NS(checkpoint=source, resume_migration=path, new_mdp_warm_start=False,
            policy_distribution_migration=False, target_policy_version=None,
            seed=4001, device="cpu", semantic_version="v3", num_envs=1, command="eval",
            experiment_id="fsm_reference_p09_stable_v2")
        runtime_cli._preflight_checkpoint(args, new)
        assert args._migration_record == verified and args._observation_layout == LAYOUT
        constructed, loaded_infos = [], []
        real_construct, real_load = cli.construct_semantic_runner, cli.load_semantic_checkpoint
        def construct(*a, **kw):
            result = real_construct(*a, **kw)
            constructed.append(result[0])
            return result
        def load(*a, **kw):
            result = real_load(*a, **kw)
            loaded_infos.append(result)
            return result
        monkeypatch.setattr(cli, "construct_semantic_runner", construct)
        monkeypatch.setattr(cli, "load_semantic_checkpoint", load)
        action, proof, unchanged = cli.checkpoint_loader(args, new)((0.,) * 372)
        target = constructed[0]
        assert proof["optimizer_updates"] == 0 and proof["observation_dimension"] == 372
        assert proof["policy_contract"] == metadata["policy_contract"]
        assert target.alg.storage.step == 0 and target.alg.transition.actions is None
        for part in ("actor", "critic"):
            expected, actual = getattr(runner.alg, part).state_dict(), getattr(target.alg, part).state_dict()
            assert expected.keys() == actual.keys()
            assert all(torch.equal(expected[key], actual[key]) for key in expected)
        assert training.state_hash(target.alg.optimizer.state_dict()) == metadata["optimizer_state_sha256"]
        assert training.state_hash(training._normalizers(target)) == metadata["normalizer_state_sha256"]
        assert training.capture_training_rng_state(seed=1001) == metadata["training_rng_state"]
        assert training.optimizer_learning_rate(target) == metadata["optimizer_learning_rate"]
        for key in ("global_policy_decisions", "ppo_updates", "optimizer_steps", "stage_requested_decisions"):
            assert loaded_infos[0][key] == metadata[key]
        for history in (0., .7):
            values = [0.] * 372
            values[195:207] = [history] * 12
            tensor = torch.tensor([values], dtype=torch.float32)
            inputs = TensorDict({"policy": tensor, "critic": tensor.clone()}, batch_size=[1])
            with torch.inference_mode():
                expected = tuple(runner.alg.actor(inputs, stochastic_output=False)[0].tolist())
            assert action(values, 1) == expected
            unchanged()
        target.alg.storage.step = 1
        with pytest.raises(RuntimeError, match="partial old rollout"):
            real_load(target, source, contract=new, seed=1001, migration=verified)
        target.alg.storage.step = 0
        critic = target.alg.storage.observations["critic"]
        target.alg.storage.observations["critic"] = critic[..., :-1]
        with pytest.raises(RuntimeError, match="policy/critic/action storage"):
            real_load(target, source, contract=new, seed=1001, migration=verified)
        target.alg.storage.observations["critic"] = critic
    finally:
        torch.set_num_threads(old_threads)


@pytest.mark.parametrize("ticks", [1, 7, 8, 9, 17, 23752, *range(23993, 24001)])
def test_current_task_window_count_and_partial_duration_are_exact(ticks):
    receipt = video.task_interval_receipt(ticks)
    expected = (ticks + 7) // 8
    assert receipt["frame_count"] == expected <= 3000
    assert receipt["encoded_duration_s"] == expected / 15 <= 200
    assert receipt["last_frame_episode_tick"] == ticks
    assert 1 <= receipt["final_interval_physics_ticks"] <= 8
    assert receipt["encoded_duration_s"] - receipt["physical_duration_s"] == pytest.approx(
        receipt["terminal_frame_display_quantization_s"])
    assert 0 <= receipt["terminal_frame_display_quantization_s"] <= 7 / 120
    assert receipt["extra_physics_ticks"] == receipt["extra_pre_frames"] == receipt["extra_post_frames"] == 0
    assert receipt["tick0_observation_retained"] and not receipt["tick0_encoded_as_extra_frame"]


@pytest.mark.parametrize("ticks", [0, -1, True, 1.5, 24001])
def test_current_task_window_never_accepts_false_or_over_horizon_ticks(ticks):
    with pytest.raises(video.SemanticVideoError):
        video.task_interval_receipt(ticks)


def interval_fixture(ticks):
    from wlr50_clean.evaluation.video_timeline import DecodedVideoFrame, LedgerFrame
    receipt = video.task_interval_receipt(ticks)
    source = {"experiment_id": video.TASK_WINDOW_EXPERIMENT, "semantic_version": "v3",
              "episode_physics_ticks": ticks, "task_interval_window": receipt}
    decoded = [DecodedVideoFrame(i, i / 15, f"{i:08X}", i == 0) for i in range(receipt["frame_count"])]
    ledger = [LedgerFrame(i, min(8 * (i + 1), ticks), min(8 * (i + 1), ticks) / 120)
              for i in range(receipt["frame_count"])]
    return source, decoded, ledger


@pytest.mark.parametrize("ticks", [17, 24, 23752, *range(23993, 24001)])
def test_actual_partial_endpoint_and_all_native_frames_bind_to_existing_publication(ticks):
    from wlr50_clean.evaluation.video_timeline import verify_native_rate_output
    source, decoded, ledger = interval_fixture(ticks)
    window = video.task_interval_action_window(source, decoded, ledger)
    assert window.is_full_source and window.expected_frame_count == (ticks + 7) // 8
    assert window.output_duration_s <= 200 and window.source_last_sim_time_s == ticks / 120
    assert window.semantic_start_sim_s == 0 and window.semantic_end_sim_s == ticks / 120
    assert ledger[0].sim_step == 8 and ledger[-1].sim_step == ticks
    assert window.maximum_ledger_to_pts_offset_deviation_s == pytest.approx(
        source["task_interval_window"]["terminal_frame_display_quantization_s"])
    identity = verify_native_rate_output(decoded, window, require_decoded_frame_identity=True,
                                        require_exact_pts_delta_identity=True)
    assert identity["decoded_frames_unchanged"] and identity["native_frame_cadence_unchanged"]


@pytest.mark.parametrize("fault", ["tick0_extra", "false_final_tick", "false_final_time",
                                  "retimed", "wrong_manifest", "lost_failure_tail"])
def test_task_interval_adapter_rejects_synthetic_clock_or_missing_endpoint(fault):
    from dataclasses import replace
    source, decoded, ledger = interval_fixture(23997)
    if fault == "tick0_extra":
        ledger.insert(0, replace(ledger[0], sim_step=0, sim_time_s=0))
    elif fault == "false_final_tick":
        ledger[-1] = replace(ledger[-1], sim_step=24000)
    elif fault == "false_final_time":
        ledger[-1] = replace(ledger[-1], sim_time_s=200)
    elif fault == "retimed":
        decoded[2] = replace(decoded[2], pts_s=decoded[2].pts_s + .01)
    elif fault == "wrong_manifest":
        source["task_interval_window"]["final_interval_physics_ticks"] = 8
    else:
        ledger.pop()
        decoded.pop()
    with pytest.raises(video.SemanticVideoError):
        video.task_interval_action_window(source, decoded, ledger)


def _capture_tick23_fixture(tmp_path, monkeypatch, *, success,
                            done_only=False, terminal_encoder_failure=False):
    # Entirely fake physics/encoder fixture: verifies wiring, not live success.
    events, frames, raw_ticks = [], [], []
    backend = NS(_episode_tick=0, _video_post_terminal_tick_count=0,
        _controller=NS(physics_tick=0), render_video_frame=lambda: events.append("render"))
    core = NS(backend=backend, decision_count=0, done=False, observation=(0.,) * 372)
    def reset(*, seed):
        events.append(("reset", seed))
        core.frame = NS(physics_tick=0, state_id="P01", info={"reset_count": 1,
            "reset_options": {}, "training_phase_snapshot": None,
            "locked_scene_snapshot": {"camera": video.CAMERA}})
    def step(action):
        assert len(action) == 12
        for tick in range(backend._episode_tick + 1, min(backend._episode_tick + 8, 23) + 1):
            before = core.frame
            backend._episode_tick = tick
            core.frame = NS(physics_tick=tick, state_id="P13" if tick == 23 else "P01", info={})
            events.append(("physics", tick))
            core.tick_observer(before, core.frame, None)
            if done_only and tick == 23:
                core.done = True
                core.frame.info["termination_reason"] = "LOCAL_TASK_TIMEOUT"
        core.decision_count += 1
        return NS(observation=core.observation, info=dict(core.frame.info))
    core.reset, core.step = reset, step
    class Physical:
        def __init__(self, root, **kwargs):
            self.evaluator = NS(snapshot={"success": False, "termination_reason": None})
        def start(self, frame):
            raw_ticks.append(frame.physics_tick)
        def observe(self, before, after, projection):
            raw_ticks.append(after.physics_tick)
            if after.physics_tick == 23 and not done_only:
                self.evaluator.snapshot = {"success": success,
                    "termination_reason": None if success else "TASK_FAILURE_BODY_CONTACT"}
        def summary(self):
            return {"task_success": self.evaluator.snapshot["success"],
                    "termination_reason": self.evaluator.snapshot["termination_reason"],
                    "observed_through_tick": raw_ticks[-1]}
        def close(self):
            pass
    class Recorder:
        def __init__(self, root):
            self.root = root
        def start(self):
            events.append(("record_start", backend._episode_tick))
            return True
        def before_render(self, *, sim_step, sim_time_s):
            assert sim_step == backend._episode_tick
            frames.append((sim_step, sim_time_s))
        def after_render(self):
            pass
        def require_healthy(self):
            if terminal_encoder_failure and backend._episode_tick == 23:
                events.append(("encoder_failure", 23))
                raise RuntimeError("terminal encoder failure")
        def finalize(self):
            events.append(("finalize", backend._episode_tick))
            return {"valid": not terminal_encoder_failure}
    monkeypatch.setattr(video, "PhysicalEvaluationRecorder", Physical)
    monkeypatch.setattr(video, "_validate_video_configuration_binding", lambda *a, **k: None)
    monkeypatch.setattr(video, "common_post_success_tick",
                        lambda *a, **k: pytest.fail("current task window added post-success physics"))
    monkeypatch.setattr(video, "reset_with_existing_settle_tail",
                        lambda *a, **k: pytest.fail("current window encoded reset context"))
    result = video.capture_semantic_video(core, role="B", seed=4001,
        output_directory=tmp_path / "source", contract={"semantic_version": "v3",
        "experiment_id": video.TASK_WINDOW_EXPERIMENT}, recorder_factory=Recorder,
        semantic_version="v3", experiment_id=video.TASK_WINDOW_EXPERIMENT)
    assert events.count(("reset", 4001)) == 1 and ("record_start", 0) in events
    assert [e[1] for e in events if isinstance(e, tuple) and e[0] == "physics"] == list(range(1, 24))
    assert raw_ticks == list(range(24))  # Includes initial tick0 evidence.
    assert frames == [(8, 8 / 120), (16, 16 / 120), (23, 23 / 120)]
    assert result["pre_action_ticks"] == result["requested_post_success_ticks"] == 0
    assert result["performed_post_success_ticks"] == result["extra_pre_action_physics_ticks"] == 0
    assert result["task_interval_window"] == video.task_interval_receipt(23)
    assert (tmp_path / "source" / "physical_video_roll_ticks.jsonl").read_text() == ""
    return NS(result=result, events=events, core=core, source=tmp_path / "source")


@pytest.mark.parametrize("success", [True, False])
def test_current_capture_keeps_real_P01_to_partial_terminal_without_PRE_or_POST(tmp_path, monkeypatch, success):
    capture = _capture_tick23_fixture(tmp_path, monkeypatch, success=success)
    result = capture.result
    assert result["physical_task_success"] is success and result["success_candidate"] is success
    assert result["diagnostic_only"] is not success
    if not success:
        assert "did not meet common physical task" in result["source_acceptance_error"]
        assert "NOT SUCCESS" in video.comparison_title(result)


def test_done_only_local_timeout_retains_partial_terminal_and_is_never_success(tmp_path, monkeypatch):
    capture = _capture_tick23_fixture(tmp_path, monkeypatch, success=False, done_only=True)
    result = capture.result
    assert capture.core.done and capture.core.frame.info["termination_reason"] == "LOCAL_TASK_TIMEOUT"
    # The evaluator has NO terminal flag; the capture's post-loop fallback must
    # still preserve the real tick23 frame. The helper checks frames8/16/23 and
    # raw observations0..23, with no PRE/POST physics.
    assert result["physical_episode"] == {
        "task_success": False, "termination_reason": None, "observed_through_tick": 23}
    assert result["physical_task_success"] is False and result["success_candidate"] is False
    assert result["diagnostic_only"] is True and "NOT SUCCESS" in video.comparison_title(result)
    decisions = [json.loads(line) for line in
                 (capture.source / "video_policy_decisions.jsonl").read_text().splitlines()]
    assert [row["end_tick"] for row in decisions] == [8, 16, 23]
    assert decisions[-1]["environment_step_returned"] is True
    assert decisions[-1]["physics_ticks"] == 7
    assert decisions[-1]["step_info"]["termination_reason"] == "LOCAL_TASK_TIMEOUT"
    assert "did not meet common physical task" in result["source_acceptance_error"]
    monkeypatch.setattr(video.subprocess, "run",
                        lambda *a, **k: pytest.fail("diagnostic cannot launch publication"))
    with pytest.raises(video.SemanticVideoError, match="failed diagnostic"):
        video.publish_success_source(capture.source, tmp_path / "semantic_B_success.mp4")


def test_terminal_encoder_failure_preserves_physical_success_but_blocks_publication(tmp_path, monkeypatch):
    capture = _capture_tick23_fixture(tmp_path, monkeypatch, success=True,
                                      terminal_encoder_failure=True)
    result = capture.result
    assert ("encoder_failure", 23) in capture.events and ("finalize", 23) in capture.events
    # require_healthy throws during the actual terminal callback, before the
    # normal post-loop summary. The finally path must retain the evaluator's
    # measured success through23, without converting an invalid video to success.
    assert result["physical_episode"] == {
        "task_success": True, "termination_reason": None, "observed_through_tick": 23}
    assert result["physical_task_success"] is True and result["success_candidate"] is False
    assert result["diagnostic_only"] is True and "NOT SUCCESS" in video.comparison_title(result)
    assert "terminal encoder failure" in result["source_acceptance_error"]
    monkeypatch.setattr(video.subprocess, "run",
                        lambda *a, **k: pytest.fail("invalid encoder output cannot publish"))
    with pytest.raises(video.SemanticVideoError, match="failed diagnostic"):
        video.publish_success_source(capture.source, tmp_path / "semantic_B_success.mp4")


def test_old_experiment_window_and_current_comparison_count_remain_distinct():
    old = {"episode_physics_ticks": 23752, "experiment_id": "all_stage_acceptance_v1"}
    current = {"episode_physics_ticks": 24000, "experiment_id": video.TASK_WINDOW_EXPERIMENT,
               "role": "C", "success_candidate": True, "diagnostic_only": False,
               "physical_episode": {"task_success": True}}
    assert video.source_frame_count(old) == 3001  # Historical packaging shape preserved.
    assert video.source_frame_count(current) == 3000
    baseline = dict(current, role="A", episode_physics_ticks=23997)
    _, count = video.comparison_filter(baseline, current)
    assert count == 3000
