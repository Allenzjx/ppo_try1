"""Independent A/B/C video capture; no optimizer or frozen-controller mutation.

All source files are immutable. Raw capture remains neutral/diagnostic until
physical success, real post-roll and independent decode/clock checks pass.
"""
from __future__ import annotations

from dataclasses import replace
import json
import math
import os
from pathlib import Path
import stat
import subprocess
import uuid

from wlr50_clean.infrastructure.video_capture import (
    ActiveViewportVideoRecorder, find_ffmpeg, sha256_file, validate_mp4,
)
from wlr50_clean.evaluation.video_timeline import (
    decode_frame_timeline, load_viewport_frame_ledger, plan_action_window,
    verify_native_rate_output,
)
from .semantic_legacy_evaluation import (
    PhysicalEvaluationRecorder, measured_observation, physical_json,
)
from .semantic_supervisor import SemanticControllerAdapter, TaskEvaluator
from .semantic_training import verified_native_effect, write_json

HZ, FPS, STRIDE = 120, 15, 8
PRE_TICKS, POST_TICKS, MAX_FRAMES = 64, 184, 3000
TASK_WINDOW_EXPERIMENT = "fsm_reference_p09_stable_v2"
CAMERA = {"eye_m": [1.45, -1.25, .8], "target_m": [.45, 0., .12]}
ROLES = {"A": "legacy_fsm_eval", "B": "semantic_prior_eval",
         "C": "semantic_residual_eval"}
ZERO12 = (0.,) * 12


class SemanticVideoError(RuntimeError):
    pass


class _PhysicalEndpoint(Exception):
    """Unwind only this capture's incomplete final decision, not the controller."""


def require(condition, message):
    if not condition:
        raise SemanticVideoError(message)


def jsonl(stream, row):
    stream.write(json.dumps(physical_json(row), allow_nan=False,
                            separators=(",", ":")) + "\n")
    stream.flush()


def frames_for_episode(episode_ticks):
    require(type(episode_ticks) is int and episode_ticks > 0, "bad episode tick count")
    return 1 + (PRE_TICKS + episode_ticks + POST_TICKS) // STRIDE


def task_interval_receipt(episode_ticks):
    """CFR samples real executed intervals; no extra physics or task deadline."""
    require(type(episode_ticks) is int and 0 < episode_ticks <= 24000,
            "task video endpoint must remain inside the unchanged 200s horizon")
    count = (episode_ticks + STRIDE - 1) // STRIDE
    final_ticks = episode_ticks - STRIDE * (count - 1)
    return {"schema": "wlr50_clean.task_interval_video_window.v1",
        "experiment_id": TASK_WINDOW_EXPERIMENT, "first_episode_tick": 0,
        "endpoint_episode_tick": episode_ticks, "frame_count": count,
        "frame_sample": "actual_executed_interval_right_endpoint",
        "first_frame_episode_tick": min(STRIDE, episode_ticks),
        "last_frame_episode_tick": episode_ticks,
        "final_interval_physics_ticks": final_ticks,
        "terminal_frame_display_quantization_s": (STRIDE - final_ticks) / HZ,
        "encoded_duration_s": count / FPS, "physical_duration_s": episode_ticks / HZ,
        "tick0_observation_retained": True, "tick0_encoded_as_extra_frame": False,
        "extra_physics_ticks": 0, "extra_pre_frames": 0, "extra_post_frames": 0}


def source_frame_count(source):
    if source.get("experiment_id") == TASK_WINDOW_EXPERIMENT:
        return task_interval_receipt(source["episode_physics_ticks"])["frame_count"]
    return frames_for_episode(source["episode_physics_ticks"])


def capture_task_interval_frame(recorder, backend, episode_tick):
    """Actual viewport at a real interval end, including an exact partial end."""
    require(type(episode_tick) is int and 0 < episode_tick <= 24000, "invalid task frame tick")
    recorder.before_render(sim_step=episode_tick, sim_time_s=episode_tick / HZ)
    backend.render_video_frame()
    recorder.after_render()
    recorder.require_healthy()


def task_interval_action_window(source, decoded, ledger):
    """Narrow adapter to the existing publication window, preserving real ticks.

    Full intervals contain8 ticks; the final interval contains1..8. Its native
    CFR display quantization is explicit, never a fabricated simulator time.
    """
    from wlr50_clean.evaluation.video_timeline import (
        ActionWindow, _sha256_float64, _sha256_text_rows,
    )
    require(source.get("experiment_id") == TASK_WINDOW_EXPERIMENT
            and source.get("semantic_version") == "v3", "wrong task interval experiment")
    endpoint = source["episode_physics_ticks"]
    receipt = task_interval_receipt(endpoint)
    require(source.get("task_interval_window") == receipt, "task interval manifest mismatch")
    count = receipt["frame_count"]
    require(2 <= count <= MAX_FRAMES and len(decoded) == len(ledger) == count,
            "task interval frame count differs")
    ticks = [min((index + 1) * STRIDE, endpoint) for index in range(count)]
    require([row.sim_step for row in ledger] == ticks, "task interval actual endpoint tick mismatch")
    require(all(frame.frame_index == row.frame_index == index
                and abs(row.sim_time_s - ticks[index] / HZ) < 1e-10
                for index, (frame, row) in enumerate(zip(decoded, ledger))),
            "task interval ledger clock/index mismatch")
    pts = [frame.pts_s for frame in decoded]
    deltas = [b - a for a, b in zip(pts, pts[1:])]
    require(all(abs(delta - 1 / FPS) < 1e-5 for delta in deltas), "task video was retimed")
    offsets = [row.sim_time_s - frame.pts_s for row, frame in zip(ledger, decoded)]
    offset = offsets[0]
    require(all(abs(value - offset) < 1e-5 for value in offsets[:-1])
            and abs((offset - offsets[-1]) -
                    receipt["terminal_frame_display_quantization_s"]) < 1e-5,
            "task terminal interval/PTS quantization mismatch")
    duration = count / FPS
    # Keep every decoded frame/hash/native PTS delta. Do not reinterpret the
    # true endpoint ledger as an ideal physical-time grid.
    return ActionWindow(
        semantic_start_sim_s=0., semantic_end_sim_s=endpoint / HZ,
        source_first_frame_index=0, source_last_frame_index=count-1,
        source_first_pts_s=pts[0], source_last_pts_s=pts[-1],
        source_first_sim_time_s=ledger[0].sim_time_s, source_last_sim_time_s=ledger[-1].sim_time_s,
        trim_start_pts_s=pts[0], trim_end_pts_s=pts[-1]+1/FPS,
        output_duration_s=duration, expected_frame_count=count,
        leading_frames_removed=0, trailing_frames_removed=0, phase_clock_origin_sim_s=0.,
        semantic_start_output_s=0., semantic_end_output_s=endpoint/HZ,
        action_frame_start_output_pts_s=0., action_frame_end_output_pts_s_exclusive=duration,
        requested_pre_roll_s=0., requested_post_roll_s=0., maximum_preserved_roll_s=0.,
        available_pre_roll_s=0., available_post_roll_s=0.,
        retained_pre_roll_s=0., retained_post_roll_s=0.,
        capture_lag_after_action_start_s=ledger[0].sim_time_s,
        packet_copy_start_is_keyframe=decoded[0].key_frame,
        source_selected_pts_sha256=_sha256_float64(pts),
        source_selected_pts_delta_sha256=_sha256_float64(deltas),
        source_selected_checksums_sha256=_sha256_text_rows(frame.checksum for frame in decoded),
        ledger_to_pts_offset_s=offset,
        maximum_ledger_to_pts_offset_deviation_s=max(abs(value-offset) for value in offsets))


def file_record(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": sha256_file(path)}


def inside(root, record):
    path = Path(record["path"])
    # Reject Windows junctions as well as ordinary symlinks.
    for component in (path, *path.parents):
        metadata = component.lstat()
        require(not stat.S_ISLNK(metadata.st_mode)
                and not (getattr(metadata, "st_file_attributes", 0)
                         & stat.FILE_ATTRIBUTE_REPARSE_POINT),
                "reparse/symlink video evidence")
    path = path.resolve(strict=True)
    require(path.is_relative_to(root.resolve()), "video evidence escaped its source")
    require(file_record(path) == dict(record), "video evidence hash/size changed")
    return path


def capture_frame(recorder, backend, global_tick):
    require(global_tick % STRIDE == 0, "render must use native 15 Hz grid")
    recorder.before_render(sim_step=global_tick, sim_time_s=global_tick / HZ)
    backend.render_video_frame()
    recorder.after_render()
    recorder.require_healthy()


def video_configuration(semantic_version, *, experiment_id=None):
    """One explicit configuration selection for capture AND independent replay."""
    from .semantic_cli import version_paths
    config = (version_paths(semantic_version) if experiment_id is None else
              version_paths(semantic_version, experiment_id=experiment_id))[2]
    result = {name: config / filename for name, filename in (
        ("task_spec_path", "stage_task_spec.yaml"),
        ("quality_score_path", "quality_score.yaml"),
        ("execution_profile", "execution_profile.yaml"),
        ("reward_config_path", "reward_config.yaml"),
        ("observation_schema_path", "observation_schema.json"))}
    if experiment_id is not None:
        # Bind all six experiment files, even though residual execution uses
        # execution_profile rather than reading action_schema a second time.
        result["action_schema_path"] = config / "action_schema.json"
    return result


def _validate_video_configuration_binding(configs, contract, *, experiment_id):
    """Exact explicit-experiment paths/bytes; no permission to change the MDP."""
    require(contract.get("experiment_id") == experiment_id, "video/runtime experiment mismatch")
    if experiment_id is None:
        return  # Preserve historical implicit v2/v3 source manifests.
    from .semantic_cli import PROJECT_ROOT
    from .semantic_policy_distribution import CONFIG_NAMES
    require(set(path.name for path in configs.values()) == CONFIG_NAMES
            and len(configs) == len(CONFIG_NAMES), "video requires all six selected configurations")
    selected = contract.get("selected_configuration")
    require(isinstance(selected, dict) and set(selected) == CONFIG_NAMES,
            "video runtime lacks exactly six selected configuration bindings")
    for path in configs.values():
        record = file_record(path)
        actual = Path(record["path"])
        require(actual.is_relative_to(PROJECT_ROOT.resolve()), "video configuration escaped project")
        relative = actual.relative_to(PROJECT_ROOT.resolve()).as_posix()
        require(selected[path.name] == {"path": relative, "sha256": record["sha256"]}
                and contract.get("files", {}).get(relative) == record["sha256"],
                "video selected configuration path/hash differs from current runtime bytes")


def reset_with_existing_settle_tail(core, recorder, roll, *, seed):
    """Observe the last 64 of the existing 180 reset steps, never add a step.

    A before-dispatch observer sees the preceding step already completed by
    the unchanged reset loop. The last receipt is observed after reset returns.
    No reader call, controller refresh, mapper advance or state write is added.
    The original per-instance method is restored even if reset/capture fails.
    """
    from .isaac_fsm_backend import SETTLE_TICKS
    require(SETTLE_TICKS == 180, "video settle contract changed")
    backend = core.backend
    original = backend._atomic_apply
    had_override = "_atomic_apply" in vars(backend)
    saved_override = vars(backend).get("_atomic_apply")
    first = SETTLE_TICKS - PRE_TICKS
    count = 0
    previous_ack = None

    def completed(index, ack):
        require(ack["physics_tick"] == first + index - 1,
                "settle-tail receipt tick gap")
        jsonl(roll, {"kind": "pre_action", "global_tick": index,
            "task_credit": False, "source": "existing_reset_settle_tail",
            "physical_hold": {"physics_tick": ack["physics_tick"],
                "applied_full12": ack["applied_full12"], "root_state_write_count": 0,
                "existing_reset_dispatch": True, "atomic_ack": ack},
            "sensor_observation_sampled": False})
        if index % STRIDE == 0:
            capture_frame(recorder, backend, index)

    def observe(adapter, command, *, physics_tick, tracking_servo_names,
                drive_feedback_bias_full12):
        nonlocal count, previous_ack
        require(physics_tick == count and count < SETTLE_TICKS,
                "video natural reset added or reordered physical dispatches")
        require(tuple(command) == ZERO12 and not tracking_servo_names
                and tuple(drive_feedback_bias_full12) == ZERO12,
                "video reset is not the existing zero-command settle")
        if count == first:
            for _ in range(3):
                backend.render_video_frame()  # Shader-only warmup; no app update.
            require(recorder.start(), "viewport capture did not start")
            capture_frame(recorder, backend, 0)
        elif count > first:
            completed(count - first, previous_ack)
        ack = original(adapter, command, physics_tick=physics_tick,
            tracking_servo_names=tracking_servo_names,
            drive_feedback_bias_full12=drive_feedback_bias_full12)
        require(tuple(ack["applied_full12"]) == ZERO12,
                "existing settle did not dispatch zero Full12")
        previous_ack = ack
        count += 1
        return ack

    backend._atomic_apply = observe
    try:
        core.reset(seed=seed)
        require(count == SETTLE_TICKS, "incomplete existing reset settle")
        completed(PRE_TICKS, previous_ack)
    finally:
        if had_override:
            backend._atomic_apply = saved_override
        else:
            del backend._atomic_apply
    return {"source": "existing_reset_settle_tail", "existing_settle_ticks": count,
            "first_observed_dispatch_tick": first,
            "last_observed_dispatch_tick": SETTLE_TICKS - 1,
            "extra_physics_ticks": 0, "extra_sensor_reads": 0,
            "extra_controller_steps": 0, "extra_state_writes": 0,
            "task_credit": False}


def refresh_semantic_core_after_preroll(core):
    """Reset only episode-local sensing/controller/encoding, never physics.

    This is a video-only legal pre-episode initialization. A uses its existing
    legacy refresh method and never enters this function.
    """
    backend = core.backend
    require(core.decision_count == 0 and core.frame.physics_tick == 0
            and core.frame.state_id == "P01" and not core.done,
            "semantic pre-roll refresh requires fresh unstepped P01")
    generation = backend._committed_reset_generation
    reset_count = backend._reset_count
    previous_adapter = backend._adapter
    refreshed = backend.refresh_video_pre_action_frame()
    raw = refreshed.info["raw_observation"]
    from .semantic_backend import CONFIG_ROOT
    controller = SemanticControllerAdapter.from_paths(
        backend.fsm_path, backend.motion_contract_path,
        task_spec_path=CONFIG_ROOT / "stage_task_spec.yaml")
    controller_frame = controller.step(raw, sim_time_s=0.)
    require(controller_frame.state_id == "P01" and controller_frame.physics_tick == 0,
            "post-roll physical state did not initialize natural P01")
    backend._controller = controller
    backend._controller_frame = controller_frame
    frame = backend._build_authoritative_frame(raw, controller_frame, previous_frame=None)
    frame = replace(frame, info={**dict(frame.info),
        "video_pre_action_refresh": refreshed.info["video_pre_action_refresh"],
        "semantic_video_initialization": {
            "task_evaluator_recreated_from_current_tick_zero": True,
            "prefix_credit_imported": False, "physical_reset_performed": False}})
    backend._authoritative_frame = frame
    require(backend._adapter is previous_adapter
            and backend._committed_reset_generation == generation
            and backend._reset_count == reset_count, "pre-roll reset physics/mapper")
    from .semantic_observation import HISTORY_GROUPS, vector, semantic_task
    nominal = vector(frame.nominal_action_full12, 12, "refreshed nominal")
    drive = vector(frame.info["drive_target_full12"], 12, "refreshed actual drive")
    core.observation_builder.reset()
    core._history = {name: ZERO12 for name in HISTORY_GROUPS}
    core._history.update(previous_applied_full12=drive,
                        previous_previous_applied_full12=drive,
                        previous_nominal_full12=nominal)
    core._semantic_frame = core.observation_builder.build(frame, core._history)
    core.bridge.reset(state_id=frame.state_id, applied_action_full12=nominal)
    core.frame = frame
    core.observation = core.observation_schema.encode(core._semantic_frame.groups)
    core.trace = []
    core._episode_return = 0.
    core._transition_evidence_count = len(semantic_task(frame)["transition_evidence"])
    return tuple(core.observation), dict(frame.info)


def post_success_hold_receipt(ack):
    """Keep the realized episode's exact mapper/bias split, stopping wheels."""
    from .isaac_fsm_backend import _full12
    from wlr50_clean.infrastructure.command_batch import SERVO_ORDER
    required = {"fsm_nominal_mapper_input_full12", "combined_post_mapper_bias_full12",
                "applied_full12", "drive_feedback_bias_requested_full12", "tracking_servo_names"}
    require(ack.get("ppo_actuation_contract") == "frozen_nominal_plus_post_mapper_residual.v1"
            and required <= ack.keys(), "post-roll lacks a verified last actuation receipt")
    nominal = _full12(ack["fsm_nominal_mapper_input_full12"], "last nominal mapper input")
    bias = _full12(ack["combined_post_mapper_bias_full12"], "last combined post-mapper bias")
    require(_full12(ack["applied_full12"], "last applied mapper command") == nominal
            and _full12(ack["drive_feedback_bias_requested_full12"], "last adapter bias") == bias,
            "post-roll receipt differs from the last actual adapter input")
    tracking = tuple(ack["tracking_servo_names"])
    require(len(set(tracking)) == len(tracking) and set(tracking) <= set(SERVO_ORDER),
            "post-roll receipt tracking names are invalid")
    return nominal[:8] + (0.,)*4, bias[:8] + (0.,)*4, tracking


def common_post_success_tick(backend, evaluator, *, episode_ticks, post_index):
    """One physical context tick authorized by the SAME independent evaluator.

    Does not change backend._done, any controller lifecycle/result, controller
    tick, episode tick or original controller object. Mapper/physics continue.
    """
    require(evaluator.snapshot.get("success") is True, "post-roll lacks common success")
    require(1 <= post_index <= POST_TICKS, "post-roll outside declared length")
    from .isaac_fsm_backend import _require_running, _validate_sensor_contract
    scene, adapter, reader = backend._scene, backend._adapter, backend._reader
    source = backend._controller_frame
    invariant = (id(backend._controller), id(source), backend._episode_tick,
                 backend._done, backend._controller.physics_tick)
    require(backend._episode_tick == episode_ticks, "episode tick changed after stop")
    _require_running(scene, "semantic physical video post-roll")
    previous_ack = dict(backend._last_atomic_ack)
    physical_tick = int(previous_ack["physics_tick"]) + 1
    # Repeat the last actual dispatch receipt, not the incoming controller
    # frame or _previous_action_full12 (which already contains PPO residual).
    # Retaining the nominal request prevents mapper compensation reset; the
    # exact combined bias keeps PPO strictly on its original post-mapper path.
    hold, bias, tracking = post_success_hold_receipt(previous_ack)
    if previous_ack.get("semantic_residual_composition") == "independent_post_mapper_residual.v1":
        from .isaac_fsm_backend import ResidualActuationPlan, _full12
        from .semantic_residual_adapter import SemanticActuationDispatch
        controller = _full12(previous_ack["bounded_controller_bias_requested_full12"], "last bounded controller")
        residual = _full12(previous_ack["independent_policy_residual_requested_full12"], "last independent residual")
        require(tuple(a + b for a, b in zip(controller, residual)) ==
                tuple(previous_ack["combined_post_mapper_bias_full12"]),
                "independent post-roll receipt composition mismatch")
        controller, residual = controller[:8] + ZERO12[8:], residual[:8] + ZERO12[8:]
        plan = ResidualActuationPlan(hold, tuple(a + b for a, b in zip(hold, residual)),
                                    residual, controller, bias)
        adapter = SemanticActuationDispatch(adapter, plan,
            policy_headroom_mode=getattr(backend, "_policy_headroom_mode", None),
            tracking_reference_mode=getattr(backend, "_tracking_reference_mode", None))
    ack = backend._atomic_apply(adapter, hold, physics_tick=physical_tick,
        tracking_servo_names=tracking, drive_feedback_bias_full12=bias)
    # The next hold must use the same receipt contract, not an unannotated ack.
    ack = {**ack,
        "ppo_actuation_contract": previous_ack["ppo_actuation_contract"],
        "fsm_nominal_mapper_input_full12": list(hold),
        "combined_post_mapper_bias_full12": list(bias),
        "video_post_success_hold": {"source_ack_physics_tick": previous_ack["physics_tick"],
            "servo_nominal_and_post_mapper_bias_preserved": True,
            "nominal_and_bias_wheels_explicitly_zero": True}}
    scene.sim.step(render=False)
    adapter.update_readback()
    logical_tick = backend._episode_sensor_tick_offset + episode_ticks + post_index
    raw = reader.read(physics_tick=logical_tick,
                      simulation_time_s=logical_tick/HZ,
                      commanded_full12=ack["drive_target_full12"])
    _validate_sensor_contract(raw, backend._dependencies.expected_contact_bodies,
                              require_finite=True)
    backend._raw_observation = raw
    backend._last_atomic_ack = ack
    backend._video_post_terminal_tick_count = post_index
    result = evaluator.observe(raw)
    require(invariant == (id(backend._controller), id(backend._controller_frame),
            backend._episode_tick, backend._done, backend._controller.physics_tick),
            "post-roll mutated controller or spoofed its terminal state")
    require(result.get("success") is True and result.get("termination_reason") is None,
            "physical post-success stability was lost")
    return {"post_index": post_index, "physical_command_tick": physical_tick,
            "task_credit": False, "root_state_write_count": 0,
            "controller_step_count": 0, "atomic_ack": ack,
            "raw_observation": measured_observation(raw),
            "physical_evaluation": result}


class EndpointObserver:
    """Common physical metrics first; exact-tick endpoint then native-grid render."""
    def __init__(self, physical, recorder, backend, *, task_window=False, height_diagnostics=None):
        self.physical, self.recorder, self.backend = physical, recorder, backend
        self.task_window = task_window
        self.height_diagnostics = height_diagnostics
        self.last_captured_tick = None
        self.last_frame = None
        self.last_global_tick = 0
        self.reason = None

    def __call__(self, before, after, projection):
        self.physical.observe(before, after, projection)
        self.last_frame = after
        self.last_global_tick = (0 if self.task_window else PRE_TICKS) + after.physics_tick
        result = self.physical.evaluator.snapshot
        terminal = result.get("success") is True or result.get("termination_reason") is not None
        if self.height_diagnostics is not None:
            self.height_diagnostics.sample(after, terminal=terminal)
        if self.task_window:
            if after.physics_tick % STRIDE == 0 or terminal:
                capture_task_interval_frame(self.recorder, self.backend, after.physics_tick)
                self.last_captured_tick = after.physics_tick
        elif self.last_global_tick % STRIDE == 0:
            capture_frame(self.recorder, self.backend, self.last_global_tick)
        if terminal:
            self.reason = "SUCCESS" if result.get("success") else result["termination_reason"]
            raise _PhysicalEndpoint
        if not self.task_window and frames_for_episode(after.physics_tick) > MAX_FRAMES:
            self.reason = "VIDEO_FULL_CONTEXT_EXCEEDS_200S"
            raise _PhysicalEndpoint


def current_video_natural_reset_proof(info, *, role, contract, entry):
    """Current A/B/C share one no-snapshot criterion, with real backend receipts."""
    require(role in ROLES and contract.get("semantic_version") == "v3"
            and contract.get("experiment_id") == TASK_WINDOW_EXPERIMENT,
            "natural video reset proof has wrong role/runtime")
    require(isinstance(entry, dict) and entry == {
        "state_id": "P01", "physics_tick": 0, "decision_count": 0, "done": False}
        and type(entry["physics_tick"]) is int and type(entry["decision_count"]) is int
        and entry["done"] is False, "video did not own initial natural P01 state")
    fields = ("seed", "reset_count", "reset_options", "training_phase_snapshot",
              "phase_snapshot_restoration", "reset_prime_tick_count",
              "sensor_tick_at_effective_entry", "fsm_and_episode_clock_at_effective_entry",
              "sensor_clock_semantics")
    require(isinstance(info, dict) and all(key in info for key in fields),
            "natural video reset receipt is incomplete")
    require(type(info["reset_count"]) is int and info["reset_count"] == 1
            and info["seed"] == 4001 and info["reset_options"] == {},
            "video requires first reset with locked seed and no explicit reset options")
    for key in ("reset_prime_tick_count", "sensor_tick_at_effective_entry",
                "fsm_and_episode_clock_at_effective_entry"):
        require(type(info[key]) is int and info[key] == 0,
                "video reset contains historical replay or a nonzero entry clock")
    require(info["sensor_clock_semantics"] == "episode_relative_tick",
            "video reset sensor clock is not natural episode-relative time")
    restoration = info["phase_snapshot_restoration"]
    require(isinstance(restoration, dict) and "requested_phase" in restoration
            and (restoration.get("snapshot_validated") is None
                 or restoration.get("snapshot_validated") is False)
            and not any(restoration.get(key) is not None for key in (
                "snapshot_path", "state_sha256", "file_sha256", "source_tick", "physical_state")),
            "video cannot start from a validated or restored phase snapshot")
    # The inherited metadata key names requested_phase, not proof that a
    # snapshot was loaded. Only the semantic backend's actual natural mode
    # plus its execution receipt may use the P01 marker.
    if restoration.get("mode") == "semantic_natural_P01":
        require(role in ("B", "C") and info.get("execution_mode") == "semantic_B_or_C"
                and info.get("supervisor_schema") == "task_semantic_v2"
                and restoration.get("requested_phase") == info["training_phase_snapshot"] == "P01"
                and restoration.get("historical_state_equality_required") is False
                and restoration.get("policy_credit_excludes_settle") is True,
                "semantic P01 marker lacks its actual natural-reset provenance")
    else:
        require(role == "A" and restoration.get("mode") == "normal_p01_reset"
                and restoration.get("requested_phase") is None
                and info["training_phase_snapshot"] is None
                and restoration.get("snapshot_validated") is False
                and info.get("execution_mode") is None,
                "legacy A reset is not the original unrequested natural P01")
    metadata = physical_json({key: info[key] for key in fields})
    metadata.update(execution_mode=info.get("execution_mode"),
                    supervisor_schema=info.get("supervisor_schema"))
    return {"schema": "wlr50_clean.current_video_natural_reset.v1",
            "role": role, "experiment_id": TASK_WINDOW_EXPERIMENT,
            "semantic_version": "v3", "entry": dict(entry), "reset_metadata": metadata}


def validate_current_video_natural_reset(source):
    proof = source.get("natural_reset_proof")
    require(isinstance(proof, dict), "current video lacks persisted natural reset proof")
    verified = current_video_natural_reset_proof(proof.get("reset_metadata"),
        role=source["role"], contract=source["runtime_contract"], entry=proof.get("entry"))
    require(proof == verified and source.get("from_phase") == "P01",
            "current video natural reset proof was altered")
    legacy_receipt = source.get("reset_evidence") or {}
    require(all(key in legacy_receipt and legacy_receipt[key] == proof["reset_metadata"][key]
                for key in ("reset_count", "reset_options", "training_phase_snapshot")),
            "video reset evidence disagrees with persisted natural reset proof")
    return proof


def capture_semantic_video(core, *, role, seed, output_directory, contract,
                           policy_loader=None, recorder_factory=ActiveViewportVideoRecorder,
                           semantic_version="v2", experiment_id=None):
    """Capture one fresh P01 episode in one live process.

    policy_loader(refreshed_observation) returns (deterministic_action_callable,
    load_provenance, unchanged_model_assertion). It must use the official
    load_semantic_checkpoint; C without that loader is rejected. A/B use zero.
    """
    require(role in ROLES and seed == 4001, "wrong role or locked video seed")
    require((role == "C") == (policy_loader is not None), "wrong checkpoint role")
    configs = (video_configuration(semantic_version) if experiment_id is None else
               video_configuration(semantic_version, experiment_id=experiment_id))
    _validate_video_configuration_binding(configs, contract, experiment_id=experiment_id)
    require(contract.get("semantic_version", "v2") == semantic_version,
            "capture version differs from runtime contract")
    root = Path(output_directory).resolve()
    root.mkdir(parents=True, exist_ok=False)
    recorder = recorder_factory(root)
    backend = core.backend
    physical = None
    observer = None
    roll = None
    decisions = None
    recorder_manifest = None
    physical_summary = None
    height_diagnostics = None
    height_diagnostic_receipt = None
    load_provenance = None
    check_model = lambda: None
    issued_decisions, completed_decisions, partial_ticks = 0, 0, 0
    error = None
    reset_info = {}
    settle_evidence = None
    natural_reset_proof = None
    task_window = experiment_id == TASK_WINDOW_EXPERIMENT
    require(not task_window or semantic_version == "v3", "current task video requires v3")
    try:
        roll = (root/"physical_video_roll_ticks.jsonl").open("x", encoding="utf-8")
        decisions = (root/"video_policy_decisions.jsonl").open("x", encoding="utf-8")
        if semantic_version == "v3" and not task_window:
            settle_evidence = reset_with_existing_settle_tail(core, recorder, roll, seed=seed)
        else:
            core.reset(seed=seed)
        require(core.frame.state_id == "P01" and core.frame.physics_tick == 0
                and core.decision_count == 0 and not core.done, "video did not reset to P01")
        reset_info = dict(core.frame.info)
        if task_window:
            natural_reset_proof = current_video_natural_reset_proof(reset_info,
                role=role, contract=contract, entry={"state_id": core.frame.state_id,
                    "physics_tick": core.frame.physics_tick,
                    "decision_count": core.decision_count, "done": core.done})
        else:
            require(reset_info.get("reset_count") == 1
                    and reset_info.get("reset_options") == {}
                    and reset_info.get("training_phase_snapshot") is None,
                    "video must own the first natural reset of a fresh process")
        camera = reset_info["locked_scene_snapshot"]["camera"]
        require(all(list(camera[key]) == value for key,value in CAMERA.items()),
                "camera differs from common A/B/C view")
        legacy_controller = backend._controller if role == "A" else None
        if task_window:
            for _ in range(3):
                backend.render_video_frame()  # Shader-only; no physical/control step.
            require(recorder.start(), "viewport capture did not start")
            # Tick0 stays in physical_observations.jsonl; no extra encoded frame.
        if semantic_version == "v3":
            observation, refreshed = tuple(core.observation), dict(core.frame.info)
        else:
            for _ in range(3):
                backend.render_video_frame()  # Shader warmup; no encoded time.
            require(recorder.start(), "viewport capture did not start")
            capture_frame(recorder, backend, 0)
            for index in range(1, PRE_TICKS+1):
                hold = backend.advance_video_pre_action_tick()
                require(hold["root_state_write_count"] == 0, "pre-roll root write")
                jsonl(roll, {"kind": "pre_action", "global_tick": index, "task_credit": False,
                            "physical_hold": hold,
                            "raw_observation": measured_observation(backend.raw_observation)})
                if index % STRIDE == 0:
                    capture_frame(recorder, backend, index)
            if role == "A":
                observation, refreshed = core.refresh_after_video_pre_action_hold()
                require(backend._controller is legacy_controller,
                        "legacy controller was replaced during video pre-roll")
            else:
                observation, refreshed = refresh_semantic_core_after_preroll(core)
        require(core.frame.physics_tick == 0 and core.frame.state_id == "P01",
                "pre-roll did not preserve logical P01 tick zero")
        reset_info = dict(refreshed)
        physical = PhysicalEvaluationRecorder(root, task_spec_path=configs["task_spec_path"],
                                               quality_score_path=configs["quality_score_path"])
        physical.start(core.frame)  # Prefix ticks cannot grant lift/cross/task credit.
        if task_window:
            from .semantic_height_diagnostics import HeightDiagnostics
            height_diagnostics = HeightDiagnostics(root, backend)
            height_diagnostics.start(core.frame)
        if policy_loader is not None:
            action, load_provenance, check_model = policy_loader(tuple(observation))
            require(load_provenance.get("checkpoint_loaded_and_verified") is True,
                    "C source lacks actual checkpoint load proof")
        else:
            action = lambda _observation, _decision: ZERO12
        observer = EndpointObserver(physical, recorder, backend, task_window=task_window,
                                    height_diagnostics=height_diagnostics)
        if role == "A":
            core.tick_callback = observer
        else:
            core.tick_observer = observer
        while not core.done:
            before_tick = backend._episode_tick
            phase = core.frame.state_id
            raw = tuple(float(value) for value in action(observation, issued_decisions))
            require(len(raw) == 12 and all(math.isfinite(value) for value in raw),
                    "policy did not return finite Full12")
            if role in ("A", "B"):
                require(raw == ZERO12, "baseline role emitted residual")
            if role == "A":
                backend.set_actuator_target_audit_request(phase, raw,
                    core.phase_actions.mask_for(phase))
            issued_decisions += 1
            try:
                step = core.step(raw)
            except _PhysicalEndpoint:
                partial_ticks = backend._episode_tick-before_tick
                jsonl(decisions, {"decision": issued_decisions, "request_phase": phase,
                    "raw_policy_action_full12": raw, "start_tick": before_tick,
                    "end_tick": backend._episode_tick, "physics_ticks": partial_ticks,
                    "environment_step_returned": False, "stop_reason": observer.reason})
                break
            completed_decisions += 1
            observation = tuple(step.observation)
            jsonl(decisions, {"decision": issued_decisions, "request_phase": phase,
                "raw_policy_action_full12": raw, "start_tick": before_tick,
                "end_tick": backend._episode_tick,
                "physics_ticks": backend._episode_tick-before_tick,
                "environment_step_returned": True, "step_info": step.info})
        endpoint = observer.last_frame
        require(endpoint is not None, "video had no episode physics")
        if task_window and observer.last_captured_tick != endpoint.physics_tick:
            capture_task_interval_frame(recorder, backend, endpoint.physics_tick)
            observer.last_captured_tick = endpoint.physics_tick
        physical_summary = physical.summary()  # Metrics end before all post-roll.
        require(physical_summary["task_success"] is True, "episode did not meet common physical task")
        require(observer.reason == "SUCCESS", "video was stopped for another reason")
        require((task_interval_receipt(endpoint.physics_tick)["frame_count"] if task_window else
                 frames_for_episode(endpoint.physics_tick)) <= MAX_FRAMES,
                "complete source would exceed 200 seconds")
        for index in range(1, (0 if task_window else POST_TICKS)+1):
            row = common_post_success_tick(backend, physical.evaluator,
                episode_ticks=endpoint.physics_tick, post_index=index)
            global_tick = PRE_TICKS + endpoint.physics_tick + index
            jsonl(roll, {"kind": "post_success", "global_tick": global_tick, **row})
            if global_tick % STRIDE == 0:
                capture_frame(recorder, backend, global_tick)
        check_model()  # Actor/critic/optimizer/normalizer hashes must be unchanged.
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        if height_diagnostics is not None:
            height_diagnostic_receipt = height_diagnostics.close(
                None if observer is None else observer.last_frame)
        if physical is not None:
            if task_window and physical_summary is None:
                # Preserve measured task outcome even if encoding failed at the
                # terminal callback; video error still prevents publication.
                try:
                    physical_summary = physical.summary()
                except Exception as exc:
                    error = error or f"physical summary failed: {type(exc).__name__}: {exc}"
            physical.close()
        if roll is not None:
            roll.close()
        if decisions is not None:
            decisions.close()
        recorder_manifest = recorder.finalize()  # MUST precede SimulationApp.close.
    endpoint = None if observer is None else observer.last_frame
    if not recorder_manifest.get("valid"):
        error = error or "recorder/source full-decode failed"
    names = ("actual_viewport_video.mp4", "viewport_frame_ledger.jsonl",
             "viewport_buffer_video_manifest.json", "viewport_first_frame.png",
             "viewport_last_frame.png", "physical_video_roll_ticks.jsonl",
             "video_policy_decisions.jsonl", "physical_observations.jsonl",
             "native_tick_audit.jsonl", "stage_transition_evidence.jsonl",
             "phase_metrics.csv", "physics_quality_metrics.csv",
             "height_diagnostics_startup.json", "height_diagnostics.jsonl")
    payload = {"schema": "wlr50_clean.semantic_video_source.v1",
        "semantic_version": semantic_version,
        "evaluation_configuration": {name: file_record(path) for name, path in configs.items()},
        "pre_action_source": ("not_encoded_natural_reset" if task_window else
                              "existing_reset_settle_tail" if semantic_version == "v3" else "additional_zero_hold"),
        "extra_pre_action_physics_ticks": 0 if semantic_version == "v3" else PRE_TICKS,
        "settle_capture_evidence": settle_evidence,
        "role": role, "mode": ROLES[role], "seed": seed, "runtime_contract": dict(contract),
        "capture_process_id": os.getpid(), "capture_process_instance_id": uuid.uuid4().hex,
        "fresh_process_single_episode": True, "episode_count": 1,
        "from_phase": "P01", "pre_action_ticks": 0 if task_window else PRE_TICKS,
        "requested_post_success_ticks": 0 if task_window else POST_TICKS,
        "performed_post_success_ticks": backend._video_post_terminal_tick_count,
        "episode_physics_ticks": None if endpoint is None else endpoint.physics_tick,
        "issued_policy_decisions": issued_decisions,
        "completed_environment_steps": completed_decisions,
        "interrupted_final_decision_ticks": partial_ticks,
        "camera": {**CAMERA, "resolution": [1280,720], "fps": FPS},
        "reset_evidence": {key: reset_info.get(key) for key in
            ("reset_count", "reset_options", "training_phase_snapshot",
             "video_pre_action_refresh", "semantic_video_initialization")},
        "checkpoint_load_provenance": load_provenance,
        "physical_episode": physical_summary, "optimizer_updates": 0,
        "physical_task_success": (None if physical_summary is None else
                                  bool(physical_summary["task_success"])),
        "success_candidate": error is None,
        "diagnostic_only": error is not None, "improved_claim": False,
        "source_acceptance_error": error,
        "stitched": False, "frame_interpolation": False, "speed_modified": False,
        "artifacts": {name: file_record(root/name) for name in names if (root/name).is_file()}}
    if experiment_id is not None:
        payload["experiment_id"] = experiment_id
    if task_window:
        payload["natural_reset_proof"] = natural_reset_proof
        payload["height_diagnostics"] = height_diagnostic_receipt
    if task_window and endpoint is not None:
        payload["task_interval_window"] = task_interval_receipt(endpoint.physics_tick)
    write_json(root/"semantic_video_source_manifest.json", payload)
    return payload


def validate_video_configuration(source):
    version = source.get("semantic_version", "v2")
    require(source["runtime_contract"].get("semantic_version", "v2") == version,
            "source/runtime semantic version mismatch")
    experiment_id = source.get("experiment_id")
    configs = (video_configuration(version) if experiment_id is None else
               video_configuration(version, experiment_id=experiment_id))
    _validate_video_configuration_binding(configs, source["runtime_contract"], experiment_id=experiment_id)
    if version == "v3" or "evaluation_configuration" in source:
        require(source.get("evaluation_configuration") ==
                {name: file_record(path) for name, path in configs.items()},
                "video evaluator/configuration differs from captured bytes")
    return configs


def validate_existing_settle_evidence(source, pre):
    require(source["pre_action_source"] == "existing_reset_settle_tail"
            and source["extra_pre_action_physics_ticks"] == 0,
            "v3 video added physical warmup")
    expected = {"source": "existing_reset_settle_tail", "existing_settle_ticks": 180,
        "first_observed_dispatch_tick": 116, "last_observed_dispatch_tick": 179,
        "extra_physics_ticks": 0, "extra_sensor_reads": 0,
        "extra_controller_steps": 0, "extra_state_writes": 0, "task_credit": False}
    require(source["settle_capture_evidence"] == expected, "invalid existing settle evidence")
    require(len(pre) == PRE_TICKS, "incomplete existing settle tail")
    for index, row in enumerate(pre, 1):
        hold = row["physical_hold"]
        ack = hold["atomic_ack"]
        require(row["global_tick"] == index and row["source"] == expected["source"]
                and row["task_credit"] is False and row["sensor_observation_sampled"] is False
                and hold["existing_reset_dispatch"] is True
                and hold["root_state_write_count"] == 0
                and hold["physics_tick"] == ack["physics_tick"] == 115 + index
                and tuple(hold["applied_full12"]) == tuple(ack["applied_full12"]) == ZERO12
                and tuple(ack["drive_feedback_bias_requested_full12"]) == ZERO12,
                "settle tail is not the original zero dispatch sequence")


def validate_semantic_video_source(root, *, expected_role=None):
    """Independently replay physical task and bind source frames to native PTS."""
    root = Path(root).resolve(strict=True)
    source = json.loads((root/"semantic_video_source_manifest.json").read_text())
    require(source["schema"] == "wlr50_clean.semantic_video_source.v1", "source schema")
    require(source["role"] in ROLES and source["mode"] == ROLES[source["role"]], "source role")
    require(expected_role is None or source["role"] == expected_role, "role mismatch")
    require(source["success_candidate"] is True and source["diagnostic_only"] is False,
            "failed diagnostic is not publishable success")
    require(source["improved_claim"] is False and source["optimizer_updates"] == 0,
            "capture cannot establish PPO improvement or update weights")
    require(source["seed"] == 4001 and source["episode_count"] == 1
            and source["fresh_process_single_episode"] is True, "not fresh paired video source")
    require(source["camera"] == {**CAMERA, "resolution": [1280,720], "fps": FPS}, "camera mismatch")
    require(all(source[key] is False for key in ("stitched","frame_interpolation","speed_modified")),
            "source timeline was synthesized")
    task_window = source.get("experiment_id") == TASK_WINDOW_EXPERIMENT
    if task_window:
        validate_current_video_natural_reset(source)
        require(source.get("semantic_version") == "v3"
                and source["pre_action_source"] == "not_encoded_natural_reset"
                and source["pre_action_ticks"] == source["requested_post_success_ticks"]
                == source["performed_post_success_ticks"] == source["extra_pre_action_physics_ticks"] == 0
                and source["settle_capture_evidence"] is None,
                "current task window added physical or encoded context")
        require(source.get("physical_task_success") is True
                and source["physical_episode"]["task_success"] is True,
                "video candidate lacks independent physical task success")
    else:
        require(source["pre_action_ticks"] == PRE_TICKS
                and source["performed_post_success_ticks"] == POST_TICKS, "incomplete real context")
    paths = {name: inside(root, record) for name,record in source["artifacts"].items()}
    endpoint = source["episode_physics_ticks"]
    require(source_frame_count(source) <= MAX_FRAMES, "video context exceeds 200 s")
    from .semantic_cli import runtime_contract
    configs = validate_video_configuration(source)
    experiment_options = ({} if source.get("experiment_id") is None else
                          {"experiment_id": source["experiment_id"]})
    require(runtime_contract(expected_head=source["runtime_contract"]["source_git_commit"],
                             semantic_version=source.get("semantic_version", "v2"), **experiment_options)
            == source["runtime_contract"], "validator runtime/task spec differs from capture")
    if source["role"] == "C":
        require(source["checkpoint_load_provenance"]["checkpoint_loaded_and_verified"] is True,
                "C checkpoint load proof missing")
    evaluator = TaskEvaluator(configs["task_spec_path"])
    count = 0
    with paths["physical_observations.jsonl"].open(encoding="utf-8") as stream:
        for count,line in enumerate(stream, 1):
            raw = json.loads(line)
            require(raw["physics_tick"] == count-1, "raw episode tick gap")
            evaluation = evaluator.observe(raw)
            require(count-1 == endpoint or not evaluation["success"], "episode continued after first success")
    require(count == endpoint+1 and evaluation["success"] is True
            and evaluation["termination_reason"] is None, "independent task replay did not succeed")
    decision_rows = [json.loads(line) for line in paths["video_policy_decisions.jsonl"].read_text().splitlines()]
    require(len(decision_rows) == source["issued_policy_decisions"], "decision ledger count")
    expected_start = 0
    completed_steps = 0
    for index, row in enumerate(decision_rows, 1):
        require(row["decision"] == index and row["start_tick"] == expected_start,
                "policy decision clock gap")
        require(1 <= row["physics_ticks"] <= STRIDE
                and row["end_tick"]-row["start_tick"] == row["physics_ticks"],
                "bad partial decision duration")
        if row["environment_step_returned"]:
            completed_steps += 1
        else:
            require(index == len(decision_rows) and row["stop_reason"] == "SUCCESS",
                    "non-final interrupted decision")
        expected_start = row["end_tick"]
    require(expected_start == endpoint and completed_steps == source["completed_environment_steps"],
            "decision ledger endpoint mismatch")
    if task_window:
        expected_ends = [min((index+1)*STRIDE, endpoint) for index in range(source_frame_count(source))]
        require([row["end_tick"] for row in decision_rows] == expected_ends,
                "task frames must match actual executed policy interval endpoints")
    require(decision_rows[-1]["physics_ticks"] == source["interrupted_final_decision_ticks"]
            and not decision_rows[-1]["environment_step_returned"], "partial endpoint was hidden")
    count = 0
    with paths["native_tick_audit.jsonl"].open(encoding="utf-8") as stream:
        for count,line in enumerate(stream, 1):
            row = json.loads(line); audit = row["native_audit"]
            require(row["episode_physics_tick"] == count, "native audit gap")
            verified_native_effect({"actuator_target_effect_audit": audit},
                                   tuple(audit["raw_policy_action_full12"]))
            if source["role"] in ("A", "B"):
                require(tuple(audit["raw_policy_action_full12"]) == ZERO12
                        and tuple(row["projected_residual_full12"]) == ZERO12,
                        "baseline source contains a residual")
            require(all(row[key] == 0 for key in (
                "in_episode_root_pose_writes","in_episode_root_velocity_writes",
                "in_episode_force_or_impulse_writes","in_episode_gravity_writes")), "forbidden state write")
    require(count == endpoint, "native audit count mismatch")
    roll_rows = [json.loads(line) for line in paths["physical_video_roll_ticks.jsonl"].read_text().splitlines()]
    pre = [row for row in roll_rows if row["kind"] == "pre_action"]
    post = [row for row in roll_rows if row["kind"] == "post_success"]
    if task_window:
        require(not roll_rows, "current task window contains extra physical PRE/POST ticks")
    else:
        require([row["global_tick"] for row in pre] == list(range(1,PRE_TICKS+1)), "pre-roll tick gap")
        if source.get("semantic_version", "v2") == "v3":
            validate_existing_settle_evidence(source, pre)
        require([row["global_tick"] for row in post] ==
                list(range(PRE_TICKS+endpoint+1, PRE_TICKS+endpoint+POST_TICKS+1)), "post-roll tick gap")
    for row in pre:
        require(row["task_credit"] is False and row["physical_hold"]["root_state_write_count"] == 0
                and tuple(row["physical_hold"]["applied_full12"]) == ZERO12,
                "pre-roll was not a real zero-command no-credit hold")
    for row in post:
        require(row["task_credit"] is False and row["root_state_write_count"] == 0
                and row["controller_step_count"] == 0, "post-roll changed task/control history")
        require(evaluator.observe(row["raw_observation"])["success"] is True,
                "independent post-roll replay lost physical stability")
    decoded = decode_frame_timeline(paths["actual_viewport_video.mp4"])
    ledger = load_viewport_frame_ledger(paths["viewport_frame_ledger.jsonl"])
    require(len(ledger) == source_frame_count(source), "encoded frame count mismatch")
    if task_window:
        window = task_interval_action_window(source, decoded, ledger)
    else:
        require([row.sim_step for row in ledger] == [STRIDE*i for i in range(len(ledger))],
                "frames were dropped/duplicated or a partial terminal tick was mislabeled")
        require(all(abs(row.sim_time_s-row.sim_step/HZ)<1e-10 for row in ledger), "ledger clock drift")
        window = plan_action_window(decoded, ledger,
            semantic_start_sim_s=PRE_TICKS/HZ, semantic_end_sim_s=(PRE_TICKS+endpoint)/HZ,
            expected_fps=FPS, requested_pre_roll_s=PRE_TICKS/HZ,
            requested_post_roll_s=1., maximum_preserved_roll_s=2.)
    require(window.is_full_source, "source context was silently trimmed")
    validation = validate_mp4(paths["actual_viewport_video.mp4"],
        expected_fps=FPS, expected_frame_count=len(ledger), maximum_duration_s=200.,
        require_sane_container_duration=False)
    require(validation["valid"] is True, "raw source did not fully decode")
    return source, window


def publication_name_allowed(role, name):
    # A requested baseline name is allowed only AFTER source validation below.
    # A physically successful C video is still a candidate, never improvement.
    names = {"A": {"fsm_original_baseline.mp4", "semantic_A_success.mp4", "fsm_baseline_clean.mp4"},
             "B": {"semantic_B_success.mp4"},
             "C": {"semantic_C_candidate.mp4", "ppo_success_clean.mp4"}}
    return name in names.get(role, set())


def comparison_title(source):
    """Task outcome is not evidence of improved stability."""
    name = {"A": "FSM baseline", "B": "Semantic prior", "C": "PPO candidate"}[source["role"]]
    success = (source.get("success_candidate") is True
               and source.get("diagnostic_only") is False
               and (source.get("physical_episode") or {}).get("task_success") is True)
    return name + (" - task SUCCESS" if success else " - diagnostic NOT SUCCESS")


def comparison_filter(left, right):
    """No retiming: the shorter completed source visibly freezes at its end."""
    frame_counts = [source_frame_count(item) for item in (left, right)]
    total = max(frame_counts)
    require(total <= MAX_FRAMES, "comparison exceeds 200 seconds")
    parts = []
    for index, (source, count) in enumerate(zip((left, right), frame_counts)):
        branch = f"[{index}:v]setsar=1"
        if count < total:
            branch += f",tpad=stop_mode=clone:stop={total-count}"
        branch += f",drawtext=text='{comparison_title(source)}':x=28:y=26:fontcolor=white:fontsize=26"
        if count < total:
            branch += (",drawtext=text='SOURCE COMPLETED - last frame held':x=28:y=64:"
                       f"fontcolor=white:fontsize=22:enable='gte(n,{count})'")
        parts.append(branch + f"[side{index}]")
    return ";".join(parts + ["[side0][side1]hstack=inputs=2[out]"]), total


def publish_success_comparison(baseline_root, candidate_root, destination):
    """Publish common-condition actual A/C successes, with no improvement claim."""
    left, _ = validate_semantic_video_source(baseline_root, expected_role="A")
    right, _ = validate_semantic_video_source(candidate_root, expected_role="C")
    for key in ("seed", "camera", "runtime_contract", "semantic_version", "experiment_id",
                "pre_action_source", "extra_pre_action_physics_ticks"):
        require(left.get(key) == right.get(key), f"comparison condition differs: {key}")
    destination = Path(destination).resolve()
    require(destination.name == "fsm_vs_ppo_success.mp4" and not destination.exists(),
            "comparison must be a new task-success video, not an improved claim")
    destination.parent.mkdir(parents=True, exist_ok=True)
    filters, count = comparison_filter(left, right)
    command = [str(find_ffmpeg()), "-hide_banner", "-nostdin", "-v", "error", "-n"]
    for source in (left, right):
        command += ["-i", source["artifacts"]["actual_viewport_video.mp4"]["path"]]
    command += ["-filter_complex", filters, "-map", "[out]", "-an", "-sn", "-dn",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
                "-fps_mode", "passthrough", "-movflags", "+faststart", str(destination)]
    completed = subprocess.run(command, capture_output=True, text=True, errors="replace")
    require(completed.returncode == 0, completed.stderr[-2000:])
    validation = validate_mp4(destination, expected_fps=FPS, expected_frame_count=count,
        expected_width=2560, expected_height=720, maximum_duration_s=200.,
        require_sane_container_duration=True)
    require(validation["valid"] is True, "comparison decode/PTS validation failed")
    result = {"schema": "wlr50_clean.semantic_video_comparison.v1",
        "sources": [file_record(Path(root)/"semantic_video_source_manifest.json")
                    for root in (baseline_root, candidate_root)],
        "video": file_record(destination), "validation": validation,
        "titles": [comparison_title(item) for item in (left, right)],
        "ffmpeg_command": command, "speed_modified": False, "frame_interpolation": False,
        "shorter_source_suffix": "visibly_labelled_last_frame_hold", "improved_claim": False}
    write_json(destination.with_suffix(".manifest.json"), result)
    return result


def publish_success_source(source_root, destination):
    source, window = validate_semantic_video_source(source_root)
    destination = Path(destination).resolve()
    require(publication_name_allowed(source["role"], destination.name),
            "only validated baseline/candidate names are supported; improvement is a separate decision")
    require(not destination.exists(), "refusing to overwrite published video")
    destination.parent.mkdir(parents=True, exist_ok=True)
    raw = Path(source["artifacts"]["actual_viewport_video.mp4"]["path"])
    ffmpeg = find_ffmpeg()
    completed = subprocess.run([str(ffmpeg), "-hide_banner", "-nostdin", "-v", "error", "-n",
        "-copyts", "-start_at_zero", "-i", str(raw), "-map", "0:v:0", "-an", "-sn", "-dn",
        "-c:v", "copy", "-movflags", "+faststart", str(destination)],
        capture_output=True, text=True, errors="replace")
    require(completed.returncode == 0, completed.stderr[-2000:])
    validation = validate_mp4(destination, expected_fps=FPS,
        expected_frame_count=window.expected_frame_count, maximum_duration_s=200.,
        require_sane_container_duration=True)
    require(validation["valid"] is True, "published MP4 duration/codec/PTS failed")
    identity = verify_native_rate_output(decode_frame_timeline(destination), window,
        require_decoded_frame_identity=True, require_exact_pts_delta_identity=True)
    record = {"schema": "wlr50_clean.semantic_video_publication.v1",
        "source_manifest": file_record(Path(source_root)/"semantic_video_source_manifest.json"),
        "video": file_record(destination), "validation": validation,
        "source_identity": identity, "improved_claim": False, "single_episode": True,
        "publication_role": source["role"],
        "policy_claim_status": "candidate_not_improved" if source["role"]=="C" else "baseline_physical_success"}
    write_json(destination.with_suffix(".manifest.json"), record)
    return record
