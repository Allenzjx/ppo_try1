"""Offline contract regressions for the reviewed semantic vector draft.

Synthetic physical observations here are explicitly NOT live vector evidence.
"""
from __future__ import annotations

from types import SimpleNamespace
from dataclasses import replace

import pytest
torch = pytest.importorskip("torch")

from test_semantic_observation_reward_env import _frame, Backend, ZERO12
from wlr50_clean.infrastructure.command_batch import (
    FULL12_ORDER,SERVO_ORDER,resolve_joint_indices,
)
from wlr50_clean.infrastructure.servo_target_mapper import ServoTargetMapper
from wlr50_clean.ppo.actuator_target_effect import (
    actuator_target_audit_request,build_actuator_target_effect_audit,
)
from wlr50_clean.ppo.isaac_fsm_backend import build_residual_actuation_plan
from wlr50_clean.ppo.semantic_backend import DEFAULT_EXECUTION_PROFILE
from wlr50_clean.ppo.semantic_env import SemanticEpisodeEnv,WRITE_COUNTERS
from wlr50_clean.ppo.semantic_vector_backend import _SemanticBatchedCommandAdapter
from wlr50_clean.ppo.semantic_vector_env import SemanticVectorRslEnv
from wlr50_clean.ppo.semantic_vector_training import (
    construct_semantic_vector_runner,train_semantic_vector,
)
from wlr50_clean.ppo.termination import TerminationSignals


class FakeVectorBackend:
    """Pure CPU tensor/clock seam; no scene, no live capacity attestation."""
    num_envs = 8
    execution_profile_path = DEFAULT_EXECUTION_PROFILE
    def __init__(self, terminal_tick=None):
        self.terminal_tick = terminal_tick
        self.resets = 0
        self.global_ticks = self.writes = self.captures = 0
        self.controllers = tuple(object() for _ in range(8))
        self.readers = tuple(object() for _ in range(8))

    def _batch(self,frames):
        return SimpleNamespace(frames=tuple(frames),physics_tick=self.tick,
            global_physics_step_count=self.global_ticks,
            batched_articulation_write_count=self.writes,
            exact_pair_capture_count=self.captures)

    def reset_all(self, *, seeds):
        assert len(seeds)==8
        self.resets += 1
        self.tick = 0
        self.requests = None
        frames = [_frame() for _ in range(8)]
        self.frames = frames
        return self._batch(frames)

    def set_actuator_target_audit_requests(self, phases, raws, masks):
        self.requests = tuple(zip(phases,raws,masks,strict=True))

    def step_physics_batch(self, actions):
        self.tick += 1
        self.global_ticks += 1
        self.writes += 1
        self.captures += 1
        frames = []
        for index,(action,request) in enumerate(zip(actions,self.requests,strict=True)):
            phase,raw,mask = request
            current = _frame(self.tick)
            if index==0 and self.tick==self.terminal_tick:
                current.termination_signals = TerminationSignals(fall=True)
            current.info["drive_target_full12"] = tuple(action)
            current.info.update(dict.fromkeys(WRITE_COUNTERS,0))
            command_tick = self.global_ticks+300
            current.info["atomic_ack"] = {"physics_tick":command_tick}
            changed = tuple(x!=0 for x in action)
            current.info["actuator_target_effect_audit"] = {
                "schema":"wlr50_clean.actuator_target_effect_audit.v1",
                "verified":True,"actual_mapping_matches_dispatch":True,
                "setter_dispatch_targets_equal":True,"same_tick_counterfactual":True,
                "physics_tick":command_tick,"source_phase_id":phase,
                "policy_request_phase":phase,"raw_policy_action_full12":list(raw),
                "phase_mask_full12":list(mask),"changed_channels_full12":changed,
                "changed_target_channel_count":sum(changed),"target_dtype":"torch.float32",
                "fixture_only_not_live_evidence":True}
            frames.append(current)
        self.frames = frames
        return self._batch(frames)


def test_rows_match_real_single_semantic_kernel_without_advancing_eight_scenes():
    backend = FakeVectorBackend()
    env = SemanticVectorRslEnv(backend,device="cpu")
    actions = torch.full((8,12),.05)
    observation,rewards,dones,extras = env.step(actions)
    single = SemanticEpisodeEnv(Backend())
    single.reset()
    step = single.step(tuple(float(x) for x in actions[0]))
    assert observation["policy"].shape==(8,single.observation_dimension)
    assert observation["policy"][0].tolist()==pytest.approx(step.observation)
    assert rewards[0].item()==pytest.approx(step.reward)
    assert extras["semantic_decisions"][0]["reward"]==step.info["reward"]
    assert extras["semantic_decisions"][0]["applied_action_full12"]==step.info["applied_action_full12"]
    assert backend.global_ticks==backend.writes==backend.captures==8
    assert not bool(dones.any())
    assert len({id(row.builder) for row in env.rows})==8
    assert len({id(row.bridge) for row in env.rows})==8


def test_peer_bootstrap_uses_final_obs_before_reset_and_keeps_task_reward_separate():
    backend = FakeVectorBackend(terminal_tick=3)
    env = SemanticVectorRslEnv(backend,device="cpu")
    calls = []
    def critic(final_obs):
        calls.append((backend.resets,final_obs["policy"].clone()))
        return torch.arange(1,9,dtype=torch.float32).reshape(8,1)
    env.bind_final_value_function(critic,gamma=env.gamma)
    reset_observation = env.get_observations()["policy"].clone()
    obs,rewards,dones,extras = env.step(torch.zeros(8,12))
    assert calls[0][0]==1 and backend.resets==2
    assert bool(dones.all())
    assert extras["true_task_terminations"].tolist()==[True]+[False]*7
    assert extras["external_peer_reset_truncations"].tolist()==[False]+[True]*7
    assert not bool(extras["time_outs"].any())
    assert extras["peer_bootstrap_credit"][0]==0
    assert torch.allclose(extras["peer_bootstrap_credit"][1:],env.gamma*torch.arange(2,9))
    assert torch.allclose(rewards,extras["environment_rewards"]+extras["peer_bootstrap_credit"])
    assert torch.equal(obs["policy"],reset_observation)
    assert not torch.equal(calls[0][1],reset_observation)
    assert torch.equal(extras["terminal_observation"]["policy"],calls[0][1])
    assert extras["semantic_decisions"][0]["reward"]["potential_after"]==0
    assert extras["semantic_decisions"][1]["reward"]["potential_after"]>0
    assert extras["semantic_decisions"][1]["environment_reward"]==pytest.approx(extras["environment_rewards"][1].item())
    assert len(env.completed_episodes)==1 and len(env.peer_truncations)==7


def test_missing_final_value_binding_fails_before_reset():
    backend = FakeVectorBackend(terminal_tick=1)
    env = SemanticVectorRslEnv(backend,device="cpu")
    with pytest.raises(RuntimeError,match="bind live critic"):
        env.step(torch.zeros(8,12))
    assert backend.resets==1


class TensorRobot:
    def __init__(self):
        self.joint_names = FULL12_ORDER
        self.body_names = ("base_link",)
        self.data = SimpleNamespace(joint_pos=torch.zeros(8,12),joint_vel=torch.zeros(8,12),
            joint_pos_target=torch.zeros(8,12),joint_vel_target=torch.zeros(8,12))
        self._joint_pos_target_sim = torch.zeros(8,12)
        self._joint_vel_target_sim = torch.zeros(8,12)
        self.writes = 0
    def set_joint_position_target(self,values,*,joint_ids):
        self.data.joint_pos_target[:,joint_ids] = values
    def set_joint_velocity_target(self,values,*,joint_ids):
        self.data.joint_vel_target[:,joint_ids] = values
    def write_data_to_sim(self):
        self._joint_pos_target_sim.copy_(self.data.joint_pos_target)
        self._joint_vel_target_sim.copy_(self.data.joint_vel_target)
        self.writes += 1


def test_actual_batched_mapper_physical_conversion_and_native_audit_row_facade():
    # Bypass only the unrelated PhysX limit-install constructor in this CPU test.
    # The actual inherited batched apply, real mapper, sign/unit conversion and
    # production native-target helper all execute unchanged.
    adapter = object.__new__(_SemanticBatchedCommandAdapter)
    adapter.robot,adapter.num_envs = TensorRobot(),8
    adapter.origins = torch.zeros(8,3)
    adapter.joint_map = resolve_joint_indices(FULL12_ORDER)
    adapter._standing_servo = torch.zeros(8,8)
    adapter.standing_pose_deg = [dict.fromkeys(SERVO_ORDER,0.) for _ in range(8)]
    adapter.mappers = [ServoTargetMapper(standing,physics_dt_s=1/120) for standing in adapter.standing_pose_deg]
    adapter._final_drive = [dict.fromkeys(SERVO_ORDER,0.) for _ in range(8)]
    adapter.write_count,adapter._last_tick = 0,None
    actions = [ZERO12[:8]+(.01*(row+1),0.,0.,0.) for row in range(8)]
    plans = [build_residual_actuation_plan(action,frozen_nominal_full12=ZERO12,
        drive_feedback_bias_full12=ZERO12,normal_drive_bias_full12=ZERO12) for action in actions]
    ack = adapter.apply_batch((ZERO12,)*8,physics_tick=100,
        tracking_servo_names=((),)*8,
        drive_feedback_bias_full12=tuple(plan.combined_post_mapper_bias_full12 for plan in plans))
    for row,plan in enumerate(plans):
        audit = build_actuator_target_effect_audit(adapter=adapter.row_adapter(row),
            actuation=plan,raw_ack=ack.rows[row],previous_final_drive_servo_deg=(0.,)*8,
            source_phase_id="P01",policy_request=actuator_target_audit_request("P01",(.05,)*12,(1,)*12))
        assert audit["verified"] is True
        assert audit["changed_target_channel_count"]==1
        assert audit["actual_native_targets"]["wheel_velocity_rad_s"][0]==pytest.approx(-.01*(row+1))
        assert adapter.mappers[row]._feedback_tick==1
    assert adapter.robot.writes==adapter.write_count==1


def test_real_official_rsl_update_keeps_failed_rows_and_bootstrapped_peers(tmp_path):
    pytest.importorskip("rsl_rl")
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        env = SemanticVectorRslEnv(FakeVectorBackend(terminal_tick=3),device="cpu")
        runner,_ = construct_semantic_vector_runner(env,seed=1001,device="cpu")
        result = train_semantic_vector(runner,env,decisions=1024,run_dir=tmp_path/"run",
            output_root=tmp_path/"out",stage="smoke",contract={"revision":"offline_fixture"},seed=1001)
        assert result["actual_policy_decisions"]==1024
        assert result["ppo_updates_this_run"]==1
        assert result["optimizer_steps_this_run"]==20
        assert result["actor_parameter_sha256_before"]!=result["actor_parameter_sha256_after"]
        assert result["finite_nonzero_gradient_observed"] is True
        assert result["telemetry"]["completed_episode_count"]==128
        assert result["telemetry"]["external_peer_truncation_count"]==128*7
        storage = torch.load(tmp_path/"run"/"rollouts"/"rollout_000001.pt",weights_only=False)
        assert storage["actions"].shape==(128,8,12)
        assert bool(storage["dones"].all())
        assert torch.isfinite(storage["returns"]).all()
    finally:
        torch.set_num_threads(old_threads)


@pytest.mark.parametrize("decisions",[1,128,10000,1025])
def test_vector_total_budget_cannot_silently_round_a_tail(decisions):
    env = SimpleNamespace(num_envs=8,_final_value_function=lambda x:x)
    runner = SimpleNamespace(cfg={"num_steps_per_env":128})
    with pytest.raises(ValueError,match="exact multiple"):
        train_semantic_vector(runner,env,decisions=decisions)


def test_actual_semantic_batch_reset_step_and_second_reset_interfaces(monkeypatch):
    # Mock only scene construction/physical integration and reader acquisition.
    # Actual reset coordinator, semantic controller, sensor validation, frame
    # builder, mapper, native audit and semantic env all execute unchanged.
    from test_semantic_observation_reward_env import _raw
    from wlr50_clean.sensing.contact_classifier import SENSED_BODIES, DEFAULT_ROLE_MAP
    from wlr50_clean.sensing.observation import (
        BodyContactObservation, PairContactObservation, RigidBodyObservation, ContactClass,
    )
    from wlr50_clean.ppo import semantic_vector_backend as module
    from wlr50_clean.ppo.isaac_fsm_backend import DEFAULT_FSM_PATH, DEFAULT_MOTION_CONTRACT_PATH

    def complete_raw(tick, row):
        raw = _raw(tick,joint=0.)
        bodies,contacts = dict(raw.bodies),dict(raw.contacts)
        for name in SENSED_BODIES:
            if name not in bodies:
                bodies[name] = RigidBodyObservation(name,(0.,0.,.2),(1.,0.,0.,0.),(0.,)*3,(0.,)*3)
            if name not in contacts:
                def pair(kind):
                    return PairContactObservation(name,kind,False,(0.,)*3,0.,0.,None,
                        ((0.,)*3,)*8,(False,)*8,0,"offline_exact_fixture",True)
                contacts[name] = BodyContactObservation(name,DEFAULT_ROLE_MAP[name],ContactClass.AIR,
                    pair("ground"),pair("obstacle"))
        return replace(raw,bodies=bodies,contacts=contacts,
            center_of_mass=replace(raw.center_of_mass,included_bodies=tuple(bodies)))

    class Reader:
        def __init__(self,row):
            self.row = row
            self.contact_classifier,self.guard_tracker,self.body_collision_detector = object(),object(),object()
        def read(self,*,physics_tick,simulation_time_s,commanded_full12):
            assert simulation_time_s==physics_tick/120
            return replace(complete_raw(physics_tick,self.row),commanded_full12=tuple(commanded_full12))

    class ContactBank:
        capture_count = 0
        def reset(self):
            self.capture_count = 0
        def capture(self,tick):
            self.capture_count += 1

    backend = object.__new__(module.SemanticVectorIsaacBackend)
    backend.num_envs,backend.env_spacing_m = 8,8.
    backend.execution_profile_path = DEFAULT_EXECUTION_PROFILE
    backend.execution_profile = module.load_execution_profile(DEFAULT_EXECUTION_PROFILE)
    backend.fsm_path,backend.motion_contract_path = DEFAULT_FSM_PATH,DEFAULT_MOTION_CONTRACT_PATH
    backend.robot = TensorRobot()
    origins = torch.zeros(8,3)
    origins[:,0] = torch.arange(8)*8
    backend.scene = SimpleNamespace(env_origins=origins,update=lambda dt:None)
    backend.sim = SimpleNamespace(step=lambda *,render:None)
    backend.simulation_app = SimpleNamespace(is_running=lambda:True)
    backend.contact_bank = ContactBank()
    backend._canonical_reset_state = SimpleNamespace(state_sha256="offline_not_live")
    backend._reset_transaction_poisoned = False
    backend._reset_count,backend.global_physics_step_count = 0,0
    backend._prepare_physical_reset = lambda:{"fixture_only_not_live":True}
    backend._make_readers = lambda:tuple(Reader(row) for row in range(8))
    monkeypatch.setattr(module._BatchedCommandAdapter,"_install_limits",lambda self:None)

    env = SemanticVectorRslEnv(backend,device="cpu")
    old_controllers,old_readers = backend.controllers,backend.readers
    assert len({id(item.evaluator) for item in old_controllers})==8
    assert all(frame.info["level_calibration_sample_count"]==0 for frame in backend.frames)
    assert all(frame.info["semantic_task"]["completed_stage_ids"]==[] for frame in backend.frames)
    actions = torch.zeros(8,12)
    actions[:,8] = torch.arange(1,9)*.02
    obs,rewards,dones,extras = env.step(actions)
    assert not bool(dones.any())
    assert obs["policy"].shape==(8,env.schema.dimension)
    assert all(info["actuator_target_effect_audit_summary"]["all_ticks_verified"] for info in extras["semantic_decisions"])
    assert all(info["no_in_episode_state_writes_verified"] for info in extras["semantic_decisions"])
    effects = [info["actuator_target_effect_audit"]["actual_native_targets"]["wheel_velocity_rad_s"][0]
               for info in extras["semantic_decisions"]]
    assert len(set(effects))==8
    env._reset_all()
    assert all(new is not old for new,old in zip(backend.controllers,old_controllers,strict=True))
    assert all(new is not old for new,old in zip(backend.readers,old_readers,strict=True))
    assert all(frame.info["semantic_task"]["completed_stage_ids"]==[] for frame in backend.frames)
    assert backend._policy_requests==(None,)*8


def test_read_only_native_row_facade_cannot_mix_or_write_peer_target_rows():
    from wlr50_clean.ppo.semantic_vector_backend import _RowTargetRobot
    robot = TensorRobot()
    for row in range(8):
        robot.data.joint_pos_target[row,:] = row+1
        robot._joint_pos_target_sim[row,:] = row+1
    facade = _RowTargetRobot(robot,3,torch.zeros(3),8)
    assert facade.data.joint_pos_target.shape==(1,12)
    assert facade._joint_pos_target_sim.shape==(1,12)
    assert bool((facade.data.joint_pos_target==4).all())
    assert bool((facade._joint_pos_target_sim==4).all())
    with pytest.raises(AttributeError):
        facade.write_data_to_sim()
