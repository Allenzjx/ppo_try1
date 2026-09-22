"""CPU-only official orchestration tests, not physical or deployed PPO credit."""
from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest
import torch

from receiving_test_bootstrap import ROOT, HERE, MODULES
from wlr50_clean.ppo.semantic_return_profile import RETURN_PROFILE
from wlr50_clean.ppo.rl_library_wrapper import construct_runner
from wlr50_clean.ppo import semantic_history_actor as old_actor
from tensordict import TensorDict

p, t, m, c, prefix = (MODULES[k] for k in ("semantic_policy_distribution", "semantic_training",
    "semantic_migration", "semantic_cli", "semantic_checkpoint_prefix_policy"))
profile = MODULES["semantic_receiving_wheel_profile"]
new_actor = MODULES["semantic_receiving_wheel_sigma"]
OLD, NEW = p.TASK_CONDITIONED_HIP_WHEEL_POLICY, profile.RECEIVING_WHEEL_POLICY
LAYOUT = "diagonal_transfer_state_v1"


@pytest.fixture(autouse=True)
def cpu_rng():
    rng, threads = torch.get_rng_state(), torch.get_num_threads()
    torch.manual_seed(1221); torch.set_num_threads(1)
    yield
    torch.set_rng_state(rng); torch.set_num_threads(threads)


def values(phase=10, raw=None):
    x = torch.zeros(372); x[phase] = 1; x[20] = .02
    x[158:158+phase] = 1; x[157] = 1
    if raw is not None: x[195:207] = torch.as_tensor(raw).clamp(-19., 19.)
    return x


class CPUCore:
    def __init__(self): self.frame = SimpleNamespace(sim_time_s=0.); self.calls = 0
    def reset(self, *, seed=1001, options=None):
        self.tick = 0; self.frame.sim_time_s = 0.; return tuple(values().tolist())
    def step(self, raw):
        self.tick += 1; self.calls += 1; self.frame.sim_time_s = self.tick / 15
        end = self.tick == 17
        return SimpleNamespace(observation=tuple(values(10, raw).tolist()), reward=1+.1*raw[9]-.03*raw[11]**2,
            terminated=end, truncated=False, info={"phase_id":"P11", "raw_policy_action_full12":raw,
            "applied_action_full12":tuple(.1*v for v in raw), "task_success":False,
            "termination_reason":"CPU_SYNTHETIC_ONLY" if end else None,
            "actuator_target_effect_audit":{"schema":"wlr50_clean.actuator_target_effect_audit.v1",
                "verified":True,"actual_mapping_matches_dispatch":True,"setter_dispatch_targets_equal":True,
                "same_tick_counterfactual":True,"raw_policy_action_full12":raw,"target_dtype":"torch.float32",
                "changed_target_channel_count":12}})
    def telemetry_summary(self): return {"CPU_SYNTHETIC_NOT_PHYSICS":True,"calls":self.calls}


def runner(version=NEW):
    env = t.SemanticRslAdapter(CPUCore(), seed=1001, device="cpu"); env.cfg["semantic_version"]="v3"
    cfg = t.semantic_runner_config(seed=1001, device="cpu", semantic_version="v3",
        policy_version=version, observation_layout=LAYOUT, return_profile=RETURN_PROFILE)
    r = construct_runner(env, cfg, log_dir=None); r.logger.writer=None
    r._semantic_policy_version=version; r._semantic_observation_layout=LAYOUT
    r._semantic_version="v3"; r._semantic_runner_config=copy.deepcopy(cfg)
    return r, env


def minimal_contract():
    return {"CPU_SYNTHETIC_ONLY_NOT_ADOPTABLE_RUNTIME":True,
        "experiment_id":"task_conditioned_hip_wheel_v1", "semantic_version":"v3",
        "training_budgets":t.training_quantity_budgets("task_conditioned_hip_wheel_v1")}


def test_registered_metadata_and_requested_actual_sigma_share_exact_profile():
    r, env = runner()
    metadata = {"runner_config":r._semantic_runner_config,"policy_contract":t._runner_policy_contract(r),
                "seed":1001,"semantic_version":"v3","policy_version":NEW}
    assert p.policy_version_from_metadata(metadata)==NEW
    assert p.supported_heteroscedastic_contract_version(metadata["policy_contract"])==NEW
    observation=env.get_observations()
    raw,audit=t.audited_history_policy_request(r.alg.actor,observation,
        lambda:r.alg.actor(observation,stochastic_output=True),stochastic=True)
    assert audit["policy_version"]==NEW and audit["sampling_draws"]==1
    assert audit["extra_model_forwards"]==0 and audit["extra_random_draws"]==0
    assert audit["receiving_sigma_multiplier_full12"]==[1.]*9+[3.,1.,3.]
    assert torch.equal(torch.tensor(audit["effective_sigma_full12"]),r.alg.actor.output_std[0])
    assert audit["selected_raw_log_probability"]==r.alg.actor.get_output_log_prob(raw)[0].item()
    cached=tuple(x.clone() for x in r.alg.actor.output_distribution_params); rng=torch.get_rng_state()
    _,det=t.audited_history_policy_request(r.alg.actor,observation,lambda:r.alg.actor(observation),stochastic=False)
    assert det["effective_sigma_full12"]==audit["effective_sigma_full12"]
    assert torch.equal(rng,torch.get_rng_state()) and all(torch.equal(a,b) for a,b in zip(cached,r.alg.actor.output_distribution_params))
    for key in ("sigma_scaling_semantics","receiving_continuation_gate"):
        bad=copy.deepcopy(metadata); bad["policy_contract"][key]="tamper"
        with pytest.raises(ValueError): p.policy_version_from_metadata(bad)


def test_official_128_update_store_current_likelihood_save_exact_reload(tmp_path):
    r,env=runner(); contract=minimal_contract()
    result=t.train_semantic(r,env,run_dir=tmp_path/"run",output_root=tmp_path/"out",stage="full_episode",
        decisions=128,contract=contract,seed=1001)
    assert result["ppo_updates_this_run"]==1 and result["optimizer_steps_this_run"]==20
    rows=[json.loads(s) for s in (tmp_path/"run/residual_and_projection_audit.jsonl").read_text().splitlines()]
    saved=torch.load(tmp_path/"run/rollouts/rollout_000001.pt",map_location="cpu",weights_only=False)
    for i,row in enumerate(rows):
        req=row["policy_request"]
        assert req["policy_version"]==NEW
        assert torch.equal(torch.tensor(req["selected_raw_full12"]),saved["actions"][i,0])
        assert torch.equal(torch.tensor(req["effective_sigma_full12"]),saved["distribution_params"][1][i,0])
        assert req["selected_raw_log_probability"]==saved["actions_log_prob"][i,0,0].item()
    likelihood=json.loads((tmp_path/"run/rollouts/update_000001_likelihood.json").read_text())
    first=likelihood["minibatches"][0]
    assert first["sigma_source"]=="current_official_Gaussian_cache_after_B_over_cap_and_receiving_FR_RR_sigma_x3"
    assert "current_network_mean_full12" in first
    assert max(abs(v-1) for v in first["ratio"])<1e-5
    checkpoint=tmp_path/"out/checkpoints/history/checkpoint_step_000000128.pt"
    restored,_=runner()
    infos=t.load_semantic_checkpoint(restored,checkpoint,contract=contract,seed=1001)
    assert infos["global_policy_decisions"]==128 and infos["ppo_updates"]==1
    assert t.state_hash(r.alg.actor.state_dict())==t.state_hash(restored.alg.actor.state_dict())
    assert t.state_hash(r.alg.optimizer.state_dict())==t.state_hash(restored.alg.optimizer.state_dict())
    assert r.alg.learning_rate==restored.alg.learning_rate
    assert restored.alg.storage.step==0 and restored.alg.transition.actions is None
