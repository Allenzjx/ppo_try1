"""Reuse the real no-credit prefix boundary; no changed nominal or snapshots."""
from types import SimpleNamespace

import pytest
import torch

from test_semantic_checkpoint_prefix import Core, ZERO
from wlr50_clean.ppo.semantic_checkpoint_prefix import (
    CheckpointPolicyPrefixRequest as Request, CheckpointPolicyPrefixRslAdapter as Adapter,
    sampling_label,
)
from wlr50_clean.ppo.semantic_migration import continuation_topology
from wlr50_clean.ppo.semantic_transfer_roles import ROLE_OBSERVATION_LAYOUT


class Core372(Core):
    observation_dimension = 372
    observation_schema = SimpleNamespace(transfer_role_features_version=ROLE_OBSERVATION_LAYOUT)

    def reset(self, *args, **kwargs):
        super().reset(*args, **kwargs)
        self.observation = (0.0,)*372
        return self.observation

    def step(self, raw):
        from dataclasses import replace
        result = super().step(raw)
        self.observation = tuple(result.observation)+(0.0,)*(372-len(result.observation))
        return replace(result, observation=self.observation)


def binding():
    return {"schema": "wlr50_clean.successful_nominal_prefix.v1", "source": "successful_nominal",
        "execution_profile_sha256": "a"*64, "stage_task_spec_sha256": "b"*64,
        "runtime_content_sha256": "c"*64, "raw_action_full12": [0.0]*12, "policy_credit": False,
        "interface_contract": {"observation_dimension": 372,
            "observation_layout": ROLE_OBSERVATION_LAYOUT, "action_dimension": 12}}


def test_nominal_rollin_keeps_controller_history_and_excludes_prefix_credit():
    core = Core372(plans=[[("P06", 8, None), ("P07", 8, None)]])
    records = []
    request = Request(target_phase="P06", source="successful_nominal")
    env = Adapter(core, seed=1001, device="cpu", evidence_sink=records.append, request=request)
    identities = (core.bridge, core.mapper, core.nominal, core.contacts)
    env.install_prefix_policy(lambda obs: ZERO, binding())
    assert core.actions == [ZERO] and env.total_decisions == 0
    assert identities == (core.bridge, core.mapper, core.nominal, core.contacts)
    assert env.core.start_record["mode"] == "successful_nominal_initialized_suffix"
    _, _, dones, extras = env.step(torch.zeros(1, 12))
    assert not dones.any() and env.total_decisions == 1
    assert extras["semantic_decisions"][0]["task_result_scope"] == "successful_nominal_initialized_suffix"
    assert continuation_topology(sampling_label(request), request.as_dict(),
        observation_layout=ROLE_OBSERVATION_LAYOUT)["phase_suffix_curriculum_implemented"]


def test_nominal_prefix_rejects_nonzero_hidden_actor():
    env = Adapter(Core372(), seed=1001, device="cpu", evidence_sink=lambda row: None,
        request=Request(source="successful_nominal"))
    with pytest.raises(RuntimeError, match="nonzero residual"):
        env.install_prefix_policy(lambda obs: (0.1,)*12, binding())


def test_formal_eval_cannot_request_nominal_prefix(tmp_path):
    from wlr50_clean.ppo import semantic_cli as cli
    args = cli.parser().parse_args(["eval", "--run-dir", str(cli.PROJECT_ROOT / "runs/ppo_task_first_recovery_v1/test"),
        "--expected-head", "a"*40, "--semantic-version", "v3", "--experiment-id", "task_first_recovery_v1",
        "--prefix-source", "successful_nominal", "--checkpoint", "unused.pt", "--seed", "4001"])
    with pytest.raises(ValueError, match="prefix requires"):
        cli.validate_request(args)
